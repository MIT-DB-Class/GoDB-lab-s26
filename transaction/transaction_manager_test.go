package transaction

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/logging"
	"mit.edu/dsg/godb/storage"
)

// setupTest creates a TransactionManager wired to a fresh BufferPool in a temp directory.
func setupTest(t *testing.T, lm storage.LogManager) (*TransactionManager, *LockManager, *storage.BufferPool) {
	t.Helper()
	bp := storage.NewBufferPool(100, storage.NewDiskStorageManager(t.TempDir()), lm)
	lockMgr := NewLockManager()
	return NewTransactionManager(lm, bp, lockMgr), lockMgr, bp
}

// TestTxnManager_Basic_Commit covers the full Begin→Commit lifecycle:
//   - TIDs are unique and monotonically increasing across successive Begin calls.
//   - Begin appends exactly one LogBeginTransaction record with the correct TxnID.
//   - Commit appends a LogCommit record after the LogBeginTransaction.
//   - All held locks are released after Commit.
//   - SS2PL: locks are not released before the commit record is durable.
func TestTxnManager_Basic_Commit(t *testing.T) {
	mlm := logging.NewMemoryLogManager()
	mlm.SetFlushOnAppend(true)
	tm, lockMgr, _ := setupTest(t, mlm)
	tag := NewTableLockTag(common.ObjectID(1))

	seen := make(map[common.TransactionID]bool)
	var prevID common.TransactionID

	for i := 0; i < 5; i++ {
		prevCount := mlm.Count()

		ctx, err := tm.Begin()
		require.NoError(t, err)
		require.NotNil(t, ctx)

		// TID must be unique and strictly greater than the previous one.
		tid := ctx.id
		assert.Greater(t, uint64(tid), uint64(prevID), "TID must increase monotonically")
		assert.False(t, seen[tid], "TID must be unique")
		seen[tid] = true
		prevID = tid

		// Begin must append exactly one LogBeginTransaction with the correct TxnID.
		require.Equal(t, prevCount+1, mlm.Count(), "Begin must append exactly 1 log record")
		rec := mlm.GetRecord(prevCount)
		assert.Equal(t, storage.LogBeginTransaction, rec.RecordType())
		assert.Equal(t, tid, rec.TxnID(), "logged TxnID must match context ID")

		// Acquire a lock so we can verify it is released on Commit.
		require.NoError(t, ctx.AcquireLock(tag, LockModeX))
		assert.True(t, lockMgr.LockHeld(tag), "lock should be held before Commit")

		require.NoError(t, tm.Commit(ctx))

		// Commit must append a LogCommit record immediately after the LogBeginTransaction.
		require.Equal(t, prevCount+2, mlm.Count(), "Begin+Commit must produce exactly 2 log records")
		assert.Equal(t, storage.LogBeginTransaction, mlm.GetRecord(prevCount).RecordType())
		assert.Equal(t, storage.LogCommit, mlm.GetRecord(prevCount+1).RecordType())
		commitRec := mlm.GetRecord(prevCount + 1)
		assert.Equal(t, tid, commitRec.TxnID(), "LogCommit must carry the correct TxnID")

		// All locks must be released after Commit.
		assert.False(t, lockMgr.LockHeld(tag), "lock must be released after Commit")
	}
}

// TestTxnManager_Basic_GroupCommit checks that a single flush unblocks multiple concurrent commits simultaneously.
func TestTxnManager_Basic_GroupCommit(t *testing.T) {
	{
		mlm2 := logging.NewMemoryLogManager()
		tm2, lockMgr2, _ := setupTest(t, mlm2)

		const numConcurrent = 5
		tags2 := make([]DBLockTag, numConcurrent)
		ctxs := make([]*TransactionContext, numConcurrent)
		for i := range ctxs {
			tags2[i] = NewTableLockTag(common.ObjectID(i + 100))
			ctx2, err := tm2.Begin()
			require.NoError(t, err)
			require.NoError(t, ctx2.AcquireLock(tags2[i], LockModeX))
			ctxs[i] = ctx2
		}

		// Start all commits concurrently; they all block in WaitUntilFlushed.
		done := make(chan struct{}, numConcurrent)
		for _, c := range ctxs {
			go func(c *TransactionContext) {
				defer func() { done <- struct{}{} }()
				assert.NoError(t, tm2.Commit(c))
			}(c)
		}

		// Give goroutines time to reach WaitUntilFlushed.
		time.Sleep(20 * time.Millisecond)

		// All locks must still be held while flush is blocked (SS2PL).
		for i, tag2 := range tags2 {
			assert.True(t, lockMgr2.LockHeld(tag2), "lock %d must be held while waiting for flush", i)
		}

		// Advance flushedUntil past any LSN the TM could have appended. This single flush
		// call unblocks all concurrent commits simultaneously — group commit.
		mlm2.SetFlushedLSN(storage.LSN(1 << 20))
		for range ctxs {
			<-done
		}

		// All locks released after group commit completes.
		for i, tag2 := range tags2 {
			assert.False(t, lockMgr2.LockHeld(tag2), "lock %d must be released after group commit", i)
		}
	}
}

// TestTxnManager_Basic_Commit_FlushFailure verifies that a failed WAL append causes
// Commit to return an error.
func TestTxnManager_Basic_Commit_FlushFailure(t *testing.T) {
	mlm := logging.NewMemoryLogManager()
	tm, _, _ := setupTest(t, mlm)

	ctx, err := tm.Begin()
	require.NoError(t, err)

	// Inject an append failure; the next Append call (the commit record) will fail.
	mlm.SetAppendErrorAfterN(0, errors.New("injected WAL failure"))

	err = tm.Commit(ctx)
	assert.Error(t, err, "Commit must return an error when the WAL append fails")
}

// TestTxnManager_Basic_Abort_NoUndo verifies that Abort() with no undoable records logs
// LogAbort and releases all held locks.
func TestTxnManager_Basic_Abort_NoUndo(t *testing.T) {
	mlm := logging.NewMemoryLogManager()
	mlm.SetFlushOnAppend(true)
	tm, lockMgr, _ := setupTest(t, mlm)
	tag := NewTableLockTag(common.ObjectID(1))

	ctx, err := tm.Begin()
	require.NoError(t, err)
	tid := ctx.id
	require.NoError(t, ctx.AcquireLock(tag, LockModeX))
	assert.True(t, lockMgr.LockHeld(tag))

	require.NoError(t, tm.Abort(ctx))

	require.Equal(t, 2, mlm.Count(), "Begin+Abort should produce 2 log records")
	assert.Equal(t, storage.LogBeginTransaction, mlm.GetRecord(0).RecordType())
	assert.Equal(t, storage.LogAbort, mlm.GetRecord(1).RecordType())
	abortRec := mlm.GetRecord(1)
	assert.Equal(t, tid, abortRec.TxnID(), "LogAbort must carry the correct TxnID")
	assert.False(t, lockMgr.LockHeld(tag), "lock must be released after Abort")
}

// TestTxnManager_Basic_TID_AcrossAbort verifies that TIDs are strictly increasing and
// unique regardless of whether transactions end via Commit or Abort.
func TestTxnManager_Basic_TID_AcrossAbort(t *testing.T) {
	mlm := logging.NewMemoryLogManager()
	mlm.SetFlushOnAppend(true)
	tm, _, _ := setupTest(t, mlm)

	seen := make(map[common.TransactionID]bool)
	var prevID common.TransactionID

	for i := 0; i < 8; i++ {
		ctx, err := tm.Begin()
		require.NoError(t, err)

		assert.Greater(t, uint64(ctx.id), uint64(prevID), "TID must increase monotonically (i=%d)", i)
		assert.False(t, seen[ctx.id], "TID must be unique (i=%d)", i)
		seen[ctx.id] = true
		prevID = ctx.id

		if i%2 == 0 {
			require.NoError(t, tm.Abort(ctx))
		} else {
			require.NoError(t, tm.Commit(ctx))
		}
	}
}
