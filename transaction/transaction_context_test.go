package transaction

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/storage"
)

// newTestContext builds a bare TransactionContext for unit tests (bypassing the TM pool).
func newTestContext(lm *LockManager) *TransactionContext {
	return &TransactionContext{
		id:         common.TransactionID(1),
		lm:         lm,
		logRecords: newLogRecordBuffer(),
		heldLocks:  make(map[DBLockTag]DBLockMode),
	}
}

// TestLogRecordBuffer_Basic_AllocAndLen verifies that len() correctly tracks the number of
// allocations across Insert, Delete, and Update records, and that each allocated record can
// be read back with the correct type, TID, and RID immediately after allocation.
func TestLogRecordBuffer_Basic_AllocAndLen(t *testing.T) {
	buf := newLogRecordBuffer()
	assert.Equal(t, 0, buf.len(), "empty buffer should have len 0")

	tid := common.TransactionID(1)
	rowData := make([]byte, common.IntSize)

	type step struct {
		rid        common.RecordID
		recordType storage.LogRecordType
	}
	var steps []step

	// Interleave Insert, Delete, and Update to verify mixed-type length tracking.
	for i := 0; i < 100; i++ {
		rid := common.RecordID{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: int32(i)}
		var rt storage.LogRecordType
		switch i % 3 {
		case 0:
			sz := storage.InsertRecordSize(rowData)
			b := buf.allocate(sz)
			storage.NewInsertRecord(b, tid, rid, rowData)
			rt = storage.LogInsert
		case 1:
			sz := storage.DeleteRecordSize()
			b := buf.allocate(sz)
			storage.NewDeleteRecord(b, tid, rid)
			rt = storage.LogDelete
		default:
			sz := storage.UpdateRecordSize(rowData, rowData)
			b := buf.allocate(sz)
			storage.NewUpdateRecord(b, tid, rid, rowData, rowData)
			rt = storage.LogUpdate
		}
		steps = append(steps, step{rid, rt})
		assert.Equal(t, i+1, buf.len(), "len should equal number of allocations after step %d", i+1)

		// Verify the just-written record is immediately readable.
		r := buf.get(i)
		assert.Equal(t, rt, r.RecordType(), "step %d: wrong record type", i)
		assert.Equal(t, rid.Slot, r.RID().Slot, "step %d: wrong slot", i)
		assert.Equal(t, tid, r.TxnID(), "step %d: wrong TxnID", i)
	}

	// All previously allocated records must still be readable.
	for i, s := range steps {
		r := buf.get(i)
		assert.Equal(t, s.recordType, r.RecordType(), "final scan record %d: wrong type", i)
		assert.Equal(t, s.rid.Slot, r.RID().Slot, "final scan record %d: wrong slot", i)
	}
}

// TestLogRecordBuffer_Basic_Pop verifies that pop() removes the last record without affecting others.
func TestLogRecordBuffer_Basic_Pop(t *testing.T) {
	buf := newLogRecordBuffer()
	rowData := make([]byte, common.IntSize)
	tid := common.TransactionID(7)

	rids := []common.RecordID{
		{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: 10},
		{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: 20},
		{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: 30},
	}
	for _, rid := range rids {
		sz := storage.InsertRecordSize(rowData)
		b := buf.allocate(sz)
		storage.NewInsertRecord(b, tid, rid, rowData)
	}
	require.Equal(t, 3, buf.len())

	buf.pop()
	assert.Equal(t, 2, buf.len(), "after one pop")
	assert.Equal(t, int32(10), buf.get(0).RID().Slot, "record 0 unchanged after pop")
	assert.Equal(t, int32(20), buf.get(1).RID().Slot, "record 1 unchanged after pop")

	buf.pop()
	assert.Equal(t, 1, buf.len(), "after two pops")
	assert.Equal(t, int32(10), buf.get(0).RID().Slot, "record 0 still intact")
}

// TestLogRecordBuffer_Basic_Reset verifies that reset() clears all records and the buffer is reusable.
func TestLogRecordBuffer_Basic_Reset(t *testing.T) {
	buf := newLogRecordBuffer()
	rowData := make([]byte, common.IntSize)
	tid := common.TransactionID(3)
	rid := common.RecordID{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: 5}

	for i := 0; i < 5; i++ {
		sz := storage.InsertRecordSize(rowData)
		b := buf.allocate(sz)
		storage.NewInsertRecord(b, tid, rid, rowData)
	}
	require.Equal(t, 5, buf.len())

	buf.reset()
	assert.Equal(t, 0, buf.len(), "len should be 0 after reset")

	// Re-use after reset: should work without panicking.
	sz := storage.InsertRecordSize(rowData)
	b := buf.allocate(sz)
	storage.NewInsertRecord(b, tid, rid, rowData)
	assert.Equal(t, 1, buf.len(), "can allocate again after reset")
	assert.Equal(t, int32(5), buf.get(0).RID().Slot)
}

// TestLogRecordBuffer_Stress_RandomOps exercises the log buffer under a random sequence of
// allocate, pop, and reset operations. Each result is verified against a simple reference
// implementation (a slice of slot+type pairs).
func TestLogRecordBuffer_Stress_RandomOps(t *testing.T) {
	type refEntry struct {
		slot       int32
		recordType storage.LogRecordType
	}

	rng := rand.New(rand.NewSource(42))
	buf := newLogRecordBuffer()
	var ref []refEntry

	tid := common.TransactionID(1)
	row := make([]byte, common.IntSize)

	const nOps = 100000
	slotCounter := int32(0)
	for op := 0; op < nOps; op++ {
		action := rng.Intn(10)
		switch {
		case len(ref) == 0 || action < 6: // ~60% allocate; always allocate when empty
			slotCounter++
			rid := common.RecordID{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: slotCounter}
			var rt storage.LogRecordType
			switch rng.Intn(3) {
			case 0:
				sz := storage.InsertRecordSize(row)
				b := buf.allocate(sz)
				storage.NewInsertRecord(b, tid, rid, row)
				rt = storage.LogInsert
			case 1:
				sz := storage.DeleteRecordSize()
				b := buf.allocate(sz)
				storage.NewDeleteRecord(b, tid, rid)
				rt = storage.LogDelete
			default:
				sz := storage.UpdateRecordSize(row, row)
				b := buf.allocate(sz)
				storage.NewUpdateRecord(b, tid, rid, row, row)
				rt = storage.LogUpdate
			}
			ref = append(ref, refEntry{slotCounter, rt})
		case action < 9: // ~30% pop
			buf.pop()
			ref = ref[:len(ref)-1]
		default: // ~10% reset
			buf.reset()
			ref = ref[:0]
		}

		require.Equal(t, len(ref), buf.len(), "op %d: len mismatch", op)

		// Spot-check the last and one random entry after every operation.
		n := buf.len()
		if n > 0 {
			last := n - 1
			r := buf.get(last)
			assert.Equal(t, ref[last].recordType, r.RecordType(), "op %d: last record type mismatch", op)
			assert.Equal(t, ref[last].slot, r.RID().Slot, "op %d: last record slot mismatch", op)

			if n > 1 {
				idx := rng.Intn(n - 1)
				r = buf.get(idx)
				assert.Equal(t, ref[idx].recordType, r.RecordType(), "op %d: spot record %d type mismatch", op, idx)
				assert.Equal(t, ref[idx].slot, r.RID().Slot, "op %d: spot record %d slot mismatch", op, idx)
			}
		}
	}

	// Final full scan: every entry in the buffer must match the reference.
	require.Equal(t, len(ref), buf.len())
	for i, e := range ref {
		r := buf.get(i)
		assert.Equal(t, e.recordType, r.RecordType(), "final record %d: wrong type", i)
		assert.Equal(t, e.slot, r.RID().Slot, "final record %d: wrong slot", i)
	}
}

// TestTxnContext_Reentrant_SameMode verifies that re-acquiring the same lock in the same mode
// is a no-op: no error and the mode in heldLocks is unchanged.
func TestTxnContext_Reentrant_SameMode(t *testing.T) {
	// All remaining lock modes — each reentrant same-mode acquire must be a no-op.
	for _, mode := range []DBLockMode{LockModeS, LockModeIS, LockModeIX, LockModeSIX, LockModeX} {
		lm := NewLockManager()
		ctx := newTestContext(lm)
		tag := NewTableLockTag(common.ObjectID(100))
		require.NoError(t, ctx.AcquireLock(tag, mode))
		gotMode, _ := ctx.HeldLock(tag)
		assert.Equal(t, mode, gotMode)

		// Re-acquire: should be a no-op.
		require.NoError(t, ctx.AcquireLock(tag, mode))
		gotMode, _ = ctx.HeldLock(tag)
		assert.Equal(t, mode, gotMode, "mode should not change on reentrant same-mode acquire")
		assert.True(t, lm.LockHeld(tag))
	}
}

// TestTxnContext_Reentrant_CoveredMode verifies that acquiring a weaker mode when a stronger
// mode is already held is a no-op: the held mode is NOT downgraded.
func TestTxnContext_Reentrant_CoveredMode(t *testing.T) {
	// All non-identity covered pairs: CoveredBy(req, held)=true but req != held.
	for _, p := range []struct{ req, held DBLockMode }{
		{LockModeS, LockModeSIX},
		{LockModeS, LockModeX},
		{LockModeIS, LockModeS},
		{LockModeIS, LockModeX},
		{LockModeIS, LockModeIX},
		{LockModeIS, LockModeSIX},
		{LockModeIX, LockModeX},
		{LockModeIX, LockModeSIX},
		{LockModeSIX, LockModeX},
	} {
		lm := NewLockManager()
		ctx := newTestContext(lm)
		tag := NewTableLockTag(common.ObjectID(200))
		require.NoError(t, ctx.AcquireLock(tag, p.held))
		require.NoError(t, ctx.AcquireLock(tag, p.req))
		gotMode, _ := ctx.HeldLock(tag)
		assert.Equal(t, p.held, gotMode,
			"held=%s covers req=%s: mode must not be downgraded", p.held, p.req)
	}
}

// TestTxnContext_Upgrade verifies that acquiring a mode not covered by the currently held mode
// goes through the LockManager and updates heldLocks.
func TestTxnContext_Upgrade(t *testing.T) {
	// All upgradeable (held→req) pairs where CoveredBy(req, held)=false.
	for _, p := range []struct{ held, req DBLockMode }{
		{LockModeIS, LockModeS},
		{LockModeIS, LockModeIX},
		{LockModeIS, LockModeSIX},
		{LockModeIS, LockModeX},
		{LockModeS, LockModeSIX},
		{LockModeS, LockModeX},
		{LockModeIX, LockModeSIX},
		{LockModeIX, LockModeX},
		{LockModeSIX, LockModeX},
	} {
		lm := NewLockManager()
		ctx := newTestContext(lm)
		tag := NewTableLockTag(common.ObjectID(300))
		require.NoError(t, ctx.AcquireLock(tag, p.held))
		gotMode, _ := ctx.HeldLock(tag)
		assert.Equal(t, p.held, gotMode)
		require.NoError(t, ctx.AcquireLock(tag, p.req))
		gotMode, _ = ctx.HeldLock(tag)
		assert.Equal(t, p.req, gotMode,
			"held=%s -> req=%s: heldLocks should reflect upgraded mode", p.held, p.req)
	}
}

// TestTxnContext_ReleaseAllLocks verifies that ReleaseAllLocks releases every held lock.
func TestTxnContext_ReleaseAllLocks(t *testing.T) {
	lm := NewLockManager()

	// Case 1: empty context — must not panic.
	emptyCtx := newTestContext(lm)
	assert.NotPanics(t, func() { emptyCtx.ReleaseAllLocks() })

	// Case 2: context holding multiple locks across different modes.
	ctx := newTestContext(lm)

	tags := []DBLockTag{
		NewTableLockTag(common.ObjectID(1)),
		NewTableLockTag(common.ObjectID(2)),
		NewTableLockTag(common.ObjectID(3)),
	}
	modes := []DBLockMode{LockModeIS, LockModeIX, LockModeS}

	for i, tag := range tags {
		require.NoError(t, ctx.AcquireLock(tag, modes[i]))
		assert.True(t, lm.LockHeld(tag))
	}

	ctx.ReleaseAllLocks()

	for _, tag := range tags {
		assert.False(t, lm.LockHeld(tag), "lock should be released after ReleaseAllLocks")
	}

	// Case 3: reentrant same-mode, covered no-op, and upgrade — verifies no double-unlock.
	lm3 := NewLockManager()
	ctx3 := newTestContext(lm3)
	ctx3.id = common.TransactionID(3)

	tag3a := NewTableLockTag(common.ObjectID(10))
	tag3b := NewTableLockTag(common.ObjectID(11))
	tag3c := NewTableLockTag(common.ObjectID(12))

	// Reentrant same-mode: only one lock entry added for tag3a.
	require.NoError(t, ctx3.AcquireLock(tag3a, LockModeIS))
	require.NoError(t, ctx3.AcquireLock(tag3a, LockModeIS)) // no-op

	// Covered no-op: acquire S, then IS (covered by S) — tag3b stays at S.
	require.NoError(t, ctx3.AcquireLock(tag3b, LockModeS))
	require.NoError(t, ctx3.AcquireLock(tag3b, LockModeIS)) // no-op
	tag3bMode, _ := ctx3.HeldLock(tag3b)
	assert.Equal(t, LockModeS, tag3bMode)

	// Upgrade: IS → X — tag3c ends at X.
	require.NoError(t, ctx3.AcquireLock(tag3c, LockModeIS))
	require.NoError(t, ctx3.AcquireLock(tag3c, LockModeX))
	tag3cMode, _ := ctx3.HeldLock(tag3c)
	assert.Equal(t, LockModeX, tag3cMode)

	assert.Equal(t, 3, len(ctx3.heldLocks), "three distinct tags, each with one entry")
	ctx3.ReleaseAllLocks()
	assert.False(t, lm3.LockHeld(tag3a), "tag3a released")
	assert.False(t, lm3.LockHeld(tag3b), "tag3b released")
	assert.False(t, lm3.LockHeld(tag3c), "tag3c released")
}

// TestTxnContext_Reset_ClearsState verifies that Reset() wipes TID, heldLocks, logRecords,
// and abortActions so the context can be safely reused for a new transaction.
func TestTxnContext_Reset_ClearsState(t *testing.T) {
	lm := NewLockManager()
	ctx := newTestContext(lm)

	// Populate state.
	tag := NewTableLockTag(common.ObjectID(5))
	require.NoError(t, ctx.AcquireLock(tag, LockModeX))

	rowData := make([]byte, common.IntSize)
	rid := common.RecordID{PageID: common.PageID{Oid: 1, PageNum: 0}, Slot: 0}
	ctx.NewInsertRecord(rid, rowData)
	ctx.NewDeleteRecord(rid)
	ctx.AddAbortTask(IndexTask{}) // dummy entry

	require.Equal(t, 1, len(ctx.heldLocks))
	require.Equal(t, 2, ctx.logRecords.len())
	require.Equal(t, 1, len(ctx.abortActions))

	// Reset.
	newID := common.TransactionID(99)
	ctx.Reset(newID)

	assert.Equal(t, newID, ctx.id)
	assert.Empty(t, ctx.heldLocks, "heldLocks must be cleared")
	assert.Equal(t, 0, ctx.logRecords.len(), "logRecords must be cleared")
	assert.Empty(t, ctx.abortActions, "abortActions must be cleared")
}

// TestTxnContext_AcquireLock_ErrorNotTracked verifies that when AcquireLock returns an error
// (deadlock), the failed tag is NOT inserted into heldLocks, while prior successful acquires
// remain unaffected.
func TestTxnContext_AcquireLock_ErrorNotTracked(t *testing.T) {
	lm := NewLockManager()
	tag1 := NewTableLockTag(common.ObjectID(1))
	tag2 := NewTableLockTag(common.ObjectID(2))

	ctx1 := newTestContext(lm)
	ctx1.id = common.TransactionID(1)
	ctx2 := newTestContext(lm)
	ctx2.id = common.TransactionID(2)

	require.NoError(t, ctx1.AcquireLock(tag1, LockModeX))
	require.NoError(t, ctx2.AcquireLock(tag2, LockModeX))

	var err1, err2 error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		err1 = ctx1.AcquireLock(tag2, LockModeX)
		if err1 != nil {
			_, held := ctx1.HeldLock(tag2)
			assert.False(t, held, "failed lock must NOT be in heldLocks (ctx1)")
			tag1Mode, _ := ctx1.HeldLock(tag1)
			assert.Equal(t, LockModeX, tag1Mode, "ctx1's prior lock must remain")
			ctx1.ReleaseAllLocks() // release to unblock the other goroutine
		}
	}()
	go func() {
		defer wg.Done()
		err2 = ctx2.AcquireLock(tag1, LockModeX)
		if err2 != nil {
			_, held := ctx2.HeldLock(tag1)
			assert.False(t, held, "failed lock must NOT be in heldLocks (ctx2)")
			tag2Mode, _ := ctx2.HeldLock(tag2)
			assert.Equal(t, LockModeX, tag2Mode, "ctx2's prior lock must remain")
			ctx2.ReleaseAllLocks() // release to unblock the other goroutine
		}
	}()
	wg.Wait()
	require.True(t, err1 != nil || err2 != nil, "at least one transaction must die in a deadlock")
}
