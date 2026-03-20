package execution

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"mit.edu/dsg/godb/catalog"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/logging"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

// txnManagerDeps bundles a full transaction stack for abort/commit integration
// tests that need both a real TransactionManager and real TableHeap access.
type txnManagerDeps struct {
	lm  *transaction.LockManager
	tm  *transaction.TransactionManager
	th  *TableHeap
	mlm *logging.MemoryLogManager
}

func makeTxnManagerDeps(t *testing.T) txnManagerDeps {
	t.Helper()
	mlm := logging.NewMemoryLogManager()
	// PeriodicWALFlusher simulates a real background WAL I/O thread. Without it,
	// WaitUntilFlushed (called by Commit) would spin forever. Without
	// SetFlushOnAppend, there is a real window between WAL append and WAL flush
	// in which WALCheckingDBFile can catch evictions that precede the flush.
	ctx, cancel := context.WithCancel(context.Background())
	go logging.PeriodicWALFlusher(ctx, mlm)
	sm := storage.NewDiskStorageManager(t.TempDir())
	bp := storage.NewBufferPool(5, sm, mlm)
	lm := transaction.NewLockManager()
	tm := transaction.NewTransactionManager(mlm, bp, lm)
	table := &catalog.Table{
		Oid:  1,
		Name: "t",
		Columns: []catalog.Column{
			{Name: "val", Type: common.IntType},
		},
	}
	th, err := NewTableHeap(table, bp, mlm, lm)
	require.NoError(t, err)
	t.Cleanup(func() {
		cancel()
		mlm.SetFlushedLSN(storage.LSN(1 << 20)) // unblock any pending WaitUntilFlushed
	})
	return txnManagerDeps{lm: lm, tm: tm, th: th, mlm: mlm}
}

// TestTxnManager_Abort_UndoesInsert inserts a tuple in a transaction and aborts.
// The abort must:
//   - Mark the slot as logically deleted.
//   - Append a LogInsertCLR before applying the physical undo.
func TestTxnManager_Abort_UndoesInsert(t *testing.T) {
	d := makeTxnManagerDeps(t)

	txn, err := d.tm.Begin()
	require.NoError(t, err)
	rid, err := d.th.InsertTuple(txn, makeTestTuple(42))
	require.NoError(t, err)

	buf := make([]byte, d.th.StorageSchema().BytesPerTuple())
	assert.NoError(t, d.th.ReadTuple(nil, rid, buf, false), "slot must not be deleted before abort")

	require.NoError(t, d.tm.Abort(txn))

	assert.ErrorIs(t, d.th.ReadTuple(nil, rid, buf, false), ErrTupleDeleted,
		"slot must be marked deleted after aborting insert")

	// WAL: [LogBeginTransaction, LogInsert, LogInsertCLR, LogAbort]
	require.Equal(t, 4, d.mlm.Count())
	assert.Equal(t, storage.LogBeginTransaction, d.mlm.GetRecord(0).RecordType())
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(1).RecordType())
	assert.Equal(t, storage.LogInsertCLR, d.mlm.GetRecord(2).RecordType())
	assert.Equal(t, storage.LogAbort, d.mlm.GetRecord(3).RecordType())
}

// TestTxnManager_Abort_UndoesDelete deletes a committed row in a new transaction
// and aborts. The abort must clear the deleted flag, restoring the row as visible.
func TestTxnManager_Abort_UndoesDelete(t *testing.T) {
	d := makeTxnManagerDeps(t)

	// Commit a row so it exists before the aborted transaction.
	setup, err := d.tm.Begin()
	require.NoError(t, err)
	rid, err := d.th.InsertTuple(setup, makeTestTuple(10))
	require.NoError(t, err)
	require.NoError(t, d.tm.Commit(setup))

	prevCount := d.mlm.Count()

	txn, err := d.tm.Begin()
	require.NoError(t, err)
	require.NoError(t, d.th.DeleteTuple(txn, rid))
	require.NoError(t, d.tm.Abort(txn))

	buf := make([]byte, d.th.StorageSchema().BytesPerTuple())
	assert.NoError(t, d.th.ReadTuple(nil, rid, buf, false), "deleted flag must be cleared after aborting delete")

	// WAL from the aborted transaction: [LogBeginTransaction, LogDelete, LogDeleteCLR, LogAbort]
	require.Equal(t, prevCount+4, d.mlm.Count())
	assert.Equal(t, storage.LogBeginTransaction, d.mlm.GetRecord(prevCount).RecordType())
	assert.Equal(t, storage.LogDelete, d.mlm.GetRecord(prevCount+1).RecordType())
	assert.Equal(t, storage.LogDeleteCLR, d.mlm.GetRecord(prevCount+2).RecordType())
	assert.Equal(t, storage.LogAbort, d.mlm.GetRecord(prevCount+3).RecordType())
}

// TestTxnManager_Abort_UndoesUpdate updates a committed row in a new transaction
// and aborts. The abort must restore the tuple to its exact before-image.
func TestTxnManager_Abort_UndoesUpdate(t *testing.T) {
	d := makeTxnManagerDeps(t)

	original := makeTestTuple(42)
	setup, err := d.tm.Begin()
	require.NoError(t, err)
	rid, err := d.th.InsertTuple(setup, original)
	require.NoError(t, err)
	require.NoError(t, d.tm.Commit(setup))

	prevCount := d.mlm.Count()

	txn, err := d.tm.Begin()
	require.NoError(t, err)
	require.NoError(t, d.th.UpdateTuple(txn, rid, makeTestTuple(99)))
	require.NoError(t, d.tm.Abort(txn))

	buf := make(storage.RawTuple, d.th.StorageSchema().BytesPerTuple())
	require.NoError(t, d.th.ReadTuple(nil, rid, buf, false))
	assert.Equal(t, original, buf, "tuple must be restored to its before-image after aborting update")

	// WAL from the aborted transaction: [LogBeginTransaction, LogUpdate, LogUpdateCLR, LogAbort]
	require.Equal(t, prevCount+4, d.mlm.Count())
	assert.Equal(t, storage.LogBeginTransaction, d.mlm.GetRecord(prevCount).RecordType())
	assert.Equal(t, storage.LogUpdate, d.mlm.GetRecord(prevCount+1).RecordType())
	assert.Equal(t, storage.LogUpdateCLR, d.mlm.GetRecord(prevCount+2).RecordType())
	assert.Equal(t, storage.LogAbort, d.mlm.GetRecord(prevCount+3).RecordType())
}

// TestTxnManager_Abort_CLRFailure tests the WAL-before-data principle for the undo
// phase: if writing the CLR fails, the physical undo must NOT be applied. The slot
// must remain in its pre-undo state, locks must remain held (SS2PL), and the WAL
// must contain only [Begin, Insert] — no CLR and no Abort record.
func TestTxnManager_Abort_CLRFailure(t *testing.T) {
	d := makeTxnManagerDeps(t)

	txn, err := d.tm.Begin()
	require.NoError(t, err)
	rid, err := d.th.InsertTuple(txn, makeTestTuple(42))
	require.NoError(t, err)

	// Inject a failure so the very next Append (the InsertCLR) fails.
	d.mlm.SetAppendErrorAfterN(0, errors.New("injected CLR failure"))

	err = d.tm.Abort(txn)
	assert.Error(t, err, "Abort must return an error when CLR append fails")

	// WAL-before-data: slot must not be deleted since the CLR was never written.
	buf := make([]byte, d.th.StorageSchema().BytesPerTuple())
	assert.NoError(t, d.th.ReadTuple(nil, rid, buf, false),
		"physical undo must not occur if the CLR failed to write (WAL-before-data)")

	// Locks must remain held after failed CLR (SS2PL: locks held until end of transaction).
	tableTag := transaction.NewTableLockTag(d.th.oid)
	assert.True(t, d.lm.LockHeld(tableTag),
		"table lock must remain held after failed Abort (SS2PL)")
	assert.True(t, d.lm.LockHeld(transaction.NewTupleLockTag(rid)),
		"tuple lock must remain held after failed Abort (SS2PL)")

	// WAL: only [LogBeginTransaction, LogInsert] — no CLR or Abort record was written.
	require.Equal(t, 2, d.mlm.Count())
	assert.Equal(t, storage.LogBeginTransaction, d.mlm.GetRecord(0).RecordType())
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(1).RecordType())
}

// TestTxnManager_Abort_LIFO_Ordering inserts three rows in sequence and aborts.
// Abort must process the undo log in strict LIFO order, so the CLRs appear in
// reverse insertion order: last inserted → first compensated.
func TestTxnManager_Abort_LIFO_Ordering(t *testing.T) {
	d := makeTxnManagerDeps(t)

	txn, err := d.tm.Begin()
	require.NoError(t, err)
	rid1, err := d.th.InsertTuple(txn, makeTestTuple(1))
	require.NoError(t, err)
	rid2, err := d.th.InsertTuple(txn, makeTestTuple(2))
	require.NoError(t, err)
	rid3, err := d.th.InsertTuple(txn, makeTestTuple(3))
	require.NoError(t, err)
	require.NoError(t, d.tm.Abort(txn))

	// All three slots must be logically deleted.
	buf := make([]byte, d.th.StorageSchema().BytesPerTuple())
	assert.ErrorIs(t, d.th.ReadTuple(nil, rid1, buf, false), ErrTupleDeleted, "rid1 must be deleted after abort")
	assert.ErrorIs(t, d.th.ReadTuple(nil, rid2, buf, false), ErrTupleDeleted, "rid2 must be deleted after abort")
	assert.ErrorIs(t, d.th.ReadTuple(nil, rid3, buf, false), ErrTupleDeleted, "rid3 must be deleted after abort")

	// WAL: [Begin, Insert(1), Insert(2), Insert(3), CLR(3), CLR(2), CLR(1), Abort]
	require.Equal(t, 8, d.mlm.Count())
	assert.Equal(t, storage.LogBeginTransaction, d.mlm.GetRecord(0).RecordType())
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(1).RecordType())
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(2).RecordType())
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(3).RecordType())
	assert.Equal(t, storage.LogInsertCLR, d.mlm.GetRecord(4).RecordType(), "LIFO: Insert(3) compensated first")
	assert.Equal(t, storage.LogInsertCLR, d.mlm.GetRecord(5).RecordType(), "LIFO: Insert(2) compensated second")
	assert.Equal(t, storage.LogInsertCLR, d.mlm.GetRecord(6).RecordType(), "LIFO: Insert(1) compensated last")
	assert.Equal(t, storage.LogAbort, d.mlm.GetRecord(7).RecordType())

	// Verify strict LIFO via RIDs: CLRs must appear in reverse insertion order.
	assert.Equal(t, rid3, d.mlm.GetRecord(4).RID(), "first CLR must undo the last insertion (rid3)")
	assert.Equal(t, rid2, d.mlm.GetRecord(5).RID(), "second CLR must undo the middle insertion (rid2)")
	assert.Equal(t, rid1, d.mlm.GetRecord(6).RID(), "third CLR must undo the first insertion (rid1)")
}

// TestTxnManager_Basic_MixedSerialTrace runs a sequential multi-transaction trace
// where each transaction performs a mix of insert, update, and delete operations
// across a shared set of rows. Committed transactions persist their writes; aborted
// transactions are fully undone in LIFO order.
func TestTxnManager_Basic_MixedSerialTrace(t *testing.T) {
	d := makeTxnManagerDeps(t)
	buf := make(storage.RawTuple, d.th.StorageSchema().BytesPerTuple())
	tableTag := transaction.NewTableLockTag(d.th.oid)

	// T1 (commit): insert three rows.
	t1, err := d.tm.Begin()
	require.NoError(t, err)
	ridA, err := d.th.InsertTuple(t1, makeTestTuple(10))
	require.NoError(t, err)
	ridB, err := d.th.InsertTuple(t1, makeTestTuple(20))
	require.NoError(t, err)
	ridC, err := d.th.InsertTuple(t1, makeTestTuple(30))
	require.NoError(t, err)
	require.NoError(t, d.tm.Commit(t1))

	// T2 (abort): update ridA, delete ridB, insert ridDAborted.
	// Abort undoes all three in LIFO order: insert first, then delete, then update.
	t2, err := d.tm.Begin()
	require.NoError(t, err)
	require.NoError(t, d.th.UpdateTuple(t2, ridA, makeTestTuple(100)))
	require.NoError(t, d.th.DeleteTuple(t2, ridB))
	ridDAborted, err := d.th.InsertTuple(t2, makeTestTuple(40))
	require.NoError(t, err)
	require.NoError(t, d.tm.Abort(t2))
	require.NoError(t, d.th.ReadTuple(nil, ridA, buf, false), "ridA: update undone by T2 abort")
	assert.Equal(t, makeTestTuple(10), buf, "ridA: update undone by T2 abort")
	require.NoError(t, d.th.ReadTuple(nil, ridB, buf, false), "ridB: delete undone by T2 abort")
	assert.Equal(t, makeTestTuple(20), buf, "ridB: delete undone by T2 abort")
	assert.ErrorIs(t, d.th.ReadTuple(nil, ridDAborted, buf, false), ErrTupleDeleted, "ridDAborted: insert undone by T2 abort")
	require.NoError(t, d.th.ReadTuple(nil, ridC, buf, false), "ridC: unaffected by T2")
	assert.Equal(t, makeTestTuple(30), buf, "ridC: unaffected by T2")

	// T3 (commit) and T4 (abort) are interleaved on a single thread.
	// Both transactions are active at the same time; their operations are
	// manually stepped in turn. T3 touches ridA, ridB, and a new ridD;
	// T4 touches ridC and a new ridEAborted. The row sets are disjoint,
	// so neither transaction blocks the other.
	t3, err := d.tm.Begin()
	require.NoError(t, err)
	t4, err := d.tm.Begin()
	require.NoError(t, err)

	require.NoError(t, d.th.DeleteTuple(t3, ridA))
	ridEAborted, err := d.th.InsertTuple(t4, makeTestTuple(50))
	require.NoError(t, err)
	require.NoError(t, d.th.UpdateTuple(t3, ridB, makeTestTuple(200)))
	require.NoError(t, d.th.UpdateTuple(t4, ridC, makeTestTuple(999)))
	ridD, err := d.th.InsertTuple(t3, makeTestTuple(40))
	require.NoError(t, err)

	require.NoError(t, d.tm.Abort(t4))
	require.NoError(t, d.tm.Commit(t3))

	assert.False(t, d.lm.LockHeld(tableTag), "locks released after T3+T4 complete")
	assert.ErrorIs(t, d.th.ReadTuple(nil, ridA, buf, false), ErrTupleDeleted, "ridA deleted by T3")
	require.NoError(t, d.th.ReadTuple(nil, ridB, buf, false), "ridB updated by T3")
	assert.Equal(t, makeTestTuple(200), buf, "ridB updated by T3")
	require.NoError(t, d.th.ReadTuple(nil, ridC, buf, false), "ridC: update undone by T4 abort")
	assert.Equal(t, makeTestTuple(30), buf, "ridC: update undone by T4 abort")
	require.NoError(t, d.th.ReadTuple(nil, ridD, buf, false), "ridD inserted by T3")
	assert.Equal(t, makeTestTuple(40), buf, "ridD inserted by T3")
	assert.ErrorIs(t, d.th.ReadTuple(nil, ridEAborted, buf, false), ErrTupleDeleted, "ridEAborted: insert undone by T4 abort")

	logging.VerifyWALOrdering(t, d.mlm)
}
