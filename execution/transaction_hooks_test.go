package execution

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"mit.edu/dsg/godb/catalog"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/indexing"
	"mit.edu/dsg/godb/logging"
	"mit.edu/dsg/godb/planner"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

// txnTestDeps bundles the dependencies for transaction integration tests.
// TransactionContext is created directly via transaction.NewTestTransactionContext to
// avoid assuming students have finished implementing TransactionManager.
type txnTestDeps struct {
	bp  *storage.BufferPool
	lm  *transaction.LockManager
	th  *TableHeap
	mlm *logging.MemoryLogManager
}

// makeTxnTestDeps wires together a fully transactional single-column (int) table.
func makeTxnTestDeps(t *testing.T) txnTestDeps {
	t.Helper()
	mlm := logging.NewMemoryLogManager()
	mlm.SetFlushOnAppend(true)
	sm := storage.NewDiskStorageManager(t.TempDir())
	bp := storage.NewBufferPool(100, sm, mlm)
	lm := transaction.NewLockManager()
	table := &catalog.Table{
		Oid:  1,
		Name: "txn_test",
		Columns: []catalog.Column{
			{Name: "id", Type: common.IntType},
		},
	}
	th, err := NewTableHeap(table, bp, mlm, lm)
	require.NoError(t, err)
	return txnTestDeps{bp: bp, lm: lm, th: th, mlm: mlm}
}

func makeTestTuple(val int64) storage.RawTuple {
	desc := storage.NewRawTupleDesc([]common.Type{common.IntType})
	raw := make([]byte, desc.BytesPerTuple())
	tup := storage.FromValues(common.NewIntValue(val))
	tup.WriteToBuffer(raw, desc)
	return raw
}

// TestTxnHeap_Insert_AcquiresIXAndTupleX verifies that InsertTuple with a
// transaction acquires IX on the table and X on the newly inserted slot,
// and releases both locks on commit.
func TestTxnHeap_Insert_AcquiresIXAndTupleX(t *testing.T) {
	d := makeTxnTestDeps(t)

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	rid, err := d.th.InsertTuple(txn, makeTestTuple(42))
	require.NoError(t, err)

	tableTag := transaction.NewTableLockTag(d.th.oid)
	tupleTag := transaction.NewTupleLockTag(rid)
	tableMode, tableHeld := txn.HeldLock(tableTag)
	assert.True(t, tableHeld, "IX table lock must be held after insert")
	assert.Equal(t, transaction.LockModeIX, tableMode, "insert must acquire IX on table")
	tupleMode, tupleHeld := txn.HeldLock(tupleTag)
	assert.True(t, tupleHeld, "X tuple lock must be held after insert")
	assert.Equal(t, transaction.LockModeX, tupleMode, "insert must acquire X on tuple")

	txn.ReleaseAllLocks()
	assert.False(t, d.lm.LockHeld(tableTag), "table lock must be released after commit")
	assert.False(t, d.lm.LockHeld(tupleTag), "tuple lock must be released after commit")
}

// TestTxnHeap_Delete_AcquiresIXAndTupleX verifies that DeleteTuple acquires IX
// on the table and X on the target slot.
func TestTxnHeap_Delete_AcquiresIXAndTupleX(t *testing.T) {
	d := makeTxnTestDeps(t)

	rid, err := d.th.InsertTuple(nil, makeTestTuple(42))
	require.NoError(t, err)

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	require.NoError(t, d.th.DeleteTuple(txn, rid))

	tableTag := transaction.NewTableLockTag(d.th.oid)
	tupleTag := transaction.NewTupleLockTag(rid)
	tableMode, tableHeld := txn.HeldLock(tableTag)
	assert.True(t, tableHeld, "IX table lock must be held after delete")
	assert.Equal(t, transaction.LockModeIX, tableMode, "delete must acquire IX on table")
	tupleMode, tupleHeld := txn.HeldLock(tupleTag)
	assert.True(t, tupleHeld, "X tuple lock must be held after delete")
	assert.Equal(t, transaction.LockModeX, tupleMode, "delete must acquire X on tuple")

	txn.ReleaseAllLocks()
	assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
	assert.False(t, d.lm.LockHeld(tupleTag), "locks must be released after commit")
}

// TestTxnHeap_Update_AcquiresIXAndTupleX verifies that UpdateTuple acquires IX
// on the table and X on the target slot, matching the locking contract of Insert
// and Delete.
func TestTxnHeap_Update_AcquiresIXAndTupleX(t *testing.T) {
	d := makeTxnTestDeps(t)

	rid, err := d.th.InsertTuple(nil, makeTestTuple(42))
	require.NoError(t, err)

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	require.NoError(t, d.th.UpdateTuple(txn, rid, makeTestTuple(99)))

	tableTag := transaction.NewTableLockTag(d.th.oid)
	tupleTag := transaction.NewTupleLockTag(rid)
	tableMode, tableHeld := txn.HeldLock(tableTag)
	assert.True(t, tableHeld, "IX table lock must be held after update")
	assert.Equal(t, transaction.LockModeIX, tableMode, "update must acquire IX on table")
	tupleMode, tupleHeld := txn.HeldLock(tupleTag)
	assert.True(t, tupleHeld, "X tuple lock must be held after update")
	assert.Equal(t, transaction.LockModeX, tupleMode, "update must acquire X on tuple")

	txn.ReleaseAllLocks()
	assert.False(t, d.lm.LockHeld(tableTag), "table lock must be released after commit")
	assert.False(t, d.lm.LockHeld(tupleTag), "tuple lock must be released after commit")
}

// TestTxnHeap_Read_AcquiresLocks verifies that ReadTuple acquires the correct
// table and tuple locks depending on the forUpdate flag:
//   - forUpdate=false: IS on the table and S on the tuple
//   - forUpdate=true:  IX on the table and X on the tuple
func TestTxnHeap_Read_AcquiresLocks(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
		desc      string
	}{
		{false, transaction.LockModeIS, transaction.LockModeS, "IS/S for regular read"},
		{true, transaction.LockModeIX, transaction.LockModeX, "IX/X for read-for-update"},
	} {
		d := makeTxnTestDeps(t)
		desc := d.th.StorageSchema()

		rid, err := d.th.InsertTuple(nil, makeTestTuple(99))
		require.NoError(t, err)

		txn := transaction.NewTestTransactionContext(d.lm, 1)
		buf := make([]byte, desc.BytesPerTuple())
		require.NoError(t, d.th.ReadTuple(txn, rid, buf, tc.forUpdate))

		tableTag := transaction.NewTableLockTag(d.th.oid)
		tupleTag := transaction.NewTupleLockTag(rid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held after read")
		assert.Equal(t, tc.tableMode, tableMode, "read must acquire correct table lock mode")
		tupleMode, tupleHeld := txn.HeldLock(tupleTag)
		assert.True(t, tupleHeld, "tuple lock must be held after read")
		assert.Equal(t, tc.tupleMode, tupleMode, "read must acquire correct tuple lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag))
		assert.False(t, d.lm.LockHeld(tupleTag))
	}
}

// TestTxnHeap_Iterator_AcquiresTable verifies that creating a full-table Iterator
// acquires and holds the requested lock on the table until commit.
func TestTxnHeap_Iterator_AcquiresTable(t *testing.T) {
	for _, mode := range []transaction.DBLockMode{transaction.LockModeS, transaction.LockModeX, transaction.LockModeSIX} {
		d := makeTxnTestDeps(t)
		desc := d.th.StorageSchema()

		_, err := d.th.InsertTuple(nil, makeTestTuple(1))
		require.NoError(t, err)

		txn := transaction.NewTestTransactionContext(d.lm, 1)
		buf := make([]byte, desc.BytesPerTuple())
		iter, err := d.th.Iterator(txn, mode, buf)
		require.NoError(t, err)
		for iter.Next() {
		}
		require.NoError(t, iter.Close())

		tableTag := transaction.NewTableLockTag(d.th.oid)
		gotMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held by the iterator's transaction")
		assert.Equal(t, mode, gotMode, "iterator must acquire the requested table lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "lock must be released after commit")
	}
}

// TestTxnHeap_Insert_LogsInsertRecord verifies that InsertTuple appends a LogInsert
// record with the correct AfterImage, and that the page LSN is advanced by the insert.
func TestTxnHeap_Insert_LogsInsertRecord(t *testing.T) {
	d := makeTxnTestDeps(t)

	// Write a priming Begin record so the Insert record lands at a byte offset > 0.
	// MemoryLogManager starts at offset 0, which is also the page's initial LSN — without
	// this, a correct implementation and a buggy one both produce page.LSN() == 0.
	_, err := d.mlm.Append(transaction.NewTestTransactionContext(d.lm, 99).NewBeginTransactionRecord())
	require.NoError(t, err)
	prevCount := d.mlm.Count()

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	row := makeTestTuple(77)
	rid, err := d.th.InsertTuple(txn, row)
	require.NoError(t, err)

	require.Equal(t, prevCount+1, d.mlm.Count(), "InsertTuple must produce exactly one log record")
	assert.Equal(t, storage.LogInsert, d.mlm.GetRecord(prevCount).RecordType())

	insertRec := d.mlm.GetRecord(prevCount)
	assert.Equal(t, row, insertRec.AfterImage(),
		"AfterImage must match the inserted row bytes")

	page, err := d.bp.GetPage(rid.PageID)
	require.NoError(t, err)
	lsn := page.LSN()
	d.bp.UnpinPage(page, false)
	assert.Greater(t, int64(lsn), int64(0),
		"Page LSN must be updated after a transactional insert")

	txn.ReleaseAllLocks()
}

// TestTxnHeap_Delete_LogsDeleteRecord verifies that DeleteTuple appends a LogDelete record.
func TestTxnHeap_Delete_LogsDeleteRecord(t *testing.T) {
	d := makeTxnTestDeps(t)

	rid, err := d.th.InsertTuple(nil, makeTestTuple(10))
	require.NoError(t, err)
	prevCount := d.mlm.Count()

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	require.NoError(t, d.th.DeleteTuple(txn, rid))

	require.Equal(t, prevCount+1, d.mlm.Count(), "DeleteTuple must produce exactly one log record")
	assert.Equal(t, storage.LogDelete, d.mlm.GetRecord(prevCount).RecordType())

	txn.ReleaseAllLocks()
}

// TestTxnHeap_Update_LogsUpdateRecord verifies that UpdateTuple appends a LogUpdate
// record, that BeforeImage captures the original value and AfterImage captures the new
// value, and that the page LSN is advanced by the update.
func TestTxnHeap_Update_LogsUpdateRecord(t *testing.T) {
	d := makeTxnTestDeps(t)

	// Write a priming Begin record so the Update record lands at a byte offset > 0.
	// MemoryLogManager starts at offset 0, which is also the page's initial LSN — without
	// this, a correct implementation and a buggy one both produce page.LSN() == 0.
	_, err := d.mlm.Append(transaction.NewTestTransactionContext(d.lm, 99).NewBeginTransactionRecord())
	require.NoError(t, err)

	before := makeTestTuple(42)
	rid, err := d.th.InsertTuple(nil, before)
	require.NoError(t, err)
	prevCount := d.mlm.Count()

	page, err := d.bp.GetPage(rid.PageID)
	require.NoError(t, err)
	lsnBefore := page.LSN()
	d.bp.UnpinPage(page, false)

	txn := transaction.NewTestTransactionContext(d.lm, 1)
	after := makeTestTuple(99)
	require.NoError(t, d.th.UpdateTuple(txn, rid, after))

	require.Equal(t, prevCount+1, d.mlm.Count(), "UpdateTuple must produce exactly one log record")
	assert.Equal(t, storage.LogUpdate, d.mlm.GetRecord(prevCount).RecordType())

	updateRec := d.mlm.GetRecord(prevCount)
	assert.Equal(t, before, updateRec.BeforeImage(),
		"BeforeImage must capture the original value (42); "+
			"if it shows 99, copy() was called before NewUpdateRecord()")
	assert.Equal(t, after, updateRec.AfterImage(),
		"AfterImage must capture the new value (99)")

	page, err = d.bp.GetPage(rid.PageID)
	require.NoError(t, err)
	lsnAfter := page.LSN()
	d.bp.UnpinPage(page, false)
	assert.Greater(t, int64(lsnAfter), int64(lsnBefore),
		"Page LSN must increase after a transactional update")

	txn.ReleaseAllLocks()
}

// TestTxnHeap_Read_DoesNotLog verifies that ReadTuple does not append any log
// records and does not modify the page's LSN, regardless of the forUpdate flag.
func TestTxnHeap_Read_DoesNotLog(t *testing.T) {
	for _, forUpdate := range []bool{false, true} {
		d := makeTxnTestDeps(t)
		desc := d.th.StorageSchema()

		rid, err := d.th.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)
		prevCount := d.mlm.Count()

		page, err := d.bp.GetPage(rid.PageID)
		require.NoError(t, err)
		lsnBefore := page.LSN()
		d.bp.UnpinPage(page, false)

		txn := transaction.NewTestTransactionContext(d.lm, 1)
		buf := make([]byte, desc.BytesPerTuple())
		require.NoError(t, d.th.ReadTuple(txn, rid, buf, forUpdate))
		assert.Equal(t, prevCount, d.mlm.Count(),
			"ReadTuple must not append any log records (forUpdate=%v)", forUpdate)
		txn.ReleaseAllLocks()

		page, err = d.bp.GetPage(rid.PageID)
		require.NoError(t, err)
		lsnAfter := page.LSN()
		d.bp.UnpinPage(page, false)
		assert.Equal(t, lsnBefore, lsnAfter,
			"Page LSN must not change after a read-only operation (forUpdate=%v)", forUpdate)
	}
}

// TestTxnHeap_Vacuum_SkipsLockedTuples verifies that VacuumPage does not reclaim a
// deleted slot while a transaction holds a lock on it, but reclaims it once the lock
// is released after commit.
func TestTxnHeap_Vacuum_SkipsLockedTuples(t *testing.T) {
	d := makeTxnTestDeps(t)

	// Insert two tuples (they will share a page on this small table).
	rid1, err := d.th.InsertTuple(nil, makeTestTuple(1))
	require.NoError(t, err)
	rid2, err := d.th.InsertTuple(nil, makeTestTuple(2))
	require.NoError(t, err)
	require.Equal(t, rid1.PageID, rid2.PageID, "both tuples must be on the same page")

	// Delete rid2 without a transaction — no lock will remain.
	require.NoError(t, d.th.DeleteTuple(nil, rid2))

	// Delete rid1 inside a transaction — txn holds X on rid1.
	txn := transaction.NewTestTransactionContext(d.lm, 1)
	require.NoError(t, d.th.DeleteTuple(txn, rid1))

	// VacuumPage: rid1 is locked → must be skipped; rid2 is unlocked → must be reclaimed.
	require.NoError(t, d.th.VacuumPage(rid1.PageID))

	page, err := d.bp.GetPage(rid1.PageID)
	require.NoError(t, err)
	hp := page.AsHeapPage()
	assert.True(t, hp.IsAllocated(rid1), "locked deleted slot must not be vacuumed")
	assert.False(t, hp.IsAllocated(rid2), "unlocked deleted slot must be vacuumed")
	d.bp.UnpinPage(page, false)

	// Release locks (simulates commit) → releases X lock on rid1.
	txn.ReleaseAllLocks()

	// VacuumPage again — rid1's lock is gone, so it must now be reclaimed.
	require.NoError(t, d.th.VacuumPage(rid1.PageID))

	page, err = d.bp.GetPage(rid1.PageID)
	require.NoError(t, err)
	hp = page.AsHeapPage()
	assert.False(t, hp.IsAllocated(rid1), "rid1 must be reclaimed after the lock is released")
	d.bp.UnpinPage(page, false)
}

// TestIndexLookup_WithTransaction_AcquiresLocks verifies that running an
// IndexLookupExecutor inside a transaction acquires the correct locks:
//   - forUpdate=false: IS on the table and S on the fetched tuple
//   - forUpdate=true:  IX on the table and X on the fetched tuple
func TestIndexLookup_WithTransaction_AcquiresLocks(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		idx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert one row and add it to the index without a transaction.
		row := makeTestTuple(42)
		rid, err := d.th.InsertTuple(nil, row)
		require.NoError(t, err)
		key := createKey(idx, common.NewIntValue(42))
		require.NoError(t, idx.InsertEntry(key, rid, nil))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		lookupNode := planner.NewIndexLookupNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(), key, tc.forUpdate,
		)
		lookupExec := NewIndexLookupExecutor(lookupNode, idx, d.th)
		require.NoError(t, lookupExec.Init(NewExecutorContext(txn)))

		found := false
		for lookupExec.Next() {
			found = true
		}
		require.NoError(t, lookupExec.Close())
		assert.True(t, found, "should find the inserted row via the index")

		tableTag := transaction.NewTableLockTag(d.th.oid)
		tupleTag := transaction.NewTupleLockTag(rid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock should be held after index lookup")
		assert.Equal(t, tc.tableMode, tableMode, "index lookup must acquire correct table lock mode")
		tupleMode, tupleHeld := txn.HeldLock(tupleTag)
		assert.True(t, tupleHeld, "tuple lock should be held after index lookup")
		assert.Equal(t, tc.tupleMode, tupleMode, "index lookup must acquire correct tuple lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag))
		assert.False(t, d.lm.LockHeld(tupleTag))
	}
}

// TestIndexLookup_SkipsStaleHeapEntry verifies the "recheck after lock" behavior of
// IndexLookupExecutor: when an index entry points to a heap tuple that has since been
// deleted, the executor skips the stale entry and returns no results.
// Locks must still be held on the skipped tuple, in the mode determined by forUpdate.
func TestIndexLookup_SkipsStaleHeapEntry(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		idx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert one row and register it in the index.
		rid, err := d.th.InsertTuple(nil, makeTestTuple(55))
		require.NoError(t, err)
		key := createKey(idx, common.NewIntValue(55))
		require.NoError(t, idx.InsertEntry(key, rid, nil))

		// Delete the heap tuple only — the index entry remains, creating a stale pointer.
		require.NoError(t, d.th.DeleteTuple(nil, rid))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		lookupNode := planner.NewIndexLookupNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(), key, tc.forUpdate,
		)
		lookupExec := NewIndexLookupExecutor(lookupNode, idx, d.th)
		require.NoError(t, lookupExec.Init(NewExecutorContext(txn)))

		assert.False(t, lookupExec.Next(),
			"executor must skip the stale index entry and return no results")
		require.NoError(t, lookupExec.Close())

		// Locks must still be held on the skipped tuple until commit.
		tableTag := transaction.NewTableLockTag(d.th.oid)
		tupleTag := transaction.NewTupleLockTag(rid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held even for a skipped deleted tuple")
		assert.Equal(t, tc.tableMode, tableMode, "index lookup must acquire correct table lock mode")
		tupleMode, tupleHeld := txn.HeldLock(tupleTag)
		assert.True(t, tupleHeld, "tuple lock must be held even for a skipped deleted tuple")
		assert.Equal(t, tc.tupleMode, tupleMode, "index lookup must acquire correct tuple lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
		assert.False(t, d.lm.LockHeld(tupleTag), "locks must be released after commit")
	}
}

// TestIndexLookup_SkipsKeyMismatch verifies that IndexLookupExecutor skips a tuple
// whose heap value no longer matches the probe key: the index maps key=42 → RID,
// but that RID was updated to 99. The executor must not return the mismatched row.
// Locks must still be held on the skipped tuple, in the mode determined by forUpdate.
func TestIndexLookup_SkipsKeyMismatch(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		idx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert row 42 and register it in the index.
		rid, err := d.th.InsertTuple(nil, makeTestTuple(42))
		require.NoError(t, err)
		key42 := createKey(idx, common.NewIntValue(42))
		require.NoError(t, idx.InsertEntry(key42, rid, nil))

		// Simulate check-and-miss: update the heap tuple to 99 without updating the index.
		// The index still maps key=42 → rid, but the tuple at rid now has value=99.
		require.NoError(t, d.th.UpdateTuple(nil, rid, makeTestTuple(99)))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		lookupNode := planner.NewIndexLookupNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(), key42, tc.forUpdate,
		)
		lookupExec := NewIndexLookupExecutor(lookupNode, idx, d.th)
		require.NoError(t, lookupExec.Init(NewExecutorContext(txn)))

		assert.False(t, lookupExec.Next(),
			"executor must skip the tuple whose indexed key no longer matches the probe key")
		require.NoError(t, lookupExec.Close())

		// Lock is acquired before the key-mismatch recheck, so it must still be held.
		tableTag := transaction.NewTableLockTag(d.th.oid)
		tupleTag := transaction.NewTupleLockTag(rid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held even for a key-mismatch skip")
		assert.Equal(t, tc.tableMode, tableMode, "index lookup must acquire correct table lock mode")
		tupleMode, tupleHeld := txn.HeldLock(tupleTag)
		assert.True(t, tupleHeld, "tuple lock must be held even for a key-mismatch skip")
		assert.Equal(t, tc.tupleMode, tupleMode, "index lookup must acquire correct tuple lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
		assert.False(t, d.lm.LockHeld(tupleTag), "locks must be released after commit")
	}
}

// TestIndexScan_AcquiresLocks verifies that running an IndexScanExecutor inside a
// transaction acquires the correct locks and releases them on commit:
//   - forUpdate=false: IS on the table and S on the fetched tuple
//   - forUpdate=true:  IX on the table and X on the fetched tuple
func TestIndexScan_AcquiresLocks(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		btreeIdx := indexing.NewMemBTreeIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		rid, err := d.th.InsertTuple(nil, makeTestTuple(42))
		require.NoError(t, err)
		require.NoError(t, btreeIdx.InsertEntry(createKey(btreeIdx, common.NewIntValue(42)), rid, nil))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		scanNode := planner.NewIndexScanNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(),
			indexing.ScanDirectionForward, createKey(btreeIdx, common.NewIntValue(42)), tc.forUpdate,
		)
		scanExec := NewIndexScanExecutor(scanNode, btreeIdx, d.th)
		require.NoError(t, scanExec.Init(NewExecutorContext(txn)))

		found := false
		for scanExec.Next() {
			found = true
		}
		require.NoError(t, scanExec.Close())
		assert.True(t, found, "should find the inserted row via the index scan")

		tableTag := transaction.NewTableLockTag(d.th.oid)
		tupleTag := transaction.NewTupleLockTag(rid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held after index scan")
		assert.Equal(t, tc.tableMode, tableMode, "index scan must acquire correct table lock mode")
		tupleMode, tupleHeld := txn.HeldLock(tupleTag)
		assert.True(t, tupleHeld, "tuple lock must be held after index scan")
		assert.Equal(t, tc.tupleMode, tupleMode, "index scan must acquire correct tuple lock mode")

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
		assert.False(t, d.lm.LockHeld(tupleTag), "locks must be released after commit")
	}
}

// TestIndexScan_SkipsStaleHeapEntry verifies the "recheck after lock" behavior for
// B-Tree range scans: a deleted heap tuple referenced by the index snapshot is skipped
// while live tuples are returned normally. Locks must still be held on the skipped
// tuple, in the mode determined by forUpdate.
func TestIndexScan_SkipsStaleHeapEntry(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		btreeIdx := indexing.NewMemBTreeIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert rows 1, 2, 3 and populate the index.
		values := []int64{1, 2, 3}
		rids := make([]common.RecordID, len(values))
		for i, v := range values {
			rid, err := d.th.InsertTuple(nil, makeTestTuple(v))
			require.NoError(t, err)
			rids[i] = rid
			require.NoError(t, btreeIdx.InsertEntry(
				createKey(btreeIdx, common.NewIntValue(v)), rid, nil))
		}

		// Delete row with value=2 from the heap only — the index entry stays (stale pointer).
		require.NoError(t, d.th.DeleteTuple(nil, rids[1]))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		startKey := createKey(btreeIdx, common.NewIntValue(1))
		scanNode := planner.NewIndexScanNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(),
			indexing.ScanDirectionForward, startKey, tc.forUpdate,
		)
		scanExec := NewIndexScanExecutor(scanNode, btreeIdx, d.th)
		require.NoError(t, scanExec.Init(NewExecutorContext(txn)))

		var foundIDs []int64
		for scanExec.Next() {
			foundIDs = append(foundIDs, scanExec.Current().GetValue(0).IntValue())
		}
		assert.Equal(t, []int64{1, 3}, foundIDs,
			"deleted tuple (2) must be skipped; others must be returned in order")

		// All three tuples were visited via ReadTuple, so all must be locked (including the deleted one).
		tableTag := transaction.NewTableLockTag(d.th.oid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held after scan")
		assert.Equal(t, tc.tableMode, tableMode, "index scan must acquire correct table lock mode")
		for i, rid := range rids {
			tupleMode, tupleHeld := txn.HeldLock(transaction.NewTupleLockTag(rid))
			assert.True(t, tupleHeld, "tuple lock must be held for rids[%d] (value=%d) even if skipped", i, values[i])
			assert.Equal(t, tc.tupleMode, tupleMode, "index scan must acquire correct tuple lock mode for rids[%d]", i)
		}

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
	}
}

// TestIndexScan_SkipsKeyMismatch verifies that IndexScanExecutor skips a tuple whose
// heap value no longer matches the B-Tree key: rids[1] was updated to 99 but the
// index still maps key=20 → rids[1]. Only unchanged rows must be returned.
// Locks must still be held on the skipped tuple, in the mode determined by forUpdate.
func TestIndexScan_SkipsKeyMismatch(t *testing.T) {
	for _, tc := range []struct {
		forUpdate bool
		tableMode transaction.DBLockMode
		tupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		d := makeTxnTestDeps(t)

		btreeIdx := indexing.NewMemBTreeIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert rows 10, 20, 30 and populate the index.
		values := []int64{10, 20, 30}
		rids := make([]common.RecordID, len(values))
		for i, v := range values {
			rid, err := d.th.InsertTuple(nil, makeTestTuple(v))
			require.NoError(t, err)
			rids[i] = rid
			require.NoError(t, btreeIdx.InsertEntry(
				createKey(btreeIdx, common.NewIntValue(v)), rid, nil))
		}

		// Simulate check-and-miss: update the row with value=20 to value=99 without
		// updating the index. The B-Tree still maps key=20 → rids[1], but rids[1]
		// now holds value=99.
		require.NoError(t, d.th.UpdateTuple(nil, rids[1], makeTestTuple(99)))

		txn := transaction.NewTestTransactionContext(d.lm, 1)

		startKey := createKey(btreeIdx, common.NewIntValue(10))
		scanNode := planner.NewIndexScanNode(
			common.ObjectID(0), d.th.oid,
			d.th.StorageSchema().GetFieldTypes(),
			indexing.ScanDirectionForward, startKey, tc.forUpdate,
		)
		scanExec := NewIndexScanExecutor(scanNode, btreeIdx, d.th)
		require.NoError(t, scanExec.Init(NewExecutorContext(txn)))

		var foundIDs []int64
		for scanExec.Next() {
			foundIDs = append(foundIDs, scanExec.Current().GetValue(0).IntValue())
		}
		assert.Equal(t, []int64{10, 30}, foundIDs,
			"tuple whose key no longer matches iter.Key() (20→99) must be skipped")

		// All three tuples were visited via ReadTuple, so all must be locked (including rids[1]).
		tableTag := transaction.NewTableLockTag(d.th.oid)
		tableMode, tableHeld := txn.HeldLock(tableTag)
		assert.True(t, tableHeld, "table lock must be held after scan")
		assert.Equal(t, tc.tableMode, tableMode, "index scan must acquire correct table lock mode")
		for i, rid := range rids {
			tupleMode, tupleHeld := txn.HeldLock(transaction.NewTupleLockTag(rid))
			assert.True(t, tupleHeld, "tuple lock must be held for rids[%d] (original value=%d) even if skipped", i, values[i])
			assert.Equal(t, tc.tupleMode, tupleMode, "index scan must acquire correct tuple lock mode for rids[%d]", i)
		}

		txn.ReleaseAllLocks()
		assert.False(t, d.lm.LockHeld(tableTag), "locks must be released after commit")
	}
}

// ============ Uncomment these test cases if you chose to implement Index Nested Loop Join in Lab 2 =================
/*
// TestIndexNestedLoopJoin_AcquiresLocks verifies that running an
// IndexNestedLoopJoinExecutor inside a transaction acquires the correct locks:
//   - S on the left table (governed by the SeqScan's lock mode)
//   - forUpdate=false: IS on the right table and S on the fetched right tuple
//   - forUpdate=true:  IX on the right table and X on the fetched right tuple
func TestIndexNestedLoopJoin_AcquiresLocks(t *testing.T) {
	for _, tc := range []struct {
		forUpdate      bool
		rightTableMode transaction.DBLockMode
		rightTupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		mlm := logging.NewMemoryLogManager()
		mlm.SetFlushOnAppend(true)
		sm := storage.NewDiskStorageManager(t.TempDir())
		bp := storage.NewBufferPool(50, sm, mlm)
		lm := transaction.NewLockManager()

		leftTable := &catalog.Table{Oid: 20, Name: "left", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}
		rightTable := &catalog.Table{Oid: 21, Name: "right", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}

		leftHeap, err := NewTableHeap(leftTable, bp, mlm, lm)
		require.NoError(t, err)
		rightHeap, err := NewTableHeap(rightTable, bp, mlm, lm)
		require.NoError(t, err)

		rightIdx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert left row: id=5.
		_, err = leftHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)

		// Insert right row: id=5. Register it in the index.
		rightRID, err := rightHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)
		key5 := createKey(rightIdx, common.NewIntValue(5))
		require.NoError(t, rightIdx.InsertEntry(key5, rightRID, nil))

		txn := transaction.NewTestTransactionContext(lm, 1)

		leftScan := NewSeqScanExecutor(
			planner.NewSeqScanNode(leftHeap.oid, leftHeap.StorageSchema().GetFieldTypes(), transaction.LockModeS),
			leftHeap,
		)
		leftKey := planner.NewColumnValueExpression(0, leftHeap.StorageSchema().GetFieldTypes(), "left.id")
		joinPlan := planner.NewIndexNestedLoopJoinNode(
			leftScan.PlanNode(),
			rightHeap.oid,
			common.ObjectID(0),
			[]planner.Expr{leftKey},
			rightHeap.StorageSchema().GetFieldTypes(),
			tc.forUpdate,
		)
		joinExec := NewIndexJoinExecutor(joinPlan, leftScan, rightIdx, rightHeap)
		require.NoError(t, joinExec.Init(NewExecutorContext(txn)))

		found := false
		for joinExec.Next() {
			found = true
		}
		require.NoError(t, joinExec.Close())
		assert.True(t, found, "join must find the matching right row")

		leftTableTag := transaction.NewTableLockTag(leftHeap.oid)
		rightTableTag := transaction.NewTableLockTag(rightHeap.oid)
		rightTupleTag := transaction.NewTupleLockTag(rightRID)
		leftTableMode, leftTableHeld := txn.HeldLock(leftTableTag)
		assert.True(t, leftTableHeld, "S table lock must be held on left table after join")
		assert.Equal(t, transaction.LockModeS, leftTableMode, "join must acquire S on left table")
		rightTableMode, rightTableHeld := txn.HeldLock(rightTableTag)
		assert.True(t, rightTableHeld, "right table lock must be held after join")
		assert.Equal(t, tc.rightTableMode, rightTableMode, "join must acquire correct lock mode on right table")
		rightTupleMode, rightTupleHeld := txn.HeldLock(rightTupleTag)
		assert.True(t, rightTupleHeld, "right tuple lock must be held after join")
		assert.Equal(t, tc.rightTupleMode, rightTupleMode, "join must acquire correct lock mode on right tuple")

		txn.ReleaseAllLocks()
		assert.False(t, lm.LockHeld(leftTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTupleTag), "locks must be released after commit")
	}
}

// TestIndexNestedLoopJoin_SkipsStaleHeapEntry verifies the "recheck after lock"
// behavior of IndexNestedLoopJoinExecutor: when an index entry on the right side
// points to a heap tuple that has since been deleted, the executor skips the stale
// entry and produces no join results.
// Locks must still be held on the skipped right tuple (lock-before-check under SS2PL),
// in the mode determined by forUpdate.
func TestIndexNestedLoopJoin_SkipsStaleHeapEntry(t *testing.T) {
	for _, tc := range []struct {
		forUpdate      bool
		rightTableMode transaction.DBLockMode
		rightTupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		mlm := logging.NewMemoryLogManager()
		mlm.SetFlushOnAppend(true)
		sm := storage.NewDiskStorageManager(t.TempDir())
		bp := storage.NewBufferPool(50, sm, mlm)
		lm := transaction.NewLockManager()

		leftTable := &catalog.Table{Oid: 20, Name: "left", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}
		rightTable := &catalog.Table{Oid: 21, Name: "right", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}

		leftHeap, err := NewTableHeap(leftTable, bp, mlm, lm)
		require.NoError(t, err)
		rightHeap, err := NewTableHeap(rightTable, bp, mlm, lm)
		require.NoError(t, err)

		rightIdx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert left row: id=5.
		_, err = leftHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)

		// Insert right row: id=5. Register it in the index.
		rightRID, err := rightHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)
		key5 := createKey(rightIdx, common.NewIntValue(5))
		require.NoError(t, rightIdx.InsertEntry(key5, rightRID, nil))

		// Delete the heap tuple only — the index entry remains, creating a stale pointer.
		require.NoError(t, rightHeap.DeleteTuple(nil, rightRID))

		txn := transaction.NewTestTransactionContext(lm, 1)

		leftScan := NewSeqScanExecutor(
			planner.NewSeqScanNode(leftHeap.oid, leftHeap.StorageSchema().GetFieldTypes(), transaction.LockModeS),
			leftHeap,
		)
		leftKey := planner.NewColumnValueExpression(0, leftHeap.StorageSchema().GetFieldTypes(), "left.id")
		joinPlan := planner.NewIndexNestedLoopJoinNode(
			leftScan.PlanNode(),
			rightHeap.oid,
			common.ObjectID(0),
			[]planner.Expr{leftKey},
			rightHeap.StorageSchema().GetFieldTypes(),
			tc.forUpdate,
		)
		joinExec := NewIndexJoinExecutor(joinPlan, leftScan, rightIdx, rightHeap)
		require.NoError(t, joinExec.Init(NewExecutorContext(txn)))

		assert.False(t, joinExec.Next(),
			"join must skip the stale index entry and return no results")
		require.NoError(t, joinExec.Close())

		// Even though the right tuple was skipped, the lock is acquired before the validity
		// check (lock-before-check paradigm), so locks must remain held until commit.
		leftTableTag := transaction.NewTableLockTag(leftHeap.oid)
		rightTableTag := transaction.NewTableLockTag(rightHeap.oid)
		rightTupleTag := transaction.NewTupleLockTag(rightRID)
		leftTableMode, leftTableHeld := txn.HeldLock(leftTableTag)
		assert.True(t, leftTableHeld, "S table lock must be held on left table after join")
		assert.Equal(t, transaction.LockModeS, leftTableMode, "join must acquire S on left table")
		rightTableMode, rightTableHeld := txn.HeldLock(rightTableTag)
		assert.True(t, rightTableHeld, "right table lock must be held even for a skipped deleted tuple")
		assert.Equal(t, tc.rightTableMode, rightTableMode, "join must acquire correct lock mode on right table")
		rightTupleMode, rightTupleHeld := txn.HeldLock(rightTupleTag)
		assert.True(t, rightTupleHeld, "right tuple lock must be held on skipped deleted right tuple")
		assert.Equal(t, tc.rightTupleMode, rightTupleMode, "join must acquire correct lock mode on right tuple")

		txn.ReleaseAllLocks()
		assert.False(t, lm.LockHeld(leftTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTupleTag), "locks must be released after commit")
	}
}

// TestIndexNestedLoopJoin_SkipsKeyMismatch verifies the check-and-miss fix in
// IndexNestedLoopJoinExecutor. The scenario: the index on the right table maps
// key=5 → rightRID, but rightRID has since been updated to value=99. The join
// must produce no results rather than emitting a mis-matched row.
// Locks must still be held on the skipped right tuple (lock-before-check under SS2PL),
// in the mode determined by forUpdate.
func TestIndexNestedLoopJoin_SkipsKeyMismatch(t *testing.T) {
	for _, tc := range []struct {
		forUpdate      bool
		rightTableMode transaction.DBLockMode
		rightTupleMode transaction.DBLockMode
	}{
		{false, transaction.LockModeIS, transaction.LockModeS},
		{true, transaction.LockModeIX, transaction.LockModeX},
	} {
		mlm := logging.NewMemoryLogManager()
		mlm.SetFlushOnAppend(true)
		sm := storage.NewDiskStorageManager(t.TempDir())
		bp := storage.NewBufferPool(50, sm, mlm)
		lm := transaction.NewLockManager()

		leftTable := &catalog.Table{Oid: 20, Name: "left", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}
		rightTable := &catalog.Table{Oid: 21, Name: "right", Columns: []catalog.Column{{Name: "id", Type: common.IntType}}}

		leftHeap, err := NewTableHeap(leftTable, bp, mlm, lm)
		require.NoError(t, err)
		rightHeap, err := NewTableHeap(rightTable, bp, mlm, lm)
		require.NoError(t, err)

		// Hash index on right.id (column 0).
		rightIdx := indexing.NewMemHashIndex(
			storage.NewRawTupleDesc([]common.Type{common.IntType}),
			[]int{0},
		)

		// Insert left row: id=5.
		_, err = leftHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)

		// Insert right row: id=5. Register it in the index.
		rightRID, err := rightHeap.InsertTuple(nil, makeTestTuple(5))
		require.NoError(t, err)
		key5 := createKey(rightIdx, common.NewIntValue(5))
		require.NoError(t, rightIdx.InsertEntry(key5, rightRID, nil))

		// Simulate check-and-miss: update right row to id=99 without updating the index.
		// The index still maps key=5 → rightRID, but rightRID now holds value=99.
		require.NoError(t, rightHeap.UpdateTuple(nil, rightRID, makeTestTuple(99)))

		txn := transaction.NewTestTransactionContext(lm, 1)

		leftScan := NewSeqScanExecutor(
			planner.NewSeqScanNode(leftHeap.oid, leftHeap.StorageSchema().GetFieldTypes(), transaction.LockModeS),
			leftHeap,
		)
		leftKey := planner.NewColumnValueExpression(0, leftHeap.StorageSchema().GetFieldTypes(), "left.id")
		joinPlan := planner.NewIndexNestedLoopJoinNode(
			leftScan.PlanNode(),
			rightHeap.oid,
			common.ObjectID(0),
			[]planner.Expr{leftKey},
			rightHeap.StorageSchema().GetFieldTypes(),
			tc.forUpdate,
		)
		joinExec := NewIndexJoinExecutor(joinPlan, leftScan, rightIdx, rightHeap)
		require.NoError(t, joinExec.Init(NewExecutorContext(txn)))

		assert.False(t, joinExec.Next(),
			"join must produce no results when right tuple's key no longer matches the probed key")
		require.NoError(t, joinExec.Close())

		// The right tuple was visited via ReadTuple before the key mismatch was detected,
		// so its locks must be held (lock-before-check under SS2PL).
		leftTableTag := transaction.NewTableLockTag(leftHeap.oid)
		rightTableTag := transaction.NewTableLockTag(rightHeap.oid)
		rightTupleTag := transaction.NewTupleLockTag(rightRID)
		leftTableMode, leftTableHeld := txn.HeldLock(leftTableTag)
		assert.True(t, leftTableHeld, "S table lock must be held on left table after join")
		assert.Equal(t, transaction.LockModeS, leftTableMode, "join must acquire S on left table")
		rightTableMode, rightTableHeld := txn.HeldLock(rightTableTag)
		assert.True(t, rightTableHeld, "right table lock must be held after join")
		assert.Equal(t, tc.rightTableMode, rightTableMode, "join must acquire correct lock mode on right table")
		rightTupleMode, rightTupleHeld := txn.HeldLock(rightTupleTag)
		assert.True(t, rightTupleHeld, "right tuple lock must be held on skipped right tuple")
		assert.Equal(t, tc.rightTupleMode, rightTupleMode, "join must acquire correct lock mode on right tuple")

		txn.ReleaseAllLocks()
		assert.False(t, lm.LockHeld(leftTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTableTag), "locks must be released after commit")
		assert.False(t, lm.LockHeld(rightTupleTag), "locks must be released after commit")
	}
}
*/

// TestBufferPool_WAL_BlocksEvictionUntilLogFlushed verifies the WAL-before-eviction
// ordering property: dirty pages are evicted in ascending LSN order as flushedUntil
// advances, each evicted page's data is correctly persisted to disk, and goroutines
// waiting on higher-LSN pages remain blocked until their threshold is crossed.
//
// Setup: 3-frame pool. Frames hold dirty pages at LSN=200, LSN=500, and LSN=1000.
// Three goroutines each fetch an uncached page, all blocking initially. SetFlushedLSN
// is called three times (201 → 501 → 1001), and at each step exactly one goroutine
// unblocks and the evicted page's LSN is verified on disk.
func TestBufferPool_WAL_BlocksEvictionUntilLogFlushed(t *testing.T) {
	mlm := logging.NewMemoryLogManager()
	sm := storage.NewDiskStorageManager(t.TempDir())
	const numFrames = 3
	bp := storage.NewBufferPool(numFrames, sm, mlm)

	oid := common.ObjectID(1)
	dbFile, err := bp.StorageManager().GetDBFile(oid)
	require.NoError(t, err)
	firstPage, err := dbFile.AllocatePage(numFrames * 2)
	require.NoError(t, err)

	// Fill the 3 frames with dirty pages at three distinct LSNs.
	pageLSNs := [numFrames]storage.LSN{200, 500, 1000}
	for i := 0; i < numFrames; i++ {
		pageID := common.PageID{Oid: oid, PageNum: int32(firstPage + i)}
		frame, err := bp.GetPage(pageID)
		require.NoError(t, err)
		frame.MonotonicallyUpdateLSN(pageLSNs[i])
		bp.UnpinPage(frame, true) // dirty, pin count → 0
	}

	// Three goroutines each fetch a different uncached page, all blocking on eviction.
	done := make(chan struct{}, numFrames)
	for i := 0; i < numFrames; i++ {
		pageID := common.PageID{Oid: oid, PageNum: int32(firstPage + numFrames + i)}
		go func(pid common.PageID) {
			defer func() { done <- struct{}{} }()
			frame, err := bp.GetPage(pid)
			assert.NoError(t, err)
			if err == nil {
				bp.UnpinPage(frame, false)
			}
		}(pageID)
	}

	time.Sleep(50 * time.Millisecond)
	select {
	case <-done:
		t.Fatal("a goroutine unblocked before any WAL flush")
	default:
	}

	// Advance flushedUntil past each LSN threshold in turn. Each step must unblock
	// exactly one goroutine, and the evicted page must be readable from disk with its
	// LSN intact (proving the dirty write occurred before the frame was reused).
	buffer := make([]byte, common.PageSize)
	for _, lsn := range []storage.LSN{201, 501, 1001} {
		mlm.SetFlushedLSN(lsn)
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("no goroutine unblocked after flushedUntil=%d", lsn)
		}

		// Only pages with LSN < flushed can be flushed to disk
		for i := 0; i < numFrames; i++ {
			err := dbFile.ReadPage(firstPage+i, buffer)
			assert.NoError(t, err)

			assert.True(t, storage.LSN(binary.LittleEndian.Uint64(buffer[:8])) < lsn,
				"page %d must be flushed before it can be evicted", firstPage+i)
		}
	}
}
