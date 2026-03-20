package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"mit.edu/dsg/godb/catalog"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/execution"
	"mit.edu/dsg/godb/indexing"
	"mit.edu/dsg/godb/logging"
	"mit.edu/dsg/godb/planner"
	"mit.edu/dsg/godb/recovery"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

// WALCheckingDBFileManager wraps a storage.DBFileManager and returns
// instrumented DBFile handles so every WritePage is checked for WAL ordering.
// Call SimulateCrash() to simulate a storage crash: all subsequent WritePage calls are
// silently dropped, and VerifyDisk() can be used to check the resulting disk
// snapshot against the flushedLSN captured at crash time.
type WALCheckingDBFileManager struct {
	inner   storage.DBFileManager
	mlm     *logging.MemoryLogManager
	t       testing.TB
	crashed atomic.Bool
}

// NewWALCheckingDBFileManager wraps inner so that every WritePage call is checked
// against mlm.FlushedUntil(); violations are reported immediately via t.Errorf.
func NewWALCheckingDBFileManager(
	inner storage.DBFileManager,
	mlm *logging.MemoryLogManager,
	t testing.TB,
) *WALCheckingDBFileManager {
	return &WALCheckingDBFileManager{inner: inner, mlm: mlm, t: t}
}

func (m *WALCheckingDBFileManager) GetDBFile(oid common.ObjectID) (storage.DBFile, error) {
	inner, err := m.inner.GetDBFile(oid)
	if err != nil {
		return nil, err
	}
	return &WALCheckingDBFile{inner: inner, mlm: m.mlm, t: m.t, crashed: &m.crashed}, nil
}

func (m *WALCheckingDBFileManager) DeleteDBFile(oid common.ObjectID) error {
	return m.inner.DeleteDBFile(oid)
}

// SimulateCrash simulates a storage crash: records the flushedLSN at this moment and
// silently drops all subsequent WritePage calls. Call VerifyDisk afterwards to
// check the WAL-before-data invariant against the captured snapshot.
func (m *WALCheckingDBFileManager) SimulateCrash() storage.LSN {
	lsn := m.mlm.FlushedUntil()
	m.crashed.Store(true)
	return lsn
}

func pageUninitialized(frame *storage.PageFrame) bool {
	for _, data := range frame.Bytes {
		if data != 0 {
			return false
		}
	}
	return true
}

// VerifyARIESDisk checks the WAL-before-data invariant against the frozen disk snapshot.
//
// For each tuple with WAL records (LSN < crashFlushedLSN), this reads the page from the
// inner storage and finds the last record at or below diskPageLSN (the last modification
// that should be reflected on disk). It then verifies the slot content matches exactly
// that record's expected state. If no record was applied (diskPageLSN < all record LSNs),
// it checks that no unapplied modification is spuriously present on disk.
// SimulateCrash() must be called before VerifyARIESDisk().
func (m *WALCheckingDBFileManager) VerifyARIESDisk(t testing.TB) {
	require.True(t, m.crashed.Load(), "VerifyARIESDisk called without a prior SimulateCrash()")
	bp := storage.NewBufferPool(1024, m.inner, storage.NoopLogManager{})
	tupleRecords := buildTupleRecords(t, m.mlm, bp)

	for rid, records := range tupleRecords {
		pid := rid.PageID
		frame, err := bp.GetPage(pid)
		require.NoError(t, err, "ARIES: GetPage(%v) failed", pid)
		// Uninitialized page means it was never flushed -- that's ok.
		if pageUninitialized(frame) {
			bp.UnpinPage(frame, false)
			continue
		}

		if !heapSlotMatchesRecords(frame, int(rid.Slot), records) {
			assert.Fail(t, fmt.Sprintf(
				"ARIES violation: slot %d on page %v does not reflect the last WAL modification",
				rid.Slot, pid))
		}
		bp.UnpinPage(frame, false)
	}
}

// WALCheckingDBFile wraps a storage.DBFile and intercepts WritePage to verify
// that the page's WAL record has been flushed before the page is written to disk.
// The page LSN lives at bytes [0:8] in little-endian order.
type WALCheckingDBFile struct {
	inner   storage.DBFile
	mlm     *logging.MemoryLogManager
	t       testing.TB
	crashed *atomic.Bool
}

func (f *WALCheckingDBFile) WritePage(pageNum int, frame []byte) error {
	// After a simulated crash, silently drop writes to preserve the disk snapshot.
	if f.crashed.Load() {
		return nil
	}
	if len(frame) >= 8 {
		pageLSN := storage.LSN(int64(binary.LittleEndian.Uint64(frame[:8])))
		if flushed := f.mlm.FlushedUntil(); pageLSN > flushed {
			f.t.Errorf(
				"WAL ordering violation: page (num=%d) written to disk with pageLSN=%d but flushedUntil=%d (buffer pool evicted without calling WaitUntilFlushed)",
				pageNum, pageLSN, flushed)
		}
	}
	return f.inner.WritePage(pageNum, frame)
}

func (f *WALCheckingDBFile) AllocatePage(numPages int) (int, error) {
	return f.inner.AllocatePage(numPages)
}
func (f *WALCheckingDBFile) ReadPage(pageNum int, frame []byte) error {
	return f.inner.ReadPage(pageNum, frame)
}
func (f *WALCheckingDBFile) Sync() error            { return f.inner.Sync() }
func (f *WALCheckingDBFile) Close() error           { return f.inner.Close() }
func (f *WALCheckingDBFile) NumPages() (int, error) { return f.inner.NumPages() }

// buildTupleRecords scans the WAL and collects per-tuple modification records with
// LSN < stopBeforeLSN. Records are appended in WAL order (ascending LSN), so the
// last element of each slice is the most recent modification.
func buildTupleRecords(t testing.TB, mlm *logging.MemoryLogManager, bp *storage.BufferPool) map[common.RecordID][]storage.LogRecord {
	tupleRecords := make(map[common.RecordID][]storage.LogRecord)
	iter, err := mlm.Iterator(0)
	require.NoError(t, err)
	defer iter.Close()
	for iter.Next() {
		lsn := iter.CurrentLSN()
		rec := iter.CurrentRecord()
		switch rec.RecordType() {
		case storage.LogInsert, storage.LogInsertCLR, storage.LogUpdate, storage.LogUpdateCLR, storage.LogDelete, storage.LogDeleteCLR:
			rid := rec.RID()
			pid := rid.PageID
			frame, err := bp.GetPage(pid)
			require.NoError(t, err, "ARIES: GetPage(%v) failed", pid)
			// Uninitialized page means it was never flushed -- that's ok.
			if pageUninitialized(frame) {
				bp.UnpinPage(frame, false)
				continue
			}
			// Changes corresponding to this log entry was purportedly not flushed
			if frame.AsHeapPage().LSN() < lsn {
				bp.UnpinPage(frame, false)
				continue
			}
			tupleRecords[rec.RID()] = append(tupleRecords[rec.RID()], rec)
			bp.UnpinPage(frame, false)
		default:
			continue
		}
	}
	return tupleRecords
}

// heapSlotMatchesRecords returns true if the slot content in rawPage matches the
// expected state described by mr. Returns false for uninitialized pages, out-of-range
// slots, or content that does not match the expected state. RowSize and NumSlots are
// read from fixed header offsets before calling AsHeapPage() to guard against panics.
func heapSlotMatchesRecords(frame *storage.PageFrame, slot int, records []storage.LogRecord) bool {
	hp := frame.AsHeapPage()
	rid := common.RecordID{Slot: int32(slot)}
	lastRecord := records[len(records)-1]
	switch lastRecord.RecordType() {
	case storage.LogInsert, storage.LogUpdate, storage.LogUpdateCLR:
		if hp.IsAllocated(rid) && !hp.IsDeleted(rid) && bytes.Equal(hp.AccessTuple(rid), lastRecord.AfterImage()) {
			return true
		}
		return false
	case storage.LogInsertCLR, storage.LogDelete:
		if hp.IsAllocated(rid) && hp.IsDeleted(rid) {
			return true
		}
		return false
	case storage.LogDeleteCLR:
		if !hp.IsAllocated(rid) || hp.IsDeleted(rid) {
			return false
		}
		for i := len(records) - 2; i >= 0; i-- {
			if records[i].RecordType() == storage.LogInsert || records[i].RecordType() == storage.LogUpdate || records[i].RecordType() == storage.LogUpdateCLR {
				lastAfterImage := records[i].AfterImage()
				return bytes.Equal(hp.AccessTuple(rid), lastAfterImage)
			}
		}
		panic("This path should not be reachable")
	default:
		panic(fmt.Sprintf("unexpected record type: %v", lastRecord.RecordType()))
	}
}

// workloadDB is a test-local database handle that bundles all components
// needed by the workload tests.
type workloadDB struct {
	Catalog            *catalog.Catalog
	BufferPool         *storage.BufferPool
	TableManager       *execution.TableManager
	TransactionManager *transaction.TransactionManager
	LockManager        *transaction.LockManager
	IndexManager       *indexing.IndexManager
	Planner            *planner.SQLPlanner
	mlm                *logging.MemoryLogManager
	spySM              *WALCheckingDBFileManager
}

// makeWorkloadDB constructs a GoDB wired to a MemoryLogManager and a
// WALCheckingDBFileManager.  The caller controls SetFlushOnAppend on the
// returned workloadDB.mlm (default off, so Commit blocks until flushedUntil is
// advanced).  Every WritePage call is automatically checked against flushedUntil;
// violations are reported immediately via t.Errorf.
func makeWorkloadDB(t *testing.T, cat *catalog.Catalog, numBufferPages int) *workloadDB {
	t.Helper()
	dbDir := t.TempDir()
	mlm := logging.NewMemoryLogManager()

	spySM := NewWALCheckingDBFileManager(storage.NewDiskStorageManager(dbDir), mlm, t)
	bufferPool := storage.NewBufferPool(numBufferPages, spySM, mlm)
	lockManager := transaction.NewLockManager()
	txnManager := transaction.NewTransactionManager(mlm, bufferPool, lockManager)

	tableManager, err := execution.NewTableManager(cat, bufferPool, mlm, lockManager)
	require.NoError(t, err)
	indexManager, err := indexing.NewIndexManager(cat)
	require.NoError(t, err)

	rm := recovery.NewNoLogRecoveryManager(bufferPool, txnManager, cat, tableManager, indexManager)
	require.NoError(t, rm.Recover())

	logicalRules := []planner.LogicalRule{&planner.PredicatePushDownRule{}}
	physicalRules := []planner.PhysicalConversionRule{
		&planner.SeqScanRule{},
		&planner.IndexScanRule{},
		&planner.IndexLookupRule{},
		&planner.BlockNestedLoopJoinRule{},
		&planner.LimitRule{},
		&planner.SubqueryRule{},
		&planner.AggregationRule{},
		&planner.ProjectionRule{},
		&planner.FilterRule{},
		&planner.SortRule{},
		&planner.InsertRule{},
		&planner.DeleteRule{},
		&planner.UpdateRule{},
		&planner.ValuesRule{},
	}

	return &workloadDB{
		Catalog:            cat,
		BufferPool:         bufferPool,
		TableManager:       tableManager,
		TransactionManager: txnManager,
		LockManager:        lockManager,
		IndexManager:       indexManager,
		Planner:            planner.NewSQLPlanner(cat, logicalRules, physicalRules),
		mlm:                mlm,
		spySM:              spySM,
	}
}

// execSQL plans and executes sqlStr within txn, returning all output rows.
// String Values are copied so they remain valid after the executor is closed.
// On error, the caller is responsible for aborting txn.
func (db *workloadDB) execSQL(txn *transaction.TransactionContext, sqlStr string) ([][]common.Value, error) {
	plan, err := db.Planner.Plan(sqlStr, true)
	if err != nil {
		return nil, err
	}
	ex, err := execution.BuildExecutorTree(plan, db.Catalog, db.TableManager, db.IndexManager)
	if err != nil {
		return nil, err
	}
	if err := ex.Init(execution.NewExecutorContext(txn)); err != nil {
		ex.Close()
		return nil, err
	}
	defer ex.Close()

	var rows [][]common.Value
	for ex.Next() {
		tup := ex.Current()
		row := make([]common.Value, tup.NumColumns())
		for i := range row {
			row[i] = tup.GetValue(i).Copy()
		}
		rows = append(rows, row)
	}
	return rows, ex.Error()
}

// startPeriodicFlusher launches a background WAL flusher and returns a stop
// function that cancels the flusher and unblocks any pending WaitUntilFlushed
// calls. Always call the returned stop function when the workload finishes.
func startPeriodicFlusher(wdb *workloadDB) func() {
	ctx, cancel := context.WithCancel(context.Background())
	go logging.PeriodicWALFlusher(ctx, wdb.mlm)
	return func() {
		cancel()
		wdb.mlm.SetFlushedLSN(storage.LSN(math.MaxInt64))
	}
}

// abortOnDeadlock asserts that err is a deadlock error, aborts txn, yields the
// scheduler, and returns true. Returns false if err is nil.
// Typical use: if abortOnDeadlock(t, wdb, txn, err) { continue }
func abortOnDeadlock(t testing.TB, wdb *workloadDB, txn *transaction.TransactionContext, err error) bool {
	if err == nil {
		return false
	}
	assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
	require.NoError(t, wdb.TransactionManager.Abort(txn))
	runtime.Gosched()
	return true
}

// newKVCatalog creates a catalog with a kv(k INT, v INT) table and a btree
// primary-key index on k.
func newKVCatalog(t *testing.T) *catalog.Catalog {
	cat, err := catalog.NewCatalog(catalog.NullPersistenceProvider{})
	require.NoError(t, err)
	_, err = cat.AddTable("kv", []catalog.Column{
		{Name: "k", Type: common.IntType},
		{Name: "v", Type: common.IntType},
	})
	require.NoError(t, err)
	_, err = cat.AddIndex("kv_pk", "kv", "btree", []string{"k"})
	require.NoError(t, err)
	return cat
}

// setupKVDB pre-populates the kv table with rows (1,initialVal)..(numKeys,initialVal)
// using nil-txn inserts so the WAL is not polluted with setup records.
func setupKVDB(t *testing.T, wdb *workloadDB, numKeys int, initialVal int64) {
	for k := 1; k <= numKeys; k++ {
		_, err := wdb.execSQL(nil, fmt.Sprintf("INSERT INTO kv VALUES (%d, %d)", k, initialVal))
		require.NoError(t, err)
	}
}

// TestWAL_FlushStress exercises the WAL-before-data invariant under a simulated
// crash. A tiny buffer pool forces aggressive page evictions, creating many windows where a buggy
// implementation could evict a modified page before writing its WAL record or updating its pageLSN.
//
// After a random delay, SimulateCrash() freezes the disk snapshot.  Workers continue
// unaware (subsequent WritePage calls are silently dropped), then VerifyARIESDisk
// checks that no page on disk contains a modification whose WAL record is not
// reflected in the page's on-disk LSN.
func TestWAL_FlushStress(t *testing.T) {
	const numIterations = 20
	for i := 0; i < numIterations; i++ {
		rng := rand.New(rand.NewSource(int64(i)))

		const (
			numKeys        = 10000
			numBufferPages = 5
			numWorkers     = 8
			opsPerTxn      = 5
		)

		wdb := makeWorkloadDB(t, newKVCatalog(t), numBufferPages)
		wdb.mlm.SetFlushOnAppend(true)

		// Setup: insert numKeys rows with initial value 0.
		for k := 1; k <= numKeys; k++ {
			txn, err := wdb.TransactionManager.Begin()
			require.NoError(t, err)
			if _, err := wdb.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, 0)", k)); err != nil {
				require.NoError(t, wdb.TransactionManager.Abort(txn))
				require.NoError(t, err)
			}
			require.NoError(t, wdb.TransactionManager.Commit(txn))
		}

		// Switch to periodic (realistic) WAL flushing.
		wdb.mlm.SetFlushOnAppend(false)
		ctx, cancelFlush := context.WithCancel(context.Background())
		go logging.PeriodicWALFlusher(ctx, wdb.mlm)

		// Pre-generate crash timing and per-worker seeds before starting workers,
		// so the crash point is not correlated with worker scheduling.
		workerSeeds := make([]int64, numWorkers)
		for i := range workerSeeds {
			workerSeeds[i] = rng.Int63()
		}
		crashDelay := time.Duration(rng.Int63n(int64(500 * time.Millisecond)))

		var stopped atomic.Bool
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				wrng := rand.New(rand.NewSource(workerSeeds[workerID]))
			Outer:
				for !stopped.Load() {
					txn, err := wdb.TransactionManager.Begin()
					if err != nil {
						return
					}
					for i := 0; i < opsPerTxn; i++ {
						key := wrng.Int63n(numKeys) + 1
						val := wrng.Int63()
						_, err = wdb.execSQL(txn, fmt.Sprintf(
							"UPDATE kv SET v = %d WHERE k = %d", val, key))
						if err != nil {
							require.NoError(t, wdb.TransactionManager.Abort(txn))
							continue Outer
						}
					}
					assert.NoError(t, wdb.TransactionManager.Commit(txn))
				}
			}(w)
		}

		time.Sleep(crashDelay)
		cancelFlush()
		wdb.spySM.SimulateCrash()
		stopped.Store(true)
		// unblock any final WaitUntilFlushed calls. storage manager will ignore flushed disk pages from here on
		wdb.mlm.SetFlushedLSN(storage.LSN(math.MaxInt64))
		wg.Wait()

		// Core crash invariant: no page on disk was modified without its pageLSN
		// being updated (and the corresponding WAL record being flushed first).
		wdb.spySM.VerifyARIESDisk(t)
		logging.VerifyWALOrdering(t, wdb.mlm)
	}
}

// TestConcurrent_Bank is an end-to-end port of the banking workload.
// Three concurrent actor types exercise multi-granularity locking on a kv(k INT, v INT)
// table and verify the total value conservation invariant:
//
//  1. Point transferors: two targeted UPDATEs (Table IX + Tuple X per row). Multiple
//     transferors touching disjoint keys run concurrently (IX+IX compatible).
//  2. IS→S scanners: point SELECT (Table IS + Tuple S via IndexLookup) then full SUM
//     (upgrades table lock IS→S). S is incompatible with IX and X, so no transfers or
//     bulk ops are in-flight during the aggregate, guaranteeing a consistent snapshot.
//     The SUM result must always equal the expected total.
//  3. Bulk operators: UPDATE all rows with a random bonus (no WHERE clause →
//     Table X via SeqScan ForUpdate + Tuple X on every row). Changes are immediately
//     visible to the next scanner.
func TestConcurrent_Bank(t *testing.T) {
	const (
		numKeys        = 50
		initialBalance = 10000
	)
	workloadTime := 10 * time.Second

	wdb := makeWorkloadDB(t, newKVCatalog(t), 100)
	setupKVDB(t, wdb, numKeys, initialBalance)
	stopFlusher := startPeriodicFlusher(wdb)

	// totalBonus tracks the cumulative per-key bonus added by bulk operators.
	// Updated BEFORE Commit (while Table-X still held), so any scanner that acquires
	// Table-S after the commit sees both the new balances and the updated totalBonus.
	var totalBonus atomic.Int64

	start := time.Now()
	var wg sync.WaitGroup

	// 1. Point transferors (8 goroutines): Table IX + Tuple X on src and dst.
	// Multiple transferors touching disjoint accounts proceed concurrently (IX+IX compatible).
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))
			for time.Since(start) < workloadTime {
				src := int64(rng.Intn(numKeys)) + 1
				dst := int64(rng.Intn(numKeys)) + 1
				if src == dst {
					continue
				}
				txn, err := wdb.TransactionManager.Begin()
				require.NoError(t, err)
				_, err = wdb.execSQL(txn, fmt.Sprintf(
					"UPDATE kv SET v = v - 1 WHERE k = %d", src))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				_, err = wdb.execSQL(txn, fmt.Sprintf(
					"UPDATE kv SET v = v + 1 WHERE k = %d", dst))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				assert.NoError(t, wdb.TransactionManager.Commit(txn))
			}
		}(i)
	}

	// 2. IS→S scanners (3 goroutines): point SELECT acquires IS+Tuple-S via IndexLookup,
	// then SUM upgrades the table lock IS→S. Because S is incompatible with IX and X,
	// no transfer or bulk op is in-flight when the aggregate runs, guaranteeing a
	// consistent snapshot. The SUM result is checked against the expected total.
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id + 200)))
			for time.Since(start) < workloadTime {
				txn, err := wdb.TransactionManager.Begin()
				require.NoError(t, err)
				k := int64(rng.Intn(numKeys)) + 1
				pointRows, err := wdb.execSQL(txn, fmt.Sprintf(
					"SELECT v FROM kv WHERE k = %d", k))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				require.Len(t, pointRows, 1, "IS→S scanner: key %d not found", k)
				sumRows, err := wdb.execSQL(txn, "SELECT SUM(kv.v) FROM kv")
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				got := sumRows[0][0].IntValue()
				expected := int64(numKeys*initialBalance) + totalBonus.Load()
				assert.Equal(t, expected, got, "IS→S scanner: sum mismatch")
				assert.NoError(t, wdb.TransactionManager.Commit(txn))
				runtime.Gosched()
			}
		}(i)
	}

	// 3. Bulk operators (2 goroutines): UPDATE all accounts (no WHERE clause).
	// SeqScan with ForUpdate=true acquires Table-X; all other actors are blocked
	// until commit. totalBonus is advanced before Commit so the invariant holds.
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id + 100)))
			for time.Since(start) < workloadTime {
				bonus := int64(rng.Intn(10) + 1)
				txn, err := wdb.TransactionManager.Begin()
				require.NoError(t, err)
				_, err = wdb.execSQL(txn, fmt.Sprintf(
					"UPDATE kv SET v = v + %d", bonus))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				// Advance totalBonus while Table-X is still held. Any scanner that
				// acquires Table-S after Commit will see both the new balances and
				// the updated totalBonus atomically from its perspective.
				totalBonus.Add(bonus * int64(numKeys))
				if err = wdb.TransactionManager.Commit(txn); err != nil {
					// Commit failed: revert the premature totalBonus increment.
					totalBonus.Add(-bonus * int64(numKeys))
					assert.NoError(t, err)
				}
				runtime.Gosched()
			}
		}(i)
	}

	wg.Wait()
	stopFlusher()

	// Final conservation check: total balance must equal initial total plus all committed bonuses.
	finalTxn, err := wdb.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := wdb.execSQL(finalTxn, "SELECT SUM(kv.v) FROM kv")
	require.NoError(t, err)
	require.NoError(t, wdb.TransactionManager.Commit(finalTxn))
	assert.Equal(t, int64(numKeys*initialBalance)+totalBonus.Load(), rows[0][0].IntValue(),
		"final sum conservation violated")

	logging.VerifyWALOrdering(t, wdb.mlm)
}

// TestAbortStorm starts a table with numKeys rows and runs concurrent workers
// that each insert into a per-worker "dirty zone" (k > numKeys) and delete a
// random original row, then explicitly abort. Concurrent scanners verify:
//  1. The original zone always has exactly numKeys rows (no phantom deletes).
//  2. The dirty zone is always empty (no dirty reads of uncommitted inserts).
func TestAbortStorm(t *testing.T) {
	const (
		numKeys        = 500
		numWorkers     = 8
		txnsPerWorker  = 5000
		opsPerTxn      = 5 // ops per txn; each op is randomly an insert or delete
		numBufferPages = 50
		insertFraction = 0.5 // probability that each op is an insert into the dirty zone
	)

	wdb := makeWorkloadDB(t, newKVCatalog(t), numBufferPages)
	setupKVDB(t, wdb, numKeys, 0)

	stopFlusher := startPeriodicFlusher(wdb)

	var stopped atomic.Bool
	var workerWg, scannerWg sync.WaitGroup

	// Scanners: verify no dirty reads via randomized point lookups (IS+Tuple-S),
	for s := 0; s < numWorkers; s++ {
		scannerWg.Add(1)
		go func(id int) {
			defer scannerWg.Done()
			rng := rand.New(rand.NewSource(int64(id * 7919)))
			for !stopped.Load() {
				txn, err := wdb.TransactionManager.Begin()
				if err != nil {
					return
				}
				// Check that a random original-zone key still exists (no dirty delete).
				origKey := rng.Int63n(int64(numKeys)) + 1
				origRows, err := wdb.execSQL(txn, fmt.Sprintf(
					"SELECT v FROM kv WHERE k = %d", origKey))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				assert.Len(t, origRows, 1,
					"dirty delete: original-zone key %d not found", origKey)
				// Check that a random dirty-zone key does not exist (no dirty insert).
				dirtyKey := rng.Int63n(int64(numWorkers*opsPerTxn)) + int64(numKeys) + 1
				dirtyRows, err := wdb.execSQL(txn, fmt.Sprintf(
					"SELECT v FROM kv WHERE k = %d", dirtyKey))
				if abortOnDeadlock(t, wdb, txn, err) {
					continue
				}
				assert.Empty(t, dirtyRows,
					"dirty insert: dirty-zone key %d unexpectedly found", dirtyKey)
				assert.NoError(t, wdb.TransactionManager.Commit(txn))
				runtime.Gosched()
			}
		}(s)
	}

	// Workers: every transaction explicitly aborts.
	for w := 0; w < numWorkers; w++ {
		workerWg.Add(1)
		go func(workerID int) {
			defer workerWg.Done()
			rng := rand.New(rand.NewSource(int64(workerID*1000003 + 42)))
			insertBase := int64(numKeys + 1 + workerID*opsPerTxn)
			for i := 0; i < txnsPerWorker; i++ {
				txn, err := wdb.TransactionManager.Begin()
				if err != nil {
					return
				}
				aborted := false
				insertIdx := int64(0)
				// Each op is randomly an insert into the dirty zone or a delete from
				// the original zone. Because all txns abort, both are undone and the
				// dirty zone slots are always available for reuse each transaction.
				for j := 0; j < opsPerTxn && !aborted; j++ {
					if rng.Float64() < insertFraction {
						_, err := wdb.execSQL(txn, fmt.Sprintf(
							"INSERT INTO kv VALUES (%d, 0)", insertBase+insertIdx))
						insertIdx++
						aborted = abortOnDeadlock(t, wdb, txn, err)
					} else {
						deleteKey := rng.Int63n(int64(numKeys)) + 1
						_, err := wdb.execSQL(txn, fmt.Sprintf(
							"DELETE FROM kv WHERE k = %d", deleteKey))
						aborted = abortOnDeadlock(t, wdb, txn, err)
					}
				}
				if !aborted {
					require.NoError(t, wdb.TransactionManager.Abort(txn))
				}
			}
		}(w)
	}

	workerWg.Wait()
	stopped.Store(true)
	scannerWg.Wait()
	stopFlusher()

	// Final state: table must be identical to its initial state.
	finalTxn, err := wdb.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := wdb.execSQL(finalTxn, "SELECT COUNT(*) FROM kv")
	require.NoError(t, err)
	require.NoError(t, wdb.TransactionManager.Commit(finalTxn))
	assert.Equal(t, int64(numKeys), rows[0][0].IntValue(), "final count wrong after abort storm")
	logging.VerifyWALOrdering(t, wdb.mlm)
}

// ycsbConfig parameterises a YCSB-style key-value workload.
type ycsbConfig struct {
	numKeys        int
	numWorkers     int
	txnsPerWorker  int
	opsPerTxn      int
	numBufferPages int
	// readFraction is the probability [0,1] that each operation is a read.
	// 0.5 means 50% reads / 50% writes.
	readFraction float64
	// zipfTheta controls key-access skew. 0 = uniform; values near 1 (e.g. 0.99)
	// give the classic YCSB Zipfian distribution where a small fraction of keys
	// receives most accesses. Uses Go's rand.NewZipf with s=1+zipfTheta, v=1.
	zipfTheta float64
}

// newKeyGen returns a key-selection function for a single worker. When
// zipfTheta == 0 the selection is uniform; otherwise it is Zipfian.
func newKeyGen(rng *rand.Rand, cfg ycsbConfig) func() int64 {
	if cfg.zipfTheta == 0 {
		return func() int64 { return rng.Int63n(int64(cfg.numKeys)) + 1 }
	}
	z := rand.NewZipf(rng, 1+cfg.zipfTheta, 1, uint64(cfg.numKeys-1))
	return func() int64 { return int64(z.Uint64()) + 1 }
}

// opEntry is one entry in a transaction's unified read/write log.
type opEntry struct {
	isWrite bool
	key     int64
	val     int64
}

// txnRecord holds the complete operation log for a single transaction,
// plus the transaction ID needed for WAL correlation.
// ops records every read and write in execution order, enabling RYOW,
// within-txn NRR, and dirty-read detection.
type txnRecord struct {
	tid       common.TransactionID
	ops       []opEntry // unified log: reads and writes in execution order
	committed bool
}

func doRead(t *testing.T, wdb *workloadDB, txn *transaction.TransactionContext, record *txnRecord, keyGen func() int64) bool {
	key := keyGen()
	valRows, err := wdb.execSQL(txn, fmt.Sprintf(
		"SELECT v FROM kv WHERE k = %d", key))
	if err != nil {
		assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
		return false
	}
	record.ops = append(record.ops, opEntry{isWrite: false, key: key, val: valRows[0][0].IntValue()})
	return true
}

func doWrite(t *testing.T, wdb *workloadDB, txn *transaction.TransactionContext, record *txnRecord, keyGen func() int64, writeCounter *atomic.Int64) bool {
	key := keyGen()
	val := writeCounter.Add(1) // globally unique so dirty-read detection is unambiguous
	_, err := wdb.execSQL(txn, fmt.Sprintf(
		"UPDATE kv SET v = %d WHERE k = %d", val, key))
	if err != nil {
		assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
		runtime.Gosched()
		return false
	}
	record.ops = append(record.ops, opEntry{isWrite: true, key: key, val: val})
	return true
}

func checkRead(t *testing.T, wdb *workloadDB, txn *transaction.TransactionContext, record *txnRecord) {
	// Compute the last write value per key so we know what RYOW should return.
	lastWrite := make(map[int64]int64)
	for _, op := range record.ops {
		if op.isWrite {
			lastWrite[op.key] = op.val
		}
	}
	// Re-read every key that was read during the transaction (no need to log these).
	for _, op := range record.ops {
		if op.isWrite {
			continue
		}
		rows, err := wdb.execSQL(txn, fmt.Sprintf("SELECT v FROM kv WHERE k = %d", op.key))
		require.NoError(t, err)
		got := rows[0][0].IntValue()
		if w, written := lastWrite[op.key]; written {
			assert.Equal(t, w, got, fmt.Sprintf("RYOW violation: txn %v wrote key %d = %d, re-read returned %d",
				record.tid, op.key, w, got))
		} else {
			assert.Equal(t, op.val, got, fmt.Sprintf("NRR violation: txn %v read key %d = %d, re-read returned %d (S-lock released early?)",
				record.tid, op.key, op.val, got))
		}
	}
}

func runYCSBTxn(t *testing.T, wdb *workloadDB, cfg ycsbConfig, txn *transaction.TransactionContext, record *txnRecord, rng *rand.Rand, writeCounter *atomic.Int64, keyGen func() int64) {
	for i := 0; i < cfg.opsPerTxn; i++ {
		var ok bool
		if rng.Float64() < cfg.readFraction {
			ok = doRead(t, wdb, txn, record, keyGen)
		} else {
			ok = doWrite(t, wdb, txn, record, keyGen, writeCounter)
		}
		if !ok {
			err := wdb.TransactionManager.Abort(txn)
			record.committed = false
			require.NoError(t, err)
			return
		}
	}
	checkRead(t, wdb, txn, record)
	err := wdb.TransactionManager.Commit(txn)
	record.committed = true
	require.NoError(t, err)
}

// runYCSBWorkload launches cfg.numWorkers goroutines. Each worker performs
// cfg.txnsPerWorker transactions; each transaction mixes reads and writes on
// random keys according to cfg.readFraction, using globally unique write values.
func runYCSBWorkload(t *testing.T, wdb *workloadDB, cfg ycsbConfig) []*txnRecord {
	var writeCounter atomic.Int64
	writeCounter.Store(1) // 0 is the initial value; writes start from 1
	allRecords := make([][]*txnRecord, cfg.numWorkers)
	var wg sync.WaitGroup
	for w := 0; w < cfg.numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID*1000003 + 13)))
			keyGen := newKeyGen(rng, cfg)
			for txnNum := 0; txnNum < cfg.txnsPerWorker; txnNum++ {
				txn, err := wdb.TransactionManager.Begin()
				require.NoError(t, err)
				rec := &txnRecord{tid: txn.ID()}
				runYCSBTxn(t, wdb, cfg, txn, rec, rng, &writeCounter, keyGen)
				allRecords[workerID] = append(allRecords[workerID], rec)
			}
		}(w)
	}
	wg.Wait()

	// Flatten per-worker slices into a single slice for the verifier.
	var records []*txnRecord
	for _, recs := range allRecords {
		records = append(records, recs...)
	}
	return records
}

type writeRecord struct {
	writtenBy    common.TransactionID
	valueWritten int64
	committed    bool
}

// verifyYCSBConsistency checks REPEATABLE_READ isolation invariants using
// the per-transaction read/write sets correlated against WAL commit order.
// For every transaction T (committed or aborted) and every key K that T read
// with value V, the following must hold:
//
//  1. V was committed before T read it: V must be either the initial value (0)
//     or appear in the committed write history of K. Violation ⇒ dirty read.
//
//  2. V is written by the transaction with largest commit LSN smaller than the commit/abort LSN of the reading txn
func verifyYCSBConsistency(t *testing.T, records []*txnRecord, mlm *logging.MemoryLogManager) {
	endLSN := make(map[common.TransactionID]storage.LSN)
	iter, err := mlm.Iterator(0)
	require.NoError(t, err)
	defer iter.Close()
	for iter.Next() {
		rec := iter.CurrentRecord()
		if rec.RecordType() == storage.LogCommit || rec.RecordType() == storage.LogAbort {
			endLSN[rec.TxnID()] = iter.CurrentLSN()
		}
	}

	valueWriter := make(map[int64]common.TransactionID)
	history := make(map[int64][]writeRecord)
	for _, rec := range records {
		for _, op := range rec.ops {
			if !op.isWrite {
				continue
			}
			valueWriter[op.val] = rec.tid
			h := history[op.key]
			// Other transactions should never read intermediate values
			if len(h) == 0 || h[len(h)-1].writtenBy != rec.tid {
				history[op.key] = append(h, writeRecord{
					writtenBy:    rec.tid,
					valueWritten: op.val,
					committed:    rec.committed,
				})
			} else {
				h[len(h)-1].valueWritten = op.val
			}
		}
	}
	for _, vals := range history {
		slices.SortFunc(vals, func(a, b writeRecord) int {
			return cmp.Compare(endLSN[valueWriter[a.valueWritten]], endLSN[valueWriter[b.valueWritten]])
		})
	}

	for _, rec := range records {
		cLSN, _ := endLSN[rec.tid]
		for opIdx, op := range rec.ops {
			if op.isWrite {
				continue
			}

			// Own-write check: T must see exactly its most-recent write to this key.
			if valueWriter[op.val] == rec.tid {
				latestOwnWrite := int64(-1)
				for j := opIdx - 1; j >= 0; j-- {
					if rec.ops[j].isWrite && rec.ops[j].key == op.key {
						latestOwnWrite = rec.ops[j].val
						break
					}
				}
				assert.Equal(t, latestOwnWrite, op.val,
					fmt.Sprintf("RYOW stale-write violation: txn %v read key %d = %d but latest own write was %d",
						rec.tid, op.key, op.val, latestOwnWrite))
				continue
			}

			wLSN, committed := endLSN[valueWriter[op.val]]
			assert.True(t, op.val == 0 || (committed && wLSN < cLSN))
			h := history[op.key]
			for i, v := range h {
				if v.valueWritten == op.val {
					for _, u := range h[i+1:] {
						if u.committed {
							assert.True(t, endLSN[valueWriter[u.valueWritten]] >= cLSN)
							break
						}
					}
					break
				}
			}
		}
	}
}

// ycsbTestHarness wires together schema creation, workload execution, and all
// invariant checkers for one YCSB configuration.
func ycsbTestHarness(t *testing.T, cfg ycsbConfig) {
	wdb := makeWorkloadDB(t, newKVCatalog(t), cfg.numBufferPages)
	setupKVDB(t, wdb, cfg.numKeys, 0)

	stopFlusher := startPeriodicFlusher(wdb)

	records := runYCSBWorkload(t, wdb, cfg)

	stopFlusher()
	verifyYCSBConsistency(t, records, wdb.mlm)
	logging.VerifyWALOrdering(t, wdb.mlm)
}

func TestConcurrent_YCSB_HotSpots(t *testing.T) {
	ycsbTestHarness(t, ycsbConfig{
		// hot-spots: tiny key space forces maximum lock contention and deadlocks,
		// stressing the deadlock detector, retry path, and within-txn NRR detection.
		numKeys:        5,
		numWorkers:     16,
		txnsPerWorker:  10000,
		opsPerTxn:      10,
		numBufferPages: 10,
		readFraction:   0.5,
	})
}

func TestConcurrent_YCSB_Balanced(t *testing.T) {
	ycsbTestHarness(t, ycsbConfig{
		// balanced: moderate contention with a mix of reads and writes.
		numKeys:        500,
		numWorkers:     16,
		txnsPerWorker:  3000,
		opsPerTxn:      10,
		numBufferPages: 100,
		readFraction:   0.5,
	})
}

func TestConcurrent_YCSB_ReadHeavy_ZipfSkew(t *testing.T) {
	ycsbTestHarness(t, ycsbConfig{
		// read-heavy with Zipf skew: a few hot keys accumulate many concurrent
		// S-locks while occasional writers queue for X-locks. Primary stress test
		// for RR — any premature S-lock release shows up as an NRR violation.
		numKeys:        500,
		numWorkers:     16,
		txnsPerWorker:  500,
		opsPerTxn:      10,
		numBufferPages: 100,
		readFraction:   0.8,
		zipfTheta:      0.99,
	})
}

func TestConcurrent_YCSB_LongTransactions(t *testing.T) {
	ycsbTestHarness(t, ycsbConfig{
		// long transactions with Zipf skew: each transaction runs 50 ops, so the
		// checkRead phase re-reads keys first seen early in the txn after many
		// intervening committed writes by other workers. Primary target: NRR
		// violations (S-lock released early) and RYOW across interleaving writes.
		numKeys:        500,
		numWorkers:     8,
		txnsPerWorker:  5000,
		opsPerTxn:      50,
		numBufferPages: 100,
		readFraction:   0.7,
		zipfTheta:      0.9,
	})
}

func TestConcurrent_YCSB_BufferPressure(t *testing.T) {
	ycsbTestHarness(t, ycsbConfig{
		// large key space squeezed through a tiny buffer: pages are constantly
		// evicted and re-fetched under concurrent load, stressing WAL-before-data
		// ordering and the buffer pool's scan-resistant eviction policy.
		numKeys:        1000,
		numWorkers:     8,
		txnsPerWorker:  1000,
		opsPerTxn:      10,
		numBufferPages: 5,
		readFraction:   0.7,
	})
}
