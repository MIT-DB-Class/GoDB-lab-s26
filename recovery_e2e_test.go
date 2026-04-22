package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
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

// recoverySpyDBFileManager wraps storage.DBFileManager for WAL-before-data checking
// and crash simulation. It uses storage.LogManager so it works with DoubleBufferLogManager.
type recoverySpyDBFileManager struct {
	inner   storage.DBFileManager
	lm      storage.LogManager
	t       testing.TB
	crashed atomic.Bool
}

func (m *recoverySpyDBFileManager) GetDBFile(oid common.ObjectID) (storage.DBFile, error) {
	inner, err := m.inner.GetDBFile(oid)
	if err != nil {
		return nil, err
	}
	return &recoverySpyDBFile{inner: inner, lm: m.lm, t: m.t, crashed: &m.crashed}, nil
}

func (m *recoverySpyDBFileManager) DeleteDBFile(oid common.ObjectID) error {
	return m.inner.DeleteDBFile(oid)
}

// SimulateCrash freezes the disk: all subsequent WritePage calls are silently dropped.
func (m *recoverySpyDBFileManager) SimulateCrash() {
	m.crashed.Store(true)
}

type recoverySpyDBFile struct {
	inner   storage.DBFile
	lm      storage.LogManager
	t       testing.TB
	crashed *atomic.Bool
}

func (f *recoverySpyDBFile) WritePage(pageNum int, frame []byte) error {
	if f.crashed.Load() {
		return nil
	}
	if len(frame) >= 8 {
		pageLSN := storage.LSN(int64(binary.LittleEndian.Uint64(frame[:8])))
		if flushed := f.lm.FlushedUntil(); pageLSN >= flushed {
			f.t.Errorf(
				"WAL ordering violation: page (num=%d) written to disk with pageLSN=%d but flushedUntil=%d",
				pageNum, pageLSN, flushed)
		}
	}
	return f.inner.WritePage(pageNum, frame)
}

func (f *recoverySpyDBFile) AllocatePage(numPages int) (int, error) {
	return f.inner.AllocatePage(numPages)
}
func (f *recoverySpyDBFile) ReadPage(pageNum int, frame []byte) error {
	return f.inner.ReadPage(pageNum, frame)
}
func (f *recoverySpyDBFile) Sync() error            { return f.inner.Sync() }
func (f *recoverySpyDBFile) Close() error           { return f.inner.Close() }
func (f *recoverySpyDBFile) NumPages() (int, error) { return f.inner.NumPages() }

// recoveryWorkloadDB is a standalone test DB for recovery e2e tests.
// It uses DoubleBufferLogManager for realistic WAL crash behavior.
type recoveryWorkloadDB struct {
	Catalog            *catalog.Catalog
	BufferPool         *storage.BufferPool
	TableManager       *execution.TableManager
	TransactionManager *transaction.TransactionManager
	LockManager        *transaction.LockManager
	IndexManager       *indexing.IndexManager
	Planner            *planner.SQLPlanner
	lm                 *logging.DoubleBufferLogManager
	closeLog           sync.Once // guards lm.Close() so it is called exactly once
	spySM              *recoverySpyDBFileManager
	dbDir              string
	logPath            string
}

// closeLM closes the log manager exactly once. Safe to call from cleanup and
// from simulateCrash.
func (wdb *recoveryWorkloadDB) closeLM() {
	wdb.closeLog.Do(func() { _ = wdb.lm.Close() })
}

func testPlannerRules() ([]planner.LogicalRule, []planner.PhysicalConversionRule) {
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
	return logicalRules, physicalRules
}

func makeRecoveryWorkloadDB(t *testing.T, cat *catalog.Catalog, numBufferPages int) *recoveryWorkloadDB {
	dbDir := t.TempDir()
	logPath := filepath.Join(dbDir, "wal.log")
	lm, err := logging.NewDoubleBufferLogManager(logPath)
	require.NoError(t, err)

	spySM := &recoverySpyDBFileManager{
		inner: storage.NewDiskStorageManager(dbDir),
		lm:    lm,
		t:     t,
	}
	bp := storage.NewBufferPool(numBufferPages, spySM, lm)
	lockMgr := transaction.NewLockManager()
	txnMgr := transaction.NewTransactionManager(lm, bp, lockMgr)

	tableManager, err := execution.NewTableManager(cat, bp, lm, lockMgr)
	require.NoError(t, err)
	indexManager, err := indexing.NewIndexManager(cat)
	require.NoError(t, err)

	rm := recovery.NewNoLogRecoveryManager(bp, txnMgr, cat, tableManager, indexManager)
	require.NoError(t, rm.Recover())

	logicalRules, physicalRules := testPlannerRules()

	wdb := &recoveryWorkloadDB{
		Catalog:            cat,
		BufferPool:         bp,
		TableManager:       tableManager,
		TransactionManager: txnMgr,
		LockManager:        lockMgr,
		IndexManager:       indexManager,
		Planner:            planner.NewSQLPlanner(cat, logicalRules, physicalRules),
		lm:                 lm,
		spySM:              spySM,
		dbDir:              dbDir,
		logPath:            logPath,
	}
	t.Cleanup(wdb.closeLM)
	return wdb
}

// simulateCrash freezes the disk, closes the log manager (flushing remaining
// in-memory buffer to disk), then truncates the log file back to the
// pre-crash flushed offset. This simulates losing the in-memory WAL tail.
//
// Ordering matters: SimulateCrash() must come BEFORE FlushedUntil() to avoid
// a race where the DoubleBufferLogManager's internal flush goroutine advances
// flushedUntil between the snapshot and the freeze, allowing pages with
// pageLSN > crashLSN to land on disk. By freezing the disk first, all
// subsequent page evictions are silently dropped, so every page on disk has
// pageLSN < crashLSN and recovery is consistent.
func simulateCrash(t *testing.T, rwdb *recoveryWorkloadDB) {
	// Freeze disk FIRST: no more page writes from this point.
	// This must come before FlushedUntil() to prevent a race where the
	// internal flush goroutine advances flushedUntil between the snapshot
	// and the disk freeze, allowing pages with pageLSN > crashLSN on disk.
	rwdb.spySM.SimulateCrash()
	// Capture the durably-flushed offset AFTER the disk is frozen.
	// All pages on disk now have pageLSN < flushedUntil <= crashLSN.
	crashLSN := rwdb.lm.FlushedUntil()
	// Shut down the log manager (flushes active buffer to disk, unblocks waiters).
	rwdb.closeLM()
	// Truncate the log file back to crashLSN — simulates losing the WAL tail
	// that was in memory but never durably flushed before the crash.
	require.NoError(t, os.Truncate(rwdb.logPath, int64(crashLSN)))
}

// recoverFromCrash opens a fresh DoubleBufferLogManager on the (truncated) log
// file and runs ARIES recovery, returning a new recoveryWorkloadDB ready for
// post-recovery verification and continued work.
func recoverFromCrash(t *testing.T, cat *catalog.Catalog, dbDir, logPath string) *recoveryWorkloadDB {
	lm, err := logging.NewDoubleBufferLogManager(logPath)
	require.NoError(t, err)

	spySM := &recoverySpyDBFileManager{
		inner: storage.NewDiskStorageManager(dbDir),
		lm:    lm,
		t:     t,
	}
	bp := storage.NewBufferPool(64, spySM, lm)
	lockMgr := transaction.NewLockManager()
	txnMgr := transaction.NewTransactionManager(lm, bp, lockMgr)

	tableManager, err := execution.NewTableManager(cat, bp, lm, lockMgr)
	require.NoError(t, err)
	indexManager, err := indexing.NewIndexManager(cat)
	require.NoError(t, err)

	rm := recovery.NewRecoveryManager(lm, bp, txnMgr, dbDir, cat, tableManager, indexManager)
	require.NoError(t, rm.Recover())

	logicalRules, physicalRules := testPlannerRules()

	wdb := &recoveryWorkloadDB{
		Catalog:            cat,
		BufferPool:         bp,
		TableManager:       tableManager,
		TransactionManager: txnMgr,
		LockManager:        lockMgr,
		IndexManager:       indexManager,
		Planner:            planner.NewSQLPlanner(cat, logicalRules, physicalRules),
		lm:                 lm,
		spySM:              spySM,
		dbDir:              dbDir,
		logPath:            logPath,
	}
	t.Cleanup(wdb.closeLM)
	return wdb
}

// crashAndRecover is a convenience wrapper: simulates crash then runs ARIES recovery.
func crashAndRecover(t *testing.T, rwdb *recoveryWorkloadDB) *recoveryWorkloadDB {
	simulateCrash(t, rwdb)
	return recoverFromCrash(t, rwdb.Catalog, rwdb.dbDir, rwdb.logPath)
}

// stopCMWithTimeout calls cm.Stop() and asserts it returns within 5 seconds.
// This ensures the CheckpointManager shuts down cleanly and is not stuck in a
// goroutine leak or blocked flush.
func stopCMWithTimeout(t *testing.T, cm *recovery.CheckpointManager) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		cm.Stop()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("CheckpointManager.Stop() did not return within 5 seconds — possible goroutine leak or blocked flush")
	}
}

func abortOnDeadlockR(t testing.TB, wdb *recoveryWorkloadDB, txn *transaction.TransactionContext, err error) bool {
	if err == nil {
		return false
	}
	assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
	require.NoError(t, wdb.TransactionManager.Abort(txn))
	runtime.Gosched()
	return true
}

func (wdb *recoveryWorkloadDB) execSQL(txn *transaction.TransactionContext, sqlStr string) ([][]common.Value, error) {
	plan, err := wdb.Planner.Plan(sqlStr, true)
	if err != nil {
		return nil, err
	}
	ex, err := execution.BuildExecutorTree(plan, wdb.Catalog, wdb.TableManager, wdb.IndexManager)
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

func (wdb *recoveryWorkloadDB) queryKV(t *testing.T) [][]common.Value {
	txn, err := wdb.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := wdb.execSQL(txn, "SELECT * FROM kv ORDER BY k")
	require.NoError(t, err)
	require.NoError(t, wdb.TransactionManager.Commit(txn))
	return rows
}

func (wdb *recoveryWorkloadDB) assertKVContains(t *testing.T, expected map[int64]int64) {
	rows := wdb.queryKV(t)
	assert.Equal(t, len(expected), len(rows), "unexpected number of rows")
	for _, row := range rows {
		k := row[0].IntValue()
		v := row[1].IntValue()
		expectedV, ok := expected[k]
		assert.True(t, ok, "unexpected key %d", k)
		if ok {
			assert.Equal(t, expectedV, v, "wrong value for key %d", k)
		}
	}
}

// setupRecoveryKVDB inserts numKeys rows into the kv table using real
// transactions so that the WAL contains INSERT records for ARIES recovery.
func setupRecoveryKVDB(t *testing.T, wdb *recoveryWorkloadDB, numKeys int, initialVal int64) {
	for k := 1; k <= numKeys; k++ {
		txn, err := wdb.TransactionManager.Begin()
		require.NoError(t, err)
		_, err = wdb.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, %d)", k, initialVal))
		require.NoError(t, err)
		require.NoError(t, wdb.TransactionManager.Commit(txn))
	}
}

// --- E2E Recovery Tests (concurrent workloads) ---

// TestRecovery_E2E_YCSB_CrashRecover runs a full YCSB workload with the
// DoubleBufferLogManager's built-in periodic WAL flushing, then crashes and
// runs ARIES recovery. Checks that the recovered state matches all committed
// writes recorded during the workload.
func TestRecovery_E2E_YCSB_CrashRecover(t *testing.T) {
	type recoveryYCSBConfig struct {
		name       string
		cfg        ycsbConfig
		checkpoint bool
	}

	configs := []recoveryYCSBConfig{
		{
			name: "NoCheckpoint",
			cfg: ycsbConfig{
				numKeys:        1000,
				numWorkers:     8,
				txnsPerWorker:  2000,
				opsPerTxn:      10,
				numBufferPages: 50,
				readFraction:   0.5,
			},
			checkpoint: false,
		},
		{
			name: "BufferPressure",
			cfg: ycsbConfig{
				numKeys:        1000,
				numWorkers:     8,
				txnsPerWorker:  1000,
				opsPerTxn:      5,
				numBufferPages: 5,
				readFraction:   0.7,
			},
			checkpoint: true,
		},
	}

	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			wdb := makeRecoveryWorkloadDB(t, newKVCatalog(t), tc.cfg.numBufferPages)
			setupRecoveryKVDB(t, wdb, tc.cfg.numKeys, 0)

			var cm *recovery.CheckpointManager
			if tc.checkpoint {
				cm = recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
				cm.Start()
			}

			// Track committed writes directly.
			var mu sync.Mutex
			committed := make(map[int64]int64, tc.cfg.numKeys)
			for k := 1; k <= tc.cfg.numKeys; k++ {
				committed[int64(k)] = 0
			}
			var writeCounter atomic.Int64
			writeCounter.Store(1)

			var wg sync.WaitGroup
			for w := 0; w < tc.cfg.numWorkers; w++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					rng := rand.New(rand.NewSource(int64(workerID*1000003 + 13)))
					keyGen := newKeyGen(rng, tc.cfg)
					for txnNum := 0; txnNum < tc.cfg.txnsPerWorker; txnNum++ {
						txn, err := wdb.TransactionManager.Begin()
						require.NoError(t, err)
						var writes []struct{ k, v int64 }
						ok := true
						for i := 0; i < tc.cfg.opsPerTxn; i++ {
							k := keyGen()
							if rng.Float64() < tc.cfg.readFraction {
								_, err := wdb.execSQL(txn, fmt.Sprintf("SELECT v FROM kv WHERE k = %d", k))
								if err != nil {
									_ = wdb.TransactionManager.Abort(txn)
									ok = false
									break
								}
							} else {
								v := writeCounter.Add(1)
								_, err := wdb.execSQL(txn, fmt.Sprintf("UPDATE kv SET v = %d WHERE k = %d", v, k))
								if err != nil {
									_ = wdb.TransactionManager.Abort(txn)
									ok = false
									break
								}
								writes = append(writes, struct{ k, v int64 }{k, v})
							}
						}
						if ok {
							if err := wdb.TransactionManager.Commit(txn); err == nil {
								mu.Lock()
								for _, w := range writes {
									committed[w.k] = w.v
								}
								mu.Unlock()
							}
						}
					}
				}(w)
			}
			wg.Wait()
			if cm != nil {
				stopCMWithTimeout(t, cm)
			}

			// Crash and recover.
			recovered := crashAndRecover(t, wdb)
			recovered.assertKVContains(t, committed)

			// Post-recovery liveness: new txn must succeed.
			txn, err := recovered.TransactionManager.Begin()
			require.NoError(t, err)
			_, err = recovered.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, 999)", tc.cfg.numKeys+1))
			require.NoError(t, err)
			require.NoError(t, recovered.TransactionManager.Commit(txn))
		})
	}
}

// TestRecovery_E2E_BankCrashRecover runs the bank conservation workload, then
// crashes and recovers. The SUM of all balances must equal the expected total
// (initial balances + all committed bonuses).
func TestRecovery_E2E_BankCrashRecover(t *testing.T) {
	const (
		numKeys        = 200
		initialBalance = 10000
	)
	workloadTime := 2 * time.Second

	wdb := makeRecoveryWorkloadDB(t, newKVCatalog(t), 100)
	setupRecoveryKVDB(t, wdb, numKeys, initialBalance)

	cm := recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
	cm.Start()

	var totalBonus atomic.Int64
	start := time.Now()
	var wg sync.WaitGroup

	// Point transferors.
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
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				_, err = wdb.execSQL(txn, fmt.Sprintf(
					"UPDATE kv SET v = v + 1 WHERE k = %d", dst))
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				assert.NoError(t, wdb.TransactionManager.Commit(txn))
			}
		}(i)
	}

	// Bulk operators.
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
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				totalBonus.Add(bonus * int64(numKeys))
				if err = wdb.TransactionManager.Commit(txn); err != nil {
					totalBonus.Add(-bonus * int64(numKeys))
					assert.NoError(t, err)
				}
				runtime.Gosched()
			}
		}(i)
	}

	// IS→S scanners: point SELECT (IS+Tuple-S) then SUM (IS→S) to assert conservation.
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
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				assert.Len(t, pointRows, 1, "IS→S scanner: key %d not found", k)
				sumRows, err := wdb.execSQL(txn, "SELECT SUM(kv.v) FROM kv")
				if abortOnDeadlockR(t, wdb, txn, err) {
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

	wg.Wait()
	stopCMWithTimeout(t, cm)

	expectedTotal := int64(numKeys*initialBalance) + totalBonus.Load()

	// Crash and recover.
	recovered := crashAndRecover(t, wdb)

	txn, err := recovered.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := recovered.execSQL(txn, "SELECT SUM(kv.v) FROM kv")
	require.NoError(t, err)
	require.NoError(t, recovered.TransactionManager.Commit(txn))

	assert.Equal(t, expectedTotal, rows[0][0].IntValue(),
		"bank conservation violated after crash recovery")
}

// TestRecovery_E2E_AbortStormCrashRecover runs the abort storm workload (all txns
// explicitly abort) and then crashes and recovers. The table must contain exactly
// the original numKeys rows — no uncommitted inserts or deletes may leak through.
func TestRecovery_E2E_AbortStormCrashRecover(t *testing.T) {
	const (
		numKeys        = 500
		numWorkers     = 8
		txnsPerWorker  = 2000
		opsPerTxn      = 5
		numBufferPages = 30
		insertFraction = 0.5
	)

	wdb := makeRecoveryWorkloadDB(t, newKVCatalog(t), numBufferPages)
	setupRecoveryKVDB(t, wdb, numKeys, 0)

	cm := recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
	cm.Start()

	var stopped atomic.Bool
	var workerWg, scannerWg sync.WaitGroup

	// Scanners: verify no dirty reads via point lookups.
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
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				assert.Len(t, origRows, 1,
					"dirty delete: original-zone key %d not found", origKey)
				// Check that a random dirty-zone key does not exist (no dirty insert).
				dirtyKey := rng.Int63n(int64(numWorkers*opsPerTxn)) + int64(numKeys) + 1
				dirtyRows, err := wdb.execSQL(txn, fmt.Sprintf(
					"SELECT v FROM kv WHERE k = %d", dirtyKey))
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				assert.Empty(t, dirtyRows,
					"dirty insert: dirty-zone key %d unexpectedly found", dirtyKey)
				assert.NoError(t, wdb.TransactionManager.Commit(txn))
				runtime.Gosched()
			}
		}(s)
	}

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
				for j := 0; j < opsPerTxn && !aborted; j++ {
					if rng.Float64() < insertFraction {
						_, err := wdb.execSQL(txn, fmt.Sprintf(
							"INSERT INTO kv VALUES (%d, 0)", insertBase+insertIdx))
						insertIdx++
						aborted = abortOnDeadlockR(t, wdb, txn, err)
					} else {
						deleteKey := rng.Int63n(int64(numKeys)) + 1
						_, err := wdb.execSQL(txn, fmt.Sprintf(
							"DELETE FROM kv WHERE k = %d", deleteKey))
						aborted = abortOnDeadlockR(t, wdb, txn, err)
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
	stopCMWithTimeout(t, cm)

	// Crash and recover.
	recovered := crashAndRecover(t, wdb)

	// All aborted — table must have exactly the original rows.
	txn, err := recovered.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := recovered.execSQL(txn, "SELECT COUNT(*) FROM kv")
	require.NoError(t, err)
	require.NoError(t, recovered.TransactionManager.Commit(txn))
	assert.Equal(t, int64(numKeys), rows[0][0].IntValue(),
		"abort storm: row count wrong after crash recovery")
}

// TestRecovery_E2E_RepeatedCrashRecover runs multiple crash-recover cycles
// interleaved with workloads. Each cycle runs a YCSB-style workload, crashes,
// recovers, verifies state, then continues with new transactions. This tests
// that CLRs from recovery are themselves recoverable, and that post-recovery
// state is clean enough for subsequent workloads.
func TestRecovery_E2E_RepeatedCrashRecover(t *testing.T) {
	const (
		numKeys    = 600
		numCycles  = 2
		numWorkers = 8
		txnsPerW   = 500
		opsPerTxn  = 5
	)

	cat := newKVCatalog(t)
	wdb := makeRecoveryWorkloadDB(t, cat, 30)

	// Seed initial data.
	txn0, err := wdb.TransactionManager.Begin()
	require.NoError(t, err)
	for k := int64(1); k <= numKeys; k++ {
		_, err := wdb.execSQL(txn0, fmt.Sprintf("INSERT INTO kv VALUES (%d, 0)", k))
		require.NoError(t, err)
	}
	require.NoError(t, wdb.TransactionManager.Commit(txn0))

	// expected tracks the ground truth across cycles; mu guards concurrent updates.
	expected := make(map[int64]int64, numKeys)
	for k := int64(1); k <= numKeys; k++ {
		expected[k] = 0
	}
	var mu sync.Mutex

	for cycle := 0; cycle < numCycles; cycle++ {
		// Fresh CM per cycle: crashAndRecover replaces wdb (new log manager).
		cm := recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
		cm.Start()

		// Run a batch of committed + uncommitted txns concurrently.
		var writeCounter atomic.Int64
		writeCounter.Store(1)

		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(int64(cycle*7919 + workerID)))
				for txnNum := 0; txnNum < txnsPerW; txnNum++ {
					txn, err := wdb.TransactionManager.Begin()
					if err != nil {
						return
					}
					doCommit := rng.Intn(2) == 0
					var txnWrites []struct{ k, v int64 }
					ok := true
					for op := 0; op < opsPerTxn; op++ {
						key := rng.Int63n(numKeys) + 1
						val := writeCounter.Add(1)
						_, err := wdb.execSQL(txn, fmt.Sprintf(
							"UPDATE kv SET v = %d WHERE k = %d", val, key))
						if err != nil {
							_ = wdb.TransactionManager.Abort(txn)
							ok = false
							break
						}
						txnWrites = append(txnWrites, struct{ k, v int64 }{key, val})
					}
					if !ok {
						continue
					}
					if doCommit {
						if err := wdb.TransactionManager.Commit(txn); err == nil {
							mu.Lock()
							for _, wr := range txnWrites {
								expected[wr.k] = wr.v
							}
							mu.Unlock()
						}
					} else {
						_ = wdb.TransactionManager.Abort(txn)
					}
				}
			}(w)
		}
		wg.Wait()
		stopCMWithTimeout(t, cm)

		// Leave an uncommitted txn in flight.
		danglingRng := rand.New(rand.NewSource(int64(cycle*31337 + 1)))
		txnDangling, err := wdb.TransactionManager.Begin()
		require.NoError(t, err)
		_, err = wdb.execSQL(txnDangling, fmt.Sprintf(
			"UPDATE kv SET v = -999 WHERE k = %d", danglingRng.Int63n(numKeys)+1))
		require.NoError(t, err)
		// NOT committed.

		// Crash and recover.
		wdb = crashAndRecover(t, wdb)
		wdb.assertKVContains(t, expected)
	}
}

// TestRecovery_E2E_DeleteReplace_CrashRecover seeds numKeys rows, runs concurrent
// workers that each txn DELETEs a random key then INSERTs it back with a new value
// (delete-replace pattern). Some txns commit, some abort on deadlock. Then crashes
// and recovers, verifying that the recovered state matches all committed writes.
//
// This tests committed DELETE redo, committed INSERT redo after DELETE,˜
// uncommitted DELETE undo (CLR), uncommitted INSERT undo after DELETE (CLR),
// and index consistency after recovery with delete-replace churn.
func TestRecovery_E2E_DeleteReplace_CrashRecover(t *testing.T) {
	const (
		numKeys        = 500
		numWorkers     = 8
		txnsPerWorker  = 1000
		numBufferPages = 30
	)

	wdb := makeRecoveryWorkloadDB(t, newKVCatalog(t), numBufferPages)
	setupRecoveryKVDB(t, wdb, numKeys, 0)

	cm := recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
	cm.Start()

	var writeCounter atomic.Int64
	writeCounter.Store(1)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID*1000003 + 77)))
			for txnNum := 0; txnNum < txnsPerWorker; txnNum++ {
				k := rng.Int63n(int64(numKeys)) + 1
				newVal := writeCounter.Add(1)

				txn, err := wdb.TransactionManager.Begin()
				if err != nil {
					return
				}
				// DELETE then re-INSERT the same key with a new value.
				delRows, err := wdb.execSQL(txn, fmt.Sprintf(
					"DELETE FROM kv WHERE k = %d", k))
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				// Skip INSERT if another delete-replace committed between our
				// index scan and our lock wait — avoids two live rows for the
				// same k (GoDB does not enforce PK uniqueness on INSERT).
				if delRows[0][0].IntValue() == 0 {
					_ = wdb.TransactionManager.Abort(txn)
					continue
				}
				_, err = wdb.execSQL(txn, fmt.Sprintf(
					"INSERT INTO kv VALUES (%d, %d)", k, newVal))
				if abortOnDeadlockR(t, wdb, txn, err) {
					continue
				}
				_ = wdb.TransactionManager.Commit(txn)
			}
		}(w)
	}
	wg.Wait()
	stopCMWithTimeout(t, cm)

	// Snapshot the authoritative pre-crash state by reading the DB before crashing.
	precrash := make(map[int64]int64, numKeys)
	for _, row := range wdb.queryKV(t) {
		precrash[row[0].IntValue()] = row[1].IntValue()
	}

	// Crash and recover.
	recovered := crashAndRecover(t, wdb)
	recovered.assertKVContains(t, precrash)

	// Post-recovery liveness: index must still answer point lookups.
	txn, err := recovered.TransactionManager.Begin()
	require.NoError(t, err)
	rows, err := recovered.execSQL(txn, fmt.Sprintf("SELECT v FROM kv WHERE k = %d", int64(numKeys/2)))
	require.NoError(t, err)
	require.NoError(t, recovered.TransactionManager.Commit(txn))
	assert.Len(t, rows, 1, "post-recovery point lookup: key %d not found", numKeys/2)
}

// TestRecovery_E2E_CrashDuringRecovery tests that recovery is itself recoverable.
// It runs two back-to-back crash-recover cycles with no intervening work between
// the first recovery and the second crash. Because Abort() writes CLRs without
// calling WaitUntilFlushed, a second crash immediately after the first recovery
// can truncate some or all CLRs from the WAL, simulating recovery interrupted
// mid-undo. The second recovery must then handle partially-undone transactions
// and still produce correct state.
func TestRecovery_E2E_CrashDuringRecovery(t *testing.T) {
	const (
		numKeys        = 500
		numBufferPages = 5
		numWorkers     = 8
		workloadTime   = 500 * time.Millisecond
		numDanglingTxn = 10
		danglingOps    = 5
	)

	wdb := makeRecoveryWorkloadDB(t, newKVCatalog(t), numBufferPages)
	setupRecoveryKVDB(t, wdb, numKeys, 0)

	// Start checkpoint manager alongside concurrent workload.
	cm := recovery.NewCheckpointManager(wdb.lm, wdb.BufferPool, wdb.TransactionManager, wdb.dbDir, 50*time.Millisecond)
	cm.Start()

	var writeCounter atomic.Int64
	writeCounter.Store(1)

	start := time.Now()
	var stopped atomic.Bool
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID*1000003 + 17)))
			for time.Since(start) < workloadTime && !stopped.Load() {
				txn, err := wdb.TransactionManager.Begin()
				if err != nil {
					return
				}
				ok := true
				for i := 0; i < 5; i++ {
					key := rng.Int63n(numKeys) + 1
					val := writeCounter.Add(1)
					_, err := wdb.execSQL(txn, fmt.Sprintf(
						"UPDATE kv SET v = %d WHERE k = %d", val, key))
					if err != nil {
						_ = wdb.TransactionManager.Abort(txn)
						ok = false
						break
					}
				}
				if ok {
					_ = wdb.TransactionManager.Commit(txn)
				}
			}
		}(w)
	}

	time.Sleep(workloadTime)
	stopped.Store(true)
	wg.Wait()
	stopCMWithTimeout(t, cm)

	// Snapshot authoritative expected state before introducing dangling txns.
	expected := make(map[int64]int64, numKeys)
	for _, row := range wdb.queryKV(t) {
		expected[row[0].IntValue()] = row[1].IntValue()
	}

	// Create dangling txns to generate CLRs during recovery.
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numDanglingTxn; i++ {
		txn, err := wdb.TransactionManager.Begin()
		require.NoError(t, err)
		for j := 0; j < danglingOps; j++ {
			key := rng.Int63n(numKeys) + 1
			_, err := wdb.execSQL(txn, fmt.Sprintf(
				"UPDATE kv SET v = -9999 WHERE k = %d", key))
			if err != nil {
				_ = wdb.TransactionManager.Abort(txn)
				break
			}
		}
	}

	// First crash + recovery: ARIES will abort all dangling txns, writing CLRs.
	simulateCrash(t, wdb)
	recovered1 := recoverFromCrash(t, wdb.Catalog, wdb.dbDir, wdb.logPath)

	// Second crash immediately: truncates WAL at FlushedUntil(), removing any
	// CLRs that were not yet durably flushed. Simulates recovery interrupted
	// mid-undo.
	simulateCrash(t, recovered1)
	recovered2 := recoverFromCrash(t, recovered1.Catalog, recovered1.dbDir, recovered1.logPath)

	// Final state must match pre-crash expected (all dangling writes undone).
	recovered2.assertKVContains(t, expected)

	// Post-recovery liveness.
	txn, err := recovered2.TransactionManager.Begin()
	require.NoError(t, err)
	_, err = recovered2.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, 42)", numKeys+1))
	require.NoError(t, err)
	require.NoError(t, recovered2.TransactionManager.Commit(txn))
}
