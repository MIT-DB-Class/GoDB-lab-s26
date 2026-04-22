package recovery

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/puzpuzpuz/xsync/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"mit.edu/dsg/godb/catalog"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/execution"
	"mit.edu/dsg/godb/indexing"
	"mit.edu/dsg/godb/logging"
	"mit.edu/dsg/godb/planner"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

// recoveryTestEnv bundles all components needed for recovery unit tests.
type recoveryTestEnv struct {
	t            *testing.T
	mlm          *logging.MemoryLogManager
	dbDir        string
	bp           *storage.BufferPool
	lm           *transaction.LockManager
	tm           *transaction.TransactionManager
	cat          *catalog.Catalog
	tableManager *execution.TableManager
	indexManager *indexing.IndexManager
	planner      *planner.SQLPlanner
	statsDSM     *storage.StatsDBFileManager
}

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
	_, err = cat.AddTable("kv2", []catalog.Column{
		{Name: "k", Type: common.IntType},
		{Name: "v", Type: common.IntType},
	})
	require.NoError(t, err)
	return cat
}

func plannerRules() ([]planner.LogicalRule, []planner.PhysicalConversionRule) {
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

func newRecoveryTestEnv(t *testing.T) *recoveryTestEnv {
	mlm := logging.NewMemoryLogManager()
	mlm.SetFlushOnAppend(true)
	dbDir := t.TempDir()

	realDSM := storage.NewDiskStorageManager(dbDir)
	statsDSM := &storage.StatsDBFileManager{
		Inner: realDSM,
		Files: xsync.NewMapOf[common.ObjectID, *storage.StatsDBFile](),
	}
	bp := storage.NewBufferPool(64, statsDSM, mlm)
	lm := transaction.NewLockManager()
	tm := transaction.NewTransactionManager(mlm, bp, lm)

	cat := newKVCatalog(t)
	tableManager, err := execution.NewTableManager(cat, bp, mlm, lm)
	require.NoError(t, err)
	indexManager, err := indexing.NewIndexManager(cat)
	require.NoError(t, err)

	logicalRules, physicalRules := plannerRules()
	p := planner.NewSQLPlanner(cat, logicalRules, physicalRules)

	// Bootstrap indexes from existing heap data.
	rm := NewNoLogRecoveryManager(bp, tm, cat, tableManager, indexManager)
	require.NoError(t, rm.Recover())

	return &recoveryTestEnv{
		t: t, mlm: mlm, dbDir: dbDir,
		bp: bp, lm: lm, tm: tm,
		cat: cat, tableManager: tableManager, indexManager: indexManager,
		planner: p, statsDSM: statsDSM,
	}
}

// crashAndRecover simulates a crash (losing the buffer pool page cache) and
// runs ARIES recovery. The MemoryLogManager and catalog survive (they represent
// durable WAL storage and schema metadata respectively).
func (env *recoveryTestEnv) crashAndRecover() *recoveryTestEnv {
	realDSM := storage.NewDiskStorageManager(env.dbDir)
	statsDSM := &storage.StatsDBFileManager{
		Inner: realDSM,
		Files: xsync.NewMapOf[common.ObjectID, *storage.StatsDBFile](),
	}
	bp := storage.NewBufferPool(64, statsDSM, env.mlm)
	lm := transaction.NewLockManager()
	tm := transaction.NewTransactionManager(env.mlm, bp, lm)

	tableManager, err := execution.NewTableManager(env.cat, bp, env.mlm, lm)
	require.NoError(env.t, err)
	indexManager, err := indexing.NewIndexManager(env.cat)
	require.NoError(env.t, err)

	// Reset counters after initialization so statsDSM reflects only what
	// the recovery process itself reads or writes, not TableManager setup I/O.
	statsDSM.Files.Range(func(_ common.ObjectID, f *storage.StatsDBFile) bool {
		f.ReadCnt.Store(0)
		f.WriteCnt.Store(0)
		return true
	})

	logicalRules, physicalRules := plannerRules()
	p := planner.NewSQLPlanner(env.cat, logicalRules, physicalRules)

	rm := NewRecoveryManager(env.mlm, bp, tm, env.dbDir, env.cat, tableManager, indexManager)
	require.NoError(env.t, rm.Recover())

	return &recoveryTestEnv{
		t: env.t, mlm: env.mlm, dbDir: env.dbDir,
		bp: bp, lm: lm, tm: tm,
		cat: env.cat, tableManager: tableManager, indexManager: indexManager,
		planner: p, statsDSM: statsDSM,
	}
}

// execSQL plans and executes a SQL statement, returning all output rows.
func (env *recoveryTestEnv) execSQL(txn *transaction.TransactionContext, sqlStr string) ([][]common.Value, error) {
	plan, err := env.planner.Plan(sqlStr, true)
	if err != nil {
		return nil, err
	}
	ex, err := execution.BuildExecutorTree(plan, env.cat, env.tableManager, env.indexManager)
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

// queryKV runs SELECT * FROM kv ORDER BY k and returns sorted (k, v) pairs.
func (env *recoveryTestEnv) queryKV() [][]common.Value {
	txn, err := env.tm.Begin()
	require.NoError(env.t, err)
	rows, err := env.execSQL(txn, "SELECT * FROM kv ORDER BY k")
	require.NoError(env.t, err)
	require.NoError(env.t, env.tm.Commit(txn))
	return rows
}

// assertKVContains checks that queryKV returns exactly the expected (k, v) pairs.
func (env *recoveryTestEnv) assertKVContains(expected map[int64]int64) {
	rows := env.queryKV()
	assert.Equal(env.t, len(expected), len(rows), "unexpected number of rows")
	for _, row := range rows {
		k := row[0].IntValue()
		v := row[1].IntValue()
		expectedV, ok := expected[k]
		assert.True(env.t, ok, "unexpected key %d", k)
		if ok {
			assert.Equal(env.t, expectedV, v, "wrong value for key %d", k)
		}
	}
}

// insertKV is a convenience for INSERT INTO kv VALUES (k, v).
func (env *recoveryTestEnv) insertKV(txn *transaction.TransactionContext, k, v int64) {
	_, err := env.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, %d)", k, v))
	require.NoError(env.t, err)
}

// assertTIDMonotone verifies that the next transaction ID issued after recovery
// is strictly greater than every TID seen in the WAL. This catches a missing
// SetNextTxnID call in the recovery manager.
func assertTIDMonotone(t *testing.T, env *recoveryTestEnv) {
	iter, err := env.mlm.Iterator(0)
	require.NoError(t, err)
	defer iter.Close()
	var maxTID common.TransactionID
	for iter.Next() {
		r := iter.CurrentRecord()
		if r.RecordType() != storage.LogBeginCheckpoint && r.RecordType() != storage.LogEndCheckpoint {
			if tid := r.TxnID(); tid > maxTID {
				maxTID = tid
			}
		}
	}
	require.NoError(t, iter.Error())

	txn, err := env.tm.Begin()
	require.NoError(t, err)
	newTID := txn.ID()
	require.NoError(t, env.tm.Commit(txn))
	assert.Greater(t, uint64(newTID), uint64(maxTID),
		"post-recovery TID (%d) must exceed all WAL TIDs (max=%d)", newTID, maxTID)
}

// TestBackgroundFlusher verifies that BackgroundFlusher periodically flushes dirty pages to disk
// under concurrent write load. It runs 8 writer goroutines for ~1 second with randomized pacing,
// samples the write count periodically to confirm the flusher is active, then verifies:
//   - The on-disk content (fresh cold-cache pool) matches the in-memory content.
//   - The flusher triggered at least one disk write (via StatsDBFileManager write counters).
//
// Stop is registered via t.Cleanup so it runs on the main goroutine at teardown,
// and a 2-second timeout asserts it returns promptly.
func TestBackgroundFlusher(t *testing.T) {
	env := newRecoveryTestEnv(t)

	bf := NewBackgroundFlusher(env.bp, 10*time.Millisecond)
	bf.Start()

	// Run 8 writer goroutines for ~1 second, pacing writes with a randomized sleep.
	var stopped atomic.Bool
	var keyCounter atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(wID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(wID)))
			for !stopped.Load() {
				time.Sleep(time.Duration(rng.Intn(5)) * time.Millisecond)
				k := keyCounter.Add(1)
				txn, err := env.tm.Begin()
				if !assert.NoError(t, err) {
					return
				}
				_, err = env.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, %d)", k, k*10))
				if !assert.NoError(t, err) {
					_ = env.tm.Abort(txn)
					return
				}
				assert.NoError(t, env.tm.Commit(txn))
			}
		}(w)
	}

	// Sample write count every 250ms; it should climb as the flusher fires.
	var prevWrites int64
	countWrites := func() int64 {
		var total int64
		env.statsDSM.Files.Range(func(_ common.ObjectID, f *storage.StatsDBFile) bool {
			total += f.WriteCnt.Load()
			return true
		})
		return total
	}
	for i := 0; i < 40; i++ {
		time.Sleep(250 * time.Millisecond)
		cur := countWrites()
		if i > 0 {
			assert.Greater(t, cur, prevWrites,
				"write count should increase each sample interval (flusher not firing?)")
		}
		prevWrites = cur
	}

	stopped.Store(true)
	wg.Wait()

	// Capture in-memory rows before discarding the buffer pool.
	inMemoryRows := env.queryKV()

	// Verify disk content by swapping in a cold-cache buffer pool on the same env.
	// env.statsDSM.Inner is the real DiskStorageManager — reuse it directly.
	// lm, cat, indexManager, and planner don't hold a buffer pool reference, so
	// only bp, tm, and tableManager need to be replaced.
	env.bp = storage.NewBufferPool(64, env.statsDSM.Inner, env.mlm)
	env.tm = transaction.NewTransactionManager(env.mlm, env.bp, env.lm)
	var err error
	env.tableManager, err = execution.NewTableManager(env.cat, env.bp, env.mlm, env.lm)
	require.NoError(t, err)
	rm := NewNoLogRecoveryManager(env.bp, env.tm, env.cat, env.tableManager, env.indexManager)
	require.NoError(t, rm.Recover())

	diskRows := env.queryKV()
	assert.Equal(t, inMemoryRows, diskRows,
		"disk content (fresh cold-cache pool) should match in-memory content")

	done := make(chan struct{})
	go func() { bf.Stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Errorf("BackgroundFlusher.Stop() hung")
	}
}

// TestCheckpointManager verifies that CheckpointManager fires at roughly the right
// rate under a concurrent write workload, that Begin/End checkpoint records are paired,
// and that Stop returns promptly.
func TestCheckpointManager(t *testing.T) {
	const (
		checkpointInterval = 50 * time.Millisecond
		workloadDuration   = 2 * time.Second
	)

	env := newRecoveryTestEnv(t)

	cm := NewCheckpointManager(env.mlm, env.bp, env.tm, env.dbDir, checkpointInterval)
	cm.Start()

	// Run writer goroutines for workloadDuration, pacing writes with a randomized sleep,
	// so the checkpoint manager sees a non-trivial ATT/DPT on each firing.
	var stopped atomic.Bool
	var keyCounter atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(wID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(wID)))
			for !stopped.Load() {
				time.Sleep(time.Duration(rng.Intn(5)) * time.Millisecond)
				k := keyCounter.Add(1)
				txn, err := env.tm.Begin()
				if !assert.NoError(t, err) {
					return
				}
				_, err = env.execSQL(txn, fmt.Sprintf("INSERT INTO kv VALUES (%d, %d)", k, k*10))
				if !assert.NoError(t, err) {
					_ = env.tm.Abort(txn)
					return
				}
				assert.NoError(t, env.tm.Commit(txn))
			}
		}(w)
	}

	time.Sleep(workloadDuration)
	stopped.Store(true)
	wg.Wait()

	// Stop should return promptly.
	done := make(chan struct{})
	go func() { cm.Stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("CheckpointManager.Stop() hung")
	}

	// Scan the log and count paired Begin/End checkpoint records.
	iter, err := env.mlm.Iterator(0)
	require.NoError(t, err)
	defer iter.Close()
	var beginCount, endCount int
	for iter.Next() {
		switch iter.CurrentRecord().RecordType() {
		case storage.LogBeginCheckpoint:
			beginCount++
		case storage.LogEndCheckpoint:
			endCount++
		}
	}
	require.NoError(t, iter.Error())

	assert.Equal(t, beginCount, endCount,
		"every BeginCheckpoint must be paired with an EndCheckpoint")

	expectedFirings := int(workloadDuration / checkpointInterval)
	assert.InDelta(t, expectedFirings, beginCount, 2,
		"checkpoint should have fired ~%d times (±2)", expectedFirings)
}

// TestRecovery_Basic exercises all fundamental ARIES recovery scenarios in one continuous test.
func TestRecovery_Basic(t *testing.T) {
	env := newRecoveryTestEnv(t)

	// Empty log: recovery on a pristine env must succeed with an empty table.
	rm := NewRecoveryManager(env.mlm, env.bp, env.tm, env.dbDir, env.cat, env.tableManager, env.indexManager)
	require.NoError(t, rm.Recover())
	env.assertKVContains(map[int64]int64{})

	// Redo committed: insert {1:10, 2:20} and commit without flushing — recovery
	// must redo both inserts.
	txn, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn, 1, 10)
	env.insertKV(txn, 2, 20)
	require.NoError(t, env.tm.Commit(txn))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20})

	// Undo uncommitted: flush {1:10, 2:20} to disk, commit {3:30} and flush, then
	// flush uncommitted {4:40} — recovery must undo the dirty uncommitted insert.
	require.NoError(t, env.bp.FlushAllPages())
	txn1, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn1, 3, 30)
	require.NoError(t, env.tm.Commit(txn1))
	require.NoError(t, env.bp.FlushAllPages())
	txn2, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn2, 4, 40) // uncommitted
	require.NoError(t, env.bp.FlushAllPages())
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30})

	// Redo+undo combined: flush uncommitted {5:50}, then commit {6:60} without
	// flushing — recovery must undo {5:50} and redo {6:60}.
	txn1, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn1, 5, 50) // uncommitted
	require.NoError(t, env.bp.FlushAllPages())
	txn2, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn2, 6, 60)
	require.NoError(t, env.tm.Commit(txn2))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30, 6: 60})

	// Abort CLR: abort before crash writes CLRs; recovery must redo the CLRs so
	// that the aborted insert remains invisible.
	txn, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn, 7, 70)
	require.NoError(t, env.tm.Abort(txn))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30, 6: 60})

	// Update redo+undo: flush current state, then commit k=1→99 without flushing
	// (needs redo), then flush uncommitted k=1→999 (needs undo back to 99).
	require.NoError(t, env.bp.FlushAllPages())
	txn2, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn2, "UPDATE kv SET v = 99 WHERE k = 1")
	require.NoError(t, err)
	require.NoError(t, env.tm.Commit(txn2))
	txn3, err := env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn3, "UPDATE kv SET v = 999 WHERE k = 1")
	require.NoError(t, err)
	require.NoError(t, env.bp.FlushAllPages())
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 6: 60})

	// Idempotent: running recovery twice must yield the same result. CLRs written
	// by the first recovery are themselves redoable by the second.
	txn, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn, 8, 80)
	require.NoError(t, env.tm.Commit(txn))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 6: 60, 8: 80})
	require.NoError(t, env.bp.FlushAllPages())
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 6: 60, 8: 80})

	// Delete redo: committed DELETE not flushed — recovery must redo LogDelete.
	txn, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn, "DELETE FROM kv WHERE k = 6")
	require.NoError(t, err)
	require.NoError(t, env.tm.Commit(txn))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 8: 80})

	// Delete undo: uncommitted DELETE flushed — recovery must undo via LogDeleteCLR.
	txn, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn, "DELETE FROM kv WHERE k = 8")
	require.NoError(t, err)
	require.NoError(t, env.bp.FlushAllPages())
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 8: 80})

	// Delete CLR redo: abort a DELETE before crash — recovery must redo the CLRs so
	// the row remains visible.
	txn, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn, "DELETE FROM kv WHERE k = 3")
	require.NoError(t, err)
	require.NoError(t, env.tm.Abort(txn))
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 8: 80})

	// Multiple uncommitted: three concurrent uncommitted transactions (INSERT, UPDATE,
	// DELETE) all flushed to disk — all must be undone by recovery.
	require.NoError(t, env.bp.FlushAllPages())
	txn1, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn1, 100, 999) // uncommitted INSERT
	txn2, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn2, "UPDATE kv SET v = 999 WHERE k = 1") // uncommitted UPDATE
	require.NoError(t, err)
	txn3, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn3, "DELETE FROM kv WHERE k = 2") // uncommitted DELETE
	require.NoError(t, err)
	require.NoError(t, env.bp.FlushAllPages())
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 99, 2: 20, 3: 30, 8: 80})

	assertTIDMonotone(t, env)
}

// TestRecovery_WithCheckpoint exercises checkpoint-aware ARIES recovery in one continuous test.
func TestRecovery_WithCheckpoint(t *testing.T) {
	env := newRecoveryTestEnv(t)

	// Commit before and after a checkpoint, leave one txn uncommitted at crash.
	// Recovery must start from the checkpoint and redo/undo correctly.
	txn1, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn1, 1, 10)
	env.insertKV(txn1, 2, 20)
	require.NoError(t, env.tm.Commit(txn1))
	cm := NewCheckpointManager(env.mlm, env.bp, env.tm, env.dbDir, time.Hour)
	_, err = cm.Checkpoint()
	require.NoError(t, err)
	txn2, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn2, 3, 30)
	require.NoError(t, env.tm.Commit(txn2))
	txn3, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn3, 4, 40) // uncommitted
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30})

	// Crash immediately after a checkpoint with an empty DPT (all pages flushed).
	// Recovery should find nothing to redo or undo, and must not write any pages.
	require.NoError(t, env.bp.FlushAllPages())
	cm = NewCheckpointManager(env.mlm, env.bp, env.tm, env.dbDir, time.Hour)
	_, err = cm.Checkpoint()
	require.NoError(t, err)
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30})
	var totalWrites int64
	env.statsDSM.Files.Range(func(_ common.ObjectID, f *storage.StatsDBFile) bool {
		totalWrites += f.WriteCnt.Load()
		return true
	})
	assert.Equal(t, int64(0), totalWrites,
		"recovery with empty DPT should not write any pages (nothing to redo or undo)")

	// Multiple checkpoints: recovery must start from the latest one.
	// kv2 is populated and flushed before any kv checkpoints so its pages
	// are absent from the DPT at crash time — recovery must not touch them.
	kv2Meta, err := env.cat.GetTableMetadata("kv2")
	require.NoError(t, err)
	txn1, err = env.tm.Begin()
	require.NoError(t, err)
	_, err = env.execSQL(txn1, "INSERT INTO kv2 VALUES (1, 100)")
	require.NoError(t, err)
	_, err = env.execSQL(txn1, "INSERT INTO kv2 VALUES (2, 200)")
	require.NoError(t, err)
	require.NoError(t, env.tm.Commit(txn1))
	require.NoError(t, env.bp.FlushAllPages()) // kv2 pages now clean on disk

	cm = NewCheckpointManager(env.mlm, env.bp, env.tm, env.dbDir, time.Hour)
	txn1, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn1, 4, 40)
	require.NoError(t, env.tm.Commit(txn1))
	_, err = cm.Checkpoint()
	require.NoError(t, err)
	txn2, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn2, 5, 50)
	require.NoError(t, env.tm.Commit(txn2))
	_, err = cm.Checkpoint()
	require.NoError(t, err)
	txn3, err = env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txn3, 6, 60) // uncommitted
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30, 4: 40, 5: 50})

	kv2Stats, ok := env.statsDSM.Files.Load(kv2Meta.Oid)
	kv2Reads := int64(0)
	if ok {
		kv2Reads = kv2Stats.ReadCnt.Load()
	}
	assert.Equal(t, int64(0), kv2Reads,
		"recovery should not read kv2 pages (only kv was dirty at crash)")

	// Transaction spanning checkpoint: a transaction that starts before the checkpoint
	// fires must be captured in the checkpoint and undone on recovery.
	txnBefore, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txnBefore, 7, 70) // uncommitted
	env.insertKV(txnBefore, 8, 80)
	cm = NewCheckpointManager(env.mlm, env.bp, env.tm, env.dbDir, time.Hour)
	_, err = cm.Checkpoint()
	require.NoError(t, err)
	txnAfter, err := env.tm.Begin()
	require.NoError(t, err)
	env.insertKV(txnAfter, 9, 90)
	require.NoError(t, env.tm.Commit(txnAfter)) // committed after checkpoint
	// Crash without committing txnBefore.
	env = env.crashAndRecover()
	env.assertKVContains(map[int64]int64{1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 9: 90})
}

// TestRecovery_CheckpointATTAccuracy verifies that the analysis phase correctly
// reconciles a checkpoint's Active Transaction Table (ATT) against the rest of
// the log. Three scenarios are exercised in sequence, each building on the
// previously committed state.
func TestRecovery_CheckpointATTAccuracy(t *testing.T) {
	env := newRecoveryTestEnv(t)

	// findBeginLSN returns the LSN of the BeginTransaction record for txnID.
	findBeginLSN := func(txnID common.TransactionID) storage.LSN {
		iter, err := env.mlm.Iterator(0)
		require.NoError(t, err)
		defer iter.Close()
		var lsn storage.LSN
		var found bool
		for iter.Next() {
			r := iter.CurrentRecord()
			if r.RecordType() == storage.LogBeginTransaction && r.TxnID() == txnID {
				lsn = iter.CurrentLSN()
				found = true
			}
		}
		require.NoError(t, iter.Error())
		require.True(t, found, "no BeginTransaction for txnID %d", txnID)
		return lsn
	}

	// beginCheckpoint appends a BeginCheckpoint record and returns its LSN.
	beginCheckpoint := func() storage.LSN {
		buf := make([]byte, storage.BeginCheckpointRecordSize())
		lsn, err := env.mlm.Append(storage.NewBeginCheckpointRecord(buf))
		require.NoError(t, err)
		return lsn
	}

	// sealCheckpoint appends an EndCheckpoint with att (empty DPT) and writes
	// the master record file pointing at beginLSN.
	sealCheckpoint := func(beginLSN storage.LSN, att []transaction.ATTEntry) {
		dataSize := 8 + 16*len(att) // 4(numATT) + entries + 4(numDPT=0)
		endBuf := make([]byte, storage.EndCheckpointRecordSize(dataSize))
		endRecord := storage.NewEndCheckpointRecord(endBuf, dataSize)
		payload := endRecord.CheckpointData()
		offset := 0
		binary.LittleEndian.PutUint32(payload[offset:], uint32(len(att)))
		offset += 4
		for _, e := range att {
			binary.LittleEndian.PutUint64(payload[offset:], uint64(e.ID))
			offset += 8
			binary.LittleEndian.PutUint64(payload[offset:], uint64(e.StartLSN))
			offset += 8
		}
		binary.LittleEndian.PutUint32(payload[offset:], 0) // numDPT = 0
		_, err := env.mlm.Append(endRecord)
		require.NoError(t, err)
		masterData := make([]byte, 8)
		binary.LittleEndian.PutUint64(masterData, uint64(beginLSN))
		require.NoError(t, os.WriteFile(filepath.Join(env.dbDir, MasterRecordFileName), masterData, 0644))
	}

	// Scenario A: a transaction commits between a BeginCheckpoint and its paired
	// EndCheckpoint (the fuzzy checkpoint race window). The EndCheckpoint's ATT
	// records the transaction as active because the ATT snapshot was taken before
	// the commit. Recovery must not treat it as active and undo its writes.
	//
	// Log: Begin | Insert | BeginCheckpoint | Commit | EndCheckpoint(ATT={T})
	{
		txn, err := env.tm.Begin()
		require.NoError(t, err)
		env.insertKV(txn, 1, 10)
		begin1 := findBeginLSN(txn.ID())

		ckptLSN := beginCheckpoint() // ATT snapshot: T still active

		require.NoError(t, env.tm.Commit(txn)) // commit lands in the race window
		require.NoError(t, env.bp.FlushAllPages())
		sealCheckpoint(ckptLSN, []transaction.ATTEntry{{ID: txn.ID(), StartLSN: begin1}})

		env = env.crashAndRecover()
		env.assertKVContains(map[int64]int64{1: 10})
	}

	// Scenario B: a transaction aborts between a BeginCheckpoint and its
	// EndCheckpoint. The EndCheckpoint's ATT still lists it as active. Recovery
	// must not re-abort it — the undo was already applied before the crash.
	//
	// Log: Begin | Insert | BeginCheckpoint | InsertCLR | Abort | EndCheckpoint(ATT={T})
	{
		txn, err := env.tm.Begin()
		require.NoError(t, err)
		env.insertKV(txn, 2, 20)
		begin2 := findBeginLSN(txn.ID())

		ckptLSN := beginCheckpoint()

		require.NoError(t, env.tm.Abort(txn)) // abort lands in the race window
		require.NoError(t, env.bp.FlushAllPages())
		sealCheckpoint(ckptLSN, []transaction.ATTEntry{{ID: txn.ID(), StartLSN: begin2}})

		env = env.crashAndRecover()
		env.assertKVContains(map[int64]int64{1: 10})
	}

	// Scenario C: the checkpoint ATT holds two transactions — one that committed
	// in the race window (T3) and one genuinely uncommitted at crash time (T4).
	// T3's rows must survive; T4's must be rolled back.
	//
	// Log: Begin T3 | Insert T3 | Begin T4 | Insert T4 | BeginCheckpoint |
	//      Commit T3 | EndCheckpoint(ATT={T3, T4})   [T4 never commits]
	{
		txn3, err := env.tm.Begin()
		require.NoError(t, err)
		env.insertKV(txn3, 3, 30)
		begin3 := findBeginLSN(txn3.ID())

		txn4, err := env.tm.Begin()
		require.NoError(t, err)
		env.insertKV(txn4, 4, 40)
		begin4 := findBeginLSN(txn4.ID())

		ckptLSN := beginCheckpoint()

		require.NoError(t, env.tm.Commit(txn3)) // T3 commits in the race window
		// T4 intentionally left uncommitted — crash occurs before its commit
		require.NoError(t, env.bp.FlushAllPages())
		sealCheckpoint(ckptLSN, []transaction.ATTEntry{
			{ID: txn3.ID(), StartLSN: begin3},
			{ID: txn4.ID(), StartLSN: begin4},
		})

		env = env.crashAndRecover()
		env.assertKVContains(map[int64]int64{1: 10, 3: 30})
	}

	assertTIDMonotone(t, env)
}
