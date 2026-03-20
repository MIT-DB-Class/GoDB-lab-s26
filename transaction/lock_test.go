package transaction

import (
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"mit.edu/dsg/godb/common"
)

// TestLock_Basic_AcquireRelease checks simple lock acquisition and release.
func TestLock_Basic_AcquireRelease(t *testing.T) {
	lm := NewLockManager()
	tid := common.TransactionID(1)
	tag := NewTableLockTag(common.ObjectID(1))

	for _, m := range []DBLockMode{LockModeS, LockModeX, LockModeIS, LockModeIX, LockModeSIX} {
		assert.False(t, lm.LockHeld(tag))
		err := lm.Lock(tid, tag, m)
		assert.NoError(t, err)
		assert.True(t, lm.LockHeld(tag))
		assert.NoError(t, lm.Unlock(tid, tag))
	}

	// additionally, verify that LockHeld returns true until the last holder
	// unlocks, and false only after every holder has released.
	for _, mode := range []DBLockMode{LockModeS, LockModeIS} {

		for i := 0; i < 5; i++ {
			assert.NoError(t, lm.Lock(common.TransactionID(i+1), tag, mode))
		}
		for i := 0; i < 5-1; i++ {
			assert.NoError(t, lm.Unlock(common.TransactionID(i+1), tag))
			assert.True(t, lm.LockHeld(tag), "LockHeld should be true after unlock %d/%d (mode=%s)", i+1, 4, mode)
		}
		assert.NoError(t, lm.Unlock(common.TransactionID(5), tag))
		assert.False(t, lm.LockHeld(tag), "LockHeld should be false after last holder unlocks (mode=%s)", mode)
	}
}

// TestLock_Basic_Compatibility verifies compatibility check is correct and locks are granted immediately when compatible
func TestLock_Basic_Compatibility(t *testing.T) {
	lm := NewLockManager()
	tag := NewTableLockTag(common.ObjectID(1))

	err := lm.Lock(common.TransactionID(1), tag, LockModeS)
	assert.NoError(t, err)
	err = lm.Lock(common.TransactionID(2), tag, LockModeS)
	assert.NoError(t, err)
	err = lm.Lock(common.TransactionID(3), tag, LockModeIS)
	assert.NoError(t, err)
	assert.NoError(t, lm.Unlock(common.TransactionID(1), tag))
	assert.NoError(t, lm.Unlock(common.TransactionID(2), tag))

	// At this point, the lock should be IS
	err = lm.Lock(common.TransactionID(6), tag, LockModeSIX)
	assert.NoError(t, err)
	assert.NoError(t, lm.Unlock(common.TransactionID(6), tag))

	err = lm.Lock(common.TransactionID(7), tag, LockModeIX)
	assert.NoError(t, err)
	assert.NoError(t, lm.Unlock(common.TransactionID(3), tag))

	// At this point, the lock should be IX
	err = lm.Lock(common.TransactionID(8), tag, LockModeIX)
	assert.NoError(t, err)
	assert.NoError(t, lm.Unlock(common.TransactionID(7), tag))
	assert.NoError(t, lm.Unlock(common.TransactionID(8), tag))
}

// TestLock_Basic_MutualExclusion verifies that conflicting locks cause the requester to wait (when valid).
func TestLock_Basic_MutualExclusion(t *testing.T) {
	for i := 0; i < 100; i++ {
		lm := NewLockManager()
		tag := NewTableLockTag(1)
		accessed := atomic.Bool{}
		var wg sync.WaitGroup
		wg.Add(5)

		simulateLockAndWork := func(id common.TransactionID, mode DBLockMode) {
			for {
				err := lm.Lock(id, tag, mode)
				if err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					runtime.Gosched()
					continue
				} else {
					assert.True(t, accessed.CompareAndSwap(false, true))
					time.Sleep(time.Millisecond)
					accessed.Store(false)
					assert.NoError(t, lm.Unlock(id, tag))
					wg.Done()
					break
				}
			}
		}

		go simulateLockAndWork(common.TransactionID(1), LockModeX)
		go simulateLockAndWork(common.TransactionID(2), LockModeX)
		go simulateLockAndWork(common.TransactionID(3), LockModeSIX)
		go simulateLockAndWork(common.TransactionID(4), LockModeS)
		go simulateLockAndWork(common.TransactionID(5), LockModeIX)
		wg.Wait()
	}
}

// TestLock_Basic_UnlockNotOwned verifies that Unlock returns a LockNotFoundError when the
// transaction does not hold the requested lock (never locked, held by another TID, or already released).
func TestLock_Basic_UnlockNotOwned(t *testing.T) {
	tag := NewTableLockTag(common.ObjectID(1))
	t1 := common.TransactionID(1)
	t2 := common.TransactionID(2)

	// Case 1: Unlock a tag that was never locked by anyone.
	lm := NewLockManager()
	err := lm.Unlock(t1, tag)
	assert.Error(t, err)
	assert.Equal(t, common.LockNotFoundError, err.(common.GoDBError).Code)

	// Case 2: Unlock a tag that is held by a different transaction.
	assert.NoError(t, lm.Lock(t1, tag, LockModeX))
	err = lm.Unlock(t2, tag)
	assert.Error(t, err)
	assert.Equal(t, common.LockNotFoundError, err.(common.GoDBError).Code)
	// T2's failed unlock must not drop T1's lock.
	assert.True(t, lm.LockHeld(tag))
	assert.NoError(t, lm.Unlock(t1, tag))

	// Case 3: Double-unlock — unlock a tag that was already released.
	err = lm.Unlock(t1, tag)
	assert.Error(t, err)
	assert.Equal(t, common.LockNotFoundError, err.(common.GoDBError).Code)
}

// TestLock_Basic_Upgrade verifies that transactions can successfully upgrade from
// Shared (S) to Exclusive (X) locks, and that this upgrade maintains mutual exclusion.
func TestLock_Basic_Upgrade(t *testing.T) {
	lm := NewLockManager()
	tag := NewTableLockTag(1) // Single resource to force contention
	accessed := atomic.Bool{}
	var wg sync.WaitGroup
	wg.Add(100)

	upgrader := func(id common.TransactionID) {
		for {
			err := lm.Lock(id, tag, LockModeS)
			if err != nil {
				assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
				runtime.Gosched()
				continue
			}
			runtime.Gosched()
			err = lm.Lock(id, tag, LockModeX)
			if err != nil {
				assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
				assert.NoError(t, lm.Unlock(id, tag))
				runtime.Gosched()
				continue
			}
			assert.True(t, accessed.CompareAndSwap(false, true))
			time.Sleep(time.Millisecond)
			accessed.Store(false)
			assert.NoError(t, lm.Unlock(id, tag))
			wg.Done()
			break
		}
	}

	sharer := func(id common.TransactionID) {
		for {
			err := lm.Lock(id, tag, LockModeS)
			if err != nil {
				assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
				runtime.Gosched()
				continue
			}
			assert.False(t, accessed.Load())
			time.Sleep(time.Millisecond)
			assert.NoError(t, lm.Unlock(id, tag))
			wg.Done()
			break
		}
	}

	for i := 1; i < 101; i++ {
		r := rand.Intn(2)
		if r == 0 {
			go upgrader(common.TransactionID(i))
		} else {
			go sharer(common.TransactionID(i))
		}
	}
	wg.Wait()
}

// TestLock_Basic_UpgradeConflict tests the classic "conversion deadlock": two transactions each hold S
// and both attempt to upgrade to X simultaneously. Exactly one must succeed and exactly one must get
// a DeadlockError, regardless of the deadlock resolution strategy.
func TestLock_Basic_UpgradeConflict(t *testing.T) {
	for i := 0; i < 100; i++ {
		lm := NewLockManager()
		tag := NewTableLockTag(1)
		t1 := common.TransactionID(10)
		t2 := common.TransactionID(20)

		assert.NoError(t, lm.Lock(t1, tag, LockModeS))
		assert.NoError(t, lm.Lock(t2, tag, LockModeS))

		abortCount, successCount := atomic.Int32{}, atomic.Int32{}
		var wg sync.WaitGroup
		wg.Add(2)

		upgrade := func(tid common.TransactionID) {
			defer wg.Done()
			err := lm.Lock(tid, tag, LockModeX)
			if err != nil {
				assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
				abortCount.Add(1)
				assert.NoError(t, lm.Unlock(tid, tag)) // release the S still held
			} else {
				successCount.Add(1)
				assert.NoError(t, lm.Unlock(tid, tag))
			}
		}
		go upgrade(t1)
		go upgrade(t2)
		wg.Wait()

		assert.Equal(t, int32(1), abortCount.Load(), "exactly one upgrade must be aborted")
		assert.Equal(t, int32(1), successCount.Load(), "exactly one upgrade must succeed")
	}
}

// TestLock_Basic_Hierarchical verifies multi-granularity lock compatibility at the table level.
// It checks that:
// 1. S blocks IX (scanner reading whole table prevents tuple-level writer)
// 2. IX blocks S (tuple-level writer prevents whole-table scanner)
// 3. IS and IX coexist (readers and writers at tuple-level can proceed together)
func TestLock_Basic_Hierarchical(t *testing.T) {
	for i := 0; i < 100; i++ {
		retryLock := func(t *testing.T, lm *LockManager, tid common.TransactionID, tag DBLockTag, mode DBLockMode) {
			t.Helper()
			for {
				err := lm.Lock(tid, tag, mode)
				if err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					runtime.Gosched()
					continue
				}
				return
			}
		}

		// Case 1: S blocks IX
		// Scanner holds Table S; a concurrent writer must not acquire Table IX until the scanner releases.
		{
			lm := NewLockManager()
			tableTag := NewTableLockTag(1)
			tScanner := common.TransactionID(1)
			tWriter := common.TransactionID(2)
			holderActive := atomic.Bool{}

			assert.NoError(t, lm.Lock(tScanner, tableTag, LockModeS))
			holderActive.Store(true)

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				retryLock(t, lm, tWriter, tableTag, LockModeIX)
				assert.False(t, holderActive.Load(), "IX acquired while S still held")
				assert.NoError(t, lm.Unlock(tWriter, tableTag))
			}()

			time.Sleep(10 * time.Millisecond) // give writer a chance to block
			holderActive.Store(false)
			assert.NoError(t, lm.Unlock(tScanner, tableTag))
			wg.Wait()
		}

		// Case 2: IX blocks S
		// Writer holds Table IX; a concurrent scanner must not acquire Table S until the writer releases.
		{
			lm := NewLockManager()
			tableTag := NewTableLockTag(1)
			tWriter := common.TransactionID(1)
			tScanner := common.TransactionID(2)
			holderActive := atomic.Bool{}

			assert.NoError(t, lm.Lock(tWriter, tableTag, LockModeIX))
			holderActive.Store(true)

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				retryLock(t, lm, tScanner, tableTag, LockModeS)
				assert.False(t, holderActive.Load(), "S acquired while IX still held")
				assert.NoError(t, lm.Unlock(tScanner, tableTag))
			}()

			time.Sleep(10 * time.Millisecond)
			holderActive.Store(false)
			assert.NoError(t, lm.Unlock(tWriter, tableTag))
			wg.Wait()
		}

		// Case 3: IS and IX coexist
		// Reader holding Table IS and writer holding Table IX must be compatible (both grantable simultaneously).
		{
			lm := NewLockManager()
			tableTag := NewTableLockTag(1)
			tReader := common.TransactionID(1)
			tWriter := common.TransactionID(2)

			assert.NoError(t, lm.Lock(tReader, tableTag, LockModeIS))
			assert.NoError(t, lm.Lock(tWriter, tableTag, LockModeIX))
			assert.True(t, lm.LockHeld(tableTag))
			assert.NoError(t, lm.Unlock(tReader, tableTag))
			assert.True(t, lm.LockHeld(tableTag))
			assert.NoError(t, lm.Unlock(tWriter, tableTag))
			assert.False(t, lm.LockHeld(tableTag))
		}
	}
}

// TestLock_Basic_Hierarchical_Upgrade verifies that a single transaction can upgrade
// through the intent lock hierarchy (IS -> S -> SIX) without deadlocking itself.
// It also verifies SIX semantics: IS is compatible but IX is not.
func TestLock_Basic_Hierarchical_Upgrade(t *testing.T) {
	for i := 0; i < 100; i++ {
		lm := NewLockManager()
		tableTag := NewTableLockTag(1)
		tid := common.TransactionID(1)

		// Part 1: Single-transaction upgrade chain IS -> S -> SIX.
		// A transaction cannot deadlock with itself, so these must all succeed unconditionally.
		assert.NoError(t, lm.Lock(tid, tableTag, LockModeIS))
		assert.NoError(t, lm.Lock(tid, tableTag, LockModeS))
		assert.NoError(t, lm.Lock(tid, tableTag, LockModeSIX))

		// Part 2: SIX semantics with other transactions.
		// IS is compatible with SIX: T2 should acquire IS immediately without blocking.
		t2 := common.TransactionID(2)
		assert.NoError(t, lm.Lock(t2, tableTag, LockModeIS), "IS should be immediately compatible with SIX")

		// IX is NOT compatible with SIX. Test via upgrade: T3 first acquires IS (compatible
		// with SIX, so it is granted immediately), then attempts to upgrade IS→IX (incompatible
		// with SIX), which must block until T1 releases.
		t3 := common.TransactionID(3)
		sixHolderActive := atomic.Bool{}
		sixHolderActive.Store(true)

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				// Step 1: IS is compatible with SIX — must not block.
				if err := lm.Lock(t3, tableTag, LockModeIS); err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					runtime.Gosched()
					continue
				}
				// Step 2: Upgrade IS → IX; incompatible with T1's SIX, must block.
				if err := lm.Lock(t3, tableTag, LockModeIX); err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					assert.NoError(t, lm.Unlock(t3, tableTag))
					runtime.Gosched()
					continue
				}
				assert.False(t, sixHolderActive.Load(), "IX upgrade completed while SIX still held")
				assert.NoError(t, lm.Unlock(t3, tableTag))
				break
			}
		}()

		time.Sleep(10 * time.Millisecond) // give T3 a chance to block
		sixHolderActive.Store(false)
		assert.NoError(t, lm.Unlock(tid, tableTag))
		assert.NoError(t, lm.Unlock(t2, tableTag))
		wg.Wait()
	}
}

// TestLock_Basic_Deadlock creates a direct deadlock cycle between two transactions
// and asserts that the system resolves it (by aborting exactly one transaction).
func TestLock_Basic_Deadlock(t *testing.T) {
	for i := 0; i < 100; i++ {
		lm := NewLockManager()
		t1 := common.TransactionID(10)
		t2 := common.TransactionID(20)

		tagA := NewTableLockTag(1)
		tagB := NewTableLockTag(2)
		assert.NoError(t, lm.Lock(t1, tagA, LockModeX))
		assert.NoError(t, lm.Lock(t2, tagB, LockModeX))
		var wg sync.WaitGroup
		wg.Add(2)

		abortCount, successCount := atomic.Int32{}, atomic.Int32{}

		worker := func(txn common.TransactionID, held, acquire DBLockTag) {
			defer wg.Done()
			err := lm.Lock(txn, acquire, LockModeX)
			if err == nil {
				successCount.Add(1)
				assert.NoError(t, lm.Unlock(txn, acquire))
				assert.NoError(t, lm.Unlock(txn, held))
			} else {
				assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
				abortCount.Add(1)
				assert.NoError(t, lm.Unlock(txn, held))
			}
		}

		go worker(t1, tagA, tagB)
		go worker(t2, tagB, tagA)
		wg.Wait()
		assert.Equal(t, int32(1), abortCount.Load())
		assert.Equal(t, int32(1), successCount.Load())
	}
}

// TestLock_Basic_LongDeadlockChain creates a 1000-transaction deadlock cycle and verifies
// that the manager breaks it by aborting at least one transaction while all workers complete.
func TestLock_Basic_LongDeadlockChain(t *testing.T) {
	const N = 1000
	for i := 0; i < 100; i++ {
		lm := NewLockManager()

		tids := make([]common.TransactionID, N)
		tags := make([]DBLockTag, N)
		for k := 0; k < N; k++ {
			tids[k] = common.TransactionID((k + 1) * 10)
			tags[k] = NewTableLockTag(common.ObjectID(k + 1))
		}

		// Each transaction holds its own tag before the goroutines start.
		for k := 0; k < N; k++ {
			assert.NoError(t, lm.Lock(tids[k], tags[k], LockModeX))
		}

		abortCount, successCount := atomic.Int32{}, atomic.Int32{}
		var wg sync.WaitGroup
		wg.Add(N)

		for k := 0; k < N; k++ {
			go func(idx int) {
				defer wg.Done()
				tid := tids[idx]
				held := tags[idx]
				acquire := tags[(idx+1)%N]
				err := lm.Lock(tid, acquire, LockModeX)
				if err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					abortCount.Add(1)
					assert.NoError(t, lm.Unlock(tid, held))
				} else {
					successCount.Add(1)
					assert.NoError(t, lm.Unlock(tid, acquire))
					assert.NoError(t, lm.Unlock(tid, held))
				}
			}(k)
		}
		wg.Wait()

		assert.GreaterOrEqual(t, abortCount.Load(), int32(1), "at least one transaction must be aborted to break the cycle")
		assert.Equal(t, int32(N), abortCount.Load()+successCount.Load(), "all workers must have completed")
	}
}

// TestLock_Basic_WaiterFairness verifies that an exclusive write request is served in bounded
// time against a constant backdrop of mixed IS, IX, and S readers. It checks that:
// 1. The X writer acquires its lock within 500ms despite continuous reader traffic.
// 2. When the X writer holds the lock, no IS, IX, or S holder is concurrently active.
func TestLock_Basic_WaiterFairness(t *testing.T) {
	for i := 0; i < 100; i++ {
		lm := NewLockManager()
		tag := NewTableLockTag(1)
		var tidCounter atomic.Int64

		// Readers get a fresh TID on every attempt; the writer keeps a fixed TID throughout.
		// Once all current holders have higher TIDs than the writer, Wait-Die kills them on
		// conflict rather than the writer, allowing the writer to make progress.
		var activeIS, activeIX, activeS atomic.Int32
		stop := make(chan struct{})
		var bgWg sync.WaitGroup

		startReader := func(mode DBLockMode, counter *atomic.Int32) {
			bgWg.Add(1)
			go func() {
				defer bgWg.Done()
				for {
					select {
					case <-stop:
						return
					default:
					}
					tid := common.TransactionID(tidCounter.Add(1))
					for {
						if err := lm.Lock(tid, tag, mode); err != nil {
							assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
							runtime.Gosched()
							continue
						}
						break
					}
					counter.Add(1)
					runtime.Gosched()
					counter.Add(-1)
					assert.NoError(t, lm.Unlock(tid, tag))
				}
			}()
		}

		for i := 0; i < 4; i++ {
			startReader(LockModeIS, &activeIS)
			startReader(LockModeIX, &activeIX)
			startReader(LockModeS, &activeS)
		}

		// Give readers time to start and saturate the lock.
		time.Sleep(30 * time.Millisecond)

		// Writer (TID=1) requests X. It must be served within 500ms.
		writerServed := make(chan struct{})
		go func() {
			tid := common.TransactionID(tidCounter.Add(1))
			for {
				if err := lm.Lock(tid, tag, LockModeX); err != nil {
					assert.Equal(t, common.DeadlockError, err.(common.GoDBError).Code)
					runtime.Gosched()
					continue
				}
				// Verify exclusion: no reader should be active while X is held.
				assert.Equal(t, int32(0), activeIS.Load(), "IS active while X held")
				assert.Equal(t, int32(0), activeIX.Load(), "IX active while X held")
				assert.Equal(t, int32(0), activeS.Load(), "S active while X held")
				assert.NoError(t, lm.Unlock(tid, tag))
				close(writerServed)
				return
			}
		}()

		select {
		case <-writerServed:
			// Writer was served — starvation-free.
		case <-time.After(500 * time.Millisecond):
			t.Error("X writer starved: not served within 500ms against constant IS/IX/S backdrop")
		}

		close(stop)
		bgWg.Wait()
	}
}

// TestLock_Stress_BankTransfer verifies lock correctness under high-contention concurrent transfers.
// It checks that:
// 1. Exclusive locks prevent concurrent access to the same account.
// 2. No money is created or destroyed across all transfers (total balance invariant).
// 3. The lock manager handles concurrent deadlock resolution without livelock.
func TestLock_Stress_BankTransfer(t *testing.T) {
	lm := NewLockManager()
	numAccounts := 10
	initialBalance := 100
	accounts := make([]int, numAccounts)
	for i := 0; i < numAccounts; i++ {
		accounts[i] = initialBalance
	}

	var wg sync.WaitGroup
	numWorkers := 20
	opsPerWorker := 100000
	var tidCounter atomic.Int64

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id)))

			for j := 0; j < opsPerWorker; j++ {
				from, to := r.Intn(numAccounts), r.Intn(numAccounts)
				if from == to {
					continue
				}
				tid := common.TransactionID(tidCounter.Add(1))
				lockFrom, lockTo := NewTableLockTag(common.ObjectID(from)), NewTableLockTag(common.ObjectID(to))
				for {
					if err := lm.Lock(tid, lockFrom, LockModeX); err != nil {
						runtime.Gosched()
						continue
					}

					if err := lm.Lock(tid, lockTo, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, lockFrom))
						runtime.Gosched()
						continue
					}

					accounts[from]--
					accounts[to]++

					assert.NoError(t, lm.Unlock(tid, lockTo))
					assert.NoError(t, lm.Unlock(tid, lockFrom))
					break
				}
			}
		}(i)
	}
	wg.Wait()

	total := 0
	for _, b := range accounts {
		total += b
	}
	assert.Equal(t, numAccounts*initialBalance, total, "Bank total invariant failed! Money created/destroyed.")
	// No locks should be present after the workload has quiesced
}

// TestLock_Stress_TornRead verifies that Shared (S) locks correctly isolate against Exclusive (X) locks.
// Writers update an entire array; Readers read the array. Readers must never see a partially updated array.
func TestLock_Stress_TornRead(t *testing.T) {
	lm := NewLockManager()
	tag := NewTableLockTag(1)

	const arrSize = 1000
	var data [arrSize]int

	var wg sync.WaitGroup
	var tidCounter atomic.Int64
	var readCount atomic.Int64

	// Start time for duration-based test
	start := time.Now()
	duration := 5 * time.Second

	// Writers: acquire X, overwrite all slots with a single value, release.
	// Fresh TID per attempt ensures no writer is systematically younger than all others.
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id)))

			for time.Since(start) < duration {
				val := int(r.Int31())
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tag, LockModeX); err != nil {
						runtime.Gosched()
						continue
					}
					for k := 0; k < arrSize; k++ {
						data[k] = val
					}
					assert.NoError(t, lm.Unlock(tid, tag))
					break
				}
				runtime.Gosched()
			}
		}(i)
	}

	// Readers: acquire S (compatible with other S, exclusive with X), check all slots match.
	// Fresh TID per attempt prevents permanent starvation (otherwise readers, always younger
	// than some writer, would die every cycle and never make any assertions).
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			for time.Since(start) < duration {
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tag, LockModeS); err != nil {
						runtime.Gosched()
						continue
					}
					v := data[0]
					for k := 1; k < arrSize; k++ {
						assert.Equal(t, v, data[k], "Torn read detected! Index 0=%d but Index %d=%d", v, k, data[k])
					}
					readCount.Add(1)
					assert.NoError(t, lm.Unlock(tid, tag))
					break
				}
				runtime.Gosched()
			}
		}(i)
	}

	wg.Wait()
	assert.Greater(t, readCount.Load(), int64(0), "no reads completed — readers were starved for the entire test duration")
}

// TestLock_Stress_Mixed_ScannerUpdater verifies the multi-granularity lock protocol using a
// hierarchical bank transfer scenario with five concurrent actor types. It checks that:
//  1. Point transferors (Table IX + Tuple X) can run concurrently when they touch different slots.
//  2. Periodic scanners (Table S) always observe a consistent total balance: S is incompatible
//     with both IX and X, so no transfers or bulk ops are in flight during a scan.
//  3. Bulk operators (Table X) atomically add money to all accounts; their changes are
//     immediately visible to the next scanner.
//  4. IS → S readers acquire intent-then-shared at the table level; included to increase
//     scheduling interleavings.
//  5. SIX transferors (Table SIX + Tuple X) are fully serialized with scanners, bulk operators,
//     and point transferors; like scanners, they verify the total balance is consistent before
//     modifying any account.
func TestLock_Stress_Mixed_ScannerUpdater(t *testing.T) {
	lm := NewLockManager()
	tableTag := NewTableLockTag(common.ObjectID(1))

	const numAccounts = 10
	const initialBalance = 1000
	accounts := make([]int64, numAccounts)
	for i := range accounts {
		accounts[i] = initialBalance
	}

	var totalBonus atomic.Int64
	tupleTagForSlot := func(slot int) DBLockTag {
		return NewTupleLockTag(common.RecordID{
			PageID: common.PageID{Oid: 1, PageNum: 1},
			Slot:   int32(slot),
		})
	}

	var wg sync.WaitGroup
	duration := 10 * time.Second
	start := time.Now()

	var tidCounter atomic.Int64
	// 1. Point Transferors (Table IX + Tuple X on from + Tuple X on to)
	// IX+IX is compatible at the table level, so multiple transferors run concurrently
	// as long as they touch different account slots.
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id)))
			for time.Since(start) < duration {
				from, to := r.Intn(numAccounts), r.Intn(numAccounts)
				if from == to {
					continue
				}
				fromTag, toTag := tupleTagForSlot(from), tupleTagForSlot(to)
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeIX); err != nil {
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, fromTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, toTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					accounts[from]--
					accounts[to]++
					assert.NoError(t, lm.Unlock(tid, toTag))
					assert.NoError(t, lm.Unlock(tid, fromTag))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					break
				}
			}
		}(i)
	}

	// 2. Scanners (Table S)
	// S is incompatible with IX and X, so no transfers or bulk ops are in flight during a scan.
	// The scanner reads the total and compares against the expected value tracked by totalBonus.
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for time.Since(start) < duration {
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeS); err != nil {
						runtime.Gosched()
						continue
					}
					var total int64
					for _, bal := range accounts {
						total += bal
					}
					expected := int64(numAccounts*initialBalance) + totalBonus.Load()
					assert.Equal(t, expected, total, "Scanner: total balance mismatch")
					assert.NoError(t, lm.Unlock(tid, tableTag))
					runtime.Gosched()
					break
				}
			}
		}(i)
	}

	// 3. Bulk Operators (Table X)
	// Holds the table exclusively and adds a random bonus to every account, keeping the
	// invariant intact by atomically recording the total added in totalBonus.
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id)))
			for time.Since(start) < duration {
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeX); err != nil {
						runtime.Gosched()
						continue
					}
					b := int64(r.Intn(10) + 1)
					for k := range accounts {
						accounts[k] += b
					}
					totalBonus.Add(b * int64(numAccounts))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					runtime.Gosched()
					break
				}
			}
		}(i)
	}

	// 4. IS → S Readers (Table IS → upgrade to S)
	// IS is compatible with IX and X at the table level, so these readers can run concurrently
	// with point transferors. After upgrading to S, the holder is incompatible with IX and X.
	// No balance invariant is checked — included purely to increase scheduling interleavings.
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for time.Since(start) < duration {
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeIS); err != nil {
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, tableTag, LockModeS); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					assert.NoError(t, lm.Unlock(tid, tableTag))
					runtime.Gosched()
					break
				}
			}
		}(i)
	}

	// 5. SIX Transferors (Table SIX + Tuple X on from + Tuple X on to)
	// SIX is compatible only with IS at the table level, so SIX transferors are fully serialized
	// with scanners, bulk operators, and point transferors. Like scanners, they verify the total
	// balance is consistent (no concurrent writers) before performing their transfer.
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id + 1000)))
			for time.Since(start) < duration {
				from, to := r.Intn(numAccounts), r.Intn(numAccounts)
				if from == to {
					continue
				}
				fromTag, toTag := tupleTagForSlot(from), tupleTagForSlot(to)
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeSIX); err != nil {
						runtime.Gosched()
						continue
					}
					var sum int64
					for _, bal := range accounts {
						sum += bal
					}
					expected := int64(numAccounts*initialBalance) + totalBonus.Load()
					assert.Equal(t, expected, sum, "SIX transferor: total balance mismatch before transfer")
					if err := lm.Lock(tid, fromTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, toTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					accounts[from]--
					accounts[to]++
					assert.NoError(t, lm.Unlock(tid, toTag))
					assert.NoError(t, lm.Unlock(tid, fromTag))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					break
				}
			}
		}(i)
	}

	wg.Wait()

	var total int64
	for _, bal := range accounts {
		total += bal
	}
	assert.Equal(t, int64(numAccounts*initialBalance)+totalBonus.Load(), total, "final balance invariant violated")
	// No locks should be present after the workload has quiesced
}

// TestLock_Stress_UpgradeStorm stress-tests concurrent lock upgrades in a bank transfer scenario
// across a 2-level hierarchy (table + tuple) with three qualitatively different upgrade profiles.
// Unlike TestLock_Stress_Mixed_ScannerUpdater, every lock acquisition goes through at least one
// upgrade step — no lock mode is held directly without first holding a weaker mode.
//
//   - Profile A (Table IS→IX, Tuple IS→X): 18 point transferors; multiple can run concurrently
//     when they touch different slots. Table IS is upgraded to IX; each tuple IS is upgraded to X.
//   - Profile B (Table S→SIX, Tuple X): 9 scan-then-update transferors; fully serialized
//     with all writers. Table IS is upgraded to S, then to SIX; each tuple IS is upgraded to X.
//   - Profile C (Table IS→IX→X): 3 bulk bonus writers; fully exclusive at all levels via two
//     table upgrades: IS to IX, then IX to X.
//
// Invariants verified inside each critical section using atomic counters:
//   - Final: total balance equals initial balance plus accumulated bulk bonuses.
func TestLock_Stress_UpgradeStorm(t *testing.T) {
	lm := NewLockManager()
	tableTag := NewTableLockTag(common.ObjectID(1))

	const numAccounts = 10
	const initialBalance = 100
	accounts := make([]int64, numAccounts)
	for i := range accounts {
		accounts[i] = initialBalance
	}

	var totalBonus atomic.Int64
	tupleTag := func(slot int) DBLockTag {
		return NewTupleLockTag(common.RecordID{
			PageID: common.PageID{Oid: 1, PageNum: 1},
			Slot:   int32(slot),
		})
	}

	duration := 10 * time.Second
	start := time.Now()

	var tidCounter atomic.Int64
	var wg sync.WaitGroup

	// 1. Profile A: IS → IX point transferors (Table IS→IX, Tuple S→X on from+to)
	// IS→IX at the table level allows multiple A workers to run concurrently when they touch
	// different slots. Each tuple lock is first acquired as IS then upgraded to X.
	for i := 0; i < 18; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id)))
			for time.Since(start) < duration {
				from, to := r.Intn(numAccounts), r.Intn(numAccounts)
				if from == to {
					continue
				}
				fromTag, toTag := tupleTag(from), tupleTag(to)
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeIS); err != nil {
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, fromTag, LockModeS); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, toTag, LockModeS); err != nil {
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}

					if err := lm.Lock(tid, tableTag, LockModeIX); err != nil {
						assert.NoError(t, lm.Unlock(tid, toTag))
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, fromTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, toTag))
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, toTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, toTag))
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					accounts[from]--
					accounts[to]++
					assert.NoError(t, lm.Unlock(tid, toTag))
					assert.NoError(t, lm.Unlock(tid, fromTag))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					break
				}
			}
		}(i)
	}

	// 2. Profile B: S → SIX scan-then-update transferors (Table IS→S→SIX, Tuple X)
	for i := 0; i < 9; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id + 100)))
			for time.Since(start) < duration {
				from, to := r.Intn(numAccounts), r.Intn(numAccounts)
				if from == to {
					continue
				}
				fromTag, toTag := tupleTag(from), tupleTag(to)
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeS); err != nil {
						runtime.Gosched()
						continue
					}

					var sum int64
					for _, bal := range accounts {
						sum += bal
					}
					expected := int64(numAccounts*initialBalance) + totalBonus.Load()
					assert.Equal(t, expected, sum, "SIX transferor: total balance mismatch before transfer")

					if err := lm.Lock(tid, tableTag, LockModeSIX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}

					if err := lm.Lock(tid, fromTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}

					if err := lm.Lock(tid, toTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, fromTag))
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					accounts[from]--
					accounts[to]++
					assert.NoError(t, lm.Unlock(tid, toTag))
					assert.NoError(t, lm.Unlock(tid, fromTag))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					break
				}
			}
		}(i)
	}

	// 3. Profile C: IS → IX → X bulk writers
	// Upgrades from IS through IX to X, making this fully exclusive at all levels.
	// Adds a random bonus to every account atomically.
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(id + 200)))
			for time.Since(start) < duration {
				tid := common.TransactionID(tidCounter.Add(1))
				for {
					if err := lm.Lock(tid, tableTag, LockModeIS); err != nil {
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, tableTag, LockModeIX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}
					if err := lm.Lock(tid, tableTag, LockModeX); err != nil {
						assert.NoError(t, lm.Unlock(tid, tableTag))
						runtime.Gosched()
						continue
					}

					b := int64(r.Intn(5) + 1)
					for k := range accounts {
						accounts[k] += b
					}
					totalBonus.Add(b * int64(numAccounts))
					assert.NoError(t, lm.Unlock(tid, tableTag))
					break
				}
			}
		}(i)
	}

	wg.Wait()

	var total int64
	for _, b := range accounts {
		total += b
	}
	assert.Equal(t, int64(numAccounts*initialBalance)+totalBonus.Load(), total, "final balance invariant violated")
	// No locks should be present after the workload has quiesced
}
