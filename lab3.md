# 6.5830/6.5831 Lab 3: Transactions & Concurrency Control

**Assigned:** [Date]
**Due:** [Date]

## Introduction
In this lab, you will transform GoDB from a single-threaded query execution engine into a full transactional
database system. The architecture follows standard design principles you have seen in class:

* **SS2PL:** GoDB uses **Strong Strict Two-Phase Locking**. A global `LockManager` enforces
  pessimistic locking. Transactions must acquire locks before reading or writing data and hold them until they commit
  or abort. This will achieve the REPEATABLE READ isolation level.
* **Steal:** Dirty pages from an uncommitted transaction can be written to disk (to free up buffer space).
  This requires us to support **Undo** operations to revert these changes if the transaction aborts.
* **No-Force:** Pages modified by a committed transaction are not forced to disk immediately. This requires us to
  support **Redo** operations (in Lab 4) to recover durability after a crash.

This lab involves three major components:

1. **Lock Manager:** Implement SS2PL and Deadlock Detection (Wait-For Graph).
2. **Transaction Management:** Implement `Begin`, `Commit`, and `Abort`.
3. **Log Generation:** Instrument executors to generate WAL records (Begin, Insert, Update, Delete, Commit, Abort).

## Logistics

* **Files to Modify:**
* `godb/storage/buffer_pool.go`
* `godb/transaction/lock.go`
* `godb/transaction/transaction_context.go`
* `godb/transaction/transaction_manager.go`
* `godb/execution/table_heap.go`
* `godb/execution/index_lookup_executor.go`
* `godb/execution/index_nested_loop_join_executor.go`
* `godb/execution/index_scan_executor.go`
---
## Part 1: The Lock Manager

**File:** `godb/transaction/lock.go`

The `LockManager` is the central authority for concurrency control. It is responsible for tracking which transactions
hold locks on which resources and mediating access when conflicts arise.

### Motivation: The Memory Problem

Lock objects reside in memory (RAM), and a naive approach might be to create a lock object for every tuple in the database.
However, this is prohibitively expensive. Consider a table with 10 million rows. If a transaction scans this table and locks
every tuple individually, we would need to allocate 10 million lock structs. If each lock (containing a mutex, a condition
variable, and a list of waiting transactions) takes ~100 bytes, that scan alone would consume **1 GB of memory** just for
locks! To save memory, we could go to the other extreme and use **Coarse-Grained Locking** (i.e., lock the entire table).
This reduces lock memory overhead, but it destroys concurrency because only one transaction can write to the table at a time.

To solve this, GoDB uses two strategies:

* Multiple Granularity Locking: This system allows multiple levels of lock granularity to co-exist. Each transaction chooses
  the level of locking that best suits their needs. A transaction reading the entire table locks the Table (1 lock
  object), while a transaction modifying a single user locks just that Tuple (1 lock object). To make this mix of
  locking levels safe, we use **Intent Locks**. Before a transaction can lock a specific tuple (fine-grained), it must
  first place an "Intent" lock on the table (coarse-grained).
* Dynamic Lock Allocation: Even with multiple granularity locking, we cannot afford to pre-allocate lock objects for every
  tuple. Instead, we only create lock objects for active locks, and lock objects are aggressively reused to represent
  other locked objects as transactions release them. If a tuple is not being accessed by any transaction, it has zero
  memory overhead in the Lock Manager.

### Implementation Tasks

You need to implement the `LockManager` struct and the following tasks:

1. **Data Structures**: Define the necessary data structures to track which transactions hold or await locks on specific
resources (`DBLockTag`). As discussed in the motivation, you must ensure that your lock manager is memory-efficient. You
should not allocate memory for locks that are not currently held or requested.

2. **`Lock(tid, tag, mode)`**: Attempts to acquire a lock on the given resource.
* **Compatibility**: The method must check if the requested mode is compatible with existing locks on the resource.
* **Blocking**: If the lock cannot be granted immediately, the transaction must block until it can proceed.
* **Upgrades**: If a transaction already holds a lock on the resource (e.g., S) and requests a stronger lock (e.g., X),
  you must handle the upgrade logic correctly.
* **Deadlock Detection**: You must implement a mechanism to detect deadlocks and return a `DeadlockError` if one occurs.
  You are free to choose your detection/resolution strategy, but it must be more sophisticated than a simple timeout. Note
  that not all schemes are appropriate for the API of GoDB.

3. **`Unlock(tid, tag)`**: Releases the lock held by the transaction. If this release makes the resource available to
waiting transactions, they should be woken up.

4. **`LockHeld(tag)`**: Checks if any transaction locks the given resource under any mode.

**Tests:**
Run `go test -v [-race] ./transaction -run TestLock_Basic` to verify correctness of individual locking behaviors (compatibility, mutual exclusion, upgrades, deadlock detection).
Run `go test -v [-race] ./transaction -run TestLock_Stress` to validate correctness under high-concurrency workloads (bank transfer, torn-read prevention, mixed MGL actors).

---

## Part 2: Transaction Context

**Files:**

* `godb/transaction/transaction_context.go`
* `godb/transaction/transaction_manager.go`

GoDB's transaction logic is split into two components:

1. **The Context (`TransactionContext`):** Holds the state of a single running transaction, including its locks and its history of changes.
2. **The Manager (`TransactionManager`):** Orchestrates the global lifecycle of transactions (Begin, Commit, Abort).

In this part, you will implement the core logic for the first part. The `TransactionContext` captures the entire state of
a single transaction. GoDB's implementation has two major optimizations over a standard textbook implementation:

#### 1. Log Record Buffering

Before we can implement transaction rollback (Atomicity), we need a history of what the transaction did. This is
naturally captured by entries in the Write-ahead Log (WAL). GoDB adopts a modern optimization used by high-performance
systems like HyPer: each transaction maintains its own local byte slice (`logRecordBuffer`) that acts as a private
**Undo Log**.

* **Simplified Aborts:** This design makes aborts significantly easier. When a transaction aborts, we have a private,
  contiguous history of exactly what it did. We simply iterate backward through the `logRecordBuffer` and apply the
  inverse of each operation. We do not need to scan a global log file or filter out records from other interleaved transactions.
* **Arena Allocation:** As a bonus, this buffer significantly reduces memory pressure as it acts as an "Arena." We
  allocate space for a new record simply by incrementing an integer offset (`len`). There is no contention on the local
  buffer as it is private to the thread running the transaction. Go's GC does not track allocations made by a transaction,
  as the lifetime of these records is tied to the transaction itself and entire buffer is simply recycled when the
  transaction completes.

#### 2. Context Recycling & sync.Pool

Transaction contexts are "heavy" objects. They contain the log buffer (which can grow large), maps for tracking locks,
and other metadata. Allocating and destroying these objects for every single transaction would create massive pressure 
on the Garbage Collector. To solve this, GoDB uses `sync.Pool` to recycle `TransactionContext` objects. When `Begin()`
is called, you should fetch a context from the pool. When `Commit()` or `Abort()` completes, return it to the pool. The
pool automatically creates new contexts only as needed and will attempt to recycle objects. This means `TransactionContext`
objects are long-lived and reused across different transactions. You must ensure that the context is properly **reset**
before it is reused. Data from a previous transaction (e.g., old locks or log records) must never "leak" into a new one.

#### A Note About Reentrancy

Database locks are typically **reentrant**—meaning if a transaction already holds a lock on a resource (e.g., a specific
tuple), requesting that lock again (or requesting it at a different, compatible mode) should immediately succeed. Without
reentrancy, transactions can easily self-deadlock. Reentrancy helps a database system stay modularized, as different modules
can just request locks as they need without needing to see global lock state. That said, reentrancy logic can be complex.
In GoDB, we handle reentrancy at the transaction context level -- transactions are responsible for remembering what locks
they hold, and automatically eliding lock acquisition when requested again.

**Implementation Task:**
In `godb/transaction/transaction_context.go`, implement the following:

1. **`logRecordBuffer`**: Implement this helper data structure according to the API given. This will be used as a basis
   for the log generation logic (already provided for you, to save you some boilerplate). You will also be using this to
   simplify your buffer logic.
2. LockManagement: Implement the lock-related methods within the TransactionContext.
3. Lifecycle: Implement `Reset()` so a transaction context is properly cleared before reuse.

**Tests:**
Run `go test -v ./transaction -run LogRecordBuffer` to verify the log record buffer data structure (allocation, pop, reset).
Run `go test -v ./transaction -run TxnContext` to verify lock reentrancy, upgrade, and lifecycle (`ReleaseAllLocks`, `Reset`) behavior.

---

## Part 3: Execution Engine Integration

**Files:** 
* `godb/storage/buffer_pool.go`
* `godb/execution/table_heap.go`
* `godb/execution/index_lookup_executor.go`
* `godb/execution/index_nested_loop_join_executor.go`
* `godb/execution/index_scan_executor.go`

Before moving on to the internals of the transaction manager, you must modify your execution and storage layer code to add the
appropriate hooks. Transactions must acquire locks **before** accessing data, and they must then generate log records
and update the LSN of the page **before** modifying it (hence the write-**ahead** log). The buffer pool must **not** flush
a dirty page until the log records corresponding to the updates have been written to disk. Note that you should keep the
original non-transactional code in place and execute it whenever the transaction context passed in is nil. 

#### The Write-Ahead Logging (WAL) API
For this lab, you should program against the `LogManager` interface defined in `godb/storage/wal.go`, and the
`LogRecord` struct defined in `godb/storage/log_record.go`. For this lab, you will never need to read log records
in your code (although we will check them in the tests), and you may assume there is a correct implementation of the
`LogManager`. You will implement all of these in lab 4!

#### Lock Mode and "For Update" Arguments
In scans, the optimal lock mode depends on the optimizer's decision (e.g., S vs. SIX) and is an argument to the
executor. In `ReadTuple`, there is the possibility that the read operation is followed by an update operation (e.g.,
UPDATE ... WHERE ... scans the table first before updating it). To avoid frequent lock updates, a "for update" argument
is provided to the executor to indicate that the executor should acquire exclusive locks when reading.

#### Index Updates

Another issue introduced by concurrent query execution is inconsistency between indexes and tables. GoDB's in-memory
indexes are not separately locked — only tuples carry locks. This creates windows where the index state may not match the
table state from a concurrent reader's perspective. To solve this, GoDB's index layer distinguishes between insert and
delete operations:
- **Inserts are applied immediately.** When a transaction inserts a row, the index entry is added right away. This is
required for own-write visibility — a transaction must be able to find its own inserts via index scans.
- **Deletes are deferred to commit time.** When a transaction deletes an index entry, the underlying tuple is changed 
immediately (under the X-lock), but the index entry is only removed when the transaction *commits*.

This asymmetry is necessary for correctness. If deletes were applied immediately and the transaction later aborted, a
concurrent reader could scan the index, see the entry as gone, and later observe it reappearing after rollback — a
violation of REPEATABLE READ isolation. To implement this, GoDB's `TransactionContext` maintains two slices of
`IndexTask` structs — `abortActions` and `commitActions` — that buffer deferred index operations for the duration of the
transaction. When `InsertEntry` is called with a live transaction, it registers an `IndexTask` of type `IndexOpUndoInsert`
on the abort stack — if the transaction later aborts, the inserted entry is removed. When `DeleteEntry` is called with a
live transaction, it registers an `IndexTask` of type `IndexOpDelete` on the commit stack instead of modifying the index
immediately. At commit time, `TransactionManager.Commit` drains the commit stack. At abort time, `Abort` drains the abort
stack in reverse, undoing any inserts the transaction made. This mechanism is already implemented for you in the index
and transaction layers, but it is your responsibility to order it correctly with the rest of your commit/abort logic in
part 4 and handle any stale reads that may arise from lazy deletes.

**Implementation Task:**
Add the appropriate hooks to each method in `TableHeap` and when flushing dirty pages in the buffer pool. Additionally,
modify your index operators to handle stale reads arising from lazy deletes. If you chose to implement IndexNestedLoopJoin in lab 2,
you should also update it and uncomment the relevant tests to ensure it is correct, although we will not test this on the
autograder.

**Tests:**
Run `go test -v ./execution -run TxnHeap` to verify that `TableHeap` correctly acquires locks (IS/IX at the table level, S/X at the tuple level) and appends the appropriate log records on insert, update, and delete.
Run `go test -v ./execution -run "TestIndexLookup_|TestIndexScan_"` to verify that index operators acquire the correct locks and handle stale heap entries caused by lazy deletes.
Run `go test -v ./execution -run BufferPool_WAL` to verify that the buffer pool does not evict dirty pages before their corresponding log records have been flushed.

---

## Part 4: Transaction Manager

**Files:** `godb/transaction/transaction_manager.go`

Now that you have a functional `TransactionContext` (Part 2) and the appropriate locking hooks in `TableHeap` (Part 3),
you are ready to implement the `TransactionManager`. Recall that GoDB enforces the **Strong Strict Two-Phase Locking
(SS2PL)** protocol, which means that transactions accumulate locks but only release them when they commit or abort.
For correctness, a transaction cannot be considered "Committed" until its log record is flushed to disk. Releasing locks
before this flush completes can lead to "dirty reads" of uncommitted data if the system crashes immediately after.
Remember to properly recycle transaction contexts in `TransactionManager`.

#### A Note About Aborts

Abort is the most complex operation in the lifecycle. We must actively restore the database to its previous state before
the transaction began. This involves an "Undo Phase" where we iterate backward through the transaction's private
`logRecordBuffer`, applying the inverse of every modification it made. However, simply undoing pages is **unsafe**. The
buffer pool manager can flush a dirty page to disk at any time, and if the database crashes at this point, the subsequent
recovery pass cannot tell if the necessary undos have been applied. Blindly applying an undo is **not** safe in a database
system. For example, undoing a disk-based index insert may delete an index entry that was logically inserted later by a different
 transaction. For this reason, **every** change made to a database page must be paired with a write-ahead log record. This
is where compensation log records (CLRs) come in. When aborting transactions, make sure you generate CLR records and update
the page LSN before undoing.

**Implementation Task:**
In `godb/transaction/transaction_manager.go`, implement `Begin`, `Commit`, and `Abort`.

**Tests:**
Run `go test -v ./transaction -run TxnManager` to verify Begin, Commit (SS2PL flush ordering, group commit), and Abort (no-undo path) at the manager level.
Run `go test -v ./execution -run TxnManager` to verify full abort correctness: undo of inserts, deletes, and updates; WAL-before-data (CLR failure handling); and LIFO undo ordering.
Run `go test -v [-race] . -run "TestWAL_FlushStress|TestConcurrent_Bank|TestAbortStorm|TestConcurrent_YCSB"` to run the end-to-end stress tests covering WAL durability, concurrent transactions, abort storms, and YCSB workloads.

---

## Grading and Submission

### 1. Submission

Create a zip file containing your `godb` directory.

```bash
zip -r lab2_submission.zip . -x "*.git*"

```

Upload this zip file to Gradescope.

We reserve the right to re-execute tests after the deadline, as concurrency bugs are often non-deterministic. We also
reserve the right to run additional hidden tests on your code. It is your responsibility to ensure that your code can
reliably pass all tests under repeated runs and different system conditions.

### 2. Lab Write-up

You should expect to complete a short write-up in-class about the lab. To get full credit, you should be prepared to answer the following:

* Basic conceptual questions about the codebase and the programming task.
* Questions about your design decisions (e.g., locking strategy in BufferPool).
* Any challenges you faced, the amount of time you spent on the lab, and feedback on the lab for future semesters.

**Grading Breakdown:**

* **60%**: Passing public unit tests.
* **40%**: Manual grading of code quality, hidden tests, and write-up.

Good luck!
