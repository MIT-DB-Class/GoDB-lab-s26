# 6.5830/6.5831 Lab 4: Logging & Recovery

**Assigned:** [Date]
**Due:** [Date]

## Introduction

In Labs 1–3, you built a complete transactional database engine — storage, execution, and concurrency control. However,
your implementation uses an in-memory write-ahead log that does not survive a process restart. In this lab, you will
complete the final piece: a durable Write-Ahead Log and the ARIES recovery protocol. These algorithms are used in
virtually every serious database system today. By the end of this lab, GoDB will survive crashes and correctly
recover to a consistent state on restart — making it a complete database system with full ACID support.

## Logistics

* **Files to Modify:**
  * `godb/storage/buffer_pool.go`
  * `godb/transaction/transaction_manager.go`
  * `godb/transaction/transaction_context.go`
  * `godb/storage/log_record.go`
  * `godb/logging/log_file_iterator.go`
  * `godb/logging/double_buffer_log.go`
  * `godb/recovery/background_flusher.go`
  * `godb/recovery/checkpoint_manager.go`
  * `godb/recovery/recovery_manager.go`
  * `godb/main.go`

---

## Part 1: Checksums and Log Record Validation

**File:** `godb/storage/log_record.go`

### Why Checksums?

Recall that the log is a series of records written to disk in sequence. Unlike pages in the buffer pool, log records are
not necessarily aligned to page boundaries nor flushed strictly page-at-a-time. When a database crashes mid-write, the
tail of the log file is unreliable: a single `write()` syscall spanning multiple disk sectors may be interrupted partway
through, leaving some sectors updated and others stale. This is known as a **torn write**. Without a way to detect this,
recovery might parse garbage as a legitimate record and silently apply corrupt before/after images to live data.

To detect torn writes, GoDB embeds a **CRC32 checksum** in every log record header. If the checksum does not match after
reading, the record is invalid: the iterator stops and surfaces an error. Note that a clean EOF (zero size field or
truncated file) and a checksum mismatch are both reasons to stop, but they produce different outcomes — a missing record
is a clean end-of-log, while a record with a bad checksum indicates corruption and should be reported as an error.
Every log record on disk has the following layout:

```
Offset  Size  Field
------  ----  -----
0       2     Total record size in bytes (little-endian uint16)
2       4     CRC32 checksum (little-endian uint32)
6       2     Record type (LogInsert, LogCommit, etc.)
8+      var   Type-specific payload (TxnID, RID, before/after images, ...)
```

The checksum covers every byte *after* the checksum field (from byte 6 through the end of the record), so it is
computed over the same region regardless of payload length.

### Implementation Tasks

In `WriteToLog(buffer []byte)`, add logic to compute the CRC32 checksum and write it into the header. Use
`crc32.ChecksumIEEE` from Go's standard `hash/crc32` package. Then implement
`AsVerifiedLogRecord(data []byte) (LogRecord, error)`. This function should attempt to parse a valid `LogRecord` from
the given byte slice and return `ErrCorruptedLogRecord` (predefined in `log_record.go`) for any form of invalidity.
Think carefully about what different conditions torn-writes can produce on log records.

**Tests:**
Run `go test -v ./storage -run TestAsVerifiedLogRecord_TornWrite` to verify that your implementation correctly identifies torn writes across all field boundaries.

---
## Part 2: The Log Manager

A `LogManager` is responsible for two things: write records to an append-only log file during normal operation, and
read them back sequentially during recovery. You will implement both. Because a transaction cannot be considered
committed until its log records are flushed to disk, commit throughput is ultimately bounded by `fsync` latency — on the
order of milliseconds. Systems that issue one fsync per commit are limited to a few hundred commits per second regardless
of CPU speed. The standard solution is to batch multiple commits into a single `fsync` to form a **group commit**. In
other words, the `LogManager` should buffer writes, and should trigger flushes when either its buffer is full or when
some tunable latency threshold is reached (e.g., 5ms). In GoDB, an LSN is the byte offset of a record's first byte
in the log file, assigned at append time — so LSNs increase monotonically. Note: Tests for this part will only work
correctly if you implement both parts of the log manager.

### Part 2a: The Double-Buffer Log Manager

**File:** `godb/logging/double_buffer_log.go`

#### Why Double Buffering?

The naive approach to implement log manager is to simply keep a fixed-size, mutex-provided buffer that is flushed when
full. This is correct but slow, as writes are blocked until the buffer is flushed. A better approach is to decouple
memory writes from disk writes by keeping two buffers (front and back). Threads always append to the front buffer, and 
the log manager atomically swaps the front and back buffer when flushing. Writers can continue to append to the front
buffer while the back buffer is being written to disk. The only time a writer blocks is when *both* buffers are
full — the active buffer is saturated and the flush buffer hasn't been written to disk yet. This usually signals that 
the system is underload, and the log manager should block callers  -- this naturally introduces **back pressure** by
slowing down the writer, allowing the system to catch up.

#### Implementation Tasks

The full `LogManager` interface — method signatures and contracts — is defined in `godb/storage/wal.go`. Read it before
starting. Two methods deserve particular attention:

- **`Append(record LogRecord) (LSN, error)`** — copies the record into the active buffer and returns its LSN. Must not
  block unless both buffers are full (back-pressure).
- **`WaitUntilFlushed(lsn LSN) error`** — blocks until the record at `lsn` (and all prior records) are durably on
  disk. You called this in the transaction manager with the commit record's LSN before returning success to the client in
  Lab 3. The buffer pool also uses this before writing a dirty page, enforcing the WAL-before-data invariant.

Implement `DoubleBufferLogManager` to satisfy this interface. Design your data structures carefully before writing code.
Think about what state is needed to coordinate two buffers, track which bytes have been flushed, and wake up threads
waiting for durability. You will need to coordinate appending threads with the background flush goroutine (Hint:
`sync.Cond` is the natural tool for this). The buffer capacity is defined by the `logBufferSize` constant in
`double_buffer_log.go`.

#### Shutdown

`Close()` must ensure all buffered data reaches disk before returning. Think carefully about the interaction between
the caller of `Close()` and the background goroutine: any bytes still in the active buffer must make it to disk,
and the goroutine must finish its current flush before it can exit cleanly. After `Close()` returns, the log file is
closed and subsequent `Append` calls should return an error.

### Part 2b: The Log File Iterator
**File:** `godb/logging/log_file_iterator.go`

#### Implementation Tasks
Implement `LogFileIterator` in `godb/logging/log_file_iterator.go`. You will need to use your
`AsVerifiedLogRecord(data []byte) (LogRecord, error)` implementation to detect torn writes and only return
valid records. Hint: use `bufio.Reader` to iterate over the given log file.

**Tests:**
Parts 2a and 2b depend on each other and are tested together.
Run `go test -v ./logging -run "TestLogFileIterator_Basic|TestLogFileIterator_FailureScenarios"` to verify the iterator correctly reads records and handles torn writes.
Run `go test -v [-race] ./logging -run "TestDoubleBuffer"` to verify the log manager's correctness and concurrency safety (buffer swaps, group commit, close-under-load).

---

## Part 3: ARIES Recovery

We will now move on to implement the ARIES recovery protocol and fuzzy checkpointing. Your implementation of ARIES in
GoDB will differ from the standard ARIES in several ways. Textbook ARIES threads a `prevLSN` pointer through each log
record, forming per-transaction chains that Undo walks backward to generate CLRs. GoDB does not use `prevLSN`. Instead,
each transaction maintains an in-memory `logRecordBuffer` (built in Lab 3). During the Redo scan, GoDB reconstructs this
buffer for each transaction by loading the log records. After Redo, the standard `TransactionManager.Abort` path handles
Undo with no second log scan. The trade-off: GoDB holds every active transaction's full undo log in RAM during
recovery — negligible for short OLTP transactions, but textbook ARIES is preferable for very long-lived transactions or
memory-constrained environments.

### Part 3a: Background Flusher
**File:** `godb/recovery/background_flusher.go`

Without periodic flushing, recovery must replay the log back to the oldest dirtied page — potentially the entire history.
GoDB bounds recovery replay by periodically flushing every page in the buffer pool to disk. Note that flushing is not
eviction — it writes the dirtied in-memory page to disk but leaves it cached. The WAL invariant is enforced inside
`FlushAllPages` — your task is solely the background goroutine lifecycle. Similar to the log manager, you will need to
maintain a background goroutine and reason about graceful shutdown.

#### Implementation Tasks
Implement the `BackgroundFlusher` struct and its `Start()`, `Stop()`, and background goroutine in `background_flusher.go`.

**Tests:**
Run `go test -v ./recovery -run TestBackgroundFlusher` to verify that the flusher fires periodically under concurrent writes and that `Stop()` returns promptly.

### Part 3b: Checkpoint

Periodically, the system writes a fuzzy checkpoint to disk. In ARIES, this generates two additional log records:

**`LogBeginCheckpoint`:** Written at the start of the checkpoint. Its LSN is eventually saved to a master record file 
after the checkpoint is complete so `Recover()` knows where to begin the Analysis scan.
**`LogEndCheckpoint`:** Written immediately after, embedding the dirty page table (DPT) and active transaction table (ATT)
snapshots. The checkpoint is complete when the end record's LSN is durable on disk.

As part of the checkpointing process, the system will compute a **truncation LSN** — the minimum of the begin LSN, all
DPT `recLSN` values, and all ATT start LSNs — which is the earliest point recovery must scan from. Any log records
written before this LSN can be safely discarded. Note that this is different from the begin LSN of the checkpoint,
which is the LSN of the `LogBeginCheckpoint` record and the starting point of the Analysis scan.

#### Implementation Tasks
Implement the full `CheckpointManager` in `recovery/checkpoint_manager.go`: the `Checkpoint()` method, and
the `Start()`, `Stop()`, and background goroutine (same lifecycle pattern as `BackgroundFlusher`).

You are responsible for designing the binary encoding of the checkpoint data embedded in `LogEndCheckpoint`.
The format must be consistent between your `Checkpoint()` (write side) and the deserialization in
`recovery_manager.go` (read side) — implement both together.

You will also need to complete the two stubs deferred from earlier labs: `GetDirtyPageTableSnapshot()` in
`godb/storage/buffer_pool.go` and `GetActiveTransactionsSnapshot()` in `godb/transaction/transaction_manager.go`.

**Hint — ATT snapshot:** Think carefully about what it means for a transaction to appear in the ATT snapshot when
its commit record's LSN is earlier than `beginCheckpointLSN`. What would Analysis conclude about that transaction?

**Hint — DPT snapshot:** Some races may cause `recoveryLSN` to be larger than actual first LSN to dirty the page. Think about
what might happen if you are not careful to avoid this race.

**Tests:**
Run `go test -v ./recovery -run TestCheckpointManager` to verify that checkpoint records are paired and fire at the expected rate.

### Part 3c: Recovery
**File:** `godb/recovery/recovery_manager.go`

Recall that textbook ARIES has three phases. Here's how they each look like mapped onto GoDB:
#### Analysis

Scan the log forward from the checkpoint LSN, rebuilding the DPT and ATT. When you encounter a `LogEndCheckpoint`, merge
its embedded tables into the running state — entries already observed in the forward scan are more recent and take precedence.
The analysis phase also computes the earliest LSN Redo must read from, and the LSN of the final record seen, where Redo stops.

#### Redo

Scan the log forward from `scanStart` to `lastRecord`. For each data-modifying record (Insert, Update, Delete, and their
CLR counterparts), fetch the affected page. If the page's on-disk LSN is less than the record's LSN, apply the operation
and update the page LSN. Note that redo should be applied unconditionally, even for uncommitted transactions.
Meanwhile, specific to GoDB, redo should reconstruct the in-memory log buffer for each active transaction. Think
carefully about how CLRs interact with undo log reconstruction — a CLR does not add to the undo log in the same way
a normal operation does.

#### Undo

By the end of Redo, each survivor's undo log replicates its crash-time state. Instead of performing Undo in standard
ARIES, you can call `transactionManager.Abort` on each surviving transaction to generate CLRs and apply undo with no
additional log scan.

#### Implementation Tasks
Implement the `Recover()` method in `recovery/recovery_manager.go`. You will likely need to implement the two stubs
deferred from earlier labs:
- **`RestartTransactionForRecovery(txnID)`** in `transaction_manager.go` — allocates a fresh `TransactionContext` for a
survivor, preserving its original `TransactionID`.
- **`BufferRecordForRecovery(record)`** in `transaction_context.go` — appends a record to the survivor's undo log, or for
a CLR pops the most recent entry (that operation was already compensated before the crash).

**Hint — transaction ID continuity:** After recovery, the database must be ready to accept new transactions. Think about
what happens if `Begin()` assigns a transaction ID that was already used before the crash. You will need to ensure the
`TransactionManager`'s ID counter is advanced past any ID seen in the WAL. Consider adding a helper method to
`TransactionManager` and calling it from `Recover()` after the analysis phase.

**Tests:**
Run `go test -v ./recovery -run TestRecovery_Basic` to verify all fundamental ARIES scenarios: redo committed, undo uncommitted, CLR redo, update redo+undo, idempotent recovery, and TID monotonicity after recovery.
Run `go test -v ./recovery -run TestRecovery_WithCheckpoint` to verify checkpoint-aware recovery: analysis starting from the latest checkpoint, empty-DPT fast path (no page reads), transactions spanning checkpoints, and correctness under multiple checkpoints.

### Part 3d: Wiring It Together

**File:** `godb/main.go`

Once all components are implemented, wire them into `main.go` (look for the `TODO` comments):

1. In `NewGoDB`, replace `NoopLogManager` with `DoubleBufferLogManager`. The log file lives at `filepath.Join(logDir, "wal.log")`. You will also need to add `"mit.edu/dsg/godb/logging"` to the import block.
2. In `NewGoDB`, replace `NewNoLogRecoveryManager` with `NewRecoveryManager`. `Recover()` is already called on the result.
3. After `Recover()` returns, start a `BackgroundFlusher` and a `CheckpointManager`.

**Tests:**
Run `go test -v [-race] . -run "TestRecovery_E2E"` to run the end-to-end crash-recovery stress tests covering bank transfers, abort storms, YCSB workloads, repeated crash-recovery cycles, delete-replace races, and crash-during-recovery scenarios.

---

Congratulations — you have built a complete database system from scratch!

---

## Grading and Submission

### 1. Submission

```bash
zip -r lab4_submission.zip . -x "*.git*"
```

Upload this zip file to Gradescope. We reserve the right to re-execute tests after the deadline, as concurrency
bugs are often non-deterministic, and to run additional hidden tests. Ensure your code reliably passes all tests
under repeated runs.

### 2. Lab Write-up

You should expect to complete a short write-up in-class about the lab. To get full credit, be prepared to answer:

* Basic ARIES questions: why three phases? Why redo uncommitted transactions? Why are CLRs necessary?
* How does GoDB's Undo differ from standard ARIES, and what is the trade-off?
* Questions about your design decisions (double-buffer swap protocol, async error handling, checkpoint encoding).
* Any challenges you faced, time spent, and feedback for future semesters.

**Grading Breakdown:**

* **60%**: Passing public unit tests.
* **40%**: Manual grading of code quality, hidden tests, and write-up.

Good luck!
