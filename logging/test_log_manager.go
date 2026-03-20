package logging

import (
	"context"
	"encoding/binary"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/storage"
)

// MemoryLogManager is an optimized in-memory implementation of LogManager for testing.
// It uses a single flat byte slice to store records contiguously at byte-offset LSNs.
type MemoryLogManager struct {
	buffer        []byte
	offsets       []int
	flushedUntil  atomic.Int64
	flushOnAppend atomic.Bool

	// appendCountdown and appendCountdownErr are protected by Mutex.
	// When appendCountdown >= 0, each Append decrements it; when it reaches 0 the
	// next call returns appendCountdownErr. -1 means the feature is disabled.
	appendCountdown    int
	appendCountdownErr error

	sync.Mutex
}

func NewMemoryLogManager() *MemoryLogManager {
	return &MemoryLogManager{
		// Start with zero length so records begin at LSN 0. Cap is pre-allocated
		// to avoid the first few resizes.
		buffer:          make([]byte, 0, 4096),
		offsets:         make([]int, 0, 128),
		appendCountdown: -1, // disabled
	}
}

func (m *MemoryLogManager) Append(record storage.LogRecord) (storage.LSN, error) {
	m.Lock()
	defer m.Unlock()

	if m.appendCountdown >= 0 {
		if m.appendCountdown == 0 {
			return 0, m.appendCountdownErr
		}
		m.appendCountdown--
	}

	lsn := len(m.buffer)
	if cap(m.buffer)-lsn < record.Size() {
		// Allocate with len=lsn so existing data is preserved and the slice length
		// correctly tracks the end of written records after we extend below.
		newBuf := make([]byte, lsn, max(lsn+record.Size(), 2*cap(m.buffer)))
		copy(newBuf, m.buffer)
		m.buffer = newBuf
	}
	// Extend the slice to cover the new record, then write.
	m.buffer = m.buffer[:lsn+record.Size()]
	record.WriteToLog(m.buffer[lsn:])

	if cap(m.offsets) == len(m.offsets) {
		newOffsets := make([]int, len(m.offsets), len(m.offsets)*2)
		copy(newOffsets, m.offsets)
		m.offsets = newOffsets
	}
	m.offsets = append(m.offsets, lsn)

	if m.flushOnAppend.Load() {
		m.flushedUntil.Store(int64(lsn))
	}
	return storage.LSN(lsn), nil
}

func (m *MemoryLogManager) WaitUntilFlushed(lsn storage.LSN) error {
	for m.flushedUntil.Load() < int64(lsn) {
		time.Sleep(time.Millisecond)
	}
	return nil
}

// Iterator returns a scanner to walk the log from a specific starting point.
func (m *MemoryLogManager) Iterator(startLSN storage.LSN) (storage.LogIterator, error) {
	return &MemoryLogIterator{
		mgr:        m,
		currOffset: int(startLSN),
	}, nil
}

func (m *MemoryLogManager) FlushedUntil() storage.LSN {
	return storage.LSN(m.flushedUntil.Load())
}

func (m *MemoryLogManager) Close() error {
	return nil
}

func (m *MemoryLogManager) SetFlushedLSN(lsn storage.LSN) {
	for {
		cur := m.flushedUntil.Load()
		if int64(lsn) <= cur {
			return
		}
		if m.flushedUntil.CompareAndSwap(cur, int64(lsn)) {
			return
		}
	}
}

// Tail returns the byte offset immediately after the last appended record,
// i.e., the position at which the next Append would begin.
func (m *MemoryLogManager) Tail() storage.LSN {
	m.Lock()
	defer m.Unlock()
	return storage.LSN(len(m.buffer))
}

// SetAppendErrorAfterN schedules an error injection: the next n Append calls succeed
// normally, and the (n+1)th call returns err. n=0 makes the very next Append fail.
// This is useful for testing partial-failure scenarios such as CLR succeeds but
// LogAbort fails.
func (m *MemoryLogManager) SetAppendErrorAfterN(n int, err error) {
	m.Lock()
	defer m.Unlock()
	m.appendCountdown = n
	m.appendCountdownErr = err
}

// SetFlushOnAppend controls whether each Append automatically advances flushedUntil
// to the LSN of the appended record, making WaitUntilFlushed return immediately.
// Enable this in tests that exercise TransactionManager.Commit() but do not need to
// test WAL durability ordering.
func (m *MemoryLogManager) SetFlushOnAppend(v bool) {
	m.flushOnAppend.Store(v)
}

// Count returns the number of log records currently stored.
func (m *MemoryLogManager) Count() int {
	m.Lock()
	defer m.Unlock()
	return len(m.offsets)
}

// GetRecord returns the i-th log record (0-based)
func (m *MemoryLogManager) GetRecord(i int) storage.LogRecord {
	m.Lock()
	defer m.Unlock()
	buf := m.buffer[m.offsets[i]:]
	recordLen := int(binary.LittleEndian.Uint16(buf))
	return storage.AsLogRecord(buf[:recordLen])
}

type MemoryLogIterator struct {
	mgr        *MemoryLogManager
	currOffset int
	current    storage.LogRecord
}

func (i *MemoryLogIterator) Next() bool {
	if !i.current.IsNil() {
		i.currOffset += i.current.Size()
	}
	if i.currOffset >= len(i.mgr.buffer) {
		return false
	}
	buf := i.mgr.buffer[i.currOffset:]
	recordLen := int(binary.LittleEndian.Uint16(buf))
	i.current = storage.AsLogRecord(buf[:recordLen])
	return true
}

func (i *MemoryLogIterator) CurrentRecord() storage.LogRecord {
	return i.current
}

func (i *MemoryLogIterator) CurrentLSN() storage.LSN {
	return storage.LSN(i.currOffset)
}

func (i *MemoryLogIterator) Error() error {
	return nil
}

func (i *MemoryLogIterator) Close() error {
	return nil
}

// PeriodicWALFlusher advances flushedUntil to the actual WAL tail every 5ms,
// simulating a background WAL I/O thread. This creates realistic windows between
// flushes where a correct buffer pool must block in WaitUntilFlushed before writing
// dirty pages, and a buggy one would be caught by WALCheckingDBFileManager.
// Cancel ctx to stop the goroutine.
func PeriodicWALFlusher(ctx context.Context, mlm *MemoryLogManager) {
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mlm.SetFlushedLSN(mlm.Tail())
		}
	}
}

// VerifyWALOrdering scans the WAL and verifies the per-transaction record
// structure using a state machine:
//
//   - Every transaction has exactly one Begin followed by exactly one End
//     (Commit or Abort).
//   - Data records (Insert/Update/Delete) appear only between Begin and End.
//   - CLRs appear only in aborted transactions, between Begin and Abort.
//   - Every aborted transaction's CLR count equals its data record count
//     (one compensation record per original modification).
func VerifyWALOrdering(t *testing.T, mlm *MemoryLogManager) {
	t.Helper()

	type txnState struct {
		hasBegin  bool
		hasEnd    bool
		aborted   bool
		dataCount int
		clrCount  int
	}
	states := make(map[common.TransactionID]*txnState)
	stateOf := func(id common.TransactionID) *txnState {
		if s, ok := states[id]; ok {
			return s
		}
		s := &txnState{}
		states[id] = s
		return s
	}

	iter, err := mlm.Iterator(0)
	require.NoError(t, err)
	defer iter.Close()

	for iter.Next() {
		rec := iter.CurrentRecord()
		id := rec.TxnID()
		s := stateOf(id)
		switch rec.RecordType() {
		case storage.LogBeginTransaction:
			assert.False(t, s.hasBegin, "txn %d: duplicate Begin record", id)
			s.hasBegin = true
		case storage.LogCommit:
			assert.True(t, s.hasBegin, "txn %d: Commit without Begin", id)
			assert.False(t, s.hasEnd, "txn %d: duplicate End record", id)
			s.hasEnd = true
		case storage.LogAbort:
			assert.True(t, s.hasBegin, "txn %d: Abort without Begin", id)
			assert.False(t, s.hasEnd, "txn %d: duplicate End record", id)
			s.hasEnd = true
			s.aborted = true
		case storage.LogInsert, storage.LogUpdate, storage.LogDelete:
			assert.True(t, s.hasBegin, "txn %d: data record before Begin", id)
			assert.False(t, s.hasEnd, "txn %d: data record after End", id)
			s.dataCount++
		case storage.LogInsertCLR, storage.LogUpdateCLR, storage.LogDeleteCLR:
			assert.True(t, s.hasBegin, "txn %d: CLR before Begin", id)
			assert.False(t, s.hasEnd, "txn %d: CLR after End", id)
			s.clrCount++
		}
	}

	for id, s := range states {
		assert.True(t, s.hasBegin, "txn %d: missing Begin", id)
		assert.True(t, s.hasEnd, "txn %d: missing End (neither Commit nor Abort)", id)
		if s.aborted {
			assert.Equal(t, s.dataCount, s.clrCount,
				"txn %d: aborted txn has %d data records but %d CLRs",
				id, s.dataCount, s.clrCount)
		}
	}
}
