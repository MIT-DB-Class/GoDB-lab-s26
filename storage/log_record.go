package storage

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"mit.edu/dsg/godb/common"
)

// LogRecord is the in-memory representation.
// fields are unexported to enforce immutability outside the logging package.
// Header Layout: Size (2) | Checksum (4) | type (2) | Type-dependent payload
// Type-dependent Payload Layout
// BeginCheckpoint: header-only
// BeginTransaction, Commit, Abort: txnID (8)
// Insert: txnID (8) | RID (12) | AfterImage (?)
// InsertCLR, Delete, DeleteCLR: txnID (8) | RID (12)
// Update: txnID (8) | RID (12) | AfterImage (?) | BeforeImage (?)
// UpdateCLR: txnID (8) | RID (12) | AfterImage (?)
// EndCheckpoint: CheckpointData (?)
type LogRecord struct {
	data []byte
}

const logRecordHeaderSize = 8

// Offsets for writing/reading
const (
	offsetSize           = 0
	offsetChecksum       = offsetSize + 2
	offsetType           = offsetChecksum + 4
	offsetTxnID          = offsetType + 2
	offsetRID            = offsetTxnID + 8
	offsetAfterImage     = offsetRID + common.RecordIDSize
	offsetCheckpointData = offsetType + 2
)

// IsNil returns true if the underlying log data is empty.
func (r LogRecord) IsNil() bool {
	return len(r.data) == 0
}

// Size returns the total size of the log record in bytes.
func (r LogRecord) Size() int {
	return len(r.data)
}

// TxnID returns the Transaction ID stored in this record.
func (r LogRecord) TxnID() common.TransactionID {
	return common.TransactionID(binary.LittleEndian.Uint64(r.data[offsetTxnID:]))
}

// RecordType returns the type identifier for this log record (e.g., LogCommit, LogUpdate).
func (r LogRecord) RecordType() LogRecordType {
	return LogRecordType(binary.LittleEndian.Uint16(r.data[offsetType:]))
}

// RID returns the RecordID associated with this log record if the log record type has this field.
func (r LogRecord) RID() common.RecordID {
	t := r.RecordType()
	common.Assert(t == LogInsertCLR || t == LogDeleteCLR || t == LogDelete || t == LogUpdate || t == LogUpdateCLR || t == LogInsert, "log type %s does not support RID()", t)
	var rid common.RecordID
	rid.LoadFrom(r.data[offsetRID:])
	return rid
}

// BeforeImage returns the tuple data representing the state before the operation if the log record type has this field.
// This is used for Undo operations.
func (r LogRecord) BeforeImage() RawTuple {
	t := r.RecordType()
	common.Assert(t == LogUpdate, "log type %s does not support BeforeImage()", t)
	size := (len(r.data) - offsetAfterImage) / 2
	return r.data[offsetAfterImage+size:]
}

// AfterImage returns the tuple data representing the state after the operation if the log record type has this field.
// This is used for Redo operations.
func (r LogRecord) AfterImage() RawTuple {
	t := r.RecordType()
	common.Assert(t == LogUpdate || t == LogUpdateCLR || t == LogInsert, "log type %s does not support AfterImage()", t)

	if t == LogUpdate {
		size := (len(r.data) - offsetAfterImage) / 2
		return r.data[offsetAfterImage : offsetAfterImage+size]
	}
	return r.data[offsetAfterImage:]
}

// IsCLR returns true if the record is a Compensation Log Record.
func (r LogRecord) IsCLR() bool {
	t := r.RecordType()
	return t == LogInsertCLR || t == LogUpdateCLR || t == LogDeleteCLR
}

// CheckpointData returns the payload of an EndCheckpoint record if the log record type has this field.
func (r LogRecord) CheckpointData() []byte {
	common.Assert(r.RecordType() == LogEndCheckpoint, "CheckpointData() can only be called on CheckpointEnd records")
	return r.data[offsetCheckpointData:]
}

// Equal reports whether r and other produce identical serialized bytes.
func (r LogRecord) Equal(other LogRecord) bool {
	if r.Size() != other.Size() {
		return false
	}
	rBuf := make([]byte, r.Size())
	r.WriteToLog(rBuf)
	otherBuf := make([]byte, other.Size())
	other.WriteToLog(otherBuf)
	return bytes.Equal(rBuf, otherBuf)
}

// WriteToLog serializes the record into the provided buffer and calculates the checksum.
// The buffer must be large enough to hold r.Size() bytes.
func (r LogRecord) WriteToLog(buffer []byte) {
	common.Assert(len(buffer) >= r.Size(), "buffer allocated must be large enough fo the record")
	copy(buffer, r.data)
	binary.LittleEndian.PutUint16(buffer[offsetSize:], uint16(r.Size()))
	// TODO: implement checksum here for lab 4
}

var ErrCorruptedLogRecord = fmt.Errorf("log record corrupted: checksum mismatch")

// AsVerifiedLogRecord parses a raw byte slice into a LogRecord and verifies its checksum.
// It returns an ErrCorruptedLogRecord if the data is too short or the checksum does not match.
func AsVerifiedLogRecord(data []byte) (LogRecord, error) {
	panic("unimplemented")
}

// AsLogRecord wraps a raw byte slice as a LogRecord without performing verification.
// Use this only when the data is known to be valid.
func AsLogRecord(buf []byte) LogRecord {
	return LogRecord{data: buf}
}

// CreateCopy creates a deep copy of the source LogRecord into the provided buffer.
func CreateCopy(buf []byte, src LogRecord) LogRecord {
	common.Assert(len(buf) >= src.Size(), "buffer too small")
	copy(buf, src.data)
	return LogRecord{data: buf[:src.Size()]}
}

// BeginTransactionRecordSize returns the size required for a BeginTransaction record.
func BeginTransactionRecordSize() int {
	return logRecordHeaderSize + 8
}

// NewBeginTransactionRecord initializes a LogBeginTransaction record in the provided buffer.
func NewBeginTransactionRecord(buf []byte, txnID common.TransactionID) LogRecord {
	size := BeginTransactionRecordSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogBeginTransaction))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	return r
}

// CommitRecordSize returns the size required for a Commit record.
func CommitRecordSize() int {
	return logRecordHeaderSize + 8
}

// NewCommitRecord initializes a LogCommit record in the provided buffer.
func NewCommitRecord(buf []byte, txnID common.TransactionID) LogRecord {
	size := CommitRecordSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogCommit))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	return r
}

// AbortRecordSize returns the size required for an Abort record.
func AbortRecordSize() int {
	return logRecordHeaderSize + 8
}

// NewAbortRecord initializes a LogAbort record in the provided buffer.
func NewAbortRecord(buf []byte, txnID common.TransactionID) LogRecord {
	size := AbortRecordSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogAbort))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	return r
}

// InsertRecordSize returns the size required for an Insert record given the row data.
func InsertRecordSize(row RawTuple) int {
	return logRecordHeaderSize + 8 + common.RecordIDSize + len(row)
}

// NewInsertRecord initializes a LogInsert record.
func NewInsertRecord(buf []byte, txnID common.TransactionID, rid common.RecordID, row RawTuple) LogRecord {
	size := InsertRecordSize(row)
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogInsert))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	rid.WriteTo(r.data[offsetRID:])
	copy(r.data[offsetAfterImage:], row)
	return r
}

// InsertCLRSize returns the size required for an Insert CLR (Compensation Log Record).
func InsertCLRSize() int {
	return logRecordHeaderSize + 8 + common.RecordIDSize
}

// NewInsertCLR initializes a LogInsertCLR to undo a previous Insert operation.
func NewInsertCLR(buf []byte, insertRecord LogRecord) LogRecord {
	size := InsertCLRSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogInsertCLR))
	copy(r.data[offsetTxnID:], insertRecord.data[offsetTxnID:offsetAfterImage])
	return r
}

// DeleteRecordSize returns the size required for a Delete record.
func DeleteRecordSize() int {
	return logRecordHeaderSize + 8 + common.RecordIDSize
}

// NewDeleteRecord initializes a LogDelete record.
func NewDeleteRecord(buf []byte, txnID common.TransactionID, rid common.RecordID) LogRecord {
	size := DeleteRecordSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogDelete))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	rid.WriteTo(r.data[offsetRID:])
	return r
}

// DeleteCLRSize returns the size required for a Delete CLR.
func DeleteCLRSize() int {
	return logRecordHeaderSize + 8 + common.RecordIDSize
}

// NewDeleteCLR initializes a LogDeleteCLR to undo a previous Delete operation.
func NewDeleteCLR(buf []byte, deleteRecord LogRecord) LogRecord {
	size := DeleteCLRSize()
	r := LogRecord{data: buf[:size]}
	copy(r.data, deleteRecord.data)
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogDeleteCLR))
	return r
}

// UpdateRecordSize returns the size required for an Update record.
func UpdateRecordSize(before, after RawTuple) int {
	common.Assert(len(before) == len(after), "before and after tuples must be the same size in the current implementation")
	return logRecordHeaderSize + 8 + common.RecordIDSize + len(before) + len(after)
}

// NewUpdateRecord initializes a LogUpdate record.
// It writes `tupleToUpdate` as the BeforeImage (for Undo).
// The AfterImage slot is left empty and must be populated by the caller.
func NewUpdateRecord(buf []byte, txnID common.TransactionID, rid common.RecordID, before, after RawTuple) LogRecord {
	size := UpdateRecordSize(before, after)
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogUpdate))
	binary.LittleEndian.PutUint64(r.data[offsetTxnID:], uint64(txnID))
	rid.WriteTo(r.data[offsetRID:])
	copy(r.data[offsetAfterImage:], after)
	copy(r.data[offsetAfterImage+len(after):], before)
	return r
}

// UpdateCLRSize returns the size required for an Update CLR.
func UpdateCLRSize(updateRecord LogRecord) int {
	beforeImage := updateRecord.BeforeImage()
	return logRecordHeaderSize + 8 + common.RecordIDSize + len(beforeImage)
}

// NewUpdateCLR initializes a LogUpdateCLR to undo a previous Update operation.
// It sets the BeforeImage of the original update as the AfterImage of the CLR.
func NewUpdateCLR(buf []byte, updateRecord LogRecord) LogRecord {
	beforeImage := updateRecord.BeforeImage()
	size := UpdateCLRSize(updateRecord)
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogUpdateCLR))
	copy(r.data[offsetTxnID:], updateRecord.data[offsetTxnID:offsetAfterImage])
	copy(r.data[offsetAfterImage:], beforeImage)
	return r
}

// BeginCheckpointRecordSize returns the size required for a BeginCheckpoint record.
func BeginCheckpointRecordSize() int {
	return logRecordHeaderSize
}

// NewBeginCheckpointRecord initializes a LogBeginCheckpoint record.
func NewBeginCheckpointRecord(buf []byte) LogRecord {
	size := BeginCheckpointRecordSize()
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogBeginCheckpoint))
	return r
}

// EndCheckpointRecordSize returns the size required for an EndCheckpoint record.
func EndCheckpointRecordSize(payloadSize int) int {
	return logRecordHeaderSize + payloadSize
}

// NewEndCheckpointRecord initializes a LogEndCheckpoint record.
// The caller is responsible for filling in the checkpoint data payload.
func NewEndCheckpointRecord(buf []byte, payloadSize int) LogRecord {
	size := EndCheckpointRecordSize(payloadSize)
	r := LogRecord{data: buf[:size]}
	binary.LittleEndian.PutUint16(r.data[offsetType:], uint16(LogEndCheckpoint))
	return r
}
