package transaction

import (
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/storage"
)

// IndexCallback is the interface implemented by indexes to support in-memory undo/commit
// operations. It covers both abort-time rollback (undo insert/delete) and commit-time deferred
// work (lazy delete). It's necessary because GoDB has in-memory indexes that do not participate
// in ARIES WAL recovery.
//
// YOU SHOULD NOT NEED TO IMPLEMENT OR MODIFY THIS INTERFACE.
type IndexCallback interface {
	Invoke(opType IndexOpType, key storage.RawTuple, rid common.RecordID)
}

type IndexOpType int

const (
	IndexOpUndoInsert IndexOpType = iota // abort: remove a key that was inserted
	IndexOpDelete                        // commit: perform a deferred deletion
)

// IndexTask represents a single index operation to execute at transaction end (abort or commit).
// It is a value struct (not a pointer) to avoid heap allocation per op.
//
// YOU SHOULD NOT NEED TO MANIPULATE THIS STRUCT.
type IndexTask struct {
	Target IndexCallback // The Index instance (as an interface)
	Type   IndexOpType
	Key    storage.RawTuple
	RID    common.RecordID
}
