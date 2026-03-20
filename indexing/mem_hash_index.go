package indexing

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/puzpuzpuz/xsync/v3"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

// bucket holds the list of RecordIDs for a specific unique Key.
// Since xsync handles the key mapping, we don't need to store the Key here.
type bucket struct {
	sync.RWMutex
	key     storage.RawTuple
	rids    []common.RecordID
	removed bool
}

// MemHashIndex is a concurrent hash index using xsync.MapOf.
// It maps a string representation of the tuple (Key) to a bucket of RecordIDs.
type MemHashIndex struct {
	// m maps string(Key.RawTuple) -> *bucket
	m        *xsync.MapOf[string, *bucket]
	metadata *IndexMetadata
}

func NewMemHashIndex(schema *storage.RawTupleDesc, projectionList []int) *MemHashIndex {
	return &MemHashIndex{
		m: xsync.NewMapOf[string, *bucket](),
		metadata: &IndexMetadata{
			KeySchema:      schema,
			ProjectionList: projectionList,
		},
	}
}

func (index *MemHashIndex) Metadata() *IndexMetadata {
	return index.metadata
}

func (index *MemHashIndex) InsertEntry(key Key, rid common.RecordID, txn *transaction.TransactionContext) error {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")
	unsafeKey := unsafe.String(unsafe.SliceData(key.RawTuple), len(key.RawTuple))

	for {
		// Try to find existing bucket using unsafe key
		// This avoids allocating 'safeKey' if the entry exists.
		b, ok := index.m.Load(unsafeKey)

		// Bucket doesn't exist (or we missed it)
		if !ok {
			// Allocate the safe string because we might store it in the map.
			safeKey := string(key.RawTuple)

			newBucket := &bucket{
				// Safe because we use this as a read-only slice
				key:     unsafe.Slice(unsafe.StringData(safeKey), len(safeKey)),
				rids:    make([]common.RecordID, 0, 1),
				removed: false,
			}

			actual, _ := index.m.LoadOrStore(safeKey, newBucket)
			b = actual
		}
		b.Lock()
		// Race with a removal -- try again
		if b.removed {
			b.Unlock()
			continue
		}
		b.rids = append(b.rids, rid)
		b.Unlock()

		if txn != nil {
			txn.AddAbortTask(transaction.IndexTask{
				Target: index,
				Type:   transaction.IndexOpUndoInsert,
				Key:    b.key,
				RID:    rid,
			})
		}
		return nil
	}
}

func (index *MemHashIndex) DeleteEntry(key Key, rid common.RecordID, txn *transaction.TransactionContext) error {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")
	unsafeKey := unsafe.String(unsafe.SliceData(key.RawTuple), len(key.RawTuple))
	b, ok := index.m.Load(unsafeKey)
	if !ok {
		return nil
	}

	b.Lock()
	defer b.Unlock()

	if txn != nil {
		// Defer deletion to commit time while X-lock is still held.
		txn.AddCommitTask(transaction.IndexTask{
			Target: index,
			Type:   transaction.IndexOpDelete,
			Key:    b.key,
			RID:    rid,
		})
		return nil
	}

	for i, r := range b.rids {
		if r == rid {
			// Found it. Remove via swap-with-last
			lastIdx := len(b.rids) - 1
			b.rids[i] = b.rids[lastIdx]
			b.rids = b.rids[:lastIdx]

			if len(b.rids) == 0 {
				b.removed = true
				index.m.Delete(unsafeKey)
			}
		}
	}

	return nil
}

func (index *MemHashIndex) ScanKey(key Key, output []common.RecordID, txn *transaction.TransactionContext) ([]common.RecordID, error) {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")

	unsafeKey := unsafe.String(unsafe.SliceData(key.RawTuple), len(key.RawTuple))

	for {
		b, ok := index.m.Load(unsafeKey)
		if !ok {
			return output, nil
		}

		b.RLock()
		if b.removed {
			b.RUnlock()
			continue
		}

		// Append all RIDs in this bucket
		output = append(output, b.rids...)
		b.RUnlock()
		return output, nil
	}
}

func (index *MemHashIndex) Scan(start Key, direction ScanDirection, txn *transaction.TransactionContext) (ScanIterator, error) {
	return nil, fmt.Errorf("range scans not supported for hash indexes")
}

func (index *MemHashIndex) Invoke(opType transaction.IndexOpType, key storage.RawTuple, rid common.RecordID) {
	switch opType {
	case transaction.IndexOpUndoInsert, transaction.IndexOpDelete:
		_ = index.DeleteEntry(index.metadata.AsKey(key), rid, nil)
	default:
		fmt.Printf("Unknown index op type: %d\n", opType)
	}
}
