package indexing

import (
	"fmt"

	"github.com/tidwall/btree"
	"mit.edu/dsg/godb/common"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

type btreeItem struct {
	key Key
	rid common.RecordID
}

// MemBTreeIndex is a B+-Tree based index implementation.
// It is a wrapper around github.com/tidwall/btree, specialized for database Keys and RecordIDs.
type MemBTreeIndex struct {
	tree     *btree.BTreeG[btreeItem]
	metadata *IndexMetadata
}

func NewMemBTreeIndex(schema *storage.RawTupleDesc, projectionList []int) *MemBTreeIndex {
	// less function defines the ordering of items in the BTree.
	// Primary order by Key, secondary order by RecordID (to support non-unique keys).
	less := func(a, b btreeItem) bool {
		cmp := a.key.Compare(b.key)
		if cmp != 0 {
			return cmp < 0
		}
		// Tie-breaker: RecordID ensures uniqueness for the Set
		if a.rid.PageID.Oid != b.rid.PageID.Oid {
			return a.rid.PageID.Oid < b.rid.PageID.Oid
		}
		if a.rid.PageID.PageNum != b.rid.PageID.PageNum {
			return a.rid.PageID.PageNum < b.rid.PageID.PageNum
		}
		return a.rid.Slot < b.rid.Slot
	}

	return &MemBTreeIndex{
		tree: btree.NewBTreeG(less),
		metadata: &IndexMetadata{
			KeySchema:      schema,
			ProjectionList: projectionList,
		},
	}
}

func (index *MemBTreeIndex) Metadata() *IndexMetadata {
	return index.metadata
}

func (index *MemBTreeIndex) InsertEntry(key Key, rid common.RecordID, txn *transaction.TransactionContext) error {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")

	// Defensive copy: The key relies on a byte slice that might change after this call.
	keyCopy := key.DeepCopy()

	item := btreeItem{key: keyCopy, rid: rid}
	index.tree.Set(item)

	if txn != nil {
		txn.AddAbortTask(transaction.IndexTask{
			Target: index,
			Type:   transaction.IndexOpUndoInsert,
			Key:    keyCopy.RawTuple,
			RID:    rid,
		})
	}

	return nil
}

func (index *MemBTreeIndex) DeleteEntry(key Key, rid common.RecordID, txn *transaction.TransactionContext) error {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")

	item := btreeItem{key: key, rid: rid}
	if txn == nil {
		// txn == nil: immediate deletion (called from Invoke/abort rollback or non-transactional path)
		index.tree.Delete(item)
		return nil
	}
	// Defer deletion to commit time while X-lock is still held.
	// This prevents concurrent readers from seeing a transiently-absent key (dirty read of index state).
	stored, found := index.tree.Get(item)
	if !found {
		return nil
	}
	txn.AddCommitTask(transaction.IndexTask{
		Target: index,
		Type:   transaction.IndexOpDelete,
		Key:    stored.key.RawTuple,
		RID:    rid,
	})

	return nil
}

func (index *MemBTreeIndex) ScanKey(key Key, output []common.RecordID, txn *transaction.TransactionContext) ([]common.RecordID, error) {
	common.Assert(key.schema == index.metadata.KeySchema, "Key schema mismatch")
	// Create a pivot to find the first entry with this key.
	// We use an empty RecordID for the pivot start.
	pivot := btreeItem{key: key, rid: common.RecordID{}}

	index.tree.Ascend(pivot, func(item btreeItem) bool {
		if !item.key.Equals(key) {
			return false // Stop iterating once the key changes
		}
		output = append(output, item.rid)
		return true // Continue
	})
	return output, nil
}

func (index *MemBTreeIndex) Scan(start Key, direction ScanDirection, txn *transaction.TransactionContext) (ScanIterator, error) {
	common.Assert(start.IsNil() || start.schema == index.metadata.KeySchema, "Key schema mismatch")
	// Use Copy-On-Write for a consistent snapshot iterator
	snapshot := index.tree.Copy()
	iter := snapshot.Iter()

	it := &MemBTreeIndexIterator{
		parent:    index,
		direction: direction,
		firstCall: true,
	}

	var zeroRID common.RecordID

	if direction == ScanDirectionForward {
		if !start.IsNil() {
			it.hasMore = iter.Seek(btreeItem{key: start, rid: zeroRID})
		} else {
			it.hasMore = iter.First()
		}
	} else {
		if start.IsNil() {
			it.hasMore = iter.Last()
		} else {
			found := iter.Seek(btreeItem{key: start, rid: zeroRID})
			if !found {
				// We went past the end, so everything in the tree is < start (or tree is empty)
				// Try positioning at Last.
				it.hasMore = iter.Last()
			} else {
				// We landed on a valid item >= start.
				// If strictly > start, we must step back to find <= start.
				// If == start, we are good.
				item := iter.Item()
				if item.key.Compare(start) > 0 {
					it.hasMore = iter.Prev()
				} else {
					// Exact match
					it.hasMore = true
				}
			}
		}
	}
	it.iter = iter
	return it, nil
}

// Invoke handles transaction rollback and commit callbacks.
func (index *MemBTreeIndex) Invoke(opType transaction.IndexOpType, key storage.RawTuple, rid common.RecordID) {
	switch opType {
	case transaction.IndexOpUndoInsert, transaction.IndexOpDelete:
		_ = index.DeleteEntry(index.metadata.AsKey(key), rid, nil)
	default:
		fmt.Printf("Unknown index op type: %d\n", opType)
	}
}

// MemBTreeIndexIterator implements ScanIterator for BTree range scans.
type MemBTreeIndexIterator struct {
	parent             *MemBTreeIndex
	iter               btree.IterG[btreeItem]
	direction          ScanDirection
	firstCall, hasMore bool
}

func (it *MemBTreeIndexIterator) Next() bool {
	if it.firstCall {
		it.firstCall = false
		return it.hasMore
	}

	if !it.hasMore {
		return false
	}

	if it.direction == ScanDirectionForward {
		it.hasMore = it.iter.Next()
	} else {
		it.hasMore = it.iter.Prev()
	}

	return it.hasMore
}

func (it *MemBTreeIndexIterator) Key() Key {
	return it.iter.Item().key
}

func (it *MemBTreeIndexIterator) Value() common.RecordID {
	return it.iter.Item().rid
}

func (it *MemBTreeIndexIterator) Error() error {
	return nil
}

func (it *MemBTreeIndexIterator) Close() error {
	it.iter.Release()
	return nil
}
