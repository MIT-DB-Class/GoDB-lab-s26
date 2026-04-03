package recovery

import (
	"mit.edu/dsg/godb/catalog"
	"mit.edu/dsg/godb/execution"
	"mit.edu/dsg/godb/indexing"
	"mit.edu/dsg/godb/storage"
	"mit.edu/dsg/godb/transaction"
)

/*
NoLogRecoveryManager is an MVP implementation that rebuilds indexes from scratch
on recovery but does not perform ARIES on WAL. This allows us to run the database
idiomatically without worry about recovery until WAL and ARIES are fully implemented.
*/
type NoLogRecoveryManager struct {
	bufferPool         *storage.BufferPool
	transactionManager *transaction.TransactionManager
	catalog            *catalog.Catalog
	indexManager       *indexing.IndexManager
	tableManager       *execution.TableManager
}

// NewNoLogRecoveryManager initializes a new NoLogRecoveryManager.
func NewNoLogRecoveryManager(
	bufferPool *storage.BufferPool,
	transactionManager *transaction.TransactionManager,
	catalog *catalog.Catalog,
	tableManager *execution.TableManager,
	indexManager *indexing.IndexManager) *NoLogRecoveryManager {
	return &NoLogRecoveryManager{
		bufferPool:         bufferPool,
		transactionManager: transactionManager,
		catalog:            catalog,
		tableManager:       tableManager,
		indexManager:       indexManager,
	}
}

func rebuildIndexes(
	catalog *catalog.Catalog,
	tableManager *execution.TableManager,
	indexManager *indexing.IndexManager) error {

	for _, tableDef := range catalog.Tables {
		// Skip tables without indexes to avoid unnecessary scans
		if len(tableDef.Indexes) == 0 {
			continue
		}

		heap, err := tableManager.GetTable(tableDef.Oid)
		if err != nil {
			return err
		}

		// Resolve all index objects for this table
		var activeIndexes []indexing.Index
		var keyBuffers [][]byte

		for _, i := range tableDef.Indexes {
			indexObject, err := indexManager.GetIndex(i.Oid)
			if err != nil {
				return err
			}
			activeIndexes = append(activeIndexes, indexObject)
			keyBuffers = append(keyBuffers, make([]byte, indexObject.Metadata().KeySchema.BytesPerTuple()))
		}

		// Scan the table to rebuild indexes.
		// We use a nil transaction because we are in recovery mode (single-threaded).
		iter, err := heap.Iterator(nil, transaction.LockModeS, make([]byte, heap.StorageSchema().BytesPerTuple()))
		if err != nil {
			return err
		}

		for iter.Next() {
			tupleBytes := iter.CurrentTuple()
			rid := iter.CurrentRID()

			for i, index := range activeIndexes {
				// Project the values from the main tuple into the key tuple
				for k, colIdx := range index.Metadata().ProjectionList {
					// Extract value from the heap tuple (using table schema)
					val := heap.StorageSchema().GetValue(tupleBytes, colIdx)
					// Write value to the key buffer (using key schema)
					index.Metadata().KeySchema.SetValue(keyBuffers[i], k, val)
				}

				key := index.Metadata().AsKey(keyBuffers[i])
				if err := index.InsertEntry(key, rid, nil); err != nil {
					_ = iter.Close()
					return err
				}
			}
		}
		_ = iter.Close()
		if err := iter.Error(); err != nil {
			return err
		}
	}
	return nil
}

// Recovery only rebuilds indexes.
func (rm *NoLogRecoveryManager) Recover() error {
	return rebuildIndexes(rm.catalog, rm.tableManager, rm.indexManager)
}
