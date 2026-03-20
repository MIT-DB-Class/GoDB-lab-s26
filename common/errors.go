package common

import "fmt"

type GoDBErrorCode int

const (
	// DuplicateObjectError indicates an attempt to create a table or index
	// that already exists in the catalog.
	DuplicateObjectError GoDBErrorCode = iota
	// NoSuchObjectError indicates a request for a table or index that does
	// not exist in the catalog.
	NoSuchObjectError
	// DeadlockError is returned by the lock manager when it detects a cycle
	// in the waits-for graph, necessitating a transaction abort.
	DeadlockError
	// LogClosedError indicates an attempt to write to the Write-Ahead Log
	// after the logging subsystem has been shut down.
	LogClosedError
	// LockNotFoundError indicates an attempt to unlock without holding a lock
	LockNotFoundError
)

func (ec GoDBErrorCode) String() string {
	switch ec {
	case DuplicateObjectError:
		return "DuplicateObjectError"
	case NoSuchObjectError:
		return "NoSuchObjectError"
	case DeadlockError:
		return "DeadlockError"
	}
	return "unknown"
}

// GoDBError is the custom error type for the database engine.
// It wraps a specific GoDBErrorCode with a detailed message.
//
// By implementing the built-in 'error' interface, it integrates seamlessly
// with Go's error handling while providing enough metadata for the
// database kernel to make architectural decisions (like aborting a transaction).
type GoDBError struct {
	Code      GoDBErrorCode
	ErrString string
}

func (e GoDBError) Error() string {
	return fmt.Sprintf("err: %s; msg: %s", e.Code.String(), e.ErrString)
}
