package storage

import (
	"github.com/your-org/haxen/control-plane/pkg/types"
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// UnitOfWork manages a collection of changes as a single transaction
type UnitOfWork interface {
	// Entity registration
	RegisterNew(entity interface{}, table string, operation func(DBTX) error)
	RegisterDirty(entity interface{}, table string, operation func(DBTX) error)
	RegisterDeleted(id interface{}, table string, operation func(DBTX) error)

	// Transaction management
	Commit() error
	Rollback() error

	// State inspection
	HasChanges() bool
	GetChangeCount() int
	IsActive() bool
}

// WorkflowUnitOfWork provides specialized operations for workflow management
type WorkflowUnitOfWork interface {
	UnitOfWork

	// High-level workflow operations
	StoreWorkflowWithExecution(ctx context.Context, workflow *types.Workflow, execution *types.WorkflowExecution) error
	UpdateWorkflowStatus(ctx context.Context, workflowID string, status string, execution *types.WorkflowExecution) error
	CompleteWorkflowWithResults(workflowID string, results map[string]interface{}) error
	StoreWorkflowWithSession(ctx context.Context, workflow *types.Workflow, session *types.Session) error
}

// ChangeType represents the type of change being tracked
type ChangeType int

const (
	ChangeTypeNew ChangeType = iota
	ChangeTypeDirty
	ChangeTypeDeleted
)

// Change represents a single database operation within the unit of work
type Change struct {
	Entity    interface{}
	Table     string
	Type      ChangeType
	Operation func(DBTX) error
	Timestamp time.Time
}

// unitOfWorkImpl implements the UnitOfWork interface
type unitOfWorkBackend interface {
	executeWorkflowInsertWithTx(ctx context.Context, tx DBTX, workflow *types.Workflow) error
	executeWorkflowExecutionInsertWithTx(ctx context.Context, tx DBTX, execution *types.WorkflowExecution) error
	executeSessionInsertWithTx(ctx context.Context, tx DBTX, session *types.Session) error
}

type unitOfWorkImpl struct {
	db      *sqlDatabase
	tx      *sqlTx
	changes []Change
	mu      sync.RWMutex
	active  bool
	backend unitOfWorkBackend
}

// workflowUnitOfWorkImpl implements the WorkflowUnitOfWork interface
type workflowUnitOfWorkImpl struct {
	*unitOfWorkImpl
}

// NewUnitOfWork creates a new unit of work instance
func NewUnitOfWork(db *sqlDatabase, backend unitOfWorkBackend) UnitOfWork {
	return &unitOfWorkImpl{
		db:      db,
		changes: make([]Change, 0),
		active:  true,
		backend: backend,
	}
}

// NewWorkflowUnitOfWork creates a new workflow-specific unit of work instance
func NewWorkflowUnitOfWork(db *sqlDatabase, backend unitOfWorkBackend) WorkflowUnitOfWork {
	baseUoW := &unitOfWorkImpl{
		db:      db,
		changes: make([]Change, 0),
		active:  true,
		backend: backend,
	}
	return &workflowUnitOfWorkImpl{
		unitOfWorkImpl: baseUoW,
	}
}

// RegisterNew registers a new entity to be inserted
func (uow *unitOfWorkImpl) RegisterNew(entity interface{}, table string, operation func(DBTX) error) {
	uow.mu.Lock()
	defer uow.mu.Unlock()

	if !uow.active {
		return
	}

	change := Change{
		Entity:    entity,
		Table:     table,
		Type:      ChangeTypeNew,
		Operation: operation,
		Timestamp: time.Now(),
	}

	uow.changes = append(uow.changes, change)
}

// RegisterDirty registers an existing entity to be updated
func (uow *unitOfWorkImpl) RegisterDirty(entity interface{}, table string, operation func(DBTX) error) {
	uow.mu.Lock()
	defer uow.mu.Unlock()

	if !uow.active {
		return
	}

	change := Change{
		Entity:    entity,
		Table:     table,
		Type:      ChangeTypeDirty,
		Operation: operation,
		Timestamp: time.Now(),
	}

	uow.changes = append(uow.changes, change)
}

// RegisterDeleted registers an entity to be deleted
func (uow *unitOfWorkImpl) RegisterDeleted(id interface{}, table string, operation func(DBTX) error) {
	uow.mu.Lock()
	defer uow.mu.Unlock()

	if !uow.active {
		return
	}

	change := Change{
		Entity:    id,
		Table:     table,
		Type:      ChangeTypeDeleted,
		Operation: operation,
		Timestamp: time.Now(),
	}

	uow.changes = append(uow.changes, change)
}

// Commit executes all registered changes in a single transaction
func (uow *unitOfWorkImpl) Commit() error {
	uow.mu.Lock()
	defer uow.mu.Unlock()

	if !uow.active {
		return fmt.Errorf("unit of work is not active")
	}

	// Always mark as inactive when commit is called, regardless of success/failure
	defer func() {
		uow.active = false
		uow.changes = nil
		uow.tx = nil
	}()

	if len(uow.changes) == 0 {
		return nil
	}

	// Implement retry logic for transient database errors
	const maxRetries = 3
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if err := uow.executeCommit(); err != nil {
			lastErr = err
			if attempt < maxRetries-1 && uow.isRetryableError(err) {
				time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
				continue
			}
			return lastErr
		}
		return nil
	}

	return lastErr
}

// executeCommit performs the actual transaction commit
func (uow *unitOfWorkImpl) executeCommit() error {
	// Begin transaction only if we don't already have one
	if uow.tx == nil {
		tx, err := uow.db.Begin()
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %w", err)
		}
		uow.tx = tx
	}

	defer func() {
		if uow.tx != nil {
			uow.tx.Rollback() // Will be no-op if tx.Commit() succeeds
		}
	}()

	// Execute all changes in order
	for i, change := range uow.changes {
		if err := change.Operation(uow.tx); err != nil {
			return fmt.Errorf("failed to execute change %d for table %s: %w", i, change.Table, err)
		}
	}

	// Commit transaction
	if err := uow.tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	// Clear the transaction reference after successful commit
	uow.tx = nil

	return nil
}

// isRetryableError determines if a database error is retryable
func (uow *unitOfWorkImpl) isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	// Common retryable SQLite errors
	retryableErrors := []string{
		"database is locked",
		"database disk image is malformed",
		"disk i/o error",
		"attempt to write a readonly database",
		"busy",
		"sqlite_busy",
		"sqlite_locked",
		"cannot start a transaction within a transaction",
		"database table is locked",
	}

	for _, retryable := range retryableErrors {
		if strings.Contains(errStr, retryable) {
			return true
		}
	}
	return false
}

// Rollback cancels all registered changes
func (uow *unitOfWorkImpl) Rollback() error {
	uow.mu.Lock()
	defer uow.mu.Unlock()

	if uow.tx != nil {
		if err := uow.tx.Rollback(); err != nil {
			return fmt.Errorf("failed to rollback transaction: %w", err)
		}
		uow.tx = nil
	}

	// Clear changes and mark as inactive
	uow.changes = nil
	uow.active = false

	return nil
}

// HasChanges returns true if there are pending changes
func (uow *unitOfWorkImpl) HasChanges() bool {
	uow.mu.RLock()
	defer uow.mu.RUnlock()
	return len(uow.changes) > 0
}

// GetChangeCount returns the number of pending changes
func (uow *unitOfWorkImpl) GetChangeCount() int {
	uow.mu.RLock()
	defer uow.mu.RUnlock()
	return len(uow.changes)
}

// IsActive returns true if the unit of work is still active
func (uow *unitOfWorkImpl) IsActive() bool {
	uow.mu.RLock()
	defer uow.mu.RUnlock()
	return uow.active
}

// StoreWorkflowWithExecution stores a workflow and its execution atomically
func (wuow *workflowUnitOfWorkImpl) StoreWorkflowWithExecution(ctx context.Context, workflow *types.Workflow, execution *types.WorkflowExecution) error {
	if !wuow.active {
		return fmt.Errorf("workflow unit of work is not active")
	}

	// Register workflow operation
	workflowOp := func(tx DBTX) error {
		return wuow.backend.executeWorkflowInsertWithTx(ctx, tx, workflow)
	}
	wuow.RegisterNew(workflow, "workflows", workflowOp)

	// Register execution operation
	executionOp := func(tx DBTX) error {
		return wuow.backend.executeWorkflowExecutionInsertWithTx(ctx, tx, execution)
	}
	wuow.RegisterNew(execution, "workflow_executions", executionOp)

	return nil
}

// UpdateWorkflowStatus updates workflow status and stores execution atomically
func (wuow *workflowUnitOfWorkImpl) UpdateWorkflowStatus(ctx context.Context, workflowID string, status string, execution *types.WorkflowExecution) error {
	if !wuow.active {
		return fmt.Errorf("workflow unit of work is not active")
	}

	// Register workflow status update
	statusOp := func(tx DBTX) error {
		query := `UPDATE workflows SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE workflow_id = ?`
		_, err := tx.Exec(query, status, workflowID)
		return err
	}
	wuow.RegisterDirty(workflowID, "workflows", statusOp)

	// Register execution if provided
	if execution != nil {
		executionOp := func(tx DBTX) error {
			return wuow.backend.executeWorkflowExecutionInsertWithTx(ctx, tx, execution)
		}
		wuow.RegisterNew(execution, "workflow_executions", executionOp)
	}

	return nil
}

// CompleteWorkflowWithResults completes a workflow with results atomically
func (wuow *workflowUnitOfWorkImpl) CompleteWorkflowWithResults(workflowID string, results map[string]interface{}) error {
	if !wuow.active {
		return fmt.Errorf("workflow unit of work is not active")
	}

	// Register workflow completion
	completeOp := func(tx DBTX) error {
		query := `UPDATE workflows SET status = 'completed', completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE workflow_id = ?`
		_, err := tx.Exec(query, workflowID)
		return err
	}
	wuow.RegisterDirty(workflowID, "workflows", completeOp)

	// Register results storage if provided
	if results != nil {
		resultsOp := func(tx DBTX) error {
			// Store results in a results table or as metadata
			// Implementation depends on your schema
			return nil // Placeholder
		}
		wuow.RegisterNew(results, "workflow_results", resultsOp)
	}

	return nil
}

// StoreWorkflowWithSession stores a workflow and session atomically
func (wuow *workflowUnitOfWorkImpl) StoreWorkflowWithSession(ctx context.Context, workflow *types.Workflow, session *types.Session) error {
	if !wuow.active {
		return fmt.Errorf("workflow unit of work is not active")
	}

	// Register workflow operation
	workflowOp := func(tx DBTX) error {
		return wuow.backend.executeWorkflowInsertWithTx(ctx, tx, workflow)
	}
	wuow.RegisterNew(workflow, "workflows", workflowOp)

	// Register session operation
	sessionOp := func(tx DBTX) error {
		return wuow.backend.executeSessionInsertWithTx(ctx, tx, session)
	}
	wuow.RegisterNew(session, "sessions", sessionOp)

	return nil
}
