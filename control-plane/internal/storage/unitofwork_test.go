//go:build integration
// +build integration

package storage

import (
	"context" // Add context import
	"encoding/json"
	"github.com/your-org/agentfield/control-plane/pkg/types"
	"os"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func setupTestDB(t *testing.T) (*LocalStorage, func()) {
	// Create temporary database file
	dbPath := "/tmp/test_unitofwork.db"
	kvPath := "/tmp/test_unitofwork.bolt"

	// Remove existing files
	os.Remove(dbPath)
	os.Remove(kvPath)

	config := StorageConfig{
		Local: LocalStorageConfig{
			DatabasePath: dbPath,
			KVStorePath:  kvPath,
		},
	}

	storage := NewLocalStorage(config.Local)
	err := storage.Initialize(context.Background(), config) // Pass context
	if err != nil {
		t.Fatalf("Failed to initialize storage: %v", err)
	}

	cleanup := func() {
		storage.Close(context.Background()) // Pass context
		os.Remove(dbPath)
		os.Remove(kvPath)
	}

	return storage, cleanup
}

func TestUnitOfWork_BasicOperations(t *testing.T) {
	storage, cleanup := setupTestDB(t)
	defer cleanup()

	// Create a unit of work
	uow := storage.NewUnitOfWork()

	// Test initial state
	if !uow.IsActive() {
		t.Error("Unit of work should be active initially")
	}

	if uow.HasChanges() {
		t.Error("Unit of work should not have changes initially")
	}

	if uow.GetChangeCount() != 0 {
		t.Error("Unit of work should have 0 changes initially")
	}

	// Register a simple operation
	testOp := func(tx DBTX) error {
		_, err := tx.Exec("SELECT 1")
		return err
	}

	uow.RegisterNew("test_entity", "test_table", testOp)

	// Check state after registration
	if !uow.HasChanges() {
		t.Error("Unit of work should have changes after registration")
	}

	if uow.GetChangeCount() != 1 {
		t.Error("Unit of work should have 1 change after registration")
	}

	// Commit the unit of work
	err := uow.Commit()
	if err != nil {
		t.Errorf("Failed to commit unit of work: %v", err)
	}

	// Check state after commit
	if uow.IsActive() {
		t.Error("Unit of work should not be active after commit")
	}
}

func TestWorkflowUnitOfWork_AtomicWorkflowExecution(t *testing.T) {
	storage, cleanup := setupTestDB(t)
	defer cleanup()

	// Create test workflow and execution
	workflow := &types.Workflow{
		WorkflowID:           "test-workflow-001",
		WorkflowName:         ptrString("Test Workflow"),
		WorkflowTags:         []string{"test", "unit-test"},
		SessionID:            ptrString("test-session-001"),
		ActorID:              ptrString("test-actor-001"),
		ParentWorkflowID:     nil,
		RootWorkflowID:       ptrString("test-workflow-001"),
		WorkflowDepth:        0,
		TotalExecutions:      1,
		SuccessfulExecutions: 0,
		FailedExecutions:     0,
		TotalDurationMS:      0,
		Status:               "running",
		StartedAt:            time.Now(),
		CompletedAt:          nil,
		CreatedAt:            time.Now(),
		UpdatedAt:            time.Now(),
	}

	execution := &types.WorkflowExecution{
		WorkflowID:          "test-workflow-001",
		ExecutionID:         "test-execution-001",
		AgentFieldRequestID: "test-request-001",
		SessionID:           ptrString("test-session-001"),
		ActorID:             ptrString("test-actor-001"),
		AgentNodeID:         "test-agent-001",
		ParentWorkflowID:    nil,
		ParentExecutionID:   nil,
		RootWorkflowID:      ptrString("test-workflow-001"),
		WorkflowDepth:       0,
		ReasonerID:          "test-reasoner",
		InputData:           []byte(`{"input": "test"}`),
		OutputData:          []byte(`{"output": "test"}`),
		InputSize:           18,
		OutputSize:          19,
		Status:              string(types.ExecutionStatusSucceeded),
		StartedAt:           time.Now().Add(-time.Minute),
		CompletedAt:         ptrTime(time.Now()),
		DurationMS:          ptrInt64(60000),
		ErrorMessage:        nil,
		RetryCount:          0,
		WorkflowName:        ptrString("Test Workflow"),
		WorkflowTags:        []string{"test", "unit-test"},
		CreatedAt:           time.Now(),
		UpdatedAt:           time.Now(),
	}

	// Create workflow unit of work
	wuow := storage.NewWorkflowUnitOfWork()

	// Store workflow and execution atomically
	err := wuow.StoreWorkflowWithExecution(context.Background(), workflow, execution)
	if err != nil {
		t.Errorf("Failed to register workflow and execution: %v", err)
	}

	// Verify changes are registered
	if wuow.GetChangeCount() != 2 {
		t.Errorf("Expected 2 changes, got %d", wuow.GetChangeCount())
	}

	// Commit the unit of work
	err = wuow.Commit()
	if err != nil {
		t.Errorf("Failed to commit workflow unit of work: %v", err)
	}

	// Verify data was stored
	retrievedExecution, err := storage.GetWorkflowExecution(context.Background(), "test-execution-001") // Pass context
	if err != nil {
		t.Errorf("Failed to retrieve workflow execution: %v", err)
	}

	if retrievedExecution.WorkflowID != "test-workflow-001" {
		t.Errorf("Expected workflow ID 'test-workflow-001', got '%s'", retrievedExecution.WorkflowID)
	}

	if retrievedExecution.Status != string(types.ExecutionStatusSucceeded) {
		t.Errorf("Expected status '%s', got '%s'", types.ExecutionStatusSucceeded, retrievedExecution.Status)
	}
}

func TestWorkflowUnitOfWork_RollbackOnError(t *testing.T) {
	storage, cleanup := setupTestDB(t)
	defer cleanup()

	// Create workflow unit of work
	wuow := storage.NewWorkflowUnitOfWork()

	// Register a valid operation
	validOp := func(tx DBTX) error {
		_, err := tx.Exec("SELECT 1")
		return err
	}
	wuow.RegisterNew("valid_entity", "test_table", validOp)

	// Register an invalid operation that will cause rollback
	invalidOp := func(tx DBTX) error {
		_, err := tx.Exec("INVALID SQL STATEMENT")
		return err
	}
	wuow.RegisterNew("invalid_entity", "test_table", invalidOp)

	// Attempt to commit - should fail and rollback
	err := wuow.Commit()
	if err == nil {
		t.Error("Expected commit to fail due to invalid SQL")
	}

	// Verify unit of work is no longer active
	if wuow.IsActive() {
		t.Error("Unit of work should not be active after failed commit")
	}
}

func TestWorkflowUnitOfWork_UpdateWorkflowStatus(t *testing.T) {
	storage, cleanup := setupTestDB(t)
	defer cleanup()

	// First, create a workflow
	workflow := &types.Workflow{
		WorkflowID:   "test-workflow-002",
		WorkflowName: ptrString("Test Workflow 2"),
		WorkflowTags: []string{"test"},
		Status:       "running",
		StartedAt:    time.Now(),
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	// Store initial workflow
	err := storage.CreateOrUpdateWorkflow(context.Background(), workflow) // Pass context
	if err != nil {
		t.Errorf("Failed to create initial workflow: %v", err)
	}

	// Create new execution
	execution := &types.WorkflowExecution{
		WorkflowID:          "test-workflow-002",
		ExecutionID:         "test-execution-002",
		AgentFieldRequestID: "test-request-002",
		AgentNodeID:         "test-agent-002",
		ReasonerID:          "test-reasoner",
		InputData:           json.RawMessage(`{"test": "input"}`),
		OutputData:          json.RawMessage(`{"test": "output"}`),
		InputSize:           15,
		OutputSize:          16,
		Status:              string(types.ExecutionStatusSucceeded),
		StartedAt:           time.Now().Add(-time.Minute),
		CompletedAt:         ptrTime(time.Now()),
		DurationMS:          ptrInt64(60000),
		WorkflowName:        ptrString("Test Workflow"),
		CreatedAt:           time.Now(),
		UpdatedAt:           time.Now(),
	}

	// Update workflow status with new execution atomically
	wuow := storage.NewWorkflowUnitOfWork()
	err = wuow.UpdateWorkflowStatus(context.Background(), "test-workflow-002", string(types.ExecutionStatusSucceeded), execution)
	if err != nil {
		t.Errorf("Failed to register workflow status update: %v", err)
	}

	// Commit the changes
	err = wuow.Commit()
	if err != nil {
		t.Errorf("Failed to commit workflow status update: %v", err)
	}

	// Verify execution was stored
	retrievedExecution, err := storage.GetWorkflowExecution(context.Background(), "test-execution-002") // Pass context
	if err != nil {
		t.Errorf("Failed to retrieve workflow execution: %v", err)
	}

	if retrievedExecution.Status != string(types.ExecutionStatusSucceeded) {
		t.Errorf("Expected execution status '%s', got '%s'", types.ExecutionStatusSucceeded, retrievedExecution.Status)
	}
}

// Helper functions
func ptrString(s string) *string {
	return &s
}

func ptrTime(t time.Time) *time.Time {
	return &t
}

func ptrInt64(i int64) *int64 {
	return &i
}
