package storage

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// TestStorageParity_CreateExecutionRecord tests that both local and postgres storage
// implement CreateExecutionRecord with the same behavior
func TestStorageParity_CreateExecutionRecord(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T) (StorageProvider, context.Context)
		cleanup func(t *testing.T, store StorageProvider, ctx context.Context)
	}{
		{
			name: "local_storage",
			setup: func(t *testing.T) (StorageProvider, context.Context) {
				ctx := context.Background()
				tempDir := t.TempDir()
				cfg := StorageConfig{
					Mode: "local",
					Local: LocalStorageConfig{
						DatabasePath: filepath.Join(tempDir, "test.db"),
						KVStorePath:  filepath.Join(tempDir, "test.bolt"),
					},
				}
				ls := NewLocalStorage(LocalStorageConfig{})
				if err := ls.Initialize(ctx, cfg); err != nil {
					if strings.Contains(strings.ToLower(err.Error()), "fts5") {
						t.Skip("sqlite3 compiled without FTS5")
					}
					require.NoError(t, err)
				}
				return ls, ctx
			},
			cleanup: func(t *testing.T, store StorageProvider, ctx context.Context) {
				_ = store.Close(ctx)
			},
		},
		// Postgres test would require a running postgres instance
		// Skipping for now but structure is ready
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store, ctx := tt.setup(t)
			defer tt.cleanup(t, store, ctx)

			exec := &types.Execution{
				ExecutionID: "exec-parity-1",
				RunID:       "run-parity-1",
				AgentNodeID: "agent-1",
				ReasonerID:  "reasoner-1",
				NodeID:      "node-1",
				Status:      string(types.ExecutionStatusPending),
				StartedAt:   time.Now().UTC(),
			}

			err := store.CreateExecutionRecord(ctx, exec)
			require.NoError(t, err)

			// Verify it can be retrieved
			retrieved, err := store.GetExecutionRecord(ctx, "exec-parity-1")
			require.NoError(t, err)
			require.NotNil(t, retrieved)
			require.Equal(t, exec.ExecutionID, retrieved.ExecutionID)
			require.Equal(t, exec.RunID, retrieved.RunID)
			require.Equal(t, exec.AgentNodeID, retrieved.AgentNodeID)
			require.Equal(t, exec.ReasonerID, retrieved.ReasonerID)
		})
	}
}

// TestStorageParity_UpdateExecutionRecord tests update behavior parity
func TestStorageParity_UpdateExecutionRecord(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	defer store.Close(ctx)

	// Create initial execution
	exec := &types.Execution{
		ExecutionID: "exec-update-parity",
		RunID:       "run-update-parity",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      string(types.ExecutionStatusPending),
		StartedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(ctx, exec))

	// Update execution
	updated, err := store.UpdateExecutionRecord(ctx, "exec-update-parity", func(e *types.Execution) (*types.Execution, error) {
		e.Status = string(types.ExecutionStatusSucceeded)
		completed := time.Now().UTC()
		e.CompletedAt = &completed
		duration := int64(1000)
		e.DurationMS = &duration
		return e, nil
	})
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, string(types.ExecutionStatusSucceeded), updated.Status)
	require.NotNil(t, updated.CompletedAt)
	require.NotNil(t, updated.DurationMS)
	require.Equal(t, int64(1000), *updated.DurationMS)

	// Verify update persisted
	retrieved, err := store.GetExecutionRecord(ctx, "exec-update-parity")
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusSucceeded), retrieved.Status)
}

// TestStorageParity_QueryExecutionRecords tests query behavior parity
func TestStorageParity_QueryExecutionRecords(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	defer store.Close(ctx)

	// Create multiple executions
	executions := []*types.Execution{
		{
			ExecutionID: "exec-query-1",
			RunID:       "run-query-1",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-1",
			NodeID:      "node-1",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   time.Now().UTC(),
		},
		{
			ExecutionID: "exec-query-2",
			RunID:       "run-query-1",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-2",
			NodeID:      "node-2",
			Status:      string(types.ExecutionStatusFailed),
			StartedAt:   time.Now().UTC(),
		},
		{
			ExecutionID: "exec-query-3",
			RunID:       "run-query-2",
			AgentNodeID: "agent-2",
			ReasonerID:  "reasoner-1",
			NodeID:      "node-3",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   time.Now().UTC(),
		},
	}

	for _, exec := range executions {
		require.NoError(t, store.CreateExecutionRecord(ctx, exec))
	}

	// Query by run_id
	results, err := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		RunID: stringPtr("run-query-1"),
	})
	require.NoError(t, err)
	require.Len(t, results, 2)

	// Query by agent_node_id
	results, err = store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		AgentNodeID: stringPtr("agent-1"),
	})
	require.NoError(t, err)
	require.Len(t, results, 2)

	// Query by status
	results, err = store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		Status: stringPtr(string(types.ExecutionStatusSucceeded)),
	})
	require.NoError(t, err)
	require.Len(t, results, 2)
}

// TestStorageParity_StoreWorkflowExecution tests workflow execution storage parity
func TestStorageParity_StoreWorkflowExecution(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	defer store.Close(ctx)

	runID := "run-wf-parity"
	workflowID := "wf-parity"
	executionID := "exec-wf-parity"

	// Create workflow run
	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: workflowID,
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, store.StoreWorkflowRun(ctx, run))

	// Store workflow execution
	exec := &types.WorkflowExecution{
		WorkflowID:          workflowID,
		ExecutionID:         executionID,
		AgentFieldRequestID: "req-1",
		RunID:               &runID,
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner-1",
		Status:              string(types.ExecutionStatusPending),
		StartedAt:           time.Now().UTC(),
		CreatedAt:           time.Now().UTC(),
		UpdatedAt:           time.Now().UTC(),
	}
	require.NoError(t, store.StoreWorkflowExecution(ctx, exec))

	// Retrieve and verify
	retrieved, err := store.GetWorkflowExecution(ctx, executionID)
	require.NoError(t, err)
	require.NotNil(t, retrieved)
	require.Equal(t, executionID, retrieved.ExecutionID)
	require.Equal(t, workflowID, retrieved.WorkflowID)
	require.NotNil(t, retrieved.RunID)
	require.Equal(t, runID, *retrieved.RunID)
}

// TestStorageParity_TransactionRollback tests that both storage backends
// handle transaction rollback correctly
func TestStorageParity_TransactionRollback(t *testing.T) {
	store, ctx := setupLocalStorage(t)
	defer store.Close(ctx)

	// Create execution
	exec := &types.Execution{
		ExecutionID: "exec-rollback",
		RunID:       "run-rollback",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      string(types.ExecutionStatusPending),
		StartedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(ctx, exec))

	// Attempt update that should fail (simulating constraint violation)
	_, err := store.UpdateExecutionRecord(ctx, "exec-rollback", func(e *types.Execution) (*types.Execution, error) {
		// Return error to simulate rollback
		return nil, &ValidationError{
			Field:   "status",
			Value:   "invalid",
			Reason:  "invalid status transition",
			Context: "UpdateExecutionRecord",
		}
	})
	require.Error(t, err)
	require.IsType(t, &ValidationError{}, err)

	// Verify original execution unchanged
	retrieved, err := store.GetExecutionRecord(ctx, "exec-rollback")
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusPending), retrieved.Status)
}

// Helper function
func stringPtr(s string) *string {
	return &s
}
