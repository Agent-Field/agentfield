package storage

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupRetryTestStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()
	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}
	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() { _ = ls.Close(ctx) })
	return ls, ctx
}

func TestRetryStaleWorkflowExecutions(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	// Create a stale workflow execution with retry_count=0
	staleExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-retry-1",
		AgentFieldRequestID: "req-1",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
	}
	err := ls.StoreWorkflowExecution(ctx, staleExec)
	require.NoError(t, err)
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:  "exec-retry-1",
		RunID:        "run-retry-1",
		AgentNodeID:  "agent-1",
		ReasonerID:   "reason-1",
		NodeID:       "agent-1",
		Status:       "running",
		StartedAt:    now.Add(-2 * time.Hour),
		CreatedAt:    now.Add(-2 * time.Hour),
		UpdatedAt:    now.Add(-2 * time.Hour),
		InputPayload: json.RawMessage(`{}`),
	}))

	// Create a stale execution that already exhausted retries
	exhaustedExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-exhausted",
		AgentFieldRequestID: "req-2",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          3,
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
	}
	err = ls.StoreWorkflowExecution(ctx, exhaustedExec)
	require.NoError(t, err)

	// Create a fresh (non-stale) execution
	freshExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-fresh",
		AgentFieldRequestID: "req-3",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now,
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           now,
		UpdatedAt:           now,
	}
	err = ls.StoreWorkflowExecution(ctx, freshExec)
	require.NoError(t, err)

	// Retry with maxRetries=3 and staleAfter=1 hour
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 3, 100)
	require.NoError(t, err)

	// Only exec-retry-1 should be retried (stale + under max retries)
	assert.Equal(t, 1, len(retriedIDs))
	assert.Equal(t, "exec-retry-1", retriedIDs[0])

	// Verify the execution was reset to pending with incremented retry_count
	retried, err := ls.GetWorkflowExecution(ctx, "exec-retry-1")
	require.NoError(t, err)
	assert.Equal(t, "pending", retried.Status)
	assert.Equal(t, 1, retried.RetryCount)
	assert.Nil(t, retried.CompletedAt)

	executionRecord, err := ls.GetExecutionRecord(ctx, "exec-retry-1")
	require.NoError(t, err)
	assert.Equal(t, "pending", executionRecord.Status)
	assert.Nil(t, executionRecord.CompletedAt)

	// Verify exhausted execution was NOT retried
	exhausted, err := ls.GetWorkflowExecution(ctx, "exec-exhausted")
	require.NoError(t, err)
	assert.Equal(t, "running", exhausted.Status)
	assert.Equal(t, 3, exhausted.RetryCount)

	// Verify fresh execution was NOT retried
	fresh, err := ls.GetWorkflowExecution(ctx, "exec-fresh")
	require.NoError(t, err)
	assert.Equal(t, "running", fresh.Status)
	assert.Equal(t, 0, fresh.RetryCount)
}

func TestRetryStaleWorkflowExecutions_DisabledWithZeroMaxRetries(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)

	// maxRetries=0 should return nil without querying
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 0, 100)
	require.NoError(t, err)
	assert.Nil(t, retriedIDs)
}

func TestRetryStaleWorkflowExecutions_NoStaleExecutions(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)

	// No executions at all — should return empty
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 3, 100)
	require.NoError(t, err)
	assert.Nil(t, retriedIDs)
}
