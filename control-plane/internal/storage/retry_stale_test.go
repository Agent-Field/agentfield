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
	// CreateExecutionRecord initializes activity timestamps to the current time;
	// backdate the paired row so both clocks are stale for this retry case.
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-retry-1", now.Add(-2*time.Hour))

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

func TestRetryStaleWorkflowExecutions_ExecutionActivityProtectsWorkflow(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-heartbeat",
		ExecutionID:         "exec-retry-heartbeat",
		AgentFieldRequestID: "req-retry-heartbeat",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	// The execution is fresh even though its paired workflow row is stale.
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-retry-heartbeat",
		RunID:       "run-retry-heartbeat",
		AgentNodeID: "agent-1",
		ReasonerID:  "reason-1",
		NodeID:      "agent-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}))

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Empty(t, retried, "fresh paired execution activity must protect a stale workflow")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status)
	require.Equal(t, 0, workflowRecord.RetryCount)
}

func TestRetryStaleWorkflowExecutions_ActivityAfterSelectionSkipsUpdate(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-selection-race",
		ExecutionID:         "exec-retry-selection-race",
		AgentFieldRequestID: "req-retry-selection-race",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	execution := &types.Execution{
		ExecutionID: "exec-retry-selection-race",
		RunID:       "run-retry-selection-race",
		AgentNodeID: "agent-1",
		ReasonerID:  "reason-1",
		NodeID:      "agent-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, execution))
	backdateExecutionUpdatedAt(t, ls, "executions", execution.ExecutionID, now.Add(-1*time.Hour))

	var heartbeatErr error
	retried, err := ls.retryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100, func() {
		_, heartbeatErr = ls.UpdateExecutionRecord(ctx, execution.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			current.Notes = append(current.Notes, types.ExecutionNote{
				Message:   "heartbeat",
				Timestamp: now,
			})
			return current, nil
		})
	})
	require.NoError(t, heartbeatErr)
	require.NoError(t, err)
	require.Empty(t, retried, "activity after selection must prevent the retry update")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status)
	require.Equal(t, 0, workflowRecord.RetryCount)

	executionRecord, err := ls.GetExecutionRecord(ctx, execution.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", executionRecord.Status)
	require.Len(t, executionRecord.Notes, 1)
}

// TestRetryStaleWorkflowExecutions_HeartbeatBetweenStatementsSparesExecution
// covers the window between the workflow UPDATE and the paired execution
// UPDATE. An AFTER UPDATE trigger on workflow_executions simulates a heartbeat
// that commits after the workflow guard has been evaluated: the execution's
// activity clock moves while the retry is between its two statements. The
// execution UPDATE must repeat the staleness predicate, so the fresh execution
// survives instead of being dragged back to pending.
func TestRetryStaleWorkflowExecutions_HeartbeatBetweenStatementsSparesExecution(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-between-statements",
		ExecutionID:         "exec-retry-between-statements",
		AgentFieldRequestID: "req-retry-between-statements",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:  workflow.ExecutionID,
		RunID:        "run-retry-between-statements",
		AgentNodeID:  "agent-1",
		ReasonerID:   "reason-1",
		NodeID:       "agent-1",
		Status:       "running",
		StartedAt:    now.Add(-2 * time.Hour),
		InputPayload: json.RawMessage(`{}`),
	}))
	backdateExecutionUpdatedAt(t, ls, "executions", workflow.ExecutionID, now.Add(-2*time.Hour))

	// The trigger stamps a heartbeat on the paired execution as soon as the
	// retry's workflow UPDATE has matched, i.e. between the retry's two
	// statements.
	db := ls.requireSQLDB()
	_, err := db.Exec(`
		CREATE TRIGGER retry_heartbeat_between_statements
		AFTER UPDATE ON workflow_executions
		FOR EACH ROW
		BEGIN
			UPDATE executions
			SET updated_at = CURRENT_TIMESTAMP,
			    status_reason = 'heartbeat-between-statements'
			WHERE execution_id = NEW.execution_id;
		END`)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = db.Exec("DROP TRIGGER IF EXISTS retry_heartbeat_between_statements")
	})

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Equal(t, []string{workflow.ExecutionID}, retried)

	executionRecord, err := ls.GetExecutionRecord(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", executionRecord.Status,
		"a heartbeat between the retry statements must not be dragged back to pending")
	require.NotNil(t, executionRecord.StatusReason)
	require.Equal(t, "heartbeat-between-statements", *executionRecord.StatusReason,
		"the between-statements heartbeat must be the surviving write")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "pending", workflowRecord.Status)
	require.Equal(t, 1, workflowRecord.RetryCount)
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
