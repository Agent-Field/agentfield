package storage

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func setupLocalStorage(t *testing.T) (*LocalStorage, context.Context) {
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
			t.Skip("sqlite3 compiled without FTS5; skipping query workflow tests")
		}
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

func TestQueryWorkflowExecutionsFiltersAndSearch(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	now := time.Now().UTC()
	runID := "run-123"

	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "wf-root",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	alphaName := "Alpha Report"
	betaName := "Beta Summary"

	runningStatus := string(types.ExecutionStatusRunning)
	succeededStatus := string(types.ExecutionStatusSucceeded)
	alphaDuration := int64(2000)
	betaDuration := int64(1000)

	executions := []*types.WorkflowExecution{
		{
			WorkflowID:          "wf-root",
			ExecutionID:         "exec-alpha",
			AgentFieldRequestID: "req-1",
			RunID:               &runID,
			AgentNodeID:         "agent-one",
			ReasonerID:          "reasoner.alpha",
			WorkflowName:        &alphaName,
			Status:              runningStatus,
			StartedAt:           now.Add(-5 * time.Minute),
			CreatedAt:           now.Add(-5 * time.Minute),
			UpdatedAt:           now.Add(-4 * time.Minute),
			DurationMS:          &alphaDuration,
		},
		{
			WorkflowID:          "wf-root",
			ExecutionID:         "exec-beta",
			AgentFieldRequestID: "req-2",
			RunID:               &runID,
			AgentNodeID:         "agent-two",
			ReasonerID:          "reasoner.beta",
			WorkflowName:        &betaName,
			Status:              succeededStatus,
			StartedAt:           now.Add(-3 * time.Minute),
			CreatedAt:           now.Add(-3 * time.Minute),
			UpdatedAt:           now.Add(-2 * time.Minute),
			DurationMS:          &betaDuration,
		},
	}

	for _, exec := range executions {
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
	}

	allExecutions, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, allExecutions, 2)

	statuses := map[string]struct{}{}
	for _, exec := range allExecutions {
		statuses[exec.Status] = struct{}{}
	}
	require.Contains(t, statuses, runningStatus)
	require.Contains(t, statuses, succeededStatus)

	results, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{Status: &runningStatus})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "exec-alpha", results[0].ExecutionID)

	agentID := "agent-two"
	results, err = ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{AgentNodeID: &agentID})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "exec-beta", results[0].ExecutionID)

	rawSearch := "Alpha*"
	results, err = ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{Search: &rawSearch})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "exec-alpha", results[0].ExecutionID)

	sortBy := "duration"
	sortOrder := "ASC"
	results, err = ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{SortBy: &sortBy, SortOrder: &sortOrder})
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, "exec-beta", results[0].ExecutionID)

	paggined, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{SortBy: &sortBy, SortOrder: &sortOrder, Limit: 1, Offset: 1})
	require.NoError(t, err)
	require.Len(t, paggined, 1)
	require.Equal(t, "exec-alpha", paggined[0].ExecutionID)
}

func TestGetReasonerExecutionHistory_EnrichesExecutionFacts(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	now := time.Now().UTC().Truncate(time.Second)
	completedAt := now.Add(2 * time.Second)
	statusReason := "provider_timeout"
	sessionID := "session-123"
	actorID := "actor-123"
	errorMessage := "upstream timed out"

	inputData := json.RawMessage(`{"input":{"ticker":"NVDA"},"context":{"analysis_group":"summary.short_form"}}`)
	outputData := json.RawMessage(`{"summary":"done"}`)
	duration := int64(2000)

	exec := &types.WorkflowExecution{
		WorkflowID:          "wf-1",
		ExecutionID:         "exec-1",
		AgentFieldRequestID: "req-1",
		AgentNodeID:         "node-1",
		ReasonerID:          "summarizer",
		InputData:           inputData,
		OutputData:          outputData,
		Status:              string(types.ExecutionStatusFailed),
		StatusReason:        &statusReason,
		ErrorMessage:        &errorMessage,
		RetryCount:          2,
		SessionID:           &sessionID,
		ActorID:             &actorID,
		StartedAt:           now,
		CompletedAt:         &completedAt,
		DurationMS:          &duration,
		CreatedAt:           now,
		UpdatedAt:           completedAt,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))

	history, err := ls.GetReasonerExecutionHistory(ctx, "node-1.summarizer", 1, 10)
	require.NoError(t, err)
	require.Len(t, history.Executions, 1)

	record := history.Executions[0]
	require.Equal(t, "exec-1", record.ExecutionID)
	require.Equal(t, "node-1", record.AgentNodeID)
	require.Equal(t, "summarizer", record.ReasonerID)
	require.Equal(t, string(types.ExecutionStatusFailed), record.Status)
	require.Equal(t, &statusReason, record.StatusReason)
	require.Equal(t, &statusReason, record.ErrorCategory)
	require.Equal(t, 2, record.RetryCount)
	require.Equal(t, &sessionID, record.SessionID)
	require.Equal(t, &actorID, record.ActorID)
	require.Equal(t, now, record.StartedAt)
	require.Equal(t, &completedAt, record.CompletedAt)
	require.Equal(t, now, record.Timestamp)
	require.Equal(t, map[string]interface{}{"ticker": "NVDA"}, record.Input)
	require.Equal(t, map[string]interface{}{"analysis_group": "summary.short_form"}, record.Context)
	require.Equal(t, map[string]interface{}{"summary": "done"}, record.Output)
	require.Equal(t, errorMessage, record.Error)
	require.Equal(t, duration, record.DurationMs)
}

func TestQueryWorkflowDAGReturnsHierarchy(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	now := time.Now().UTC()
	runID := "root-run"

	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "wf-root",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	root := &types.WorkflowExecution{
		WorkflowID:          "wf-root",
		ExecutionID:         "root-exec",
		AgentFieldRequestID: "req-root",
		RunID:               &runID,
		AgentNodeID:         "agent-root",
		ReasonerID:          "root",
		Status:              string(types.ExecutionStatusRunning),
		StartedAt:           now.Add(-2 * time.Minute),
		CreatedAt:           now.Add(-2 * time.Minute),
		UpdatedAt:           now.Add(-2 * time.Minute),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, root))

	parentID := root.ExecutionID
	child := &types.WorkflowExecution{
		WorkflowID:          "wf-root",
		ExecutionID:         "child-exec",
		AgentFieldRequestID: "req-child",
		RunID:               &runID,
		AgentNodeID:         "agent-child",
		ReasonerID:          "child",
		ParentExecutionID:   &parentID,
		RootWorkflowID:      &root.WorkflowID,
		WorkflowDepth:       1,
		Status:              string(types.ExecutionStatusRunning),
		StartedAt:           now.Add(-time.Minute),
		CreatedAt:           now.Add(-time.Minute),
		UpdatedAt:           now.Add(-time.Minute),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, child))

	dagExecutions, err := ls.QueryWorkflowDAG(ctx, "wf-root")
	require.NoError(t, err)
	require.Len(t, dagExecutions, 2)
	require.Equal(t, "root-exec", dagExecutions[0].ExecutionID)
	require.Equal(t, "child-exec", dagExecutions[1].ExecutionID)
}

func TestSanitizeFTS5Query(t *testing.T) {
	sanitized := sanitizeFTS5Query("\"Alpha\" AND (Beta*) OR NOT Gamma")
	require.Equal(t, "\"Alpha Beta Gamma\"", sanitized)

	sanitized = sanitizeFTS5Query("")
	require.Equal(t, "", sanitized)
}

func TestValidateExecutionStateTransition(t *testing.T) {
	require.NoError(t, validateExecutionStateTransition(string(types.ExecutionStatusPending), string(types.ExecutionStatusRunning)))
	require.NoError(t, validateExecutionStateTransition(string(types.ExecutionStatusRunning), string(types.ExecutionStatusRunning)))
	require.NoError(t, validateExecutionStateTransition(string(types.ExecutionStatusRunning), string(types.ExecutionStatusWaiting)))
	require.NoError(t, validateExecutionStateTransition(string(types.ExecutionStatusWaiting), string(types.ExecutionStatusRunning)))
	require.NoError(t, validateExecutionStateTransition(string(types.ExecutionStatusWaiting), string(types.ExecutionStatusCancelled)))

	err := validateExecutionStateTransition(string(types.ExecutionStatusRunning), string(types.ExecutionStatusPending))
	require.Error(t, err)
	var transitionErr *InvalidExecutionStateTransitionError
	require.ErrorAs(t, err, &transitionErr)
	require.Equal(t, string(types.ExecutionStatusRunning), transitionErr.CurrentState)
	require.Equal(t, string(types.ExecutionStatusPending), transitionErr.NewState)

	err = validateExecutionStateTransition(string(types.ExecutionStatusQueued), string(types.ExecutionStatusWaiting))
	require.Error(t, err)
}
