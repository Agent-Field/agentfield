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

func TestLocalStorageCleanupWorkflowByRunID(t *testing.T) {
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
		if strings.Contains(err.Error(), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping cleanup test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	runID := "run_cleanup_test"
	workflowID := "wf_cleanup_test"
	now := time.Now().UTC()

	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: workflowID,
		Status:         string(types.ExecutionStatusRunning),
		TotalSteps:     1,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	if err := ls.StoreWorkflowRun(ctx, run); err != nil {
		t.Fatalf("store workflow run: %v", err)
	}

	workflow := &types.Workflow{
		WorkflowID:    workflowID,
		WorkflowName:  nil,
		WorkflowTags:  []string{},
		WorkflowDepth: 0,
		Status:        string(types.ExecutionStatusRunning),
		StartedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	if err := ls.CreateOrUpdateWorkflow(ctx, workflow); err != nil {
		t.Fatalf("store workflow definition: %v", err)
	}

	exec := &types.WorkflowExecution{
		WorkflowID:          workflowID,
		ExecutionID:         "exec_cleanup_test",
		AgentFieldRequestID: "req_cleanup_test",
		RunID:               &runID,
		AgentNodeID:         "agent_cleanup",
		ReasonerID:          "reasoner.cleanup",
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
		InputSize:           0,
		OutputSize:          0,
		Status:              string(types.ExecutionStatusRunning),
		StartedAt:           now,
		CreatedAt:           now,
		UpdatedAt:           now,
		WorkflowDepth:       0,
		WorkflowTags:        []string{},
	}
	if err := ls.StoreWorkflowExecution(ctx, exec); err != nil {
		t.Fatalf("store workflow execution: %v", err)
	}

	step := &types.WorkflowStep{
		StepID:    "step_cleanup",
		RunID:     runID,
		Status:    string(types.ExecutionStatusPending),
		Attempt:   0,
		Priority:  0,
		NotBefore: now,
		Metadata:  json.RawMessage("{}"),
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := ls.StoreWorkflowStep(ctx, step); err != nil {
		t.Fatalf("store workflow step: %v", err)
	}

	event := &types.WorkflowRunEvent{
		RunID:            runID,
		Sequence:         1,
		PreviousSequence: 0,
		EventType:        "test",
		Payload:          json.RawMessage("{}"),
		EmittedAt:        now,
	}
	if err := ls.StoreWorkflowRunEvent(ctx, event); err != nil {
		t.Fatalf("store workflow run event: %v", err)
	}

	result, err := ls.CleanupWorkflow(ctx, runID, false)
	if err != nil {
		t.Fatalf("cleanup workflow by run id: %v", err)
	}

	if !result.Success {
		t.Fatalf("expected cleanup success, got %#v", result)
	}

	if result.WorkflowID != workflowID {
		t.Fatalf("expected workflow id %q, got %q", workflowID, result.WorkflowID)
	}

	if result.DeletedRecords["workflow_runs"] == 0 {
		t.Fatalf("expected workflow_runs to be deleted, got %#v", result.DeletedRecords)
	}
	if result.DeletedRecords["workflow_executions"] == 0 {
		t.Fatalf("expected workflow_executions to be deleted, got %#v", result.DeletedRecords)
	}

	// Run should be gone
	fetchedRun, err := ls.GetWorkflowRun(ctx, runID)
	if err != nil {
		t.Fatalf("get workflow run after cleanup: %v", err)
	}
	if fetchedRun != nil {
		t.Fatalf("expected workflow run to be deleted")
	}

	// Workflow definition should also be removed
	if _, err := ls.GetWorkflow(ctx, workflowID); err == nil {
		t.Fatalf("expected workflow definition to be deleted")
	}
}

func TestLocalStorageCleanupOldExecutions(t *testing.T) {
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
		if strings.Contains(err.Error(), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping old execution cleanup test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	const workflowID = "wf_cleanup_window"
	oldCompleted := time.Now().Add(-2 * time.Hour).UTC()
	recentCompleted := time.Now().Add(-15 * time.Minute).UTC()

	insertExecution := func(executionID string, completedAt time.Time) {
		exec := &types.WorkflowExecution{
			WorkflowID:          workflowID,
			ExecutionID:         executionID,
			AgentFieldRequestID: executionID + "_req",
			AgentNodeID:         "agent",
			ReasonerID:          "reasoner",
			Status:              "completed",
			StartedAt:           completedAt,
			CreatedAt:           completedAt,
			UpdatedAt:           completedAt,
			WorkflowDepth:       0,
			WorkflowTags:        []string{},
		}
		exec.CompletedAt = &completedAt
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
	}

	insertExecution("old-exec", oldCompleted)
	insertExecution("recent-exec", recentCompleted)

	deleted, err := ls.CleanupOldExecutions(ctx, time.Hour, 10)
	require.NoError(t, err)
	require.Equal(t, 1, deleted)

	stillThere, err := ls.GetWorkflowExecution(ctx, "recent-exec")
	require.NoError(t, err)
	require.NotNil(t, stillThere)

	removed, err := ls.GetWorkflowExecution(ctx, "old-exec")
	require.NoError(t, err)
	require.Nil(t, removed)
}
