package storage

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/pkg/types"
)

func TestLocalStorageStoreWorkflowExecutionPersistsLifecycleFields(t *testing.T) {
	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "haxen.db"),
			KVStorePath:  filepath.Join(tempDir, "haxen.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping local storage persistence test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	now := time.Now().UTC()
	runID := "run_test"
	workflowID := "wf_test"

	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: workflowID,
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}

	if err := ls.StoreWorkflowRun(ctx, run); err != nil {
		t.Fatalf("store workflow run: %v", err)
	}

	execID := "exec_test"
	haxenRequestID := "req_test"
	agentID := "agent_1"
	reasonerID := "reasoner.alpha"

	exec := &types.WorkflowExecution{
		WorkflowID:     workflowID,
		ExecutionID:    execID,
		HaxenRequestID: haxenRequestID,
		RunID:          &runID,
		AgentNodeID:    agentID,
		ReasonerID:     reasonerID,
		Status:         string(types.ExecutionStatusPending),
		StartedAt:      now,
		CreatedAt:      now,
		UpdatedAt:      now,
	}

	if err := ls.StoreWorkflowExecution(ctx, exec); err != nil {
		t.Fatalf("store workflow execution: %v", err)
	}

	stored, err := ls.GetWorkflowExecution(ctx, execID)
	if err != nil {
		t.Fatalf("get workflow execution: %v", err)
	}
	if stored == nil {
		t.Fatalf("expected workflow execution to be stored")
	}

	if stored.RunID == nil || *stored.RunID != runID {
		t.Fatalf("expected run_id %q, got %v", runID, stored.RunID)
	}
	if stored.StateVersion != exec.StateVersion {
		t.Fatalf("expected state_version %d, got %d", exec.StateVersion, stored.StateVersion)
	}
	if stored.LastEventSequence != exec.LastEventSequence {
		t.Fatalf("expected last_event_sequence %d, got %d", exec.LastEventSequence, stored.LastEventSequence)
	}
	if stored.ActiveChildren != 0 {
		t.Fatalf("expected active_children 0, got %d", stored.ActiveChildren)
	}
	if stored.PendingChildren != 0 {
		t.Fatalf("expected pending_children 0, got %d", stored.PendingChildren)
	}
}
