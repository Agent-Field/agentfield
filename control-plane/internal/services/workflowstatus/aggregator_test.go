package workflowstatus

import (
	"github.com/your-org/haxen/control-plane/pkg/types"
	"testing"
	"time"
)

func makeExecution(status string, startedAt time.Time, completedAt *time.Time, parent bool) *types.WorkflowExecution {
	exec := &types.WorkflowExecution{
		WorkflowID:  "wf",
		ExecutionID: startedAt.Format(time.RFC3339Nano),
		Status:      status,
		StartedAt:   startedAt,
		UpdatedAt:   startedAt,
	}
	if completedAt != nil {
		exec.CompletedAt = completedAt
		exec.UpdatedAt = *completedAt
	}
	if parent {
		exec.ParentExecutionID = nil
	} else {
		id := "parent"
		exec.ParentExecutionID = &id
	}
	return exec
}

func TestAggregateExecutions_AllSucceeded(t *testing.T) {
	start := time.Now().Add(-2 * time.Minute)
	end := start.Add(10 * time.Second)

	executions := []*types.WorkflowExecution{
		makeExecution("succeeded", start, &end, true),
		makeExecution("succeeded", start.Add(time.Second), &end, false),
	}

	agg := AggregateExecutions(executions, nil)

	if agg.Status != string(types.ExecutionStatusSucceeded) {
		t.Fatalf("expected succeeded, got %s", agg.Status)
	}
	if !agg.Terminal {
		t.Fatal("expected terminal workflow")
	}
	if agg.ActiveExecutions != 0 {
		t.Fatalf("expected no active executions, got %d", agg.ActiveExecutions)
	}
}

func TestAggregateExecutions_RunningOverridesSuccess(t *testing.T) {
	now := time.Now()
	executions := []*types.WorkflowExecution{
		makeExecution("succeeded", now.Add(-5*time.Second), &now, true),
		makeExecution("running", now.Add(-2*time.Second), nil, false),
	}

	agg := AggregateExecutions(executions, nil)
	if agg.Status != string(types.ExecutionStatusRunning) {
		t.Fatalf("expected running, got %s", agg.Status)
	}
	if agg.ActiveExecutions != 1 {
		t.Fatalf("expected one active execution, got %d", agg.ActiveExecutions)
	}
	if agg.Terminal {
		t.Fatal("expected non-terminal workflow")
	}
}

func TestAggregateExecutions_PrioritisesFailures(t *testing.T) {
	now := time.Now()
	executions := []*types.WorkflowExecution{
		makeExecution("running", now.Add(-4*time.Second), nil, true),
		makeExecution("failed", now.Add(-3*time.Second), nil, false),
	}

	agg := AggregateExecutions(executions, nil)
	if agg.Status != string(types.ExecutionStatusRunning) {
		t.Fatalf("expected running while root execution is running, got %s", agg.Status)
	}
	if agg.Counts[string(types.ExecutionStatusFailed)] != 1 {
		t.Fatalf("expected failure count to include child failure")
	}
}

func TestAggregateExecutions_QueuedOnly(t *testing.T) {
	now := time.Now()
	executions := []*types.WorkflowExecution{
		makeExecution("queued", now, nil, true),
	}

	agg := AggregateExecutions(executions, nil)
	if agg.Status != string(types.ExecutionStatusQueued) {
		t.Fatalf("expected queued, got %s", agg.Status)
	}
	if agg.ActiveExecutions != 1 {
		t.Fatalf("expected active executions, got %d", agg.ActiveExecutions)
	}
	if agg.Terminal {
		t.Fatal("expected non-terminal workflow")
	}
}

func TestAggregateExecutions_TimeoutBeatsCancelled(t *testing.T) {
	now := time.Now()
	executions := []*types.WorkflowExecution{
		makeExecution("cancelled", now, nil, true),
		makeExecution("timeout", now.Add(time.Second), nil, false),
	}

	agg := AggregateExecutions(executions, nil)
	if agg.Status != string(types.ExecutionStatusCancelled) {
		t.Fatalf("expected workflow to reflect root cancelled status, got %s", agg.Status)
	}
}

func TestAggregateExecutions_PendingStepKeepsWorkflowRunning(t *testing.T) {
	now := time.Now()
	end := now.Add(2 * time.Second)

	executions := []*types.WorkflowExecution{
		makeExecution("succeeded", now.Add(-5*time.Second), &end, true),
	}

	steps := []*types.WorkflowStep{
		{
			StepID:    "step-1",
			RunID:     "wf",
			Status:    "pending",
			NotBefore: now,
			CreatedAt: now,
			UpdatedAt: now,
		},
	}

	agg := AggregateExecutions(executions, steps)

	if agg.Status != string(types.ExecutionStatusRunning) {
		t.Fatalf("expected running due to pending step, got %s", agg.Status)
	}
	if agg.Terminal {
		t.Fatal("expected non-terminal due to pending step")
	}
	if agg.ActiveSteps != 1 {
		t.Fatalf("expected 1 active step, got %d", agg.ActiveSteps)
	}
}

func TestAggregateExecutions_RootSuccessOverridesFailedChildren(t *testing.T) {
	now := time.Now()
	end := now.Add(time.Second)

	executions := []*types.WorkflowExecution{
		makeExecution("succeeded", now, &end, true),
		makeExecution("failed", now.Add(100*time.Millisecond), &end, false),
	}

	agg := AggregateExecutions(executions, nil)

	if agg.Status != string(types.ExecutionStatusSucceeded) {
		t.Fatalf("expected workflow to remain succeeded, got %s", agg.Status)
	}
	if !agg.Terminal {
		t.Fatal("expected workflow to be terminal")
	}
}
