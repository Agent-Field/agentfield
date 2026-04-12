package storage

import (
	"context"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestUpdateWorkflowExecutionErrorPaths(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          "wf-update-errors",
		ExecutionID:         "exec-update-errors",
		AgentFieldRequestID: "req-update-errors",
		AgentNodeID:         "agent-update-errors",
		ReasonerID:          "reasoner-update-errors",
		Status:              "running",
	}))

	t.Run("update function error is wrapped", func(t *testing.T) {
		err := ls.UpdateWorkflowExecution(ctx, "exec-update-errors", func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			require.NotNil(t, execution)
			return nil, errors.New("boom")
		})
		require.EqualError(t, err, "update function failed for execution exec-update-errors: boom")
	})

	t.Run("execution id cannot change", func(t *testing.T) {
		err := ls.UpdateWorkflowExecution(ctx, "exec-update-errors", func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			updated := *execution
			updated.ExecutionID = "exec-updated"
			return &updated, nil
		})
		require.EqualError(t, err, "update function cannot change execution ID: expected exec-update-errors, got exec-updated")
	})

	t.Run("invalid state transition is rejected", func(t *testing.T) {
		err := ls.UpdateWorkflowExecution(ctx, "exec-update-errors", func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			updated := *execution
			updated.Status = "pending"
			return &updated, nil
		})
		require.EqualError(t, err, "failed to store updated workflow execution: invalid execution state transition for exec-update-errors: cannot change from running to pending - transition not allowed by workflow execution state machine")
	})

	t.Run("cancelled context fails fast", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()

		err := ls.UpdateWorkflowExecution(cancelled, "exec-update-errors", func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			return execution, nil
		})
		require.EqualError(t, err, "context cancelled during update workflow execution: context canceled")
	})
}
