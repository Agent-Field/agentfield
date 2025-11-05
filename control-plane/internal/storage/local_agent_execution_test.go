package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestLocalStorageStoreExecutionRoundTrip(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	now := time.Now().UTC().Truncate(time.Millisecond)
	sessionID := "session-1"
	userID := "user-1"
	nodeID := "node-1"
	errMsg := "example failure"
	usd := 1.23

	exec := &types.AgentExecution{
		WorkflowID:   "workflow-alpha",
		SessionID:    &sessionID,
		AgentNodeID:  "agent-1",
		ReasonerID:   "reasoner.alpha",
		InputData:    json.RawMessage(`{"prompt":"hello"}`),
		OutputData:   json.RawMessage(`{"result":"world"}`),
		InputSize:    6,
		OutputSize:   5,
		DurationMS:   1234,
		Status:       "succeeded",
		ErrorMessage: &errMsg,
		UserID:       &userID,
		NodeID:       &nodeID,
		Metadata: types.ExecutionMetadata{
			Cost: &types.CostMetadata{USD: &usd, Currency: "USD"},
			Custom: map[string]interface{}{
				"invocation": float64(1),
			},
		},
		CreatedAt: now,
	}

	require.NoError(t, ls.StoreExecution(ctx, exec))
	require.Greater(t, exec.ID, int64(0))

	stored, err := ls.GetExecution(ctx, exec.ID)
	require.NoError(t, err)
	require.Equal(t, exec.ID, stored.ID)
	require.Equal(t, exec.WorkflowID, stored.WorkflowID)
	require.Equal(t, exec.AgentNodeID, stored.AgentNodeID)
	require.Equal(t, exec.ReasonerID, stored.ReasonerID)
	require.Equal(t, exec.Status, stored.Status)
	require.Equal(t, exec.InputSize, stored.InputSize)
	require.Equal(t, exec.OutputSize, stored.OutputSize)
	require.Equal(t, exec.DurationMS, stored.DurationMS)
	require.NotNil(t, stored.SessionID)
	require.Equal(t, sessionID, *stored.SessionID)
	require.NotNil(t, stored.ErrorMessage)
	require.Equal(t, errMsg, *stored.ErrorMessage)
	require.NotNil(t, stored.UserID)
	require.Equal(t, userID, *stored.UserID)
	require.NotNil(t, stored.NodeID)
	require.Equal(t, nodeID, *stored.NodeID)
	require.WithinDuration(t, exec.CreatedAt, stored.CreatedAt, time.Second)

	require.NotNil(t, stored.Metadata.Cost)
	require.NotNil(t, stored.Metadata.Cost.USD)
	require.InDelta(t, usd, *stored.Metadata.Cost.USD, 1e-9)
	require.Contains(t, stored.Metadata.Custom, "invocation")
}

func TestLocalStorageQueryExecutionsAppliesFilters(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	baseTime := time.Now().UTC().Add(-time.Minute)
	statusRunning := "running"
	statusFailed := "failed"
	sessionA := "session-a"
	sessionB := "session-b"

	executions := []*types.AgentExecution{
		{
			WorkflowID:  "workflow-shared",
			SessionID:   &sessionA,
			AgentNodeID: "agent-A",
			ReasonerID:  "reasoner.alpha",
			InputData:   json.RawMessage(`{"seed":1}`),
			OutputData:  json.RawMessage(`{"out":1}`),
			InputSize:   10,
			OutputSize:  4,
			DurationMS:  200,
			Status:      statusRunning,
			CreatedAt:   baseTime,
		},
		{
			WorkflowID:  "workflow-shared",
			SessionID:   &sessionB,
			AgentNodeID: "agent-B",
			ReasonerID:  "reasoner.beta",
			InputData:   json.RawMessage(`{"seed":2}`),
			OutputData:  json.RawMessage(`{"out":2}`),
			InputSize:   11,
			OutputSize:  5,
			DurationMS:  400,
			Status:      statusFailed,
			CreatedAt:   baseTime.Add(30 * time.Second),
		},
	}

	for _, exec := range executions {
		require.NoError(t, ls.StoreExecution(ctx, exec))
	}

	all, err := ls.QueryExecutions(ctx, types.ExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, all, 2)
	require.True(t, all[0].CreatedAt.After(all[1].CreatedAt) || all[0].CreatedAt.Equal(all[1].CreatedAt))

	filtered, err := ls.QueryExecutions(ctx, types.ExecutionFilters{Status: &statusRunning})
	require.NoError(t, err)
	require.Len(t, filtered, 1)
	require.Equal(t, statusRunning, filtered[0].Status)

	agentNode := "agent-B"
	filtered, err = ls.QueryExecutions(ctx, types.ExecutionFilters{AgentNodeID: &agentNode})
	require.NoError(t, err)
	require.Len(t, filtered, 1)
	require.Equal(t, "agent-B", filtered[0].AgentNodeID)

	limitResults, err := ls.QueryExecutions(ctx, types.ExecutionFilters{Limit: 1, Offset: 1})
	require.NoError(t, err)
	require.Len(t, limitResults, 1)
	require.NotEqual(t, all[0].ID, limitResults[0].ID)
}

func TestLocalStorageStoreExecutionHonoursContextCancellation(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()

	err := ls.StoreExecution(cancelledCtx, &types.AgentExecution{WorkflowID: "wf", AgentNodeID: "agent", ReasonerID: "r", CreatedAt: time.Now()})
	require.Error(t, err)
	require.Contains(t, err.Error(), "context cancelled")
}
