package storage

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestExecutionRecordMutationAndQueryBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)

	exec := &types.Execution{
		ExecutionID:        "exec-record-branches",
		RunID:              "run-record-branches",
		ParentExecutionID:  branchStringPtr("exec-parent"),
		AgentNodeID:        "agent-branches",
		ReasonerID:         "reasoner-branches",
		NodeID:             "node-branches",
		Status:             "running",
		StatusReason:       strPtr("booting"),
		InputPayload:       json.RawMessage(`{"input":1}`),
		ResultPayload:      json.RawMessage(`{"result":1}`),
		InputURI:           strPtr("s3://input"),
		ResultURI:          strPtr("s3://result"),
		SessionID:          strPtr("session-branches"),
		ActorID:            strPtr("actor-branches"),
		StartedAt:          now,
		CompletedAt:        pointerTime(now.Add(time.Minute)),
		DurationMS:         branchInt64Ptr(60000),
		Notes:              []types.ExecutionNote{{Message: "created", Timestamp: now}},
	}

	require.EqualError(t, ls.CreateExecutionRecord(ctx, nil), "nil execution payload")
	require.NoError(t, ls.CreateExecutionRecord(ctx, exec))

	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{
		ExecutionID: "exec-record-branches",
		URL:         "https://example.com/webhook",
		Status:      types.ExecutionWebhookStatusPending,
	}))
	require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
		ExecutionID: "exec-record-branches",
		EventType:   "delivered",
		Status:      types.ExecutionWebhookStatusDelivered,
	}))

	require.EqualError(t, func() error {
		_, err := ls.UpdateExecutionRecord(ctx, "exec-record-branches", nil)
		return err
	}(), "nil updater")

	unchanged, err := ls.UpdateExecutionRecord(ctx, "exec-record-branches", func(current *types.Execution) (*types.Execution, error) {
		require.NotNil(t, current)
		return nil, nil
	})
	require.NoError(t, err)
	require.True(t, unchanged.WebhookRegistered)
	require.Len(t, unchanged.WebhookEvents, 1)

	_, err = ls.UpdateExecutionRecord(ctx, "exec-record-branches", func(current *types.Execution) (*types.Execution, error) {
		return nil, errors.New("boom")
	})
	require.EqualError(t, err, "boom")

	updated, err := ls.UpdateExecutionRecord(ctx, "exec-record-branches", func(current *types.Execution) (*types.Execution, error) {
		next := *current
		next.Status = "succeeded"
		next.Notes = append(next.Notes, types.ExecutionNote{Message: "completed", Timestamp: now.Add(time.Minute)})
		next.ResultPayload = json.RawMessage(`{"result":2}`)
		return &next, nil
	})
	require.NoError(t, err)
	require.Equal(t, "succeeded", updated.Status)
	require.Len(t, updated.Notes, 2)
	require.True(t, updated.WebhookRegistered)

	filter := types.ExecutionFilter{
		ExecutionID:       branchStringPtr("exec-record-branches"),
		RunID:             branchStringPtr("run-record-branches"),
		ParentExecutionID: branchStringPtr("exec-parent"),
		AgentNodeID:       branchStringPtr("agent-branches"),
		ReasonerID:        branchStringPtr("reasoner-branches"),
		Status:            branchStringPtr("succeeded"),
		SessionID:         branchStringPtr("session-branches"),
		ActorID:           branchStringPtr("actor-branches"),
		StartTime:         pointerTime(now.Add(-time.Minute)),
		EndTime:           pointerTime(now.Add(2 * time.Minute)),
		ExcludePayloads: true,
		SortBy:          "execution_id",
		SortDescending:  false,
		Limit:           1,
	}
	results, err := ls.QueryExecutionRecords(ctx, filter)
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Empty(t, results[0].InputPayload)
	require.Empty(t, results[0].ResultPayload)
	require.True(t, results[0].WebhookRegistered)

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.QueryExecutionRecords(cancelled, types.ExecutionFilter{})
	require.ErrorContains(t, err, "query executions")

	_, err = ls.db.ExecContext(ctx, `UPDATE executions SET notes = ? WHERE execution_id = ?`, []byte(`{"broken":`), "exec-record-branches")
	require.NoError(t, err)
	_, err = ls.QueryExecutionRecords(ctx, types.ExecutionFilter{ExecutionID: branchStringPtr("exec-record-branches")})
	require.ErrorContains(t, err, "unmarshal notes")
}

func branchStringPtr(value string) *string {
	return &value
}

func branchInt64Ptr(value int64) *int64 {
	return &value
}
