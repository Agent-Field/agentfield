package storage

import (
	"context"
	"database/sql"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type stubResult struct {
	rows int64
	err  error
}

func (r stubResult) LastInsertId() (int64, error) { return 0, nil }
func (r stubResult) RowsAffected() (int64, error) { return r.rows, r.err }

type stubDBTX struct {
	execResult sql.Result
	execErr    error
}

func (s stubDBTX) ExecContext(context.Context, string, ...interface{}) (sql.Result, error) {
	return s.execResult, s.execErr
}

func (s stubDBTX) Exec(string, ...interface{}) (sql.Result, error) { return s.execResult, s.execErr }
func (s stubDBTX) QueryRowContext(context.Context, string, ...interface{}) *sql.Row {
	return nil
}
func (s stubDBTX) QueryContext(context.Context, string, ...interface{}) (*sql.Rows, error) {
	return nil, nil
}
func (s stubDBTX) Query(string, ...interface{}) (*sql.Rows, error) { return nil, nil }
func (s stubDBTX) QueryRow(string, ...interface{}) *sql.Row        { return nil }

func testAgentNode(now time.Time) *types.AgentNode {
	return &types.AgentNode{
		ID:         "agent-storage93",
		GroupID:    "group-storage93",
		TeamID:     "team-storage93",
		BaseURL:    "https://agent-storage93.example",
		Version:    "",
		Reasoners:  []types.ReasonerDefinition{{ID: "reasoner.storage93"}},
		Skills:     []types.SkillDefinition{{ID: "skill.storage93"}},
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   now,
		RegisteredAt:    now,
		Metadata: types.AgentMetadata{
			Custom: map[string]interface{}{"tier": "gold"},
		},
	}
}

func TestStorage93DeleteHelpersReturnExpectedErrors(t *testing.T) {
	ls := &LocalStorage{}
	ctx := context.Background()

	tests := []struct {
		name    string
		run     func(DBTX) (int, error)
		empty   func() (int, error)
	}{
		{
			name:  "execution vcs",
			run:   func(tx DBTX) (int, error) { return ls.deleteExecutionVCs(ctx, tx, []string{"wf-1"}) },
			empty: func() (int, error) { return ls.deleteExecutionVCs(ctx, stubDBTX{}, nil) },
		},
		{
			name:  "workflow vcs",
			run:   func(tx DBTX) (int, error) { return ls.deleteWorkflowVCs(ctx, tx, []string{"wf-1"}) },
			empty: func() (int, error) { return ls.deleteWorkflowVCs(ctx, stubDBTX{}, nil) },
		},
		{
			name:  "execution webhook events",
			run:   func(tx DBTX) (int, error) { return ls.deleteExecutionWebhookEvents(ctx, tx, []string{"run-1"}) },
			empty: func() (int, error) { return ls.deleteExecutionWebhookEvents(ctx, stubDBTX{}, nil) },
		},
		{
			name:  "execution webhooks",
			run:   func(tx DBTX) (int, error) { return ls.deleteExecutionWebhooks(ctx, tx, []string{"run-1"}) },
			empty: func() (int, error) { return ls.deleteExecutionWebhooks(ctx, stubDBTX{}, nil) },
		},
		{
			name:  "executions",
			run:   func(tx DBTX) (int, error) { return ls.deleteExecutions(ctx, tx, []string{"run-1"}) },
			empty: func() (int, error) { return ls.deleteExecutions(ctx, stubDBTX{}, nil) },
		},
		{
			name:  "workflow executions",
			run:   func(tx DBTX) (int, error) { return ls.deleteWorkflowExecutions(ctx, tx, []string{"wf-1"}, []string{"run-1"}) },
			empty: func() (int, error) { return ls.deleteWorkflowExecutions(ctx, stubDBTX{}, nil, nil) },
		},
		{
			name:  "workflow runs",
			run:   func(tx DBTX) (int, error) { return ls.deleteWorkflowRuns(ctx, tx, "wf-1", []string{"wf-2"}, []string{"run-1"}) },
			empty: func() (int, error) { return ls.deleteWorkflowRuns(ctx, stubDBTX{}, "", nil, nil) },
		},
		{
			name:  "workflows",
			run:   func(tx DBTX) (int, error) { return ls.deleteWorkflows(ctx, tx, []string{"wf-1"}) },
			empty: func() (int, error) { return ls.deleteWorkflows(ctx, stubDBTX{}, nil) },
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, err := tc.empty()
			require.NoError(t, err)
			require.Zero(t, n)

			_, err = tc.run(stubDBTX{execErr: errors.New("boom")})
			require.EqualError(t, err, "boom")

			_, err = tc.run(stubDBTX{execResult: stubResult{err: errors.New("rows boom")}})
			require.EqualError(t, err, "rows boom")

			n, err = tc.run(stubDBTX{execResult: stubResult{rows: 3}})
			require.NoError(t, err)
			require.Equal(t, 3, n)
		})
	}
}

func TestStorage93AgentAndQueryBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	defaultAgent := testAgentNode(now)
	defaultAgent.DeploymentType = ""
	require.NoError(t, ls.RegisterAgent(ctx, defaultAgent))

	versioned := testAgentNode(now.Add(time.Minute))
	versioned.ID = "agent-storage93-serverless"
	versioned.Version = "v2"
	versioned.DeploymentType = ""
	versioned.BaseURL = "https://serverless.example/root/"
	versioned.Metadata = types.AgentMetadata{Custom: map[string]interface{}{"serverless": "true"}}
	require.NoError(t, ls.RegisterAgent(ctx, versioned))

	got, err := ls.GetAgent(ctx, defaultAgent.ID)
	require.NoError(t, err)
	require.Equal(t, "long_running", got.DeploymentType)
	require.Equal(t, 100, got.TrafficWeight)

	health := types.HealthStatusActive
	agents, err := ls.ListAgents(ctx, types.AgentFilters{
		TeamID:       &defaultAgent.TeamID,
		GroupID:      &defaultAgent.GroupID,
		HealthStatus: &health,
	})
	require.NoError(t, err)
	require.Len(t, agents, 2)

	byVersion, err := ls.GetAgentVersion(ctx, versioned.ID, versioned.Version)
	_, err = ls.db.ExecContext(ctx, `UPDATE agent_nodes SET deployment_type = '' WHERE id = ? AND version = ?`, versioned.ID, versioned.Version)
	require.NoError(t, err)
	byVersion, err = ls.GetAgentVersion(ctx, versioned.ID, versioned.Version)
	require.NoError(t, err)
	require.Equal(t, "serverless", byVersion.DeploymentType)
	require.NotNil(t, byVersion.InvocationURL)
	require.Equal(t, "https://serverless.example/root/execute", *byVersion.InvocationURL)

	groups, err := ls.ListAgentGroups(ctx, defaultAgent.TeamID)
	require.NoError(t, err)
	require.Len(t, groups, 1)
	require.ElementsMatch(t, []string{"", "v2"}, groups[0].Versions)

	require.NoError(t, ls.UpdateAgentLifecycleStatus(ctx, defaultAgent.ID, types.AgentStatusOffline))
	require.NoError(t, ls.UpdateAgentVersion(ctx, defaultAgent.ID, "v1"))
	require.NoError(t, ls.UpdateAgentTrafficWeight(ctx, defaultAgent.ID, "v1", 25))

	updated, err := ls.GetAgentVersion(ctx, defaultAgent.ID, "v1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusOffline, updated.LifecycleStatus)
	require.Equal(t, 25, updated.TrafficWeight)

	require.NoError(t, ls.DeleteAgentVersion(ctx, defaultAgent.ID, "v1"))
	_, err = ls.GetAgentVersion(ctx, defaultAgent.ID, "v1")
	require.ErrorContains(t, err, "not found")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	require.EqualError(t, ls.DeleteAgentVersion(cancelled, defaultAgent.ID, versioned.Version), "context cancelled during delete agent version: context canceled")
	require.EqualError(t, ls.UpdateAgentVersion(cancelled, defaultAgent.ID, "v3"), "context cancelled during update agent version: context canceled")
	require.EqualError(t, ls.UpdateAgentLifecycleStatus(cancelled, defaultAgent.ID, types.AgentStatusReady), "context cancelled during update agent lifecycle status: context canceled")
	require.EqualError(t, ls.UpdateAgentTrafficWeight(cancelled, defaultAgent.ID, versioned.Version, 50), "context cancelled during update traffic weight: context canceled")
	require.EqualError(t, ls.UpdateAgentTrafficWeight(ctx, defaultAgent.ID, "missing", 50), "agent (id=agent-storage93, version=missing) not found")

	corruptCases := []struct {
		name    string
		column  string
		payload string
		msg     string
	}{
		{name: "reasoners", column: "reasoners", payload: "{", msg: "failed to unmarshal agent reasoners"},
		{name: "skills", column: "skills", payload: "{", msg: "failed to unmarshal agent skills"},
		{name: "communication config", column: "communication_config", payload: "{", msg: "failed to unmarshal agent communication config"},
		{name: "features", column: "features", payload: "{", msg: "failed to unmarshal agent features"},
		{name: "metadata", column: "metadata", payload: "{", msg: "failed to unmarshal agent metadata"},
		{name: "proposed tags", column: "proposed_tags", payload: "{", msg: "failed to unmarshal agent proposed tags"},
		{name: "approved tags", column: "approved_tags", payload: "{", msg: "failed to unmarshal agent approved tags"},
	}

	for _, tc := range corruptCases {
		t.Run(tc.name, func(t *testing.T) {
			agent := testAgentNode(now.Add(2 * time.Minute))
			agent.ID = "agent-corrupt-" + strings.ReplaceAll(tc.column, "_", "-")
			require.NoError(t, ls.RegisterAgent(ctx, agent))
			_, err := ls.db.ExecContext(ctx, "UPDATE agent_nodes SET "+tc.column+" = ? WHERE id = ? AND version = ''", tc.payload, agent.ID)
			require.NoError(t, err)
			_, err = ls.GetAgent(ctx, agent.ID)
			require.ErrorContains(t, err, tc.msg)
		})
	}

	search := "storage93"
	sortBy := "started_at"
	sortOrder := "asc"
	sessionID := "session-storage93"
	actorID := "actor-storage93"
	parentID := "exec-parent-storage93"
	status := string(types.ExecutionStatusRunning)
	approvalRequestID := "approval-123"
	startTime := now.Add(-time.Hour)
	endTime := now.Add(time.Hour)

	for i, executionID := range []string{"exec-parent-storage93", "exec-child-storage93"} {
		exec := &types.WorkflowExecution{
			WorkflowID:          "wf-storage93",
			ExecutionID:         executionID,
			AgentFieldRequestID: "req-storage93-" + executionID,
			SessionID:           &sessionID,
			ActorID:             &actorID,
			AgentNodeID:         defaultAgent.ID,
			ReasonerID:          "reasoner.storage93",
			Status:              status,
			StartedAt:           now.Add(time.Duration(i) * time.Minute),
			CreatedAt:           now.Add(time.Duration(i) * time.Minute),
			UpdatedAt:           now.Add(time.Duration(i) * time.Minute),
			WorkflowName:        ptrString("Storage93 Workflow"),
		}
		if i == 1 {
			exec.ParentExecutionID = &parentID
			exec.ApprovalRequestID = &approvalRequestID
		}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
	}

	ls.ftsEnabled = false
	results, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{
		SessionID:         &sessionID,
		ActorID:           &actorID,
		ParentExecutionID: &parentID,
		AgentNodeID:       &defaultAgent.ID,
		Status:            &status,
		ApprovalRequestID: &approvalRequestID,
		StartTime:         &startTime,
		EndTime:           &endTime,
		Search:            &search,
		SortBy:            &sortBy,
		SortOrder:         &sortOrder,
		Limit:             10,
		Offset:            -1,
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "exec-child-storage93", results[0].ExecutionID)
}

func TestStorage93ExecutionRecordAndWebhookBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	sessionID := "session-run93"
	actorID := "actor-run93"
	rootRun := "run-93-root"
	childRun := "run-93-child"

	root := &types.Execution{
		ExecutionID: "exec-run93-root",
		RunID:       rootRun,
		AgentNodeID: "agent-93",
		ReasonerID:  "reasoner-93",
		NodeID:      "node-93",
		Status:      string(types.ExecutionStatusRunning),
		SessionID:   &sessionID,
		ActorID:     &actorID,
		StartedAt:   now.Add(-10 * time.Minute),
		CreatedAt:   now.Add(-10 * time.Minute),
		UpdatedAt:   now.Add(-9 * time.Minute),
	}
	child := &types.Execution{
		ExecutionID:       "exec-run93-child",
		RunID:             rootRun,
		ParentExecutionID: &root.ExecutionID,
		AgentNodeID:       "agent-93",
		ReasonerID:        "reasoner-93-child",
		NodeID:            "node-93-child",
		Status:            string(types.ExecutionStatusPending),
		SessionID:         &sessionID,
		ActorID:           &actorID,
		StartedAt:         now.Add(-8 * time.Minute),
		CreatedAt:         now.Add(-8 * time.Minute),
		UpdatedAt:         now.Add(-7 * time.Minute),
	}
	other := &types.Execution{
		ExecutionID: "exec-run93-other",
		RunID:       childRun,
		AgentNodeID: "agent-93-other",
		ReasonerID:  "reasoner-93-other",
		NodeID:      "node-93-other",
		Status:      string(types.ExecutionStatusSucceeded),
		StartedAt:   now.Add(-6 * time.Minute),
		CompletedAt: ptrTime(now.Add(-5 * time.Minute)),
		CreatedAt:   now.Add(-6 * time.Minute),
		UpdatedAt:   now.Add(-5 * time.Minute),
	}

	for _, exec := range []*types.Execution{root, child, other} {
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	}

	hookSecret := "secret-93"
	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{
		ExecutionID: root.ExecutionID,
		URL:         "https://example.com/hook",
		Secret:      &hookSecret,
		Headers:     map[string]string{"X-Test": "storage93"},
		Status:      types.ExecutionWebhookStatusPending,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))

	registered, err := ls.ListExecutionWebhooksRegistered(ctx, []string{"", root.ExecutionID, root.ExecutionID, "missing"})
	require.NoError(t, err)
	require.True(t, registered[root.ExecutionID])
	require.False(t, registered["missing"])

	_, err = ls.db.ExecContext(ctx, `UPDATE execution_webhooks SET headers = ? WHERE execution_id = ?`, "{", root.ExecutionID)
	require.NoError(t, err)
	_, err = ls.ListDueExecutionWebhooks(ctx, -1)
	require.ErrorContains(t, err, "unmarshal webhook headers")

	_, err = ls.db.ExecContext(ctx, `UPDATE execution_webhooks SET headers = ?, next_attempt_at = NULL WHERE execution_id = ?`, `{"X-Test":"storage93"}`, root.ExecutionID)
	require.NoError(t, err)
	due, err := ls.ListDueExecutionWebhooks(ctx, -1)
	require.NoError(t, err)
	require.Len(t, due, 1)

	search := "run-93"
	status := string(types.ExecutionStatusRunning)
	startTime := now.Add(-20 * time.Minute)
	endTime := now
	sortBy := "status"
	summaries, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{
		SessionID:       &sessionID,
		ActorID:         &actorID,
		StartTime:       &startTime,
		EndTime:         &endTime,
		Search:          &search,
		SortBy:          sortBy,
		SortDescending:  false,
		Limit:           10,
		Offset:          -1,
	})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, summaries, 1)
	require.Equal(t, rootRun, summaries[0].RunID)
	require.Equal(t, 2, summaries[0].TotalExecutions)
	require.Equal(t, 2, summaries[0].ActiveExecutions)
	require.Equal(t, 1, summaries[0].MaxDepth)
	require.NotNil(t, summaries[0].RootExecutionID)
	require.Equal(t, root.ExecutionID, *summaries[0].RootExecutionID)

	_, err = ls.QueryExecutionRecords(ctx, types.ExecutionFilter{
		RunID:       &rootRun,
		SessionID:   &sessionID,
		ActorID:     &actorID,
		AgentNodeID: &root.AgentNodeID,
		ReasonerID:  &root.ReasonerID,
		Status:      &status,
		StartTime:   &startTime,
		EndTime:     &endTime,
		Search:      &search,
		Limit:       10,
		Offset:      -1,
	})
	require.NoError(t, err)

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.MarkStaleExecutions(cancelled, time.Hour, 1)
	require.EqualError(t, err, "context cancelled before marking stale executions: context canceled")
	_, err = ls.MarkStaleWorkflowExecutions(cancelled, time.Hour, 1)
	require.EqualError(t, err, "context cancelled before marking stale workflow executions: context canceled")
	_, err = ls.RetryStaleWorkflowExecutions(cancelled, time.Hour, 1, 1)
	require.EqualError(t, err, "context cancelled: context canceled")
}

func TestStorage93EventsVectorAndObservabilityBranches(t *testing.T) {
	t.Run("events extra branches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC()

		putStoredEvent(t, ls, &types.MemoryChangeEvent{
			ID:        "evt-old",
			Type:      "memory_changed",
			Timestamp: now.Add(-72 * time.Hour),
			Scope:     "agent",
			ScopeID:   "agent-93",
			Key:       "memory/old",
			Action:    "set",
		})
		putStoredEvent(t, ls, &types.MemoryChangeEvent{
			ID:        "evt-new",
			Type:      "memory_changed",
			Timestamp: now,
			Scope:     "agent",
			ScopeID:   "agent-93",
			Key:       "memory/new",
			Action:    "set",
		})

		badPattern := "["
		history, err := ls.GetEventHistory(ctx, types.EventFilter{Patterns: []string{badPattern}})
		require.NoError(t, err)
		require.Empty(t, history)

		ls.cleanupExpiredEvents()
		history, err = ls.GetEventHistory(ctx, types.EventFilter{})
		require.NoError(t, err)
		require.Len(t, history, 1)

		var noKV LocalStorage
		noKV.cleanupExpiredEvents()
	})

	t.Run("vector store extra branches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		store, ok := ls.vectorStore.(*sqliteVectorStore)
		require.True(t, ok)

		record := &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "scope-93",
			Key:       "doc-93",
			Embedding: []float32{1, 2},
			Metadata:  map[string]interface{}{"bad": math.NaN()},
		}
		err := store.Set(ctx, record)
		require.ErrorContains(t, err, "marshal metadata")

		got, err := store.Get(ctx, "session", "scope-93", "missing")
		require.NoError(t, err)
		require.Nil(t, got)

		require.NoError(t, store.Set(ctx, &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "scope-93",
			Key:       "doc-good",
			Embedding: []float32{1, 2},
			Metadata:  map[string]interface{}{"kind": "doc"},
		}))
		results, err := store.Search(ctx, "session", "scope-93", []float32{1, 2, 3}, 10, map[string]interface{}{"kind": "doc"})
		require.NoError(t, err)
		require.Empty(t, results)

		deleted, err := store.DeleteByPrefix(ctx, "session", "scope-93", "missing-")
		require.NoError(t, err)
		require.Zero(t, deleted)
	})

	t.Run("observability extra branches", func(t *testing.T) {
		ls, ctx := setupObservabilityTestStorage(t)

		secret := ""
		require.NoError(t, ls.SetObservabilityWebhook(ctx, &types.ObservabilityWebhookConfig{
			URL:     "https://example.com/obs",
			Secret:  &secret,
			Enabled: true,
		}))
		got, err := ls.GetObservabilityWebhook(ctx)
		require.NoError(t, err)
		require.Nil(t, got.Secret)

		_, err = ls.db.ExecContext(ctx, `UPDATE observability_webhooks SET headers = ? WHERE id = ?`, "{", observabilityWebhookGlobalID)
		require.NoError(t, err)
		_, err = ls.GetObservabilityWebhook(ctx)
		require.ErrorContains(t, err, "unmarshal observability webhook headers")

		err = ls.AddToDeadLetterQueue(ctx, &types.ObservabilityEvent{
			EventType:   "execution.failed",
			EventSource: "execution",
			Timestamp:   "not-a-time",
			Data:        map[string]interface{}{"bad": math.NaN()},
		}, "boom", 1)
		require.ErrorContains(t, err, "marshal event payload")

		require.NoError(t, ls.AddToDeadLetterQueue(ctx, &types.ObservabilityEvent{
			EventType:   "execution.failed",
			EventSource: "execution",
			Timestamp:   "not-a-time",
			Data:        map[string]interface{}{"ok": true},
		}, "boom", 1))

		count, err := ls.GetDeadLetterQueueCount(ctx)
		require.NoError(t, err)
		require.EqualValues(t, 1, count)

		entries, err := ls.GetDeadLetterQueue(ctx, -1, -1)
		require.NoError(t, err)
		require.Len(t, entries, 1)
		require.NoError(t, ls.DeleteFromDeadLetterQueue(ctx, []int64{entries[0].ID}))
		require.NoError(t, ls.ClearDeadLetterQueue(ctx))
	})
}
