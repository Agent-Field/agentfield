package storage

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func strp(s string) *string { return &s }

func testWorkflow(id, sessionID, actorID, status string, started time.Time) *types.Workflow {
	name := "workflow-" + id
	return &types.Workflow{
		WorkflowID:           id,
		WorkflowName:         &name,
		WorkflowTags:         []string{"tag-" + id},
		SessionID:            &sessionID,
		ActorID:              &actorID,
		RootWorkflowID:       &id,
		TotalExecutions:      2,
		SuccessfulExecutions: 1,
		FailedExecutions:     0,
		TotalDurationMS:      42,
		Status:               status,
		StartedAt:            started,
		CreatedAt:            started,
		UpdatedAt:            started.Add(time.Minute),
	}
}

func testSession(id, actorID string, started time.Time) *types.Session {
	name := "session-" + id
	return &types.Session{
		SessionID:       id,
		ActorID:         &actorID,
		SessionName:     &name,
		RootSessionID:   &id,
		TotalWorkflows:  1,
		TotalExecutions: 2,
		TotalDurationMS: 42,
		StartedAt:       started,
		LastActivityAt:  started.Add(time.Minute),
		CreatedAt:       started,
		UpdatedAt:       started.Add(2 * time.Minute),
	}
}

func testWorkflowExecution(id, workflowID string, started time.Time) *types.WorkflowExecution {
	return &types.WorkflowExecution{
		WorkflowID:          workflowID,
		ExecutionID:         id,
		AgentFieldRequestID: "req-" + id,
		AgentNodeID:         "agent-" + id,
		ReasonerID:          "reasoner-" + id,
		Status:              string(types.ExecutionStatusPending),
		StartedAt:           started,
		CreatedAt:           started,
		UpdatedAt:           started,
	}
}

func TestLocalStorageWorkflowSessionAndAgentCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	t.Run("health checks and postgres constructor", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.HealthCheck(cancelled), "context cancelled during health check: context canceled")
		require.NoError(t, ls.HealthCheck(ctx))
		require.EqualError(t, (&LocalStorage{}).HealthCheck(ctx), "database connection is not initialized")

		pg := NewPostgresStorage(PostgresStorageConfig{Host: "localhost"})
		require.Equal(t, "postgres", pg.mode)
		require.Equal(t, VectorDistanceCosine, pg.vectorMetric)
		require.NotNil(t, pg.cache)
	})

	t.Run("workflow query and transaction helpers", func(t *testing.T) {
		alpha := testWorkflow("wf-alpha", "session-a", "actor-a", "running", now.Add(-2*time.Hour))
		beta := testWorkflow("wf-beta", "session-b", "actor-b", "succeeded", now.Add(-time.Hour))
		require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, alpha))
		require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, beta))

		sortedByName := "workflow_name"
		sortAsc := "asc"
		results, err := ls.QueryWorkflows(ctx, types.WorkflowFilters{SortBy: &sortedByName, SortOrder: &sortAsc, Limit: 1})
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Equal(t, "wf-alpha", results[0].WorkflowID)

		status := "succeeded"
		actorID := "actor-b"
		results, err = ls.QueryWorkflows(ctx, types.WorkflowFilters{Status: &status, ActorID: &actorID})
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Equal(t, "wf-beta", results[0].WorkflowID)
		require.Equal(t, []string{"tag-wf-beta"}, results[0].WorkflowTags)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err = ls.QueryWorkflows(cancelled, types.WorkflowFilters{})
		require.EqualError(t, err, "context cancelled during query workflows: context canceled")

		tx, err := ls.db.BeginTx(ctx, nil)
		require.NoError(t, err)
		defer func() { _ = tx.Rollback() }()

		gamma := testWorkflow("wf-gamma", "session-c", "actor-c", "pending", now)
		require.NoError(t, ls.executeWorkflowInsertWithTx(ctx, tx, gamma))

		exec := testWorkflowExecution("exec-gamma", gamma.WorkflowID, now)
		require.NoError(t, ls.executeWorkflowExecutionInsertWithTx(ctx, tx, exec))

		storedExec, err := ls.getWorkflowExecutionWithTx(ctx, tx, exec.ExecutionID)
		require.NoError(t, err)
		require.NotNil(t, storedExec)
		require.Equal(t, exec.ExecutionID, storedExec.ExecutionID)
	})

	t.Run("session lifecycle and queries", func(t *testing.T) {
		first := testSession("session-a", "actor-a", now.Add(-2*time.Hour))
		second := testSession("session-b", "actor-b", now.Add(-time.Hour))
		require.NoError(t, ls.CreateOrUpdateSession(ctx, first))
		require.NoError(t, ls.CreateOrUpdateSession(ctx, second))

		got, err := ls.GetSession(ctx, first.SessionID)
		require.NoError(t, err)
		require.Equal(t, first.SessionID, got.SessionID)

		actorID := "actor-b"
		results, err := ls.QuerySessions(ctx, types.SessionFilters{ActorID: &actorID, Limit: 1})
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Equal(t, second.SessionID, results[0].SessionID)

		_, err = ls.GetSession(ctx, "missing-session")
		require.EqualError(t, err, "session with ID missing-session not found")

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.CreateOrUpdateSession(cancelled, first), "context cancelled during create or update session: context canceled")
		_, err = ls.GetSession(cancelled, first.SessionID)
		require.EqualError(t, err, "context cancelled during get session: context canceled")
		_, err = ls.QuerySessions(cancelled, types.SessionFilters{})
		require.EqualError(t, err, "context cancelled during query sessions: context canceled")

		tx, err := ls.db.BeginTx(ctx, nil)
		require.NoError(t, err)
		defer func() { _ = tx.Rollback() }()
		third := testSession("session-c", "actor-c", now)
		require.NoError(t, ls.executeSessionInsertWithTx(ctx, tx, third))
	})

	t.Run("agent listings and update operations", func(t *testing.T) {
		primary := makeTestAgent("agent-shared", "")
		primary.GroupID = "group-1"
		primary.TeamID = "team-1"
		primary.LastHeartbeat = now
		primary.RegisteredAt = now
		primary.DeploymentType = ""

		versioned := makeTestAgent("agent-shared", "v2")
		versioned.GroupID = "group-1"
		versioned.TeamID = "team-1"
		versioned.LastHeartbeat = now.Add(time.Minute)
		versioned.RegisteredAt = now.Add(time.Minute)

		other := makeTestAgent("agent-other", "")
		other.GroupID = "group-2"
		other.TeamID = "team-1"
		other.LastHeartbeat = now.Add(2 * time.Minute)
		other.RegisteredAt = now.Add(2 * time.Minute)

		require.NoError(t, ls.RegisterAgent(ctx, primary))
		require.NoError(t, ls.RegisterAgent(ctx, versioned))
		require.NoError(t, ls.RegisterAgent(ctx, other))

		versions, err := ls.ListAgentVersions(ctx, "agent-shared")
		require.NoError(t, err)
		require.Len(t, versions, 1)
		require.Equal(t, "v2", versions[0].Version)

		byGroup, err := ls.ListAgentsByGroup(ctx, "group-1")
		require.NoError(t, err)
		require.Len(t, byGroup, 2)

		groups, err := ls.ListAgentGroups(ctx, "team-1")
		require.NoError(t, err)
		require.Len(t, groups, 2)

		require.True(t, isDatabaseLockError(errors.New("database is locked")))
		require.True(t, isDatabaseLockError(errors.New("SQLITE_BUSY")))
		require.False(t, isDatabaseLockError(nil))
		require.False(t, isDatabaseLockError(context.Canceled))

		require.NoError(t, ls.UpdateAgentHealth(ctx, "agent-shared", types.HealthStatusDegraded))
		agent, err := ls.GetAgent(ctx, "agent-shared")
		require.NoError(t, err)
		require.Equal(t, types.HealthStatusDegraded, agent.HealthStatus)
		require.Equal(t, "long_running", agent.DeploymentType)

		require.NoError(t, ls.UpdateAgentHealthAtomic(ctx, "agent-shared", types.HealthStatusActive, nil))
		agent, err = ls.GetAgent(ctx, "agent-shared")
		require.NoError(t, err)
		require.Equal(t, types.HealthStatusActive, agent.HealthStatus)

		expectedHeartbeat := primary.LastHeartbeat
		err = ls.UpdateAgentHealthAtomic(ctx, "agent-shared", types.HealthStatusInactive, &expectedHeartbeat)
		require.Error(t, err)

		newHeartbeat := now.Add(3 * time.Minute)
		require.NoError(t, ls.UpdateAgentHeartbeat(ctx, "agent-shared", "", newHeartbeat))
		agent, err = ls.GetAgent(ctx, "agent-shared")
		require.NoError(t, err)
		require.WithinDuration(t, newHeartbeat, agent.LastHeartbeat, time.Second)

		require.NoError(t, ls.UpdateAgentLifecycleStatus(ctx, "agent-shared", types.AgentStatusDegraded))
		agent, err = ls.GetAgent(ctx, "agent-shared")
		require.NoError(t, err)
		require.Equal(t, types.AgentStatusDegraded, agent.LifecycleStatus)

		require.NoError(t, ls.UpdateAgentVersion(ctx, "agent-other", "v9"))
		updatedVersion, err := ls.GetAgentVersion(ctx, "agent-other", "v9")
		require.NoError(t, err)
		require.Equal(t, "v9", updatedVersion.Version)

		require.NoError(t, ls.UpdateAgentTrafficWeight(ctx, "agent-shared", "v2", 55))
		versionedAgent, err := ls.GetAgentVersion(ctx, "agent-shared", "v2")
		require.NoError(t, err)
		require.Equal(t, 55, versionedAgent.TrafficWeight)
		require.EqualError(t, ls.UpdateAgentTrafficWeight(ctx, "missing", "v0", 1), "agent (id=missing, version=v0) not found")

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err = ls.ListAgentVersions(cancelled, "agent-shared")
		require.EqualError(t, err, "context cancelled during list agent versions: context canceled")
		_, err = ls.ListAgentGroups(cancelled, "team-1")
		require.EqualError(t, err, "context cancelled during list agent groups: context canceled")
		require.EqualError(t, ls.UpdateAgentHealth(cancelled, "agent-shared", types.HealthStatusActive), "context cancelled during update agent health: context canceled")
		require.EqualError(t, ls.UpdateAgentHealthAtomic(cancelled, "agent-shared", types.HealthStatusActive, nil), "context cancelled during update agent health atomic: context canceled")
		require.EqualError(t, ls.UpdateAgentHeartbeat(cancelled, "agent-shared", "", now), "context cancelled during update agent heartbeat: context canceled")
		require.EqualError(t, ls.UpdateAgentLifecycleStatus(cancelled, "agent-shared", types.AgentStatusReady), "context cancelled during update agent lifecycle status: context canceled")
		require.EqualError(t, ls.UpdateAgentVersion(cancelled, "agent-shared", "v3"), "context cancelled during update agent version: context canceled")
		require.EqualError(t, ls.UpdateAgentTrafficWeight(cancelled, "agent-shared", "", 1), "context cancelled during update traffic weight: context canceled")
	})
}
