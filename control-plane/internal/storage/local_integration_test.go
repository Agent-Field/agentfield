package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// setupIntegrationStorage creates a LocalStorage backed by a real SQLite + BoltDB in a temp dir.
// It skips if the sqlite3 binary does not include FTS5.
func setupIntegrationStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()
	ls, ctx := setupLocalStorage(t) // reuse existing helper from local_query_test.go
	return ls, ctx
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent CRUD
// ─────────────────────────────────────────────────────────────────────────────

func TestIntegration_AgentCRUD_RegisterAndGet(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	agent := makeTestAgent("agent-001", "")
	require.NoError(t, ls.RegisterAgent(ctx, agent))

	got, err := ls.GetAgent(ctx, "agent-001")
	require.NoError(t, err)
	require.NotNil(t, got)
	require.Equal(t, "agent-001", got.ID)
	require.Equal(t, agent.BaseURL, got.BaseURL)
	require.Equal(t, agent.GroupID, got.GroupID)
	require.Equal(t, agent.TeamID, got.TeamID)
}

func TestIntegration_AgentCRUD_RegisterSameIDUpdates(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	agent := makeTestAgent("agent-upsert", "")
	require.NoError(t, ls.RegisterAgent(ctx, agent))

	// Re-register with a different base URL — same (id, version) key, should upsert.
	agent.BaseURL = "http://updated-host:9999"
	require.NoError(t, ls.RegisterAgent(ctx, agent))

	got, err := ls.GetAgent(ctx, "agent-upsert")
	require.NoError(t, err)
	require.Equal(t, "http://updated-host:9999", got.BaseURL)
}

func TestIntegration_AgentCRUD_ListAgents(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("list-agent-%d", i)
		require.NoError(t, ls.RegisterAgent(ctx, makeTestAgent(id, "")))
	}

	agents, err := ls.ListAgents(ctx, types.AgentFilters{})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(agents), 3, "expected at least 3 agents")
}

func TestIntegration_AgentCRUD_DeleteVersion(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	// Register a versioned agent.
	agent := makeTestAgent("agent-del", "v1.0")
	require.NoError(t, ls.RegisterAgent(ctx, agent))

	// Verify it exists.
	got, err := ls.GetAgentVersion(ctx, "agent-del", "v1.0")
	require.NoError(t, err)
	require.NotNil(t, got)

	// Delete it.
	require.NoError(t, ls.DeleteAgentVersion(ctx, "agent-del", "v1.0"))

	// Verify it is gone.
	after, err := ls.GetAgentVersion(ctx, "agent-del", "v1.0")
	// GetAgentVersion returns nil, nil on not-found (same pattern as GetWorkflowExecution)
	// Accept both: nil result without error, or a "not found" error.
	if err == nil {
		require.Nil(t, after, "agent should be nil after deletion")
	} else {
		require.True(t, strings.Contains(err.Error(), "not found") ||
			strings.Contains(err.Error(), "no rows"),
			"unexpected error: %v", err)
	}
}

func TestIntegration_AgentCRUD_GetNonExistent(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	// GetAgent should return nil, nil when the agent is not found
	// (same convention as GetWorkflowExecution).
	got, err := ls.GetAgent(ctx, "non-existent-agent-xyz")
	if err == nil {
		require.Nil(t, got, "expected nil agent for unknown ID")
	} else {
		// Some implementations return an error for not-found.
		require.True(t,
			strings.Contains(err.Error(), "not found") ||
				strings.Contains(err.Error(), "no rows"),
			"unexpected error: %v", err)
	}
}

func TestIntegration_AgentCRUD_EmptyDatabaseListReturnsEmpty(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	agents, err := ls.ListAgents(ctx, types.AgentFilters{})
	require.NoError(t, err)
	require.NotNil(t, agents, "empty list should not be nil")
	require.Empty(t, agents)
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution Records (AgentExecution)
// ─────────────────────────────────────────────────────────────────────────────

func TestIntegration_Execution_StoreAndGet(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	exec := makeTestExecution("wf-exec-001", "agent-a", "succeeded")
	require.NoError(t, ls.StoreExecution(ctx, exec))
	require.Greater(t, exec.ID, int64(0), "ID should be set after store")

	got, err := ls.GetExecution(ctx, exec.ID)
	require.NoError(t, err)
	require.NotNil(t, got)
	require.Equal(t, exec.ID, got.ID)
	require.Equal(t, "wf-exec-001", got.WorkflowID)
	require.Equal(t, "agent-a", got.AgentNodeID)
	require.Equal(t, "succeeded", got.Status)
}

func TestIntegration_Execution_QueryByAgentID(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-q1", "query-agent", "running")))
	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-q2", "query-agent", "failed")))
	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-q3", "other-agent", "succeeded")))

	agentID := "query-agent"
	results, err := ls.QueryExecutions(ctx, types.ExecutionFilters{AgentNodeID: &agentID})
	require.NoError(t, err)
	require.Len(t, results, 2, "should return only executions for query-agent")
	for _, r := range results {
		require.Equal(t, "query-agent", r.AgentNodeID)
	}
}

func TestIntegration_Execution_QueryByStatus(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-s1", "status-agent", "running")))
	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-s2", "status-agent", "succeeded")))
	require.NoError(t, ls.StoreExecution(ctx, makeTestExecution("wf-s3", "status-agent", "succeeded")))

	status := "succeeded"
	results, err := ls.QueryExecutions(ctx, types.ExecutionFilters{Status: &status})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(results), 2)
	for _, r := range results {
		require.Equal(t, "succeeded", r.Status)
	}
}

func TestIntegration_Execution_QueryWithPagination(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	workflowID := "wf-page"
	for i := 0; i < 5; i++ {
		exec := makeTestExecution(workflowID, "page-agent", "succeeded")
		require.NoError(t, ls.StoreExecution(ctx, exec))
	}

	page1, err := ls.QueryExecutions(ctx, types.ExecutionFilters{
		WorkflowID: &workflowID,
		Limit:      3,
		Offset:     0,
	})
	require.NoError(t, err)
	require.Len(t, page1, 3)

	page2, err := ls.QueryExecutions(ctx, types.ExecutionFilters{
		WorkflowID: &workflowID,
		Limit:      3,
		Offset:     3,
	})
	require.NoError(t, err)
	require.Len(t, page2, 2)
}

func TestIntegration_Execution_EmptyQueryReturnsEmptySlice(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	wfID := "non-existent-workflow"
	results, err := ls.QueryExecutions(ctx, types.ExecutionFilters{WorkflowID: &wfID})
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Empty(t, results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Executions
// ─────────────────────────────────────────────────────────────────────────────

func TestIntegration_WorkflowExecution_StoreAndGet(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	now := time.Now().UTC().Truncate(time.Millisecond)
	runID := "run-wf-001"
	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "wf-root-001",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	exec := makeTestWorkflowExecution("wf-root-001", "exec-wf-001", runID, "agent-wf")
	require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))

	got, err := ls.GetWorkflowExecution(ctx, "exec-wf-001")
	require.NoError(t, err)
	require.NotNil(t, got)
	require.Equal(t, "exec-wf-001", got.ExecutionID)
	require.Equal(t, "wf-root-001", got.WorkflowID)
	require.Equal(t, "agent-wf", got.AgentNodeID)
	require.NotNil(t, got.RunID)
	require.Equal(t, runID, *got.RunID)
}

func TestIntegration_WorkflowExecution_QueryByWorkflowID(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	now := time.Now().UTC()
	runID := "run-query-wf"
	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "query-wf-001",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	for i := 0; i < 3; i++ {
		exec := makeTestWorkflowExecution("query-wf-001", fmt.Sprintf("exec-q-%d", i), runID, "wf-agent")
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
	}

	wfID := "query-wf-001"
	results, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{WorkflowID: &wfID})
	require.NoError(t, err)
	require.Len(t, results, 3)
	for _, r := range results {
		require.Equal(t, "query-wf-001", r.WorkflowID)
	}
}

func TestIntegration_WorkflowExecution_UpdateStatus(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	now := time.Now().UTC()
	runID := "run-update-wf"
	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "update-wf-001",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	exec := makeTestWorkflowExecution("update-wf-001", "exec-update-001", runID, "update-agent")
	exec.Status = string(types.ExecutionStatusPending)
	require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))

	// pending → running (valid transition)
	err := ls.UpdateWorkflowExecution(ctx, "exec-update-001", func(e *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		e.Status = string(types.ExecutionStatusRunning)
		return e, nil
	})
	require.NoError(t, err)

	// running → succeeded (valid transition)
	err = ls.UpdateWorkflowExecution(ctx, "exec-update-001", func(e *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		e.Status = string(types.ExecutionStatusSucceeded)
		return e, nil
	})
	require.NoError(t, err)

	updated, err := ls.GetWorkflowExecution(ctx, "exec-update-001")
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, string(types.ExecutionStatusSucceeded), updated.Status)
}

func TestIntegration_WorkflowExecution_GetNonExistent(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	got, err := ls.GetWorkflowExecution(ctx, "does-not-exist-xyz")
	// Convention from the codebase: returns nil, nil for not-found
	require.NoError(t, err)
	require.Nil(t, got)
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory Operations
// ─────────────────────────────────────────────────────────────────────────────

func TestIntegration_Memory_SetAndGetRoundTrip(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	scopes := []string{"global", "session", "workflow", "actor"}
	for _, scope := range scopes {
		t.Run(scope, func(t *testing.T) {
			mem := makeTestMemory(scope, "scope-id-1", "key-1", `{"value":"hello"}`)
			require.NoError(t, ls.SetMemory(ctx, mem))

			got, err := ls.GetMemory(ctx, scope, "scope-id-1", "key-1")
			require.NoError(t, err)
			require.NotNil(t, got)
			require.Equal(t, scope, got.Scope)
			require.Equal(t, "scope-id-1", got.ScopeID)
			require.Equal(t, "key-1", got.Key)
			require.Equal(t, json.RawMessage(`{"value":"hello"}`), got.Data)
		})
	}
}

func TestIntegration_Memory_ScopeIsolation(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	// Set a key in "global" scope.
	mem := makeTestMemory("global", "iso-scope-id", "iso-key", `{"data":"in-global"}`)
	require.NoError(t, ls.SetMemory(ctx, mem))

	// Getting the same key from a different scope should fail.
	_, err := ls.GetMemory(ctx, "session", "iso-scope-id", "iso-key")
	require.Error(t, err, "reading a key from a different scope should return an error")
}

func TestIntegration_Memory_DeleteThenGetReturnsError(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	mem := makeTestMemory("global", "del-scope-id", "del-key", `{"x":1}`)
	require.NoError(t, ls.SetMemory(ctx, mem))

	// Confirm it exists.
	got, err := ls.GetMemory(ctx, "global", "del-scope-id", "del-key")
	require.NoError(t, err)
	require.NotNil(t, got)

	// Delete it.
	require.NoError(t, ls.DeleteMemory(ctx, "global", "del-scope-id", "del-key"))

	// Should now be gone.
	_, err = ls.GetMemory(ctx, "global", "del-scope-id", "del-key")
	require.Error(t, err, "get after delete should return an error")
}

func TestIntegration_Memory_ListByScope(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	scopeID := "list-scope-id"
	for i := 0; i < 4; i++ {
		key := fmt.Sprintf("list-key-%d", i)
		mem := makeTestMemory("session", scopeID, key, fmt.Sprintf(`{"i":%d}`, i))
		require.NoError(t, ls.SetMemory(ctx, mem))
	}

	memories, err := ls.ListMemory(ctx, "session", scopeID)
	require.NoError(t, err)
	require.Len(t, memories, 4)
}

func TestIntegration_Memory_OverwriteReturnsLatest(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	mem1 := makeTestMemory("workflow", "ow-scope", "ow-key", `{"v":1}`)
	require.NoError(t, ls.SetMemory(ctx, mem1))

	mem2 := makeTestMemory("workflow", "ow-scope", "ow-key", `{"v":999}`)
	require.NoError(t, ls.SetMemory(ctx, mem2))

	got, err := ls.GetMemory(ctx, "workflow", "ow-scope", "ow-key")
	require.NoError(t, err)
	require.Equal(t, json.RawMessage(`{"v":999}`), got.Data)
}

func TestIntegration_Memory_EmptyListReturnsEmpty(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	memories, err := ls.ListMemory(ctx, "global", "no-such-scope-id")
	require.NoError(t, err)
	require.NotNil(t, memories)
	require.Empty(t, memories)
}

func TestIntegration_Memory_DifferentScopeIDsAreIsolated(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	mem1 := makeTestMemory("actor", "sid-A", "key-common", `{"owner":"A"}`)
	mem2 := makeTestMemory("actor", "sid-B", "key-common", `{"owner":"B"}`)
	require.NoError(t, ls.SetMemory(ctx, mem1))
	require.NoError(t, ls.SetMemory(ctx, mem2))

	gotA, err := ls.GetMemory(ctx, "actor", "sid-A", "key-common")
	require.NoError(t, err)
	require.Equal(t, json.RawMessage(`{"owner":"A"}`), gotA.Data)

	gotB, err := ls.GetMemory(ctx, "actor", "sid-B", "key-common")
	require.NoError(t, err)
	require.Equal(t, json.RawMessage(`{"owner":"B"}`), gotB.Data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariant / Invariant Properties
// ─────────────────────────────────────────────────────────────────────────────

func TestIntegration_Invariants_EmptyDBReturnsEmptyNotNil(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	agents, err := ls.ListAgents(ctx, types.AgentFilters{})
	require.NoError(t, err)
	require.NotNil(t, agents)

	wfID := "empty-wf"
	execs, err := ls.QueryExecutions(ctx, types.ExecutionFilters{WorkflowID: &wfID})
	require.NoError(t, err)
	require.NotNil(t, execs)

	wfExecs, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{WorkflowID: &wfID})
	require.NoError(t, err)
	require.NotNil(t, wfExecs)
}

func TestIntegration_Invariants_StoredTimestampsAreUTC(t *testing.T) {
	ls, ctx := setupIntegrationStorage(t)

	exec := makeTestExecution("wf-ts-check", "ts-agent", "succeeded")
	require.NoError(t, ls.StoreExecution(ctx, exec))

	got, err := ls.GetExecution(ctx, exec.ID)
	require.NoError(t, err)
	require.Equal(t, time.UTC, got.CreatedAt.Location(),
		"CreatedAt should be in UTC, got %v", got.CreatedAt.Location())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func makeTestAgent(id, version string) *types.AgentNode {
	now := time.Now().UTC()
	return &types.AgentNode{
		ID:              id,
		Version:         version,
		GroupID:         "test-group",
		TeamID:          "test-team",
		BaseURL:         "http://localhost:8080",
		DeploymentType:  "long_running",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   now,
		RegisteredAt:    now,
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}
}

func makeTestExecution(workflowID, agentNodeID, status string) *types.AgentExecution {
	now := time.Now().UTC()
	return &types.AgentExecution{
		WorkflowID:  workflowID,
		AgentNodeID: agentNodeID,
		ReasonerID:  "reasoner.test",
		InputData:   json.RawMessage(`{"input":"test"}`),
		OutputData:  json.RawMessage(`{"output":"test"}`),
		InputSize:   10,
		OutputSize:  10,
		DurationMS:  100,
		Status:      status,
		CreatedAt:   now,
	}
}

func makeTestWorkflowExecution(workflowID, executionID, runID, agentNodeID string) *types.WorkflowExecution {
	now := time.Now().UTC()
	runIDCopy := runID
	return &types.WorkflowExecution{
		WorkflowID:          workflowID,
		ExecutionID:         executionID,
		AgentFieldRequestID: "req-" + executionID,
		RunID:               &runIDCopy,
		AgentNodeID:         agentNodeID,
		ReasonerID:          "reasoner.wf-test",
		Status:              string(types.ExecutionStatusPending),
		StartedAt:           now,
		CreatedAt:           now,
		UpdatedAt:           now,
	}
}

func makeTestMemory(scope, scopeID, key string, data string) *types.Memory {
	now := time.Now().UTC()
	return &types.Memory{
		Scope:       scope,
		ScopeID:     scopeID,
		Key:         key,
		Data:        json.RawMessage(data),
		AccessLevel: "private",
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}
