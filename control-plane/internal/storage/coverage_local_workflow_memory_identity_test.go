package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/boltdb/bolt"
	"github.com/stretchr/testify/require"
)

func TestWorkflowReadQueryAndCleanupBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)

	nameAlpha := "Alpha Workflow"
	nameBeta := "Beta Workflow"
	sessionA := "session-a"
	sessionB := "session-b"
	actorA := "actor-a"
	actorB := "actor-b"
	statusRunning := "running"
	statusSucceeded := "succeeded"

	workflows := []*types.Workflow{
		{
			WorkflowID:         "wf-alpha",
			WorkflowName:       &nameAlpha,
			WorkflowTags:       []string{"alpha", "shared"},
			SessionID:          &sessionA,
			ActorID:            &actorA,
			TotalExecutions:    2,
			SuccessfulExecutions: 1,
			FailedExecutions:   0,
			TotalDurationMS:    200,
			Status:             statusRunning,
			StartedAt:          now.Add(-2 * time.Minute),
			CreatedAt:          now.Add(-2 * time.Minute),
			UpdatedAt:          now.Add(-90 * time.Second),
		},
		{
			WorkflowID:         "wf-beta",
			WorkflowName:       &nameBeta,
			WorkflowTags:       []string{"beta"},
			SessionID:          &sessionB,
			ActorID:            &actorB,
			TotalExecutions:    4,
			SuccessfulExecutions: 4,
			FailedExecutions:   0,
			TotalDurationMS:    100,
			Status:             statusSucceeded,
			StartedAt:          now.Add(-time.Minute),
			CreatedAt:          now.Add(-time.Minute),
			UpdatedAt:          now.Add(-30 * time.Second),
		},
	}

	for _, workflow := range workflows {
		require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, workflow))
	}

	got, err := ls.GetWorkflow(ctx, "wf-alpha")
	require.NoError(t, err)
	require.Equal(t, []string{"alpha", "shared"}, got.WorkflowTags)

	_, err = ls.GetWorkflow(ctx, "wf-missing")
	require.EqualError(t, err, "workflow with ID wf-missing not found")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.GetWorkflow(cancelled, "wf-alpha")
	require.EqualError(t, err, "context cancelled during get workflow: context canceled")

	results, err := ls.QueryWorkflows(ctx, types.WorkflowFilters{
		SessionID: &sessionA,
		ActorID:   &actorA,
		Status:    &statusRunning,
		StartTime: ptrTime(now.Add(-3 * time.Minute)),
		EndTime:   ptrTime(now),
		Limit:     1,
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "wf-alpha", results[0].WorkflowID)

	sortBy := "duration"
	sortOrder := "asc"
	results, err = ls.QueryWorkflows(ctx, types.WorkflowFilters{SortBy: &sortBy, SortOrder: &sortOrder})
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, "wf-beta", results[0].WorkflowID)

	results, err = ls.QueryWorkflows(cancelled, types.WorkflowFilters{})
	require.EqualError(t, err, "context cancelled during query workflows: context canceled")
	require.Nil(t, results)

	_, err = ls.db.ExecContext(ctx, `UPDATE workflows SET workflow_tags = ? WHERE workflow_id = ?`, []byte(`{"bad":`), "wf-beta")
	require.NoError(t, err)
	_, err = ls.GetWorkflow(ctx, "wf-beta")
	require.ErrorContains(t, err, "failed to unmarshal workflow tags")
	_, err = ls.QueryWorkflows(ctx, types.WorkflowFilters{})
	require.ErrorContains(t, err, "failed to unmarshal workflow tags")

	runID := "run-cleanup-dry"
	require.NoError(t, ls.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "wf-alpha",
		Status:         statusRunning,
		CreatedAt:      now,
		UpdatedAt:      now,
	}))
	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          "wf-alpha",
		ExecutionID:         "exec-cleanup-dry",
		AgentFieldRequestID: "req-cleanup-dry",
		RunID:               &runID,
		AgentNodeID:         "agent-cleanup",
		ReasonerID:          "reasoner-cleanup",
		Status:              statusRunning,
		CreatedAt:           now,
		UpdatedAt:           now,
	}))

	dryRun, err := ls.CleanupWorkflow(ctx, runID, true)
	require.NoError(t, err)
	require.True(t, dryRun.Success)
	require.True(t, dryRun.DryRun)
	require.Greater(t, dryRun.DeletedRecords["workflow_runs"], 0)
	require.Greater(t, dryRun.DeletedRecords["workflow_executions"], 0)

	stillThere, err := ls.GetWorkflowExecution(ctx, "exec-cleanup-dry")
	require.NoError(t, err)
	require.NotNil(t, stillThere)

	result, err := ls.CleanupWorkflow(ctx, "   ", false)
	require.EqualError(t, err, "workflow ID cannot be empty")
	require.NotNil(t, result)
	require.False(t, result.Success)

	cancelledCleanup, cancelCleanup := context.WithCancel(ctx)
	cancelCleanup()
	_, err = ls.CleanupWorkflow(cancelledCleanup, runID, false)
	require.EqualError(t, err, "context cancelled during workflow cleanup: context canceled")
}

func TestMemoryErrorBranchesCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	require.NoError(t, ls.SetMemory(ctx, &types.Memory{
		Scope:     "session",
		ScopeID:   "scope-ok",
		Key:       "key-ok",
		Data:      json.RawMessage(`{"ok":true}`),
		UpdatedAt: time.Now().UTC(),
	}))

	memories, err := ls.ListMemory(ctx, "session", "scope-ok")
	require.NoError(t, err)
	require.Len(t, memories, 1)

	err = ls.SetMemory(ctx, &types.Memory{Scope: "missing", ScopeID: "scope", Key: "key"})
	require.EqualError(t, err, "BoltDB bucket 'missing' not found")

	_, err = ls.GetMemory(ctx, "missing", "scope", "key")
	require.EqualError(t, err, "BoltDB bucket 'missing' not found")

	err = ls.DeleteMemory(ctx, "missing", "scope", "key")
	require.EqualError(t, err, "BoltDB bucket 'missing' not found")

	_, err = ls.ListMemory(ctx, "missing", "scope")
	require.EqualError(t, err, "BoltDB bucket 'missing' not found")

	require.NoError(t, ls.kvStore.Update(func(tx *bolt.Tx) error {
		return tx.Bucket([]byte("session")).Put([]byte("scope-bad:key-bad"), []byte(`{"broken":`))
	}))

	_, err = ls.GetMemory(ctx, "session", "scope-bad", "key-bad")
	require.ErrorContains(t, err, "failed to unmarshal memory from BoltDB")

	_, err = ls.ListMemory(ctx, "session", "scope-bad")
	require.ErrorContains(t, err, "failed to unmarshal memory from BoltDB")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	err = ls.SetMemory(cancelled, &types.Memory{Scope: "session", ScopeID: "scope", Key: "key"})
	require.EqualError(t, err, "context cancelled before BoltDB SetMemory operation: context canceled")
	_, err = ls.GetMemory(cancelled, "session", "scope", "key")
	require.EqualError(t, err, "context cancelled before BoltDB GetMemory operation: context canceled")
	err = ls.DeleteMemory(cancelled, "session", "scope", "key")
	require.EqualError(t, err, "context cancelled before BoltDB DeleteMemory operation: context canceled")
	_, err = ls.ListMemory(cancelled, "session", "scope")
	require.EqualError(t, err, "context cancelled before BoltDB ListMemory operation: context canceled")
}

func TestDIDRegistryAndAgentDIDBranches(t *testing.T) {
	t.Run("custom DID registry schema supports roundtrip operations", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE did_registry (
				did TEXT PRIMARY KEY,
				did_document TEXT,
				public_key TEXT,
				private_key_ref TEXT,
				derivation_path TEXT,
				created_at TIMESTAMP,
				updated_at TIMESTAMP,
				status TEXT
			)
		`)
		require.NoError(t, err)

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		ctx := context.Background()

		require.NoError(t, ls.StoreDID(ctx, "did:example:1", `{"id":"did:example:1"}`, "pub-1", "ref-1", "m/1"))
		require.NoError(t, ls.StoreDID(ctx, "did:example:2", `{"id":"did:example:2"}`, "pub-2", "ref-2", "m/2"))
		require.EqualError(t, ls.StoreDID(ctx, "did:example:1", `{"id":"did:example:1"}`, "pub-1", "ref-1", "m/1"), "duplicate registry DID detected: did:example:1 already exists")

		entry, err := ls.GetDID(ctx, "did:example:1")
		require.NoError(t, err)
		require.Equal(t, "pub-1", entry.PublicKey)

		entries, err := ls.ListDIDs(ctx)
		require.NoError(t, err)
		require.Len(t, entries, 2)

		_, err = ls.GetDID(ctx, "did:missing")
		require.EqualError(t, err, "DID did:missing not found")

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreDID(cancelled, "did:cancelled", "{}", "pub", "ref", "m/3"), "context cancelled during store DID: context canceled")
		_, err = ls.GetDID(cancelled, "did:example:1")
		require.EqualError(t, err, "context cancelled during get DID: context canceled")
		_, err = ls.ListDIDs(cancelled)
		require.EqualError(t, err, "context cancelled during list DIDs: context canceled")
	})

	t.Run("agent DID readers surface JSON parsing errors", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC()

		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-json", "did:root:json", []byte("seed"), now, now))
		require.NoError(t, ls.StoreAgentDID(ctx, "agent-json", "did:agent:json", "srv-json", `{"kid":"agent-json"}`, 3))

		_, err := ls.GetAgentDID(ctx, "missing-agent")
		require.EqualError(t, err, "agent DID for missing-agent not found")

		_, err = ls.db.ExecContext(ctx, `UPDATE agent_dids SET reasoners = ? WHERE agent_node_id = ?`, "{", "agent-json")
		require.NoError(t, err)
		_, err = ls.GetAgentDID(ctx, "agent-json")
		require.ErrorContains(t, err, "failed to parse reasoners JSON")

		_, err = ls.db.ExecContext(ctx, `UPDATE agent_dids SET reasoners = ?, skills = ? WHERE agent_node_id = ?`, "{}", "{", "agent-json")
		require.NoError(t, err)
		_, err = ls.ListAgentDIDs(ctx)
		require.ErrorContains(t, err, "failed to parse skills JSON")
	})
}

