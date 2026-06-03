package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestStorageClinchEventAdditionalBranches(t *testing.T) {
	t.Run("local event helpers cover empty history marshal and closed store paths", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		history, err := ls.GetEventHistory(ctx, types.EventFilter{})
		require.NoError(t, err)
		require.Empty(t, history)

		err = ls.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:    "memory_changed",
			Scope:   "agent",
			ScopeID: "agent-marshaling",
			Key:     "memory/bad",
			Action:  "set",
			Data:    json.RawMessage(`{`),
		})
		require.EqualError(t, err, "failed to marshal event: json: error calling MarshalJSON for type json.RawMessage: unexpected end of JSON input")

		require.NoError(t, ls.kvStore.Close())

		err = ls.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:    "memory_changed",
			Scope:   "agent",
			ScopeID: "agent-closed",
			Key:     "memory/key",
			Action:  "set",
		})
		require.ErrorContains(t, err, "database not open")

		_, err = ls.GetEventHistory(ctx, types.EventFilter{})
		require.ErrorContains(t, err, "failed to get event history")

		ls.kvStore = nil
		ls.cleanupExpiredEvents()
	})

	t.Run("postgres event history preserves nullable fields", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE memory_events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				scope TEXT NOT NULL,
				scope_id TEXT NOT NULL,
				key TEXT NOT NULL,
				event_type TEXT,
				action TEXT,
				data BLOB,
				previous_data BLOB,
				metadata BLOB,
				timestamp TIMESTAMP NOT NULL
			)
		`)
		require.NoError(t, err)

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "postgres"), mode: "postgres"}
		_, err = ls.db.ExecContext(context.Background(), `
			INSERT INTO memory_events(scope, scope_id, key, event_type, action, data, previous_data, metadata, timestamp)
			VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
		`, "session", "scope-null", "doc-null", time.Now().UTC())
		require.NoError(t, err)

		events, err := ls.getEventHistoryPostgres(context.Background(), types.EventFilter{ScopeID: ptrString("scope-null")})
		require.NoError(t, err)
		require.Len(t, events, 1)
		require.Empty(t, events[0].Type)
		require.Empty(t, events[0].Action)
		require.Nil(t, events[0].Data)
		require.Nil(t, events[0].PreviousData)
		require.Equal(t, types.EventMetadata{}, events[0].Metadata)
	})
}

func TestStorageClinchCleanupOldExecutionsBranches(t *testing.T) {
	t.Run("returns early for cancelled context and no matches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		deleted, err := ls.CleanupOldExecutions(cancelled, time.Hour, 10)
		require.Zero(t, deleted)
		require.EqualError(t, err, "context cancelled during cleanup old executions: context canceled")

		deleted, err = ls.CleanupOldExecutions(ctx, time.Hour, 10)
		require.NoError(t, err)
		require.Zero(t, deleted)
	})

	t.Run("surfaces query failure when sql db is closed", func(t *testing.T) {
		tempDir := t.TempDir()
		rawDB, err := sql.Open("sqlite3", filepath.Join(tempDir, "closed.db"))
		require.NoError(t, err)
		_, err = rawDB.Exec(`CREATE TABLE workflow_executions (execution_id TEXT PRIMARY KEY, status TEXT, completed_at TIMESTAMP)`)
		require.NoError(t, err)
		require.NoError(t, rawDB.Close())

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		deleted, err := ls.CleanupOldExecutions(context.Background(), time.Hour, 1)
		require.Zero(t, deleted)
		require.ErrorContains(t, err, "failed to query old executions for cleanup")
	})
}

func TestStorageClinchAccessPolicyErrorBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)

	policy := &types.AccessPolicy{
		Name:           "storage-clinch-enabled",
		CallerTags:     []string{"ops"},
		TargetTags:     []string{"payments"},
		AllowFunctions: []string{"charge"},
		DenyFunctions:  []string{"refund"},
		Constraints: map[string]types.AccessConstraint{
			"amount": {Operator: "<=", Value: 10},
		},
		Action:    "allow",
		Priority:  5,
		Enabled:   true,
		CreatedAt: now,
		UpdatedAt: now,
	}
	require.NoError(t, ls.CreateAccessPolicy(ctx, policy))

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err := ls.GetAccessPolicies(cancelled)
	require.EqualError(t, err, "context cancelled during get access policies: context canceled")
	_, err = ls.GetAccessPolicyByID(cancelled, policy.ID)
	require.EqualError(t, err, "context cancelled during get access policy: context canceled")
	require.EqualError(t, ls.UpdateAccessPolicy(cancelled, policy), "context cancelled during update access policy: context canceled")
	require.EqualError(t, ls.DeleteAccessPolicy(cancelled, policy.ID), "context cancelled during delete access policy: context canceled")

	_, err = ls.db.ExecContext(ctx, `UPDATE access_policies SET caller_tags = ? WHERE id = ?`, "{", policy.ID)
	require.NoError(t, err)

	_, err = ls.GetAccessPolicies(ctx)
	require.EqualError(t, err, "failed to unmarshal access policy "+strconv.FormatInt(policy.ID, 10)+": failed to unmarshal caller_tags: unexpected end of JSON input")

	_, err = ls.GetAccessPolicyByID(ctx, policy.ID)
	require.EqualError(t, err, "failed to unmarshal access policy "+strconv.FormatInt(policy.ID, 10)+": failed to unmarshal caller_tags: unexpected end of JSON input")

	require.NoError(t, ls.db.DB.Close())
	_, err = ls.GetAccessPolicies(ctx)
	require.ErrorContains(t, err, "failed to get access policies")
}

func TestStorageClinchCreateSchemaAndHealthBranches(t *testing.T) {
	t.Run("health check reports sqlite integrity failure", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "postgres"), mode: "postgres"}
		require.NoError(t, rawDB.Close())

		err = ls.HealthCheck(context.Background())
		require.ErrorContains(t, err, "database is unhealthy")
	})

	t.Run("create schema returns bucket initialization errors", func(t *testing.T) {
		ls := newSchemaTestStorage(t)
		require.NoError(t, ls.kvStore.Close())

		err := ls.createSchema(context.Background())
		require.ErrorContains(t, err, "database not open")
	})
}

func TestStorageClinchConfigAndExecutionLogBranches(t *testing.T) {
	t.Run("config helpers cover postgres placeholder path and missing rows", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE config_storage (
				key TEXT PRIMARY KEY,
				value TEXT NOT NULL,
				version INTEGER NOT NULL,
				created_by TEXT,
				updated_by TEXT,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP NOT NULL
			)
		`)
		require.NoError(t, err)

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "postgres"), mode: "postgres"}
		ctx := context.Background()

		require.NoError(t, ls.SetConfig(ctx, "ui.theme", "light", "alice"))
		require.NoError(t, ls.SetConfig(ctx, "ui.theme", "dark", "bob"))

		entry, err := ls.GetConfig(ctx, "ui.theme")
		require.NoError(t, err)
		require.Equal(t, "dark", entry.Value)
		require.Equal(t, 2, entry.Version)
		require.Equal(t, "alice", entry.CreatedBy)
		require.Equal(t, "bob", entry.UpdatedBy)

		entries, err := ls.ListConfigs(ctx)
		require.NoError(t, err)
		require.Len(t, entries, 1)

		missing, err := ls.GetConfig(ctx, "missing")
		require.NoError(t, err)
		require.Nil(t, missing)

		require.NoError(t, ls.DeleteConfig(ctx, "ui.theme"))
		require.EqualError(t, ls.DeleteConfig(ctx, "ui.theme"), `config "ui.theme" not found`)
	})

	t.Run("execution log listing and pruning cover filters defaults and errors", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)

		entryOne := &types.ExecutionLogEntry{
			ExecutionID:     "exec-log-branches",
			AgentNodeID:     "agent-a",
			Level:           "warn",
			Source:          "sdk.logger",
			Message:         "first warning",
			SystemGenerated: true,
			EmittedAt:       now.Add(-3 * time.Minute),
		}
		entryTwo := &types.ExecutionLogEntry{
			ExecutionID: "exec-log-branches",
			WorkflowID:  "wf-explicit",
			AgentNodeID: "agent-b",
			Level:       "error",
			Source:      "worker",
			Message:     "second error",
			Attributes:  json.RawMessage(`{"trace":"abc"}`),
			EmittedAt:   now.Add(-2 * time.Minute),
		}
		entryThree := &types.ExecutionLogEntry{
			ExecutionID: "exec-log-branches",
			AgentNodeID: "agent-b",
			Level:       "info",
			Source:      "worker",
			Message:     "third info",
			EmittedAt:   now.Add(-1 * time.Minute),
		}

		require.NoError(t, ls.StoreExecutionLogEntries(ctx, "exec-log-branches", []*types.ExecutionLogEntry{entryOne, entryTwo, entryThree}))
		require.NoError(t, ls.StoreExecutionLogEntries(ctx, "exec-log-branches", nil))

		// Force the recorded_at fallback branch on read.
		_, err := ls.db.ExecContext(ctx, `UPDATE execution_logs SET recorded_at = NULL WHERE event_id = ?`, entryOne.EventID)
		require.NoError(t, err)

		latestTwo, err := ls.ListExecutionLogEntries(ctx, "exec-log-branches", nil, 2, nil, nil, nil, "")
		require.NoError(t, err)
		require.Len(t, latestTwo, 2)
		require.Equal(t, int64(2), latestTwo[0].Sequence)
		require.Equal(t, int64(3), latestTwo[1].Sequence)

		after := int64(1)
		filtered, err := ls.ListExecutionLogEntries(ctx, "exec-log-branches", &after, 10, []string{"error", "info"}, []string{"agent-b"}, []string{"worker"}, "error")
		require.NoError(t, err)
		require.Len(t, filtered, 1)
		require.Equal(t, entryTwo.EventID, filtered[0].EventID)
		require.Equal(t, entryTwo.EmittedAt, filtered[0].RecordedAt)
		require.JSONEq(t, `{"trace":"abc"}`, string(filtered[0].Attributes))

		require.NoError(t, ls.PruneExecutionLogEntries(ctx, "exec-log-branches", 2, now.Add(-150*time.Second)))
		remaining, err := ls.ListExecutionLogEntries(ctx, "exec-log-branches", nil, 10, nil, nil, nil, "")
		require.NoError(t, err)
		require.Len(t, remaining, 2)

		require.NoError(t, ls.db.DB.Close())
		err = ls.PruneExecutionLogEntries(ctx, "exec-log-branches", 1, time.Time{})
		require.ErrorContains(t, err, "failed to prune execution logs by count")
		_, err = ls.ListExecutionLogEntries(ctx, "exec-log-branches", nil, 10, nil, nil, nil, "")
		require.ErrorContains(t, err, "failed to query execution logs")
	})
}

func TestStorageClinchAgentUpdateErrorBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	agent := &types.AgentNode{
		ID:              "agent-update-errors",
		GroupID:         "group-a",
		TeamID:          "team-a",
		BaseURL:         "https://agent.example.com",
		Version:         "v1",
		TrafficWeight:   100,
		DeploymentType:  "long_running",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now().UTC(),
		RegisteredAt:    time.Now().UTC(),
	}
	require.NoError(t, ls.RegisterAgent(ctx, agent))

	require.NoError(t, ls.db.DB.Close())

	err := ls.UpdateAgentHealth(ctx, agent.ID, types.HealthStatusInactive)
	require.ErrorContains(t, err, "failed to begin transaction for agent health update")

	err = ls.UpdateAgentHeartbeat(ctx, agent.ID, agent.Version, time.Now().UTC())
	require.ErrorContains(t, err, "failed to begin transaction for agent heartbeat update")

	err = ls.UpdateAgentLifecycleStatus(ctx, agent.ID, types.AgentStatusOffline)
	require.ErrorContains(t, err, "failed to begin transaction for agent lifecycle update")

	err = ls.UpdateAgentVersion(ctx, agent.ID, "v2")
	require.ErrorContains(t, err, "failed to begin transaction for agent version update")
}

func TestStorageClinchRegisterWorkflowAndDIDBranches(t *testing.T) {
	t.Run("register agent and workflow surface marshal and database failures", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		badReasoner := &types.AgentNode{
			ID:                 "agent-bad-reasoner",
			BaseURL:            "https://agent.example.com",
			DeploymentType:     "long_running",
			HealthStatus:       types.HealthStatusActive,
			LifecycleStatus:    types.AgentStatusReady,
			LastHeartbeat:      time.Now().UTC(),
			RegisteredAt:       time.Now().UTC(),
			CommunicationConfig: types.CommunicationConfig{Protocols: []string{"http"}},
			Reasoners:          []types.ReasonerDefinition{{ID: "r1", InputSchema: json.RawMessage(`{`)}},
		}
		require.EqualError(t, ls.RegisterAgent(ctx, badReasoner), "failed to marshal reasoners: json: error calling MarshalJSON for type json.RawMessage: unexpected end of JSON input")

		badSkill := &types.AgentNode{
			ID:                 "agent-bad-skill",
			BaseURL:            "https://agent.example.com",
			DeploymentType:     "long_running",
			HealthStatus:       types.HealthStatusActive,
			LifecycleStatus:    types.AgentStatusReady,
			LastHeartbeat:      time.Now().UTC(),
			RegisteredAt:       time.Now().UTC(),
			CommunicationConfig: types.CommunicationConfig{Protocols: []string{"http"}},
			Skills:             []types.SkillDefinition{{ID: "s1", InputSchema: json.RawMessage(`{`)}},
		}
		require.EqualError(t, ls.RegisterAgent(ctx, badSkill), "failed to marshal skills: json: error calling MarshalJSON for type json.RawMessage: unexpected end of JSON input")

		badMetadata := &types.AgentNode{
			ID:                 "agent-bad-metadata",
			BaseURL:            "https://agent.example.com",
			DeploymentType:     "long_running",
			HealthStatus:       types.HealthStatusActive,
			LifecycleStatus:    types.AgentStatusReady,
			LastHeartbeat:      time.Now().UTC(),
			RegisteredAt:       time.Now().UTC(),
			CommunicationConfig: types.CommunicationConfig{Protocols: []string{"http"}},
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{"bad": func() {}},
			},
		}
		require.EqualError(t, ls.RegisterAgent(ctx, badMetadata), "failed to marshal agent metadata: json: unsupported type: func()")

		require.NoError(t, ls.db.DB.Close())
		err := ls.RegisterAgent(ctx, &types.AgentNode{})
		require.ErrorContains(t, err, "failed to begin transaction for agent registration")

		workflowLS, workflowCtx := setupLocalStorage(t)
		cancelled, cancel := context.WithCancel(workflowCtx)
		cancel()
		err = workflowLS.CreateOrUpdateWorkflow(cancelled, &types.Workflow{})
		require.EqualError(t, err, "context cancelled during create or update workflow: context canceled")

		require.NoError(t, workflowLS.db.DB.Close())
		err = workflowLS.CreateOrUpdateWorkflow(workflowCtx, &types.Workflow{WorkflowID: "wf-closed"})
		require.ErrorContains(t, err, "failed to create or update workflow")
	})

	t.Run("did helpers cover validation and corrupted json branches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC()
		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-did-branches", "did:root:branches", []byte("seed"), now, now))

		require.EqualError(t, ls.StoreAgentDID(ctx, "", "did:agent:x", "srv-did-branches", `{"kid":"x"}`, 1), "validation failed for agent_node_id='': agent ID cannot be empty (context: StoreAgentDID)")
		require.EqualError(t, ls.StoreAgentDID(ctx, "agent-x", "", "srv-did-branches", `{"kid":"x"}`, 1), "validation failed for did='': agent DID cannot be empty (context: StoreAgentDID)")
		require.EqualError(t, ls.StoreAgentDID(ctx, "agent-x", "did:agent:x", "srv-did-branches", "", 1), "validation failed for public_key_jwk='': public key JWK cannot be empty (context: StoreAgentDID)")

		require.NoError(t, ls.StoreAgentDID(ctx, "agent-json-branches", "did:agent:branches", "srv-did-branches", `{"kid":"ok"}`, 2))
		_, err := ls.db.ExecContext(ctx, `UPDATE agent_dids SET reasoners = ?, skills = ? WHERE agent_node_id = ?`, "{", "{", "agent-json-branches")
		require.NoError(t, err)

		_, err = ls.GetAgentDID(ctx, "agent-json-branches")
		require.EqualError(t, err, "failed to parse reasoners JSON: unexpected end of JSON input")
		_, err = ls.ListAgentDIDs(ctx)
		require.EqualError(t, err, "failed to parse reasoners JSON: unexpected end of JSON input")

		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "", "did:agent:branches", "reasoner", "summarize", 1), "validation failed for component_did='': component DID cannot be empty (context: StoreComponentDID)")
		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:x", "did:agent:branches", "", "summarize", 1), "validation failed for component_type='': component type cannot be empty (context: StoreComponentDID)")
		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:x", "did:agent:branches", "reasoner", "", 1), "validation failed for component_name='': component name cannot be empty (context: StoreComponentDID)")
	})
}
