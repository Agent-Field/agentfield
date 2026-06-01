package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestReasonerMetricsAndHistoryCoverage(t *testing.T) {
	t.Run("queries metrics and history via direct and tx paths", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		now := time.Now().UTC().Truncate(time.Second)
		completed := "completed"
		failed := string(types.ExecutionStatusFailed)
		statusReason := "provider_error"
		runID := "run-metrics"
		workflowName := "Reasoner Workflow"
		reasonerID := "node-1.reasoner-a"

		records := []*types.WorkflowExecution{
			{
				WorkflowID:          "wf-1",
				ExecutionID:         "exec-metrics-1",
				AgentFieldRequestID: "req-1",
				RunID:               &runID,
				AgentNodeID:         "node-1",
				ReasonerID:          "reasoner-a",
				WorkflowName:        &workflowName,
				InputData:           json.RawMessage(`{"ticker":"NVDA"}`),
				OutputData:          json.RawMessage(`{"bad"`),
				Status:              completed,
				StartedAt:           now.Add(-1 * time.Hour),
				CompletedAt:         timePtrLocal(now.Add(-59 * time.Minute)),
				DurationMS:          int64Ptr(1200),
				CreatedAt:           now.Add(-1 * time.Hour),
				UpdatedAt:           now.Add(-59 * time.Minute),
			},
			{
				WorkflowID:          "wf-1",
				ExecutionID:         "exec-metrics-2",
				AgentFieldRequestID: "req-2",
				RunID:               &runID,
				AgentNodeID:         "node-1",
				ReasonerID:          "reasoner-a",
				InputData:           json.RawMessage(`{"input":{"ticker":"AAPL"},"context":{"mode":"debug"}}`),
				OutputData:          json.RawMessage(`{"summary":"done"}`),
				Status:              failed,
				StatusReason:        &statusReason,
				ErrorMessage:        strPtr("upstream timeout"),
				RetryCount:          2,
				SessionID:           strPtr("session-1"),
				ActorID:             strPtr("actor-1"),
				StartedAt:           now.Add(-10 * time.Minute),
				CompletedAt:         timePtrLocal(now.Add(-9 * time.Minute)),
				DurationMS:          int64Ptr(2400),
				CreatedAt:           now.Add(-10 * time.Minute),
				UpdatedAt:           now.Add(-9 * time.Minute),
			},
			{
				WorkflowID:          "wf-2",
				ExecutionID:         "exec-other",
				AgentFieldRequestID: "req-3",
				AgentNodeID:         "node-2",
				ReasonerID:          "reasoner-b",
				Status:              completed,
				StartedAt:           now,
				CreatedAt:           now,
				UpdatedAt:           now,
			},
		}

		for _, record := range records {
			require.NoError(t, ls.StoreWorkflowExecution(ctx, record))
		}

		metrics, err := ls.GetReasonerPerformanceMetrics(ctx, reasonerID)
		require.NoError(t, err)
		require.Equal(t, 2, metrics.TotalExecutions)
		require.Equal(t, 2, metrics.ExecutionsLast24h)
		require.InDelta(t, 0.5, metrics.SuccessRate, 0.001)
		require.Equal(t, 1800, metrics.AvgResponseTimeMs)
		require.Len(t, metrics.RecentExecutions, 2)
		require.Equal(t, "exec-metrics-2", metrics.RecentExecutions[0].ExecutionID)

		tx, err := ls.db.Begin()
		require.NoError(t, err)

		metricsTx, err := ls.executeReasonerMetricsQuery(tx, "node-1", "reasoner-a")
		require.NoError(t, err)
		require.Equal(t, metrics.TotalExecutions, metricsTx.TotalExecutions)
		require.Len(t, metricsTx.RecentExecutions, 2)

		history, err := ls.executeReasonerHistoryQuery(tx, "node-1", "reasoner-a", 1, 1, 0)
		require.NoError(t, err)
		require.Equal(t, 2, history.Total)
		require.True(t, history.HasMore)
		require.Len(t, history.Executions, 1)
		require.Equal(t, "exec-metrics-2", history.Executions[0].ExecutionID)
		require.Equal(t, map[string]interface{}{"ticker": "AAPL"}, history.Executions[0].Input)
		require.Equal(t, map[string]interface{}{"mode": "debug"}, history.Executions[0].Context)
		require.Equal(t, map[string]interface{}{"summary": "done"}, history.Executions[0].Output)
		require.NoError(t, tx.Rollback())

		history, err = ls.GetReasonerExecutionHistory(ctx, reasonerID, 1, 10)
		require.NoError(t, err)
		require.Equal(t, 2, history.Total)
		require.Len(t, history.Executions, 2)
		require.Equal(t, "exec-metrics-1", history.Executions[1].ExecutionID)
		require.Equal(t, map[string]interface{}{"ticker": "NVDA"}, history.Executions[1].Input)
		require.Equal(t, map[string]interface{}{"raw": `{"bad"`}, history.Executions[1].Output)
	})

	t.Run("rejects invalid and cancelled contexts", func(t *testing.T) {
		ls, _ := setupLocalStorage(t)

		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := ls.GetReasonerPerformanceMetrics(cancelled, "node.reasoner")
		require.EqualError(t, err, "context cancelled during get reasoner performance metrics: context canceled")

		_, err = ls.GetReasonerPerformanceMetrics(context.Background(), "malformed")
		require.EqualError(t, err, "invalid reasoner_id format, expected 'node_id.reasoner_id'")

		_, err = ls.GetReasonerExecutionHistory(context.Background(), "malformed", 1, 10)
		require.EqualError(t, err, "invalid reasoner_id format, expected 'node_id.reasoner_id'")
	})
}

func TestExecutionEventAndLogCoverage(t *testing.T) {
	t.Run("stores and lists workflow execution events", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		require.NotNil(t, ls.GetExecutionEventBus())
		require.NotNil(t, ls.GetWorkflowExecutionEventBus())
		require.NotNil(t, ls.GetExecutionLogEventBus())
		_, err := ls.BeginTransaction()
		require.EqualError(t, err, "transactions not fully implemented for LocalStorage")
		require.EqualError(t, ls.StoreWorkflowExecutionEvent(ctx, nil), "workflow execution event is nil")

		runID := "run-events"
		parentID := "parent-exec"
		status := string(types.ExecutionStatusRunning)
		statusReason := "scheduled"
		emittedAt := time.Now().UTC().Truncate(time.Second)

		event := &types.WorkflowExecutionEvent{
			ExecutionID:       "exec-events-1",
			WorkflowID:        "wf-events",
			RunID:             &runID,
			ParentExecutionID: &parentID,
			Sequence:          1,
			PreviousSequence:  0,
			EventType:         "status.changed",
			Status:            &status,
			StatusReason:      &statusReason,
			EmittedAt:         emittedAt,
		}
		require.NoError(t, ls.StoreWorkflowExecutionEvent(ctx, event))
		require.NotZero(t, event.EventID)
		require.False(t, event.RecordedAt.IsZero())
		_, err = ls.db.ExecContext(ctx, `UPDATE workflow_execution_events SET recorded_at = ? WHERE event_id = ?`, event.RecordedAt, event.EventID)
		require.NoError(t, err)

		stored, err := ls.ListWorkflowExecutionEvents(ctx, "exec-events-1", nil, 10)
		require.NoError(t, err)
		require.Len(t, stored, 1)
		require.Equal(t, json.RawMessage("{}"), stored[0].Payload)
		require.Equal(t, &status, stored[0].Status)

		after := int64(1)
		stored, err = ls.ListWorkflowExecutionEvents(ctx, "exec-events-1", &after, 10)
		require.NoError(t, err)
		require.Empty(t, stored)
	})

	t.Run("stores lists and prunes execution logs", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		require.EqualError(t, ls.StoreExecutionLogEntry(ctx, nil), "execution log entry is nil")
		require.EqualError(t, ls.StoreExecutionLogEntry(ctx, &types.ExecutionLogEntry{}), "execution_id is required")

		runID := "run-logs"
		rootWorkflowID := "root-wf"
		parentExecutionID := "parent-exec"
		reasonerID := "reasoner-logs"
		eventType := "token"
		sdkLanguage := "go"
		spanID := "span-1"
		stepID := "step-1"
		errorCategory := "timeout"
		attempt := 2
		now := time.Now().UTC().Truncate(time.Second)

		entryOne := &types.ExecutionLogEntry{
			ExecutionID:       "exec-logs",
			WorkflowID:        "wf-logs",
			RunID:             &runID,
			RootWorkflowID:    &rootWorkflowID,
			ParentExecutionID: &parentExecutionID,
			AgentNodeID:       "node-1",
			ReasonerID:        &reasonerID,
			Level:             "warn",
			Source:            "sdk.logger",
			EventType:         &eventType,
			Message:           "first log",
			Attributes:        json.RawMessage(`{"request_id":"abc"}`),
			SystemGenerated:   true,
			SDKLanguage:       &sdkLanguage,
			Attempt:           &attempt,
			SpanID:            &spanID,
			StepID:            &stepID,
			ErrorCategory:     &errorCategory,
			EmittedAt:         now.Add(-2 * time.Hour),
		}
		require.NoError(t, ls.StoreExecutionLogEntry(ctx, entryOne))
		require.NotZero(t, entryOne.EventID)
		require.Equal(t, int64(1), entryOne.Sequence)

		entryTwo := &types.ExecutionLogEntry{
			ExecutionID: "exec-logs",
			WorkflowID:  "wf-logs",
			AgentNodeID: "node-2",
			Message:     "second log",
			EmittedAt:   now.Add(-1 * time.Hour),
		}
		entryThree := &types.ExecutionLogEntry{
			ExecutionID: "exec-logs",
			WorkflowID:  "wf-logs",
			AgentNodeID: "node-1",
			Level:       "error",
			Source:      "worker",
			Message:     "third log",
			Attributes:  json.RawMessage(`{"raw":true}`),
			EmittedAt:   now,
		}
		require.NoError(t, ls.StoreExecutionLogEntries(ctx, "exec-logs", []*types.ExecutionLogEntry{nil, entryTwo, entryThree}))

		tail, err := ls.ListExecutionLogEntries(ctx, "exec-logs", nil, 2, []string{"warn", "error"}, []string{"node-1"}, []string{"sdk.logger", "worker"}, "log")
		require.NoError(t, err)
		require.Len(t, tail, 2)
		require.Equal(t, int64(1), tail[0].Sequence)
		require.Equal(t, int64(3), tail[1].Sequence)

		after := int64(1)
		filtered, err := ls.ListExecutionLogEntries(ctx, "exec-logs", &after, 10, nil, nil, nil, "")
		require.NoError(t, err)
		require.Len(t, filtered, 2)
		require.Equal(t, int64(2), filtered[0].Sequence)
		require.Equal(t, "", filtered[0].Level)
		require.Equal(t, "", filtered[0].Source)
		require.Equal(t, json.RawMessage("{}"), filtered[0].Attributes)

		require.NoError(t, ls.PruneExecutionLogEntries(ctx, "exec-logs", 1, now.Add(-90*time.Minute)))
		remaining, err := ls.ListExecutionLogEntries(ctx, "exec-logs", nil, 10, nil, nil, nil, "")
		require.NoError(t, err)
		require.Len(t, remaining, 1)
		require.Equal(t, "third log", remaining[0].Message)
		require.NoError(t, ls.PruneExecutionLogEntries(ctx, "", 1, time.Time{}))
	})

	t.Run("stores and batches execution webhook events", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		require.EqualError(t, ls.StoreExecutionWebhookEvent(ctx, nil), "execution webhook event is nil")

		statusCode := 202
		responseBody := "accepted"
		errMsg := "temporary failure"

		require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
			ExecutionID:  "exec-wh-1",
			EventType:    "delivery_attempt",
			Status:       "delivered",
			HTTPStatus:   &statusCode,
			Payload:      json.RawMessage(`{"attempt":1}`),
			ResponseBody: &responseBody,
		}))
		require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
			ExecutionID:  "exec-wh-2",
			EventType:    "delivery_attempt",
			Status:       "failed",
			ErrorMessage: &errMsg,
		}))

		single, err := ls.ListExecutionWebhookEvents(ctx, "exec-wh-1")
		require.NoError(t, err)
		require.Len(t, single, 1)
		require.Equal(t, statusCode, *single[0].HTTPStatus)
		require.Equal(t, json.RawMessage(`{"attempt":1}`), single[0].Payload)

		batch, err := ls.ListExecutionWebhookEventsBatch(ctx, []string{"exec-wh-1", "exec-wh-2", "exec-wh-1", " "})
		require.NoError(t, err)
		require.Len(t, batch, 2)
		require.Len(t, batch["exec-wh-1"], 1)
		require.Equal(t, json.RawMessage("{}"), batch["exec-wh-2"][0].Payload)

		empty, err := ls.ListExecutionWebhookEventsBatch(ctx, nil)
		require.NoError(t, err)
		require.Empty(t, empty)
	})
}

func TestDIDAndVCCoverage(t *testing.T) {
	t.Run("agentfield server and agent did operations cover success and validation", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreAgentFieldServerDID(cancelled, "srv-1", "did:root:1", []byte("seed"), now, now), "context cancelled during store af server DID: context canceled")

		tests := []struct {
			name string
			run  func() error
			want string
		}{
			{
				name: "empty server id",
				run:  func() error { return ls.StoreAgentFieldServerDID(ctx, "", "did:root:1", []byte("seed"), now, now) },
				want: "validation failed for agentfield_server_id='': af server ID cannot be empty (context: StoreAgentFieldServerDID)",
			},
			{
				name: "empty root did",
				run:  func() error { return ls.StoreAgentFieldServerDID(ctx, "srv-1", "", []byte("seed"), now, now) },
				want: "validation failed for root_did='': root DID cannot be empty (context: StoreAgentFieldServerDID)",
			},
			{
				name: "empty seed",
				run:  func() error { return ls.StoreAgentFieldServerDID(ctx, "srv-1", "did:root:1", nil, now, now) },
				want: "validation failed for master_seed='<encrypted>': master seed cannot be empty (context: StoreAgentFieldServerDID)",
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				require.EqualError(t, tc.run(), tc.want)
			})
		}

		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-1", "did:root:1", []byte("seed"), now, now))
		info, err := ls.GetAgentFieldServerDID(ctx, "srv-1")
		require.NoError(t, err)
		require.Equal(t, "did:root:1", info.RootDID)
		infos, err := ls.ListAgentFieldServerDIDs(ctx)
		require.NoError(t, err)
		require.Len(t, infos, 1)

		require.EqualError(t, ls.validateAgentFieldServerExists(ctx, ""), "validation failed for agentfield_server_id='': af server ID cannot be empty (context: pre-storage validation)")
		require.EqualError(t, ls.validateAgentFieldServerExists(ctx, "missing"), "foreign key constraint violation in agent_dids.agentfield_server_id: referenced did_registry 'missing' does not exist (operation: INSERT)")

		require.EqualError(t, ls.StoreAgentDID(ctx, "agent-1", "did:agent:1", "missing", `{"kty":"OKP"}`, 7), "pre-storage validation failed: foreign key constraint violation in agent_dids.agentfield_server_id: referenced did_registry 'missing' does not exist (operation: INSERT)")
		require.NoError(t, ls.StoreAgentDID(ctx, "agent-1", "did:agent:1", "srv-1", `{"kty":"OKP"}`, 7))
		require.EqualError(t, ls.StoreAgentDID(ctx, "agent-1", "did:agent:1", "srv-1", `{"kty":"OKP"}`, 7), "duplicate agent DID detected: agent:agent-1@srv-1 already exists")

		agentInfo, err := ls.GetAgentDID(ctx, "agent-1")
		require.NoError(t, err)
		require.Equal(t, "did:agent:1", agentInfo.DID)
		require.Equal(t, map[string]types.ReasonerDIDInfo{}, agentInfo.Reasoners)
		require.Equal(t, map[string]types.SkillDIDInfo{}, agentInfo.Skills)
		agentInfos, err := ls.ListAgentDIDs(ctx)
		require.NoError(t, err)
		require.Len(t, agentInfos, 1)

		require.EqualError(t, ls.validateAgentDIDExists(ctx, ""), "validation failed for agent_did='': agent DID cannot be empty (context: pre-storage validation)")
		require.EqualError(t, ls.validateAgentDIDExists(ctx, "missing"), "foreign key constraint violation in component_dids.agent_did: referenced agent_dids 'missing' does not exist (operation: INSERT)")
		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:1", "missing", "reasoner", "summarize", 4), "pre-storage validation failed: foreign key constraint violation in component_dids.agent_did: referenced agent_dids 'missing' does not exist (operation: INSERT)")
		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:1", "did:agent:1", "invalid", "summarize", 4), "validation failed for component_type='invalid': component type must be 'reasoner' or 'skill' (context: StoreComponentDID)")
		require.NoError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:1", "did:agent:1", "reasoner", "summarize", 4))
		require.EqualError(t, ls.StoreComponentDID(ctx, "component-id", "did:component:1", "did:agent:1", "reasoner", "summarize", 4), "duplicate component DID detected: component:reasoner/summarize@did:agent:1 already exists")

		componentInfo, err := ls.GetComponentDID(ctx, "summarize")
		require.NoError(t, err)
		require.Equal(t, 4, componentInfo.DerivationIndex)
		components, err := ls.ListComponentDIDs(ctx, "did:agent:1")
		require.NoError(t, err)
		require.Len(t, components, 1)
		allComponents, err := ls.ListComponentDIDs(ctx, "")
		require.NoError(t, err)
		require.Len(t, allComponents, 1)
	})

	t.Run("store agent did with components commits atomically", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC()
		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-2", "did:root:2", []byte("seed"), now, now))

		components := []ComponentDIDRequest{
			{ComponentDID: "did:component:r1", ComponentType: "reasoner", ComponentName: "reasoner-one", PublicKeyJWK: `{"kid":"r1"}`, DerivationIndex: 9},
			{ComponentDID: "did:component:s1", ComponentType: "skill", ComponentName: "skill-one", PublicKeyJWK: `{"kid":"s1"}`, DerivationIndex: 10},
		}
		require.NoError(t, ls.StoreAgentDIDWithComponents(ctx, "agent-2", "did:agent:2", "srv-2", `{"kid":"agent"}`, 8, components))

		agentInfo, err := ls.GetAgentDID(ctx, "agent-2")
		require.NoError(t, err)
		require.Equal(t, "did:agent:2", agentInfo.DID)
		listedComponents, err := ls.ListComponentDIDs(ctx, "did:agent:2")
		require.NoError(t, err)
		require.Len(t, listedComponents, 2)
	})

	t.Run("did registry and did document operations", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreDID(cancelled, "did:example:1", "{}", "pub", "ref", "m/1"), "context cancelled during store DID: context canceled")

		require.EqualError(t, ls.StoreDID(ctx, "did:example:1", `{"id":"did:example:1"}`, "pub", "ref", "m/1"), "failed to store DID: table did_registry has no column named did")
		_, err := ls.GetDID(ctx, "did:example:1")
		require.EqualError(t, err, "failed to get DID: no such column: did")
		_, err = ls.ListDIDs(ctx)
		require.EqualError(t, err, "failed to list DIDs: no such column: did")

		record := &types.DIDDocumentRecord{
			DID:          "did:web:example.com:agents:agent-1",
			AgentID:      "agent-1",
			DIDDocument:  json.RawMessage(`{"id":"did:web:example.com:agents:agent-1"}`),
			PublicKeyJWK: `{"kid":"doc-1"}`,
			CreatedAt:    now,
			UpdatedAt:    now,
		}
		require.NoError(t, ls.StoreDIDDocument(ctx, record))
		storedRecord, err := ls.GetDIDDocument(ctx, record.DID)
		require.NoError(t, err)
		require.False(t, storedRecord.IsRevoked())
		byAgentID, err := ls.GetDIDDocumentByAgentID(ctx, "agent-1")
		require.NoError(t, err)
		require.Equal(t, record.DID, byAgentID.DID)
		require.NoError(t, ls.RevokeDIDDocument(ctx, record.DID))
		storedRecord, err = ls.GetDIDDocument(ctx, record.DID)
		require.NoError(t, err)
		require.True(t, storedRecord.IsRevoked())
		records, err := ls.ListDIDDocuments(ctx)
		require.NoError(t, err)
		require.Len(t, records, 1)
		_, err = ls.GetDIDDocumentByAgentID(ctx, "agent-1")
		require.EqualError(t, err, "DID document not found for agent: agent-1")
		require.EqualError(t, ls.RevokeDIDDocument(ctx, "missing"), "DID document not found: missing")
	})

	t.Run("execution and workflow vc operations", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)
		runID := "run-vc"
		workflowName := "VC Workflow"
		status := string(types.ExecutionStatusRunning)

		require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
			WorkflowID:          "wf-vc",
			ExecutionID:         "exec-vc-1",
			AgentFieldRequestID: "req-vc",
			RunID:               &runID,
			AgentNodeID:         "agent-vc",
			ReasonerID:          "reasoner-vc",
			WorkflowName:        &workflowName,
			Status:              status,
			StartedAt:           now,
			CreatedAt:           now,
			UpdatedAt:           now,
		}))

		require.NoError(t, ls.StoreExecutionVC(ctx, "vc-1", "exec-vc-1", "wf-vc", "session-vc", "issuer-1", "target-1", "caller-1", "in-1", "out-1", "verified", []byte(`{"vc":1}`), "sig-1", "s3://one", 123))
		require.NoError(t, ls.StoreExecutionVC(ctx, "vc-2", "exec-vc-2", "wf-vc", "session-vc", "issuer-2", "target-2", "caller-2", "in-2", "out-2", string(types.ExecutionStatusFailed), []byte(`{"vc":2}`), "sig-2", "s3://two", 456))
		require.NoError(t, ls.StoreExecutionVC(ctx, "vc-1", "exec-vc-1", "wf-vc", "session-vc", "issuer-1", "target-1", "caller-1", "in-1", "out-1", "rotated", []byte(`{"vc":1,"updated":true}`), "sig-1b", "s3://updated", 789))

		vcInfo, err := ls.GetExecutionVC(ctx, "vc-1")
		require.NoError(t, err)
		require.Equal(t, "rotated", vcInfo.Status)
		fullDoc, signature, err := ls.GetFullExecutionVC("vc-1")
		require.NoError(t, err)
		require.Equal(t, json.RawMessage(`{"vc":1,"updated":true}`), fullDoc)
		require.Equal(t, "sig-1b", signature)

		search := "vc workflow"
		listed, err := ls.ListExecutionVCs(ctx, types.VCFilters{
			WorkflowID:  strPtr("wf-vc"),
			SessionID:   strPtr("session-vc"),
			IssuerDID:   strPtr("issuer-1"),
			TargetDID:   strPtr("target-1"),
			CallerDID:   strPtr("caller-1"),
			AgentNodeID: strPtr("agent-vc"),
			Search:      &search,
			Limit:       10,
		})
		require.NoError(t, err)
		require.Len(t, listed, 1)
		require.Equal(t, workflowName, *listed[0].WorkflowName)

		total, err := ls.CountExecutionVCs(ctx, types.VCFilters{WorkflowID: strPtr("wf-vc")})
		require.NoError(t, err)
		require.Equal(t, 2, total)

		_, err = ls.ListWorkflowVCStatusSummaries(ctx, []string{"wf-vc"})
		require.EqualError(t, err, "failed to scan workflow VC status summary: sql: Scan error on column index 4, name \"last_created_at\": unsupported Scan, storing driver.Value type string into type *time.Time")
		emptySummaries, err := ls.ListWorkflowVCStatusSummaries(ctx, nil)
		require.NoError(t, err)
		require.Empty(t, emptySummaries)

		startTime := now.Add(-1 * time.Minute)
		endTime := now
		require.NoError(t, ls.StoreWorkflowVC(ctx, "wvc-1", "wf-vc", "session-vc", []string{"vc-1", "vc-2"}, "running", &startTime, nil, 3, 1, "s3://workflow", 111))
		require.NoError(t, ls.StoreWorkflowVC(ctx, "wvc-1", "wf-vc", "session-vc", []string{"vc-1", "vc-2"}, "completed", &startTime, &endTime, 3, 3, "s3://workflow-final", 222))
		workflowVC, err := ls.GetWorkflowVC(ctx, "wvc-1")
		require.NoError(t, err)
		require.Equal(t, "completed", workflowVC.Status)
		require.Equal(t, []string{"vc-1", "vc-2"}, workflowVC.ComponentVCIDs)
		workflowVCs, err := ls.ListWorkflowVCs(ctx, "wf-vc")
		require.NoError(t, err)
		require.Len(t, workflowVCs, 1)
		allWorkflowVCs, err := ls.ListWorkflowVCs(ctx, "")
		require.NoError(t, err)
		require.Len(t, allWorkflowVCs, 1)
	})
}

func TestAccessPolicyAgentTagAndHelperCoverage(t *testing.T) {
	t.Run("covers env and sql builder helpers", func(t *testing.T) {
		t.Setenv("AGENTFIELD_TEST_INT", "")
		require.Equal(t, 5, resolveEnvInt("AGENTFIELD_TEST_INT", 5))
		t.Setenv("AGENTFIELD_TEST_INT", "9")
		require.Equal(t, 9, resolveEnvInt("AGENTFIELD_TEST_INT", 5))
		t.Setenv("AGENTFIELD_TEST_INT", "oops")
		require.Equal(t, 5, resolveEnvInt("AGENTFIELD_TEST_INT", 5))

		require.Equal(t, `"my""db"`, quotePostgresIdentifier(`my"db`))
		require.False(t, isPostgresDatabaseMissingError(nil))
		require.True(t, isPostgresDatabaseMissingError(assertErrString("database does not exist")))
		require.False(t, isPostgresDatabaseAlreadyExistsError(nil))
		require.True(t, isPostgresDatabaseAlreadyExistsError(assertErrString("database already exists")))

		require.Contains(t, buildExecutionVCTableSQL("execution_vcs", true), "CREATE TABLE IF NOT EXISTS execution_vcs")
		require.Contains(t, buildExecutionVCTableSQL("execution_vcs", false), "FOREIGN KEY (parent_vc_id)")
		require.Contains(t, buildWorkflowVCTableSQL("workflow_vcs", true), "CREATE TABLE IF NOT EXISTS workflow_vcs")
		require.Contains(t, buildWorkflowVCTableSQL("workflow_vcs", false), "component_vc_ids TEXT DEFAULT '[]'")
	})

	t.Run("reconstructs agent tags and filters by lifecycle status", func(t *testing.T) {
		agent := &types.AgentNode{
			Reasoners: []types.ReasonerDefinition{
				{ID: "r1", ApprovedTags: []string{"approved-a", "approved-b"}, ProposedTags: []string{"proposed-a"}},
				{ID: "r2", ApprovedTags: []string{"approved-b"}, Tags: []string{"fallback-r2"}},
			},
			Skills: []types.SkillDefinition{
				{ID: "s1", ApprovedTags: []string{"approved-c"}, ProposedTags: []string{"proposed-s1"}},
				{ID: "s2", Tags: []string{"fallback-s2"}},
			},
		}
		reconstructAgentLevelTags(agent)
		require.ElementsMatch(t, []string{"approved-a", "approved-b", "approved-c"}, agent.ApprovedTags)
		require.ElementsMatch(t, []string{"proposed-a", "fallback-r2", "proposed-s1", "fallback-s2"}, agent.ProposedTags)

		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC()
		invocationURL := "https://agent.example.com/invoke"
		readyAgent := &types.AgentNode{
			ID:               "agent-ready",
			GroupID:          "group-a",
			TeamID:           "team-a",
			BaseURL:          "https://agent-ready.example.com",
			Version:          "1.0.0",
			TrafficWeight:    100,
			DeploymentType:   "serverless",
			InvocationURL:    &invocationURL,
			Reasoners:        []types.ReasonerDefinition{{ID: "summarize"}},
			Skills:           []types.SkillDefinition{{ID: "tool-a"}},
			HealthStatus:     types.HealthStatusActive,
			LifecycleStatus:  types.AgentStatusReady,
			LastHeartbeat:    now,
			RegisteredAt:     now,
			CommunicationConfig: types.CommunicationConfig{Protocols: []string{"http"}},
		}
		offlineAgent := &types.AgentNode{
			ID:               "agent-offline",
			GroupID:          "group-b",
			TeamID:           "team-b",
			BaseURL:          "https://agent-offline.example.com",
			Version:          "1.0.0",
			TrafficWeight:    50,
			DeploymentType:   "long_running",
			Reasoners:        []types.ReasonerDefinition{{ID: "offline"}},
			Skills:           []types.SkillDefinition{},
			HealthStatus:     types.HealthStatusInactive,
			LifecycleStatus:  types.AgentStatusOffline,
			LastHeartbeat:    now.Add(-time.Hour),
			RegisteredAt:     now.Add(-time.Hour),
			CommunicationConfig: types.CommunicationConfig{Protocols: []string{"grpc"}},
		}
		require.NoError(t, ls.RegisterAgent(ctx, readyAgent))
		require.NoError(t, ls.RegisterAgent(ctx, offlineAgent))

		filtered, err := ls.ListAgentsByLifecycleStatus(ctx, types.AgentStatusReady)
		require.NoError(t, err)
		require.Len(t, filtered, 1)
		require.Equal(t, "agent-ready", filtered[0].ID)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err = ls.ListAgentsByLifecycleStatus(cancelled, types.AgentStatusReady)
		require.EqualError(t, err, "context cancelled during list agents by lifecycle status: context canceled")
	})

	t.Run("stores access policies and agent tag vcs", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)
		description := "allow finance calls"
		policy := &types.AccessPolicy{
			Name:           "finance-allow",
			CallerTags:     []string{"finance", "trusted"},
			TargetTags:     []string{"payments"},
			AllowFunctions: []string{"charge"},
			DenyFunctions:  []string{"refund"},
			Constraints: map[string]types.AccessConstraint{
				"amount": {Operator: "<=", Value: 1000},
			},
			Action:      "allow",
			Priority:    10,
			Enabled:     true,
			Description: &description,
			CreatedAt:   now,
			UpdatedAt:   now,
		}
		require.NoError(t, ls.CreateAccessPolicy(ctx, policy))
		require.NotZero(t, policy.ID)
		require.EqualError(t, ls.CreateAccessPolicy(ctx, policy), `access policy with name "finance-allow" already exists`)

		policies, err := ls.GetAccessPolicies(ctx)
		require.NoError(t, err)
		require.Len(t, policies, 1)
		require.Equal(t, "finance-allow", policies[0].Name)

		loaded, err := ls.GetAccessPolicyByID(ctx, policy.ID)
		require.NoError(t, err)
		require.Equal(t, []string{"finance", "trusted"}, loaded.CallerTags)
		require.Equal(t, "allow", loaded.Action)

		policy.Priority = 20
		policy.Enabled = false
		policy.UpdatedAt = now.Add(time.Minute)
		require.NoError(t, ls.UpdateAccessPolicy(ctx, policy))
		updated, err := ls.GetAccessPolicyByID(ctx, policy.ID)
		require.NoError(t, err)
		require.Equal(t, 20, updated.Priority)
		require.False(t, updated.Enabled)

		emptyPolicies, err := ls.GetAccessPolicies(ctx)
		require.NoError(t, err)
		require.Empty(t, emptyPolicies)

		missing := *policy
		missing.ID = 999999
		require.EqualError(t, ls.UpdateAccessPolicy(ctx, &missing), "access policy with ID 999999 not found")
		require.EqualError(t, ls.DeleteAccessPolicy(ctx, missing.ID), "access policy with ID 999999 not found")
		require.NoError(t, ls.DeleteAccessPolicy(ctx, policy.ID))
		_, err = ls.GetAccessPolicyByID(ctx, policy.ID)
		require.Error(t, err)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.CreateAccessPolicy(cancelled, policy), "context cancelled during create access policy: context canceled")
		require.EqualError(t, ls.createAccessPolicyPostgres(cancelled, policy), "context cancelled during create access policy: context canceled")

		expiresAt := now.Add(24 * time.Hour)
		require.NoError(t, ls.StoreAgentTagVC(ctx, "agent-1", "did:agent:1", "vc-tag-1", `{"vc":"one"}`, "sig-1", now, &expiresAt))
		require.NoError(t, ls.StoreAgentTagVC(ctx, "agent-1", "did:agent:1", "vc-tag-2", `{"vc":"two"}`, "sig-2", now.Add(time.Minute), nil))
		tagVC, err := ls.GetAgentTagVC(ctx, "agent-1")
		require.NoError(t, err)
		require.Equal(t, "vc-tag-2", tagVC.VCID)
		listed, err := ls.ListAgentTagVCs(ctx)
		require.NoError(t, err)
		require.Len(t, listed, 1)
		require.NoError(t, ls.RevokeAgentTagVC(ctx, "agent-1"))
		listed, err = ls.ListAgentTagVCs(ctx)
		require.NoError(t, err)
		require.Empty(t, listed)
		require.EqualError(t, ls.RevokeAgentTagVC(ctx, "agent-1"), "no active agent tag VC found for agent agent-1")
		_, err = ls.GetAgentTagVC(ctx, "missing")
		require.EqualError(t, err, "agent tag VC not found for agent missing")
		require.EqualError(t, ls.StoreAgentTagVC(cancelled, "agent-2", "did:agent:2", "vc-tag-3", `{}`, "sig", now, nil), "context cancelled during store agent tag VC: context canceled")

		badPolicy := &types.AccessPolicy{}
		require.EqualError(t, unmarshalAccessPolicyJSON(badPolicy, "{", "", "", "", ""), "failed to unmarshal caller_tags: unexpected end of JSON input")
		_, _, _, _, _, err = marshalAccessPolicyJSON(&types.AccessPolicy{
			Constraints: map[string]types.AccessConstraint{
				"bad": {Operator: "==", Value: func() {}},
			},
		})
		require.EqualError(t, err, "constraints: json: unsupported type: func()")
	})
}

func TestWorkflowRunAndSchemaCoverage(t *testing.T) {
	t.Run("stores workflow runs events steps and loads executions", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)

		require.EqualError(t, ls.StoreWorkflowRun(ctx, nil), "workflow run cannot be nil")
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreWorkflowRun(cancelled, &types.WorkflowRun{}), "context canceled")

		rootExecutionID := "exec-root"
		completedAt := now.Add(time.Minute)
		run := &types.WorkflowRun{
			RunID:             "run-coverage",
			RootWorkflowID:    "wf-root",
			RootExecutionID:   &rootExecutionID,
			Status:            string(types.ExecutionStatusRunning),
			TotalSteps:        3,
			CompletedSteps:    1,
			FailedSteps:       0,
			StateVersion:      2,
			LastEventSequence: 4,
			Metadata:          json.RawMessage(`{"kind":"coverage"}`),
			CreatedAt:         now,
			UpdatedAt:         now,
			CompletedAt:       &completedAt,
		}
		require.NoError(t, ls.StoreWorkflowRun(ctx, run))

		loadedRun, err := ls.GetWorkflowRun(ctx, run.RunID)
		require.NoError(t, err)
		require.Equal(t, run.RunID, loadedRun.RunID)
		require.Equal(t, json.RawMessage(`{"kind":"coverage"}`), loadedRun.Metadata)
		require.Equal(t, &rootExecutionID, loadedRun.RootExecutionID)
		require.Equal(t, &completedAt, loadedRun.CompletedAt)
		missingRun, err := ls.GetWorkflowRun(ctx, "missing")
		require.NoError(t, err)
		require.Nil(t, missingRun)
		_, err = ls.GetWorkflowRun(ctx, "")
		require.EqualError(t, err, "run_id cannot be empty")

		require.EqualError(t, ls.StoreWorkflowRunEvent(ctx, nil), "workflow run event cannot be nil")
		require.EqualError(t, ls.StoreWorkflowRunEvent(cancelled, &types.WorkflowRunEvent{}), "context canceled")
		status := string(types.ExecutionStatusRunning)
		statusReason := "queued"
		require.NoError(t, ls.StoreWorkflowRunEvent(ctx, &types.WorkflowRunEvent{
			RunID:            run.RunID,
			Sequence:         1,
			PreviousSequence: 0,
			EventType:        "workflow.started",
			Status:           &status,
			StatusReason:     &statusReason,
			EmittedAt:        now,
		}))
		var payload string
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT payload FROM workflow_run_events WHERE run_id = ?`, run.RunID).Scan(&payload))
		require.Equal(t, "{}", payload)

		require.EqualError(t, ls.StoreWorkflowStep(ctx, nil), "workflow step cannot be nil")
		require.EqualError(t, ls.StoreWorkflowStep(cancelled, &types.WorkflowStep{}), "context canceled")
		step := &types.WorkflowStep{
			StepID:       "step-1",
			RunID:        run.RunID,
			ExecutionID:  strPtr("exec-step-1"),
			AgentNodeID:  strPtr("agent-step"),
			Target:       strPtr("agent.execute"),
			Status:       "pending",
			Attempt:      1,
			Priority:     5,
			InputURI:     strPtr("s3://input"),
			ResultURI:    strPtr("s3://result"),
			ErrorMessage: strPtr(""),
		}
		require.NoError(t, ls.StoreWorkflowStep(ctx, step))
		var storedMetadata string
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT metadata FROM workflow_steps WHERE step_id = ?`, step.StepID).Scan(&storedMetadata))
		require.Equal(t, "{}", storedMetadata)

		approvalStatus := "pending"
		notes := []types.ExecutionNote{{Message: "note one", Timestamp: now}}
		tags := []string{"one", "two"}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
			WorkflowID:           "wf-root",
			ExecutionID:          "exec-loaded",
			AgentFieldRequestID:  "req-loaded",
			RunID:                &run.RunID,
			SessionID:            strPtr("session-coverage"),
			ActorID:              strPtr("actor-coverage"),
			AgentNodeID:          "agent-loaded",
			ReasonerID:           "reasoner-loaded",
			Status:               string(types.ExecutionStatusWaiting),
			PendingTerminalStatus: strPtr(string(types.ExecutionStatusCancelled)),
			StatusReason:         strPtr("human_approval"),
			LeaseOwner:           strPtr("lease-owner"),
			LeaseExpiresAt:       timePtrLocal(now.Add(5 * time.Minute)),
			ApprovalRequestID:    strPtr("approval-1"),
			ApprovalRequestURL:   strPtr("https://approve"),
			ApprovalStatus:       &approvalStatus,
			ApprovalResponse:     strPtr(`{"approved":false}`),
			ApprovalRequestedAt:  timePtrLocal(now.Add(-time.Minute)),
			ApprovalRespondedAt:  timePtrLocal(now),
			ApprovalCallbackURL:  strPtr("https://callback"),
			ApprovalExpiresAt:    timePtrLocal(now.Add(10 * time.Minute)),
			WorkflowName:         strPtr("Loaded Workflow"),
			WorkflowTags:         tags,
			Notes:                notes,
			StartedAt:            now,
			CreatedAt:            now,
			UpdatedAt:            now,
		}))
		_, err = ls.db.ExecContext(ctx, `UPDATE workflow_executions SET input_data = ?, output_data = ? WHERE execution_id = ?`, `not-json`, `also-bad`, "exec-loaded")
		require.NoError(t, err)

		execution, err := ls.getWorkflowExecutionByID(ctx, ls.db, "exec-loaded")
		require.NoError(t, err)
		require.Equal(t, run.RunID, *execution.RunID)
		require.NotNil(t, execution.PendingTerminalStatus)
		require.Equal(t, approvalStatus, *execution.ApprovalStatus)
		require.Contains(t, string(execution.InputData), "corrupted_json_data")
		require.Contains(t, string(execution.OutputData), "corrupted_json_data")
		require.Equal(t, tags, execution.WorkflowTags)
		require.Len(t, execution.Notes, 1)

		missingExec, err := ls.getWorkflowExecutionByID(ctx, ls.db, "missing-exec")
		require.NoError(t, err)
		require.Nil(t, missingExec)
	})

	t.Run("migrates legacy vc schemas in place", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)

		_, err := ls.db.ExecContext(ctx, `DROP TABLE IF EXISTS execution_vcs`)
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `DROP TABLE IF EXISTS workflow_vcs`)
		require.NoError(t, err)

		_, err = ls.db.ExecContext(ctx, `
			CREATE TABLE execution_vcs (
				vc_id TEXT PRIMARY KEY,
				execution_id TEXT NOT NULL,
				workflow_id TEXT NOT NULL,
				session_id TEXT NOT NULL,
				issuer_did TEXT NOT NULL,
				target_did TEXT,
				caller_did TEXT NOT NULL,
				vc_document TEXT NOT NULL,
				signature TEXT NOT NULL,
				storage_uri TEXT DEFAULT '',
				document_size_bytes INTEGER DEFAULT 0,
				input_hash TEXT NOT NULL,
				output_hash TEXT NOT NULL,
				status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'revoked')),
				parent_vc_id TEXT,
				child_vc_ids TEXT DEFAULT '[]',
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (parent_vc_id) REFERENCES component_dids(did)
			)
		`)
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `
			INSERT INTO execution_vcs (
				vc_id, execution_id, workflow_id, session_id, issuer_did, target_did, caller_did,
				vc_document, signature, input_hash, output_hash, status, child_vc_ids
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, "vc-legacy", "exec-legacy", "wf-legacy", "session-legacy", "issuer", "target", "caller", `{"legacy":true}`, "sig", "in", "out", "failed", `["child"]`)
		require.NoError(t, err)

		_, err = ls.db.ExecContext(ctx, `
			CREATE TABLE workflow_vcs (
				workflow_vc_id TEXT PRIMARY KEY,
				workflow_id TEXT NOT NULL,
				session_id TEXT NOT NULL,
				component_vc_ids TEXT DEFAULT '[]',
				status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
				start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				end_time TIMESTAMP,
				total_steps INTEGER DEFAULT 0,
				completed_steps INTEGER DEFAULT 0,
				storage_uri TEXT DEFAULT '',
				document_size_bytes INTEGER DEFAULT 0,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		`)
		require.NoError(t, err)
		_, err = ls.db.ExecContext(ctx, `
			INSERT INTO workflow_vcs (
				workflow_vc_id, workflow_id, session_id, component_vc_ids, status, total_steps, completed_steps
			) VALUES (?, ?, ?, ?, ?, ?, ?)
		`, "wvc-legacy", "wf-legacy", "session-legacy", `["vc-legacy"]`, "failed", 2, 1)
		require.NoError(t, err)

		require.NoError(t, ls.ensureExecutionVCSchema())
		require.NoError(t, ls.ensureWorkflowVCSchema())

		var executionSchema string
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT sql FROM sqlite_master WHERE type='table' AND name='execution_vcs'`).Scan(&executionSchema))
		require.Contains(t, executionSchema, "status IN ('unknown', 'pending', 'queued', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked')")
		require.NotContains(t, executionSchema, "REFERENCES component_dids")

		var workflowSchema string
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT sql FROM sqlite_master WHERE type='table' AND name='workflow_vcs'`).Scan(&workflowSchema))
		require.Contains(t, workflowSchema, "status IN ('unknown', 'pending', 'in_progress', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout')")

		vc, err := ls.GetExecutionVC(ctx, "vc-legacy")
		require.NoError(t, err)
		require.Equal(t, "failed", vc.Status)
		workflowVC, err := ls.GetWorkflowVC(ctx, "wvc-legacy")
		require.NoError(t, err)
		require.Equal(t, "failed", workflowVC.Status)
		require.Equal(t, []string{"vc-legacy"}, workflowVC.ComponentVCIDs)
	})
}

func int64Ptr(v int64) *int64 {
	return &v
}

func timePtrLocal(v time.Time) *time.Time {
	return &v
}

func assertErrString(msg string) error {
	return &testStringError{msg: msg}
}

type testStringError struct {
	msg string
}

func (e *testStringError) Error() string {
	return e.msg
}
