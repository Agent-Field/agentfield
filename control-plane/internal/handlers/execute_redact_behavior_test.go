package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// These tests pin the security-observable behavior added in #701: execution
// input/output payloads must NOT appear in the published execution event data
// when redaction is enabled (the safe default), and MUST appear when an
// operator has explicitly opted out. They assert the emitted event contents —
// the thing a log/SSE consumer actually sees — rather than internal state.

// captureNextEvent subscribes to the given bus, runs action, and returns the
// first published event (or fails the test on timeout). Subscribing to the
// per-store bus keeps each test isolated from the process-global event bus.
func captureNextEvent(t *testing.T, bus *events.ExecutionEventBus, action func()) events.ExecutionEvent {
	t.Helper()
	ch := bus.Subscribe("redact-behavior-test")
	defer bus.Unsubscribe("redact-behavior-test")

	action()

	select {
	case ev := <-ch:
		return ev
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for execution event to be published")
		return events.ExecutionEvent{}
	}
}

func eventDataMap(t *testing.T, ev events.ExecutionEvent) map[string]interface{} {
	t.Helper()
	data, ok := ev.Data.(map[string]interface{})
	require.True(t, ok, "event data should be a map, got %T", ev.Data)
	return data
}

// envelope is the persisted input shape: an input payload plus caller context.
const redactEnvelopeInput = `{"input":{"prompt":"my-secret-prompt"},"context":{"api_key":"super-secret"}}`

func newRunningExecution(id string) *types.Execution {
	now := time.Now().UTC()
	return &types.Execution{
		ExecutionID:  id,
		RunID:        "run-" + id,
		AgentNodeID:  "node-1",
		ReasonerID:   "reasoner-a",
		Status:       types.ExecutionStatusRunning,
		InputPayload: []byte(redactEnvelopeInput),
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
}

func redactTestAgent() *types.AgentNode {
	return &types.AgentNode{
		ID:              "node-1",
		BaseURL:         "https://example.com",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}
}

func TestCompleteExecution_RedactionBehavior(t *testing.T) {
	cases := []struct {
		name        string
		redact      bool
		wantPayload bool
	}{
		{name: "redaction enabled omits payloads", redact: true, wantPayload: false},
		{name: "redaction disabled includes payloads", redact: false, wantPayload: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			agent := redactTestAgent()
			store := newTestExecutionStorage(agent)
			payloads := services.NewFilePayloadStore(t.TempDir())
			controller := newExecutionController(store, payloads, nil, time.Second, "")
			controller.redactPayloads = tc.redact

			require.NoError(t, store.CreateExecutionRecord(context.Background(), newRunningExecution("exec-complete")))

			plan := &preparedExecution{
				exec: &types.Execution{
					ExecutionID:  "exec-complete",
					RunID:        "run-exec-complete",
					InputPayload: []byte(redactEnvelopeInput),
				},
				agent:  agent,
				target: &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a"},
			}

			ev := captureNextEvent(t, store.GetExecutionEventBus(), func() {
				require.NoError(t, controller.completeExecution(context.Background(), plan, []byte(`{"answer":"the-secret-result"}`), 50*time.Millisecond))
			})

			data := eventDataMap(t, ev)
			require.Equal(t, string(types.ExecutionStatusSucceeded), ev.Status)

			if tc.wantPayload {
				require.Contains(t, data, "result", "result must be present when redaction is disabled")
				require.Contains(t, data, "input", "input must be present when redaction is disabled")
				require.Contains(t, data, "context", "context must be present when redaction is disabled")
				result, _ := data["result"].(map[string]interface{})
				require.Equal(t, "the-secret-result", result["answer"])
				ctxData, _ := data["context"].(map[string]interface{})
				require.Equal(t, "super-secret", ctxData["api_key"])
			} else {
				require.NotContains(t, data, "result", "result must be redacted by default")
				require.NotContains(t, data, "input", "input must be redacted by default")
				require.NotContains(t, data, "context", "context must be redacted by default")
			}
		})
	}
}

func TestFailExecution_RedactionBehavior(t *testing.T) {
	cases := []struct {
		name        string
		redact      bool
		wantPayload bool
	}{
		{name: "redaction enabled omits payloads but keeps error", redact: true, wantPayload: false},
		{name: "redaction disabled includes payloads", redact: false, wantPayload: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			agent := redactTestAgent()
			store := newTestExecutionStorage(agent)
			payloads := services.NewFilePayloadStore(t.TempDir())
			controller := newExecutionController(store, payloads, nil, time.Second, "")
			controller.redactPayloads = tc.redact

			require.NoError(t, store.CreateExecutionRecord(context.Background(), newRunningExecution("exec-fail")))

			plan := &preparedExecution{
				exec: &types.Execution{
					ExecutionID:  "exec-fail",
					RunID:        "run-exec-fail",
					InputPayload: []byte(redactEnvelopeInput),
				},
				agent:  agent,
				target: &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a"},
			}

			ev := captureNextEvent(t, store.GetExecutionEventBus(), func() {
				require.NoError(t, controller.failExecution(context.Background(), plan, errors.New("agent exploded"), 50*time.Millisecond, []byte(`{"detail":"sensitive-boom"}`)))
			})

			data := eventDataMap(t, ev)
			require.Equal(t, string(types.ExecutionStatusFailed), ev.Status)
			// The error message is diagnostic metadata, not payload data, so it
			// is always present regardless of redaction.
			require.Contains(t, data, "error")

			if tc.wantPayload {
				require.Contains(t, data, "result")
				require.Contains(t, data, "input")
				result, _ := data["result"].(map[string]interface{})
				require.Equal(t, "sensitive-boom", result["detail"])
			} else {
				require.NotContains(t, data, "result", "result must be redacted by default")
				require.NotContains(t, data, "input", "input must be redacted by default")
			}
		})
	}
}

func TestCompleteReplayHit_RedactionBehavior(t *testing.T) {
	cases := []struct {
		name        string
		redact      bool
		wantPayload bool
	}{
		{name: "redaction enabled omits payloads but keeps replay metadata", redact: true, wantPayload: false},
		{name: "redaction disabled includes payloads", redact: false, wantPayload: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			agent := redactTestAgent()
			store := newTestExecutionStorage(agent)
			payloads := services.NewFilePayloadStore(t.TempDir())
			controller := newExecutionController(store, payloads, nil, time.Second, "")
			controller.redactPayloads = tc.redact

			require.NoError(t, store.CreateExecutionRecord(context.Background(), newRunningExecution("exec-replay")))

			plan := &preparedExecution{
				exec: &types.Execution{
					ExecutionID:  "exec-replay",
					RunID:        "run-exec-replay",
					InputPayload: []byte(redactEnvelopeInput),
				},
				agent:  agent,
				target: &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a"},
				replayHit: &replayHit{
					SourceExecutionID: "src-exec",
					SourceRunID:       "src-run",
					Result:            json.RawMessage(`{"cached":"secret-cached-value"}`),
				},
			}

			ev := captureNextEvent(t, store.GetExecutionEventBus(), func() {
				require.NoError(t, controller.completeReplayHit(context.Background(), plan))
			})

			data := eventDataMap(t, ev)
			require.Equal(t, string(types.ExecutionStatusSucceeded), ev.Status)
			// Replay provenance metadata is always emitted.
			require.Contains(t, data, "replay")

			if tc.wantPayload {
				require.Contains(t, data, "result")
				require.Contains(t, data, "input")
				result, _ := data["result"].(map[string]interface{})
				require.Equal(t, "secret-cached-value", result["cached"])
			} else {
				require.NotContains(t, data, "result", "result must be redacted by default")
				require.NotContains(t, data, "input", "input must be redacted by default")
			}
		})
	}
}

// TestHandleStatusUpdate_RedactionBehavior drives the real HTTP handler so the
// redaction path exercised by agent status callbacks is covered. The handler
// builds its controller from the package-level default, so we toggle that.
func TestHandleStatusUpdate_RedactionBehavior(t *testing.T) {
	cases := []struct {
		name        string
		redact      bool
		wantPayload bool
	}{
		{name: "redaction enabled omits payloads", redact: true, wantPayload: false},
		{name: "redaction disabled includes payloads", redact: false, wantPayload: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)

			original := defaultRedactPayloads
			defer func() { defaultRedactPayloads = original }()
			SetRedactPayloads(tc.redact)

			agent := redactTestAgent()
			store := newTestExecutionStorage(agent)
			payloads := services.NewFilePayloadStore(t.TempDir())

			require.NoError(t, store.CreateExecutionRecord(context.Background(), newRunningExecution("exec-status")))

			router := gin.New()
			router.PUT("/api/v1/executions/:execution_id/status",
				UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

			ev := captureNextEvent(t, store.GetExecutionEventBus(), func() {
				reqBody := `{"status":"succeeded","result":{"output":"secret-output"},"duration_ms":10}`
				req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-status/status", strings.NewReader(reqBody))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)
				require.Equal(t, http.StatusOK, resp.Code)
			})

			data := eventDataMap(t, ev)

			if tc.wantPayload {
				require.Contains(t, data, "result", "result must be present when redaction is disabled")
				require.Contains(t, data, "input", "input must be present when redaction is disabled")
			} else {
				require.NotContains(t, data, "result", "result must be redacted by default")
				require.NotContains(t, data, "input", "input must be redacted by default")
			}
		})
	}
}
