package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmitWorkflowEvent_PopulatesAndSendsPayload(t *testing.T) {
	eventCh := make(chan types.WorkflowExecutionEvent, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/workflow/executions/events", r.URL.Path)
		require.Equal(t, "Bearer token-123", r.Header.Get("Authorization"))
		var event types.WorkflowExecutionEvent
		require.NoError(t, json.NewDecoder(r.Body).Decode(&event))
		eventCh <- event
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Token:         "token-123",
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = server.Client()

	durationMS := int64(42)
	a.emitWorkflowEvent(ExecutionContext{
		ExecutionID:       "exec-1",
		WorkflowID:        "wf-1",
		RunID:             "run-1",
		ReasonerName:      "demo",
		ParentExecutionID: "parent-exec",
		ParentWorkflowID:  "parent-wf",
	}, "failed", map[string]any{"input": true}, map[string]any{"result": true}, fmt.Errorf("boom"), durationMS)

	select {
	case event := <-eventCh:
		assert.Equal(t, "exec-1", event.ExecutionID)
		assert.Equal(t, "wf-1", event.WorkflowID)
		assert.Equal(t, "run-1", event.RunID)
		assert.Equal(t, "demo", event.ReasonerID)
		assert.Equal(t, "demo", event.Type)
		assert.Equal(t, "node-1", event.AgentNodeID)
		assert.Equal(t, "failed", event.Status)
		require.NotNil(t, event.ParentExecutionID)
		assert.Equal(t, "parent-exec", *event.ParentExecutionID)
		require.NotNil(t, event.ParentWorkflowID)
		assert.Equal(t, "parent-wf", *event.ParentWorkflowID)
		assert.Equal(t, map[string]any{"input": true}, event.InputData)
		assert.Equal(t, map[string]any{"result": true}, event.Result)
		assert.Equal(t, "boom", event.Error)
		require.NotNil(t, event.DurationMS)
		assert.Equal(t, durationMS, *event.DurationMS)
	default:
		t.Fatal("expected workflow event")
	}
}

func TestSendWorkflowEvent_AndPostExecutionStatus_ErrorPaths(t *testing.T) {
	t.Run("sendWorkflowEvent returns server errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "denied", http.StatusForbidden)
		}))
		defer server.Close()

		a := &Agent{
			cfg:        Config{AgentFieldURL: server.URL},
			httpClient: server.Client(),
			logger:     log.New(io.Discard, "", 0),
		}

		err := a.sendWorkflowEvent(types.WorkflowExecutionEvent{ExecutionID: "exec-1"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "server returned 403")
	})

	t.Run("postExecutionStatus surfaces request creation failures", func(t *testing.T) {
			a := &Agent{
				httpClient: http.DefaultClient,
				logger:     log.New(io.Discard, "", 0),
			}

		err := a.postExecutionStatus(context.Background(), "http://[::1", []byte(`{"ok":true}`))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "create status request")
	})
}

func TestExecutionLogger_WithSourceAndNormalizeContext(t *testing.T) {
	a := &Agent{cfg: Config{NodeID: "node-1"}}

	logger := a.ExecutionLogger(contextWithExecution(context.Background(), ExecutionContext{RunID: "run-1"}))
	require.NotNil(t, logger)
	derived := logger.WithSource("custom.source")
	require.NotNil(t, derived)
	assert.Equal(t, "custom.source", derived.source)

	normalized := a.normalizeExecutionContext(ExecutionContext{RunID: "run-1"})
	assert.Equal(t, "run-1", normalized.WorkflowID)
	assert.Equal(t, "run-1", normalized.RootWorkflowID)
	assert.Equal(t, "node-1", normalized.AgentNodeID)

	assert.Nil(t, (*ExecutionLogger)(nil).WithSource("ignored"))
}
