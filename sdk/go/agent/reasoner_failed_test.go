package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReasonerFailed_ErrorAndErrorsAs(t *testing.T) {
	var err error = &ReasonerFailed{Message: "the work failed"}
	assert.Equal(t, "the work failed", err.Error())

	var rf *ReasonerFailed
	require.True(t, errors.As(err, &rf))
	assert.Equal(t, "the work failed", rf.Message)
}

// newStatusCapturingAgent returns an agent whose async status callbacks are
// delivered to statusCh, plus the server (caller must Close it).
func newStatusCapturingAgent(t *testing.T, statusCh chan map[string]any, failFirst bool, attempts *int32) (*Agent, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Contains(t, r.URL.Path, "/api/v1/executions/")
		if !strings.HasSuffix(r.URL.Path, "/status") {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if attempts != nil {
			n := atomic.AddInt32(attempts, 1)
			if failFirst && n == 1 {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
		}
		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		statusCh <- payload
		w.WriteHeader(http.StatusNoContent)
	}))

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = server.Client()
	return a, server
}

// TestReasonerFailed_AsyncCarriesResultAndDetails maps to the contract:
// handler returns ReasonerFailed{Result: X} -> the posted failed-status body
// contains status=failed, error=Message, result=X (+ error_details).
func TestReasonerFailed_AsyncCarriesResultAndDetails(t *testing.T) {
	statusCh := make(chan map[string]any, 1)
	a, server := newStatusCapturingAgent(t, statusCh, false, nil)
	defer server.Close()

	a.executeReasonerAsync(&Reasoner{
		Name: "fails-with-result",
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			return nil, &ReasonerFailed{
				Message:      "build failed",
				Result:       map[string]any{"issues_done": 0, "pr": "none"},
				ErrorDetails: map[string]any{"reason": "no_merge"},
			}
		},
	}, map[string]any{"x": 1}, ExecutionContext{ExecutionID: "exec-1", RunID: "run-1", WorkflowID: "wf-1"})

	select {
	case payload := <-statusCh:
		assert.Equal(t, "failed", payload["status"])
		assert.Equal(t, "build failed", payload["error"])
		result, ok := payload["result"].(map[string]any)
		require.True(t, ok, "result must be carried onto the failed status")
		assert.EqualValues(t, 0, result["issues_done"])
		assert.Equal(t, "none", result["pr"])
		details, ok := payload["error_details"].(map[string]any)
		require.True(t, ok, "error_details must be carried")
		assert.Equal(t, "no_merge", details["reason"])
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for status callback")
	}
}

// TestReasonerFailed_AsyncPlainErrorHasNoResultKey maps to the contract:
// a plain error -> no result key.
func TestReasonerFailed_AsyncPlainErrorHasNoResultKey(t *testing.T) {
	statusCh := make(chan map[string]any, 1)
	a, server := newStatusCapturingAgent(t, statusCh, false, nil)
	defer server.Close()

	a.executeReasonerAsync(&Reasoner{
		Name: "plain-error",
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			return nil, fmt.Errorf("boom")
		},
	}, map[string]any{"x": 1}, ExecutionContext{ExecutionID: "exec-2", RunID: "run-2", WorkflowID: "wf-2"})

	select {
	case payload := <-statusCh:
		assert.Equal(t, "failed", payload["status"])
		assert.Equal(t, "boom", payload["error"])
		_, hasResult := payload["result"]
		assert.False(t, hasResult, "plain error must not carry a result key")
		_, hasDetails := payload["error_details"]
		assert.False(t, hasDetails, "plain error must not carry an error_details key")
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for status callback")
	}
}

// TestReasonerFailed_AsyncRetriesOn5xxAndDeliversResult maps to the contract:
// the post is still retried (5x) on 5xx, and the result survives the retry.
func TestReasonerFailed_AsyncRetriesOn5xxAndDeliversResult(t *testing.T) {
	statusCh := make(chan map[string]any, 1)
	var attempts int32
	a, server := newStatusCapturingAgent(t, statusCh, true, &attempts)
	defer server.Close()

	a.executeReasonerAsync(&Reasoner{
		Name: "retryable",
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			return nil, &ReasonerFailed{Message: "x", Result: map[string]any{"k": "v"}}
		},
	}, map[string]any{"x": 1}, ExecutionContext{ExecutionID: "exec-3", RunID: "run-3", WorkflowID: "wf-3"})

	select {
	case payload := <-statusCh:
		assert.Equal(t, "failed", payload["status"])
		result, ok := payload["result"].(map[string]any)
		require.True(t, ok, "result must survive the retry")
		assert.Equal(t, "v", result["k"])
		assert.GreaterOrEqual(t, atomic.LoadInt32(&attempts), int32(2), "a 5xx must have triggered at least one retry")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for retried status callback")
	}
}

// TestReasonerFailed_SyncReasonerResponseCarriesResult exercises the sync
// handleReasoner path (agent has no control-plane URL, so no async dispatch).
func TestReasonerFailed_SyncReasonerResponseCarriesResult(t *testing.T) {
	a, err := New(Config{NodeID: "node-1", Version: "1.0.0", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	a.RegisterReasoner("failing", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, &ReasonerFailed{
			Message:      "sync fail",
			Result:       map[string]any{"k": "v"},
			ErrorDetails: map[string]any{"reason": "boom"},
		}
	})

	req := httptest.NewRequest(http.MethodPost, "/reasoners/failing", bytes.NewBufferString(`{"input":{}}`))
	rec := httptest.NewRecorder()
	a.Handler().ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	var resp map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "sync fail", resp["error"])
	result, ok := resp["result"].(map[string]any)
	require.True(t, ok, "sync error response must carry the result")
	assert.Equal(t, "v", result["k"])
	details, ok := resp["error_details"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "boom", details["reason"])
}

// TestReasonerFailed_SyncExecuteResponseCarriesResult exercises the sync
// handleExecute path.
func TestReasonerFailed_SyncExecuteResponseCarriesResult(t *testing.T) {
	a, err := New(Config{NodeID: "node-1", Version: "1.0.0", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	a.RegisterReasoner("failing", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, &ReasonerFailed{Message: "exec fail", Result: map[string]any{"n": 2}}
	})

	req := httptest.NewRequest(http.MethodPost, "/execute/failing", bytes.NewBufferString(`{"input":{}}`))
	rec := httptest.NewRecorder()
	a.Handler().ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	var resp map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "exec fail", resp["error"])
	result, ok := resp["result"].(map[string]any)
	require.True(t, ok, "sync execute error response must carry the result")
	assert.EqualValues(t, 2, result["n"])
}

// TestReasonerFailed_SyncPlainErrorHasNoResultKey guards the negative: a plain
// error on the sync path must not synthesize a result key.
func TestReasonerFailed_SyncPlainErrorHasNoResultKey(t *testing.T) {
	a, err := New(Config{NodeID: "node-1", Version: "1.0.0", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	a.RegisterReasoner("plain", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, fmt.Errorf("plain boom")
	})

	req := httptest.NewRequest(http.MethodPost, "/reasoners/plain", bytes.NewBufferString(`{"input":{}}`))
	rec := httptest.NewRecorder()
	a.Handler().ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	var resp map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "plain boom", resp["error"])
	_, hasResult := resp["result"]
	assert.False(t, hasResult)
}
