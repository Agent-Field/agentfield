package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHandleExecute_ErrorBranchesAndInputVariant(t *testing.T) {
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "http://example.com",
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
		return input, nil
	})

	req := httptest.NewRequest(http.MethodGet, "/execute/echo", nil)
	resp := httptest.NewRecorder()
	a.handleExecute(resp, req)
	assert.Equal(t, http.StatusMethodNotAllowed, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/execute/echo", strings.NewReader("{"))
	resp = httptest.NewRecorder()
	a.handleExecute(resp, req)
	assert.Equal(t, http.StatusBadRequest, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/execute", strings.NewReader(`{}`))
	resp = httptest.NewRecorder()
	a.handleExecute(resp, req)
	assert.Equal(t, http.StatusBadRequest, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/execute/missing", strings.NewReader(`{}`))
	resp = httptest.NewRecorder()
	a.handleExecute(resp, req)
	assert.Equal(t, http.StatusNotFound, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/execute", strings.NewReader(`{"reasoner":"echo","input":"hello"}`))
	resp = httptest.NewRecorder()
	a.handleExecute(resp, req)
	assert.Equal(t, http.StatusOK, resp.Code)
	assert.JSONEq(t, `{"value":"hello"}`, resp.Body.String())
}

func TestHandleReasoner_ErrorBranchesAndAsyncAcceptance(t *testing.T) {
	statusCh := make(chan map[string]any, 1)
	var statusPosts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/status"):
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			statusPosts.Add(1)
			statusCh <- payload
			w.WriteHeader(http.StatusNoContent)
		case strings.HasSuffix(r.URL.Path, "/logs"):
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = server.Client()
	a.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": input["ok"]}, nil
	})
	a.RegisterReasoner("denied", func(context.Context, map[string]any) (any, error) {
		return nil, &ExecuteError{StatusCode: http.StatusForbidden, Message: "denied", ErrorDetails: map[string]any{"kind": "policy"}}
	})

	req := httptest.NewRequest(http.MethodGet, "/reasoners/echo", nil)
	resp := httptest.NewRecorder()
	a.handleReasoner(resp, req)
	assert.Equal(t, http.StatusMethodNotAllowed, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/reasoners/", bytes.NewBufferString(`{}`))
	resp = httptest.NewRecorder()
	a.handleReasoner(resp, req)
	assert.Equal(t, http.StatusNotFound, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/reasoners/echo", bytes.NewBufferString(`{`))
	resp = httptest.NewRecorder()
	a.handleReasoner(resp, req)
	assert.Equal(t, http.StatusBadRequest, resp.Code)

	req = httptest.NewRequest(http.MethodPost, "/reasoners/denied", bytes.NewBufferString(`{}`))
	resp = httptest.NewRecorder()
	a.handleReasoner(resp, req)
	assert.Equal(t, http.StatusForbidden, resp.Code)
	assert.JSONEq(t, `{"error":"denied","error_details":{"kind":"policy"}}`, resp.Body.String())

	req = httptest.NewRequest(http.MethodPost, "/reasoners/echo", bytes.NewBufferString(`{"ok":true}`))
	req.Header.Set("X-Execution-ID", "exec-async")
	req.Header.Set("X-Run-ID", "run-async")
	resp = httptest.NewRecorder()
	a.handleReasoner(resp, req)
	assert.Equal(t, http.StatusAccepted, resp.Code)
	assert.JSONEq(t, `{"status":"processing","execution_id":"exec-async","run_id":"run-async","reasoner_name":"echo"}`, resp.Body.String())

	select {
	case payload := <-statusCh:
		assert.Equal(t, "succeeded", payload["status"])
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for async status payload")
	}
	assert.Equal(t, int32(1), statusPosts.Load())
}

func TestWriteJSONAndRawToMap_HandleEdgeCases(t *testing.T) {
	resp := httptest.NewRecorder()
	writeJSON(resp, http.StatusNoContent, nil)
	assert.Equal(t, http.StatusNoContent, resp.Code)
	assert.Empty(t, resp.Body.String())

	resp = httptest.NewRecorder()
	writeJSON(resp, http.StatusOK, map[string]any{"bad": math.NaN()})
	assert.Equal(t, http.StatusOK, resp.Code)
	assert.Contains(t, resp.Body.String(), "{}")

	assert.Equal(t, map[string]any{}, rawToMap(nil))
	assert.Equal(t, map[string]any{}, rawToMap(json.RawMessage(`{`)))
	assert.Equal(t, map[string]any{"ok": true}, rawToMap(json.RawMessage(`{"ok":true}`)))
}

func TestPostExecutionStatus_RetriesAfterServerFailure(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts.Add(1)
		if attempt == 1 {
			http.Error(w, "retry", http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	a := &Agent{
		cfg: Config{Token: "token-123"},
		httpClient: server.Client(),
		logger: log.New(io.Discard, "", 0),
	}

	err := a.postExecutionStatus(context.Background(), server.URL, []byte(`{"ok":true}`))
	require.NoError(t, err)
	assert.Equal(t, int32(2), attempts.Load())
}
