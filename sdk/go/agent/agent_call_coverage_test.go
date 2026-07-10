package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests close the async-submit + poll error/edge branches of Agent.Call
// that the happy-path tests in agent_test.go do not reach. Each is derived from
// the behavior contract of the async Call rewrite (submit -> poll -> terminal
// mapping, error propagation, ctx handling), not from mirroring the code.

// callErrBody is a response body whose Read always fails, used to exercise the
// io.ReadAll error branches on both the submit response and each status poll.
type callErrBody struct{}

func (callErrBody) Read([]byte) (int, error) { return 0, errors.New("simulated body read failure") }
func (callErrBody) Close() error             { return nil }

// newCallTestAgent builds a minimal Agent for exercising Call's helpers. The
// AgentFieldURL is a harmless placeholder — the submit/poll base URL is passed
// explicitly to the helper under test, so no real network is used unless a
// test wires a fake transport.
func newCallTestAgent(t *testing.T, agentFieldURL string) *Agent {
	t.Helper()
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: agentFieldURL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	return a
}

// --- nextCallPollInterval clamps -------------------------------------------

// Contract: the jittered poll interval is always clamped into
// [callMinPollInterval, callMaxPollInterval]. A tiny input clamps up to the
// floor; a huge input clamps down to the ceiling; an in-range input is returned
// jittered without clamping.
func TestNextCallPollInterval_Clamps(t *testing.T) {
	// Jitter is uniform(0.8, 1.2)*current, so these bounds hold for every draw.
	for i := 0; i < 64; i++ {
		// 1ms * 1.2 = 1.2ms, always below the 50ms floor -> clamps up.
		assert.Equal(t, callMinPollInterval, nextCallPollInterval(1*time.Millisecond),
			"sub-floor interval must clamp up to callMinPollInterval")

		// 10s * 0.8 = 8s, always above the 4s ceiling -> clamps down.
		assert.Equal(t, callMaxPollInterval, nextCallPollInterval(10*time.Second),
			"super-ceiling interval must clamp down to callMaxPollInterval")

		// 1s jittered stays within [0.8s, 1.2s] -> no clamp, value returned as-is.
		mid := nextCallPollInterval(1 * time.Second)
		assert.GreaterOrEqual(t, mid, 800*time.Millisecond)
		assert.LessOrEqual(t, mid, 1200*time.Millisecond)
	}
}

// --- submitAsyncExecution error branches -----------------------------------

// Contract: an unbuildable submit request surfaces a "build request" error
// before any network I/O.
func TestSubmitAsyncExecution_BuildRequestError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")

	_, _, err := a.submitAsyncExecution(
		context.Background(), "://bad", "target.node",
		[]byte(`{"input":{}}`), ExecutionContext{}, "run-1",
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "build request")
}

// Contract: a transport failure on the async submit surfaces a
// "perform execute call" error.
func TestSubmitAsyncExecution_TransportError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial tcp: connection refused")
	})}

	_, _, err := a.submitAsyncExecution(
		context.Background(), "http://cp.example", "target.node",
		[]byte(`{"input":{}}`), ExecutionContext{}, "run-1",
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "perform execute call")
}

// Contract: a submit response whose body cannot be read surfaces a
// "read execute response" error.
func TestSubmitAsyncExecution_ReadBodyError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusAccepted,
			Header:     make(http.Header),
			Body:       callErrBody{},
		}, nil
	})}

	_, _, err := a.submitAsyncExecution(
		context.Background(), "http://cp.example", "target.node",
		[]byte(`{"input":{}}`), ExecutionContext{}, "run-1",
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "read execute response")
}

// Contract: when the control plane rejects the submit with a structured JSON
// error body, Call returns an *ExecuteError carrying the HTTP status and the
// control plane's "error" message (and error_details).
func TestCall_SubmitStructuredError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		w.WriteHeader(http.StatusForbidden)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error":         "permission denied",
			"error_details": map[string]any{"code": "forbidden"},
		})
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)

	var execErr *ExecuteError
	require.ErrorAs(t, err, &execErr)
	assert.Equal(t, http.StatusForbidden, execErr.StatusCode)
	assert.Contains(t, execErr.Message, "permission denied")
	assert.NotNil(t, execErr.ErrorDetails)
}

// Contract: a 2xx submit whose body is not valid JSON surfaces a
// "decode execute response" error.
func TestCall_SubmitDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("this is definitely not json"))
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	_, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode execute response")
}

// --- awaitExecutionResult error branches -----------------------------------

// Contract: a context already cancelled before the wait loop starts aborts
// immediately with the context error, without polling.
func TestAwaitExecutionResult_ContextAlreadyCancelled(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := a.awaitExecutionResult(
		ctx, "http://cp.example", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.ErrorIs(t, err, context.Canceled)
}

// Contract: an unbuildable status-poll request surfaces a "build status
// request" error.
func TestAwaitExecutionResult_BuildRequestError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")

	result, err := a.awaitExecutionResult(
		context.Background(), "://bad", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "build status request")
}

// Contract: when the caller's context is cancelled while a poll request is in
// flight, the wait surfaces the context error (not a generic transport error),
// preserving the ctx-cancel contract.
func TestAwaitExecutionResult_ContextCancelDuringPoll(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.httpClient = &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		<-req.Context().Done()
		return nil, req.Context().Err()
	})}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	result, err := a.awaitExecutionResult(
		ctx, "http://cp.example", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// Contract: a transport failure on a status poll (with a live context)
// surfaces a "poll execution status" error.
func TestAwaitExecutionResult_TransportError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("connection reset by peer")
	})}

	result, err := a.awaitExecutionResult(
		context.Background(), "http://cp.example", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "poll execution status")
}

// Contract: a status-poll response whose body cannot be read surfaces a
// "read execution status" error.
func TestAwaitExecutionResult_ReadBodyError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       callErrBody{},
		}, nil
	})}

	result, err := a.awaitExecutionResult(
		context.Background(), "http://cp.example", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "read execution status")
}

// Contract: an HTTP error status on a status poll aborts the wait with an
// *ExecuteError carrying that status (Python raise_for_status parity).
func TestCall_PollReturnsErrorStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		default:
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("upstream boom"))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)

	var execErr *ExecuteError
	require.ErrorAs(t, err, &execErr)
	assert.Equal(t, http.StatusInternalServerError, execErr.StatusCode)
	assert.Contains(t, execErr.Message, "execution status failed")
}

// Contract: a status-poll body that is not valid JSON surfaces a
// "decode execute response" error.
func TestCall_PollDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		default:
			_, _ = w.Write([]byte("<<not json>>"))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	_, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode execute response")
}

// Contract: a succeeded execution whose "result" field is not a JSON object
// (so it cannot decode into the result map) surfaces a "decode execute
// response" error rather than a silent empty result.
func TestCall_SucceededResultDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		default:
			// result is a JSON string, not an object -> unmarshal into
			// map[string]any fails.
			_, _ = w.Write([]byte(`{"status":"succeeded","result":"not-an-object"}`))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	_, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode execute response")
}

// Contract: a succeeded execution with a null result yields a nil result map
// and no error (the len/"null" guard skips decoding).
func TestCall_SucceededNullResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		default:
			_, _ = w.Write([]byte(`{"status":"succeeded","result":null}`))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.NoError(t, err)
	assert.Nil(t, result)
}
