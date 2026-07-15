package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"syscall"
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

// Contract: an AMBIGUOUS submit transport failure (a plain error that does not
// prove the request never reached the server) is NOT retried — re-POSTing a
// possibly-accepted execute/async would double-run the target — and surfaces a
// "perform execute call (not retried ...)" error on the first attempt.
func TestSubmitAsyncExecution_TransportError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	var attempts atomic.Int32
	a.callSubmitClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		attempts.Add(1)
		return nil, errors.New("read: connection reset by peer")
	})}

	_, _, err := a.submitAsyncExecution(
		context.Background(), "http://cp.example", "target.node",
		[]byte(`{"input":{}}`), ExecutionContext{}, "run-1",
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "perform execute call")
	assert.Contains(t, err.Error(), "not retried")
	assert.Equal(t, int32(1), attempts.Load(), "ambiguous submit failure must not be retried")
}

// Contract: a submit response whose body cannot be read surfaces a
// "read execute response" error.
func TestSubmitAsyncExecution_ReadBodyError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.callSubmitClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
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
	a.callPollClient = &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
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

// Contract: a PERSISTENT poll transport failure is retried (not fatal on the
// first blip) until the retry window is exceeded, then fails with the
// "control plane unreachable for Xs" error that wraps the underlying cause —
// not the raw first error. The window is shrunk via env so the test is fast.
func TestAwaitExecutionResult_TransportError(t *testing.T) {
	t.Setenv(envCallRetryWindowSeconds, "1")
	a := newCallTestAgent(t, "http://placeholder.invalid")
	var attempts atomic.Int32
	a.callPollClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		attempts.Add(1)
		return nil, errors.New("connection reset by peer")
	})}

	start := time.Now()
	result, err := a.awaitExecutionResult(
		context.Background(), "http://cp.example", "target.node", "exec-1", "run-1", ExecutionContext{},
	)
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "control plane unreachable")
	assert.Contains(t, err.Error(), "poll execution status")
	assert.Greater(t, attempts.Load(), int32(1), "a transient poll failure must be retried, not fatal on the first blip")
	assert.Less(t, time.Since(start), 20*time.Second, "must fail promptly once the window is exceeded")
}

// Contract: a status-poll response whose body cannot be read surfaces a
// "read execution status" error.
func TestAwaitExecutionResult_ReadBodyError(t *testing.T) {
	a := newCallTestAgent(t, "http://placeholder.invalid")
	a.callPollClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
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

// Contract: a PERMANENT (non-transient) HTTP error status on a status poll —
// e.g. 403 auth — aborts the wait immediately with an *ExecuteError carrying
// that status. (Transient statuses like 5xx/429 are instead retried; see
// TestCall_PollRetriesTransient5xxThenSucceeds.)
func TestCall_PollReturnsErrorStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		default:
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte("permission denied"))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)

	var execErr *ExecuteError
	require.ErrorAs(t, err, &execErr)
	assert.Equal(t, http.StatusForbidden, execErr.StatusCode)
	assert.Contains(t, execErr.Message, "execution status failed")
}

// Contract: a status-poll body that is not valid JSON surfaces a
// "decode execute response" error.
func TestCall_PollDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
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
		switch r.Method {
		case http.MethodPost:
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
		switch r.Method {
		case http.MethodPost:
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

// --- Call resilience to transient control-plane outages --------------------
//
// These tests exercise the Part 4 behavior contract: a transient poll blip or a
// dial-only submit failure must NOT kill a long-running cross-node call, while a
// non-idempotent submit is never blindly re-sent on an ambiguous failure.

// Contract: the fake CP serves 2 successful polls, then 3 consecutive transient
// 5xx polls (well within the retry window), then recovers and completes → the
// call SUCCEEDS with the correct result, and each transient failure is logged at
// warn with the call.outbound.poll_retry event.
func TestCall_PollRetriesTransient5xxThenSucceeds(t *testing.T) {
	var polls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/v1/execute/async/"):
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1", "run_id": "run-1", "status": "queued",
			})
		case r.Method == http.MethodGet:
			switch n := polls.Add(1); {
			case n <= 2: // two healthy "running" polls
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "running"})
			case n <= 5: // three consecutive transient failures
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte("overloaded"))
			default: // recovery
				_ = json.NewEncoder(w).Encode(map[string]any{
					"status": "succeeded",
					"result": map[string]any{"ok": true},
				})
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)

	var result map[string]any
	stdout, _, err := captureOutput(t, func() error {
		var callErr error
		result, callErr = a.Call(context.Background(), "target.node", map[string]any{})
		return callErr
	})
	require.NoError(t, err, "transient poll failures within the window must not fail the call")
	require.NotNil(t, result)
	assert.Equal(t, true, result["ok"])
	assert.GreaterOrEqual(t, polls.Load(), int32(6), "must have retried past the transient failures")

	var warnRetries int
	for _, line := range strings.Split(strings.TrimSpace(stdout), "\n") {
		if line == "" {
			continue
		}
		var entry ExecutionLogEntry
		if json.Unmarshal([]byte(line), &entry) != nil {
			continue
		}
		if entry.EventType == "call.outbound.poll_retry" && entry.Level == "warn" {
			warnRetries++
			assert.NotNil(t, entry.Attributes["attempt"], "warn retry log must carry the attempt count")
		}
	}
	assert.GreaterOrEqual(t, warnRetries, 3, "each transient failure must emit a warn retry log")
}

// Contract: when the CP stays down past the (env-shortened) retry window, the
// call fails with the "control plane unreachable for Xs" error — not the raw
// first poll error — after having retried (5xx is transient).
func TestCall_PollUnreachableAfterWindow(t *testing.T) {
	t.Setenv(envCallRetryWindowSeconds, "1")
	var polls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1", "run_id": "run-1", "status": "queued",
			})
		default:
			polls.Add(1)
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("down"))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)
	start := time.Now()
	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "control plane unreachable")
	assert.Greater(t, polls.Load(), int32(1), "a transient 5xx must be retried, not fatal on the first")
	assert.Less(t, time.Since(start), 20*time.Second, "must fail promptly once the window is exceeded")
}

// Contract: a 404 on a just-submitted execution is retried within a bounded
// (shorter) window, then fails with a DISTINCT not-found error (StatusCode 404)
// rather than the generic unreachable error.
func TestCall_Poll404BoundedThenDistinctError(t *testing.T) {
	t.Setenv(envCallRetryWindowSeconds, "1") // also shrinks the 404 window to 1s
	var gets atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-404", "run_id": "run-1", "status": "queued",
			})
		default:
			gets.Add(1)
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte("not found"))
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)
	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)

	var execErr *ExecuteError
	require.ErrorAs(t, err, &execErr)
	assert.Equal(t, http.StatusNotFound, execErr.StatusCode)
	assert.Contains(t, execErr.Message, "not found on control plane")
	assert.Greater(t, gets.Load(), int32(1), "404 right after submit must be retried within the bounded window")
}

// Contract: a submit whose connection is REFUSED twice (proving the request
// never reached the server) is safely retried and then succeeds — creating
// EXACTLY ONE execution on the CP (a refused dial writes no request bytes, so a
// re-POST cannot double-run the target).
func TestSubmit_ConnectionRefusedRetriesToExactlyOneExecution(t *testing.T) {
	var executePosts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/v1/execute/async/"):
			executePosts.Add(1)
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1", "run_id": "run-1", "status": "queued",
			})
		case r.Method == http.MethodGet:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "succeeded", "result": map[string]any{"ok": true},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	a := newCallTestAgent(t, server.URL)
	var dials atomic.Int32
	realTransport := http.DefaultTransport
	a.callSubmitClient = &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		if dials.Add(1) <= 2 {
			return nil, &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}
		}
		return realTransport.RoundTrip(req)
	})}

	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, true, result["ok"])
	assert.Equal(t, int32(1), executePosts.Load(), "connection-refused retries must create exactly one execution")
	assert.Equal(t, int32(3), dials.Load(), "two refusals then one accepted submit")
}

// Contract: an AMBIGUOUS submit failure — the server accepts the connection but
// never responds (the request MAY have been accepted) — is NOT retried. The
// generous submit timeout (shrunk here via env) bounds the wait; exactly ONE
// request reaches the server and the call fails with a clear message.
func TestSubmit_AmbiguousTimeoutNoRetry(t *testing.T) {
	t.Setenv(envCallSubmitTimeoutSeconds, "1")
	var executeRequests atomic.Int32
	// release lets the blocked handler return so server.Close() does not hang.
	// The two defers run LIFO: close(release) first, then server.Close().
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			executeRequests.Add(1)
			// Accept the connection but never respond until the test ends or the
			// client disconnects — modelling a hung-but-listening CP.
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	defer close(release)

	a := newCallTestAgent(t, server.URL)
	start := time.Now()
	result, err := a.Call(context.Background(), "target.node", map[string]any{})
	require.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "not retried")
	assert.Equal(t, int32(1), executeRequests.Load(), "ambiguous submit timeout must not be retried")
	assert.GreaterOrEqual(t, time.Since(start), 900*time.Millisecond, "submit must wait for its timeout")
	assert.Less(t, time.Since(start), 10*time.Second)
}

// Contract: the resilience knobs are read from the environment (once, cached on
// the agent), and invalid/unparseable values fall back to the documented
// defaults.
func TestCallResilienceEnvOverrides(t *testing.T) {
	t.Setenv(envCallRetryWindowSeconds, "7")
	t.Setenv(envCallSubmitTimeoutSeconds, "11")
	t.Setenv(envCallPollTimeoutSeconds, "13")
	ov := newCallTestAgent(t, "http://placeholder.invalid")
	assert.Equal(t, 7*time.Second, ov.callRetryWindow)
	assert.Equal(t, 11*time.Second, ov.callSubmitClient.Timeout)
	assert.Equal(t, 13*time.Second, ov.callPollClient.Timeout)

	// Invalid / non-positive values fall back to defaults.
	t.Setenv(envCallRetryWindowSeconds, "not-a-number")
	t.Setenv(envCallSubmitTimeoutSeconds, "-5")
	t.Setenv(envCallPollTimeoutSeconds, "0")
	fb := newCallTestAgent(t, "http://placeholder.invalid")
	assert.Equal(t, defaultCallRetryWindow, fb.callRetryWindow)
	assert.Equal(t, defaultCallSubmitTimeout, fb.callSubmitClient.Timeout)
	assert.Equal(t, defaultCallPollTimeout, fb.callPollClient.Timeout)
}

// Contract: requestNeverReachedServer returns true ONLY for errors that prove no
// request bytes reached the server (connection-refused, DNS failure, dial-phase
// net.OpError), so a non-idempotent submit is retried only when safe. Caller
// cancellation, an ambiguous post-dial timeout, and non-dial network errors all
// return false.
func TestRequestNeverReachedServer(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"caller cancelled", context.Canceled, false},
		{"caller deadline (ambiguous timeout)", context.DeadlineExceeded, false},
		{"wrapped deadline", &url.Error{Op: "Post", URL: "http://x", Err: context.DeadlineExceeded}, false},
		{"connection refused", syscall.ECONNREFUSED, true},
		{"wrapped connection refused", &url.Error{Op: "Post", URL: "http://x", Err: &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}}, true},
		{"dns failure", &net.DNSError{Err: "no such host", Name: "cp.invalid"}, true},
		{"wrapped dns failure", &url.Error{Op: "Post", URL: "http://x", Err: &net.DNSError{Err: "no such host"}}, true},
		{"dial-phase op error (non-econnrefused)", &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("network is unreachable")}, true},
		{"post-dial read error", &net.OpError{Op: "read", Net: "tcp", Err: errors.New("connection reset by peer")}, false},
		{"opaque error", errors.New("something went wrong"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, requestNeverReachedServer(tt.err))
		})
	}
}

// Contract: isTransientPollStatus treats 408/429/5xx as retryable and every
// other status (including 404 and permanent 4xx) as non-transient.
func TestIsTransientPollStatus(t *testing.T) {
	transient := []int{http.StatusRequestTimeout, http.StatusTooManyRequests, 500, 502, 503, 504}
	for _, code := range transient {
		assert.True(t, isTransientPollStatus(code), "status %d should be transient", code)
	}
	permanent := []int{200, 201, 400, http.StatusForbidden, http.StatusUnauthorized, http.StatusNotFound, 409}
	for _, code := range permanent {
		assert.False(t, isTransientPollStatus(code), "status %d should not be transient", code)
	}
}

// Contract: nextRetryBackoff doubles the backoff, clamps to callRetryBackoffMax,
// and floors a sub-minimum input at callRetryBackoffMin.
func TestNextRetryBackoff(t *testing.T) {
	assert.Equal(t, 1*time.Second, nextRetryBackoff(callRetryBackoffMin)) // 500ms -> 1s
	assert.Equal(t, callRetryBackoffMax, nextRetryBackoff(callRetryBackoffMax))
	assert.Equal(t, callRetryBackoffMax, nextRetryBackoff(callRetryBackoffMax+time.Second))
	assert.Equal(t, callRetryBackoffMin, nextRetryBackoff(1*time.Nanosecond)) // 2ns clamps up to the floor
}

// Contract: sleepCtx returns true when the full duration elapses (or is
// non-positive on a live ctx) and false when ctx is cancelled — before or
// during the wait.
func TestSleepCtx(t *testing.T) {
	assert.True(t, sleepCtx(context.Background(), 0), "non-positive duration on a live ctx returns true")
	assert.True(t, sleepCtx(context.Background(), 5*time.Millisecond), "elapsed sleep returns true")

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	assert.False(t, sleepCtx(cancelled, 0), "non-positive duration on a cancelled ctx returns false")
	assert.False(t, sleepCtx(cancelled, time.Hour), "already-cancelled ctx returns false immediately")

	ctx, cancel2 := context.WithCancel(context.Background())
	go func() { time.Sleep(20 * time.Millisecond); cancel2() }()
	start := time.Now()
	assert.False(t, sleepCtx(ctx, time.Hour), "cancellation during the wait returns false")
	assert.Less(t, time.Since(start), time.Second, "must return promptly on cancellation")
}

// Contract: when a submit's connection is refused for longer than the retry
// window, submitAsyncExecution stops retrying and fails with the "control plane
// unreachable" error (having retried more than once), never creating an
// execution.
func TestSubmit_ConnectionRefusedBeyondWindow(t *testing.T) {
	t.Setenv(envCallRetryWindowSeconds, "1")
	a := newCallTestAgent(t, "http://placeholder.invalid")
	var attempts atomic.Int32
	a.callSubmitClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		attempts.Add(1)
		return nil, &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}
	})}

	start := time.Now()
	_, _, err := a.submitAsyncExecution(
		context.Background(), "http://cp.example", "target.node",
		[]byte(`{"input":{}}`), ExecutionContext{}, "run-1",
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "control plane unreachable")
	assert.Greater(t, attempts.Load(), int32(1), "connection-refused submit must be retried within the window")
	assert.Less(t, time.Since(start), 20*time.Second, "must fail promptly once the window is exceeded")
}
