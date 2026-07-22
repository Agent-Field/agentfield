package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// runOverviewJSON builds the `/api/v1/agentic/run/:run_id` envelope `af wait`
// polls, with a single root execution at the given status.
func runOverviewJSON(runID, status, resultJSON string) string {
	return fmt.Sprintf(`{"ok":true,"data":{"run_id":%q,"executions":[`+
		`{"execution_id":"exec-1","parent_execution_id":null,"status":%q,"result":%s}]}}`,
		runID, status, resultJSON)
}

func TestRunWait(t *testing.T) {
	newOpts := func(stdout, stderr *bytes.Buffer) *waitOptions {
		return &waitOptions{
			timeout:      2 * time.Second,
			pollInterval: 5 * time.Millisecond,
			outputFormat: "json",
			stdout:       stdout,
			stderr:       stderr,
			stdoutTTY:    false,
		}
	}

	t.Run("succeeded run exits 0 and prints status and result", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/api/v1/agentic/run/run-1", r.URL.Path)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(runOverviewJSON("run-1", "succeeded", `{"answer":42}`)))
		})

		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "run-1", newOpts(&stdout, &stderr))
		require.NoError(t, err)

		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(stdout.Bytes(), &payload))
		require.Equal(t, "succeeded", payload["status"])
		result, ok := payload["result"].(map[string]interface{})
		require.True(t, ok)
		require.EqualValues(t, 42, result["answer"])
	})

	t.Run("failed run exits 1 and reports the failed status", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(runOverviewJSON("run-2", "failed", `null`)))
		})

		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "run-2", newOpts(&stdout, &stderr))
		require.Equal(t, 1, ExitCode(err))

		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(stdout.Bytes(), &payload))
		require.Equal(t, "failed", payload["status"])
	})

	t.Run("keeps polling until the run reaches a terminal state", func(t *testing.T) {
		var calls int32
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			// First two polls: still running (404 then running); then succeeded.
			switch atomic.AddInt32(&calls, 1) {
			case 1:
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"ok":false,"error":{"code":"run_not_found"}}`))
			case 2:
				_, _ = w.Write([]byte(runOverviewJSON("run-3", "running", `null`)))
			default:
				_, _ = w.Write([]byte(runOverviewJSON("run-3", "succeeded", `{"done":true}`)))
			}
		})

		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "run-3", newOpts(&stdout, &stderr))
		require.NoError(t, err)
		require.GreaterOrEqual(t, atomic.LoadInt32(&calls), int32(3))

		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(stdout.Bytes(), &payload))
		require.Equal(t, "succeeded", payload["status"])
	})

	t.Run("times out with exit code 2 when the run never finishes", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(runOverviewJSON("run-4", "running", `null`)))
		})

		var stdout, stderr bytes.Buffer
		opts := newOpts(&stdout, &stderr)
		opts.timeout = 30 * time.Millisecond
		opts.pollInterval = 5 * time.Millisecond
		err := runWait(context.Background(), "run-4", opts)
		require.Equal(t, 2, ExitCode(err))
		require.Contains(t, stderr.String(), "timed out")
	})
}

// waitTestOpts builds waitOptions with sub-second budgets so command-level and
// error-path tests never sleep a real second.
func waitTestOpts(stdout, stderr *bytes.Buffer, format string) *waitOptions {
	return &waitOptions{
		timeout:      2 * time.Second,
		pollInterval: 5 * time.Millisecond,
		outputFormat: format,
		stdout:       stdout,
		stderr:       stderr,
		stdoutTTY:    false,
	}
}

func TestRunWaitOutputAndErrorPaths(t *testing.T) {
	t.Run("pretty output prints status and result", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(runOverviewJSON("run-p", "succeeded", `{"answer":7}`)))
		})
		var stdout, stderr bytes.Buffer
		require.NoError(t, runWait(context.Background(), "run-p", waitTestOpts(&stdout, &stderr, "pretty")))
		require.Contains(t, stdout.String(), "succeeded")
		require.Contains(t, stdout.String(), "answer")
	})

	t.Run("invalid output format exits 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "run-x", waitTestOpts(&stdout, &stderr, "csv"))
		require.Equal(t, 2, ExitCode(err))
	})

	t.Run("empty run id exits 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "   ", waitTestOpts(&stdout, &stderr, "json"))
		require.Equal(t, 2, ExitCode(err))
	})

	t.Run("server error surfaces exit code 3", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"ok":false,"error":{"code":"boom"}}`))
		})
		var stdout, stderr bytes.Buffer
		err := runWait(context.Background(), "run-e", waitTestOpts(&stdout, &stderr, "json"))
		require.Equal(t, 3, ExitCode(err))
	})

	t.Run("nil opts falls back to defaults and still prints", func(t *testing.T) {
		withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(runOverviewJSON("run-d", "succeeded", `{"ok":true}`)))
		})
		out := captureOutput(t, func() {
			require.NoError(t, runWait(context.Background(), "run-d", nil))
		})
		require.Contains(t, out, "succeeded")
	})
}

func TestRootExecutionResult(t *testing.T) {
	ptr := func(s string) *string { return &s }

	t.Run("prefers the root execution (no parent)", func(t *testing.T) {
		execs := []waitExecution{
			{ExecutionID: "child", ParentExecutionID: ptr("root"), Result: json.RawMessage(`{"who":"child"}`)},
			{ExecutionID: "root", ParentExecutionID: nil, Result: json.RawMessage(`{"who":"root"}`)},
		}
		got, ok := rootExecutionResult(execs).(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, "root", got["who"])
	})

	t.Run("falls back to the last execution when no explicit root", func(t *testing.T) {
		execs := []waitExecution{
			{ExecutionID: "a", ParentExecutionID: ptr("x"), Result: json.RawMessage(`{"n":1}`)},
			{ExecutionID: "b", ParentExecutionID: ptr("y"), Result: json.RawMessage(`{"n":2}`)},
		}
		got, ok := rootExecutionResult(execs).(map[string]interface{})
		require.True(t, ok)
		require.EqualValues(t, 2, got["n"])
	})

	t.Run("non-JSON result is returned verbatim as a string", func(t *testing.T) {
		execs := []waitExecution{{ParentExecutionID: nil, Result: json.RawMessage(`not-json`)}}
		require.Equal(t, "not-json", rootExecutionResult(execs))
	})

	t.Run("empty result is nil", func(t *testing.T) {
		execs := []waitExecution{{ParentExecutionID: nil}}
		require.Nil(t, rootExecutionResult(execs))
	})
}

func TestNewWaitCommandExecute(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/agentic/run/run-cmd", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(runOverviewJSON("run-cmd", "succeeded", `{"ok":true}`)))
	})
	cmd := NewWaitCommand()
	cmd.SetArgs([]string{"run-cmd", "-o", "json"})
	out := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, out, `"succeeded"`)
}
