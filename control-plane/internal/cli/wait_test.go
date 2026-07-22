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
