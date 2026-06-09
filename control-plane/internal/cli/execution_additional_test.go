package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

func TestDefaultExecutionActionOptionsAndFlags(t *testing.T) {
	t.Setenv("AGENTFIELD_SERVER", "http://env-server")
	t.Setenv("AGENTFIELD_TOKEN", "env-token")

	opts := defaultExecutionActionOptions()
	require.Equal(t, "http://env-server", opts.serverURL)
	require.Equal(t, "env-token", opts.token)
	require.Equal(t, 15*time.Second, opts.timeout)

	cmd := &cobra.Command{Use: "test"}
	bindExecutionActionFlags(cmd, &opts)
	require.NoError(t, cmd.ParseFlags([]string{
		"--server", "http://flag-server",
		"--token", "flag-token",
		"--timeout", "3s",
		"--json",
	}))
	require.Equal(t, "http://flag-server", opts.serverURL)
	require.Equal(t, "flag-token", opts.token)
	require.Equal(t, 3*time.Second, opts.timeout)
	require.True(t, opts.jsonOutput)
}

func TestRunExecutionActionSuccessAndErrors(t *testing.T) {
	t.Run("success with human output and auth header", func(t *testing.T) {
		var gotAuth string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuth = r.Header.Get("Authorization")
			require.Equal(t, "/api/v1/executions/ex-1/cancel", r.URL.Path)
			require.Equal(t, "application/json", r.Header.Get("Content-Type"))
			var payload map[string]string
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			require.Equal(t, "operator request", payload["reason"])
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"ex-1","previous_status":"running","reason":"operator request"}`))
		}))
		defer server.Close()

		var out bytes.Buffer
		oldStdout := os.Stdout
		r, w, err := os.Pipe()
		require.NoError(t, err)
		os.Stdout = w

		_, err = runExecutionAction(executionActionConfig{
			actionName:  "cancel",
			successVerb: "cancelled",
			endpoint:    "/api/v1/executions/%s/cancel",
			executionID: "ex-1",
			withReason:  true,
			opts: &executionActionOptions{
				serverURL: server.URL + "/",
				token:     "secret-token",
				timeout:   time.Second,
				reason:    "operator request",
			},
		})
		require.NoError(t, err)
		require.NoError(t, w.Close())
		_, err = out.ReadFrom(r)
		require.NoError(t, err)
		os.Stdout = oldStdout
		require.Equal(t, "Bearer secret-token", gotAuth)
		require.Contains(t, out.String(), "Execution ex-1 cancelled (was: running)")
		require.Contains(t, out.String(), "Reason: operator request")
	})

	t.Run("success with json output", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"ex-2"}`))
		}))
		defer server.Close()

		output := captureOutput(t, func() {
			parsed, err := runExecutionAction(executionActionConfig{
				actionName:  "resume",
				successVerb: "resumed",
				endpoint:    "/api/v1/executions/%s/resume",
				executionID: "ex-2",
				opts: &executionActionOptions{
					serverURL:  server.URL,
					timeout:    time.Second,
					jsonOutput: true,
				},
			})
			require.NoError(t, err)
			require.Equal(t, "ex-2", parsed["execution_id"])
		})
		require.Contains(t, output, `"execution_id": "ex-2"`)
	})

	t.Run("not found and conflict errors", func(t *testing.T) {
		cases := []struct {
			name       string
			statusCode int
			body       string
			want       string
		}{
			{name: "not found", statusCode: http.StatusNotFound, body: `{"message":"missing"}`, want: "execution ex-3 not found: missing"},
			{name: "conflict", statusCode: http.StatusConflict, body: `{"error":"already paused"}`, want: "cannot pause execution ex-3: already paused"},
			{name: "generic", statusCode: http.StatusBadGateway, body: `{}`, want: "failed to pause execution ex-3 (502)"},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(tc.statusCode)
					_, _ = w.Write([]byte(tc.body))
				}))
				defer server.Close()

				_, err := runExecutionAction(executionActionConfig{
					actionName:  "pause",
					successVerb: "paused",
					endpoint:    "/api/v1/executions/%s/pause",
					executionID: "ex-3",
					opts: &executionActionOptions{
						serverURL: server.URL,
						timeout:   time.Second,
					},
				})
				require.ErrorContains(t, err, tc.want)
			})
		}
	})

	t.Run("request and decode failures", func(t *testing.T) {
		_, err := runExecutionAction(executionActionConfig{
			actionName:  "resume",
			successVerb: "resumed",
			endpoint:    "/api/v1/executions/%s/resume",
			executionID: "ex-4",
			opts: &executionActionOptions{
				serverURL: "http://[::1",
				timeout:   time.Millisecond,
			},
		})
		require.ErrorContains(t, err, "build request")

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`not-json`))
		}))
		defer server.Close()

		_, err = runExecutionAction(executionActionConfig{
			actionName:  "resume",
			successVerb: "resumed",
			endpoint:    "/api/v1/executions/%s/resume",
			executionID: "ex-4",
			opts: &executionActionOptions{
				serverURL: server.URL,
				timeout:   time.Second,
			},
		})
		require.ErrorContains(t, err, "decode response")
	})
}

func TestPrintExecutionActionHumanOutput(t *testing.T) {
	tests := []struct {
		name   string
		parsed map[string]any
		want   string
	}{
		{
			name:   "id and previous status",
			parsed: map[string]any{"execution_id": "ex-1", "previous_status": "running"},
			want:   "Execution ex-1 resumed (was: running)",
		},
		{
			name:   "only id",
			parsed: map[string]any{"execution_id": "ex-2"},
			want:   "Execution ex-2 resumed",
		},
		{
			name:   "fallback",
			parsed: map[string]any{},
			want:   "Execution resumed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := captureOutput(t, func() {
				printExecutionActionHumanOutput(tt.parsed, "resumed")
			})
			require.Contains(t, output, tt.want)
		})
	}
}

func TestResumeExecutionCommand(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/executions/ex-9/resume", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"execution_id":"ex-9"}`))
	}))
	defer server.Close()

	cmd := newResumeExecutionCommand()
	cmd.SetArgs([]string{"ex-9", "--server", server.URL, "--timeout", "1s"})
	output := captureOutput(t, func() {
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, "Execution ex-9 resumed")
}
