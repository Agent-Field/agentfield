package cli

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// Contract: every `af agent exec <verb> --id X` call proxies to the matching
// /api/v1/executions/X/<verb> endpoint and emits an AgentResponse envelope
// with ok:true and meta.status_code on 2xx.
func TestAgentExecHappyPaths(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		wantMethod string
		wantPath   string
		wantBody   map[string]interface{}
	}{
		{
			name:       "pause with reason",
			args:       []string{"exec", "pause", "--id", "exec_1", "--reason", "manual review"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/pause",
			wantBody:   map[string]interface{}{"reason": "manual review"},
		},
		{
			name:       "pause without reason sends empty object",
			args:       []string{"exec", "pause", "--id", "exec_1"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/pause",
			wantBody:   map[string]interface{}{},
		},
		{
			name:       "resume has no body",
			args:       []string{"exec", "resume", "--id", "exec_1"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/resume",
			wantBody:   nil,
		},
		{
			name:       "cancel with reason",
			args:       []string{"exec", "cancel", "--id", "exec_1", "--reason", "wrong input"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/cancel",
			wantBody:   map[string]interface{}{"reason": "wrong input"},
		},
		{
			name:       "restart sends default scope and reuse",
			args:       []string{"exec", "restart", "--id", "exec_1"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/restart",
			wantBody:   map[string]interface{}{"scope": "workflow", "reuse": "succeeded-before"},
		},
		{
			name:       "approval status is a GET",
			args:       []string{"exec", "approval-status", "--id", "exec_1"},
			wantMethod: http.MethodGet,
			wantPath:   "/api/v1/executions/exec_1/approval-status",
			wantBody:   nil,
		},
		{
			name:       "approve sends decision and reason",
			args:       []string{"exec", "approve", "--id", "exec_1", "--decision", "approved", "--reason", "lgtm"},
			wantMethod: http.MethodPost,
			wantPath:   "/api/v1/executions/exec_1/approval-response",
			wantBody:   map[string]interface{}{"decision": "approved", "reason": "lgtm"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotMethod, gotPath string
			var gotBody []byte

			oldTransport := http.DefaultTransport
			http.DefaultTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				gotMethod = req.Method
				gotPath = req.URL.Path
				if req.Body != nil {
					gotBody, _ = io.ReadAll(req.Body)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"execution_id":"exec_1","status":"ok"}`)),
					Request:    req,
				}, nil
			})
			defer func() { http.DefaultTransport = oldTransport }()

			oldServer, oldFormat, oldTimeout := serverURL, outputFormat, requestTimeout
			serverURL, outputFormat, requestTimeout = "http://agent.test", "json", 1
			defer func() {
				serverURL, outputFormat, requestTimeout = oldServer, oldFormat, oldTimeout
			}()

			output := captureOutput(t, func() {
				cmd := NewAgentCommand()
				cmd.SetArgs(tt.args)
				require.NoError(t, cmd.Execute())
			})

			require.Equal(t, tt.wantMethod, gotMethod)
			require.Equal(t, tt.wantPath, gotPath)

			if tt.wantBody == nil {
				require.Empty(t, gotBody)
			} else {
				var decoded map[string]interface{}
				require.NoError(t, json.Unmarshal(gotBody, &decoded))
				require.Equal(t, tt.wantBody, decoded)
			}

			require.Contains(t, output, `"ok": true`)
			require.Contains(t, output, `"status_code": 200`)
			require.Contains(t, output, `"execution_id": "exec_1"`)
		})
	}
}

// Contract: `af agent query -r events` forwards execution_id/run_id/since/
// until filters to POST /api/v1/agentic/query.
func TestAgentQueryEventsSendsFilters(t *testing.T) {
	var gotBody []byte

	oldTransport := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"ok":true,"data":{"resource":"events","results":[]}}`)),
			Request:    req,
		}, nil
	})
	defer func() { http.DefaultTransport = oldTransport }()

	oldServer, oldFormat, oldTimeout := serverURL, outputFormat, requestTimeout
	serverURL, outputFormat, requestTimeout = "http://agent.test", "json", 1
	defer func() {
		serverURL, outputFormat, requestTimeout = oldServer, oldFormat, oldTimeout
	}()

	_ = captureOutput(t, func() {
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"query", "-r", "events", "--execution-id", "exec_1", "--run-id", "run_1", "--since", "2026-07-01T00:00:00Z", "--limit", "5"})
		require.NoError(t, cmd.Execute())
	})

	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(gotBody, &payload))
	require.Equal(t, "events", payload["resource"])
	filters := payload["filters"].(map[string]interface{})
	require.Equal(t, "exec_1", filters["execution_id"])
	require.Equal(t, "run_1", filters["run_id"])
	require.Equal(t, "2026-07-01T00:00:00Z", filters["since"])
	require.Equal(t, float64(5), payload["limit"])
}

// Contract: legacy string errors normalize into {code,message,hint} — the
// error string becomes the code only when a sibling message carries the
// human detail; otherwise the code derives from the HTTP status.
func TestStructuredErrorFromString(t *testing.T) {
	t.Run("bare string error derives code from status", func(t *testing.T) {
		got := structuredErrorFromString(map[string]interface{}{}, "execution exec_x not found", http.StatusNotFound)
		require.Equal(t, "not_found", got["code"])
		require.Equal(t, "execution exec_x not found", got["message"])
		require.NotEmpty(t, got["hint"])
	})

	t.Run("error code with sibling message keeps the code", func(t *testing.T) {
		payload := map[string]interface{}{"message": "execution is in 'running' state"}
		got := structuredErrorFromString(payload, "invalid_state", http.StatusConflict)
		require.Equal(t, "invalid_state", got["code"])
		require.Equal(t, "execution is in 'running' state", got["message"])
	})

	t.Run("blank sibling message falls back to derived code", func(t *testing.T) {
		payload := map[string]interface{}{"message": "  "}
		got := structuredErrorFromString(payload, "boom", http.StatusBadRequest)
		require.Equal(t, "bad_request", got["code"])
		require.Equal(t, "boom", got["message"])
	})
}

func TestDefaultCodeForStatus(t *testing.T) {
	cases := map[int]string{
		http.StatusBadRequest:          "bad_request",
		http.StatusUnauthorized:        "unauthorized",
		http.StatusForbidden:           "forbidden",
		http.StatusNotFound:            "not_found",
		http.StatusConflict:            "conflict",
		http.StatusInternalServerError: "server_error",
		http.StatusBadGateway:          "server_error",
		http.StatusTeapot:              "request_failed",
	}
	for status, want := range cases {
		require.Equal(t, want, defaultCodeForStatus(status), "status %d", status)
	}
}

// Contract: exec subcommand without a verb lists the available verbs.
func TestAgentExecListsVerbs(t *testing.T) {
	oldFormat := outputFormat
	outputFormat = "json"
	defer func() { outputFormat = oldFormat }()

	output := captureOutput(t, func() {
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"exec"})
		require.NoError(t, cmd.Execute())
	})
	for _, verb := range []string{"pause", "resume", "cancel", "restart", "approval-status", "approve"} {
		require.Contains(t, output, verb)
	}
}

// Contract: validation and server error paths exit non-zero and print a
// structured {ok:false,error:{code,message,hint}} envelope.
func TestAgentExecExitOutputs(t *testing.T) {
	cases := []struct {
		name  string
		wants []string
	}{
		{name: "exec-pause-missing-id", wants: []string{`"code": "missing_required_flag"`, `"--id is required"`}},
		{name: "exec-approve-missing-decision", wants: []string{`"code": "missing_required_flag"`, `"--decision is required"`}},
		{name: "exec-approve-invalid-decision", wants: []string{`"code": "invalid_flag_value"`}},
		{name: "exec-restart-invalid-input", wants: []string{`"code": "invalid_flag_value"`}},
		{name: "exec-pause-not-found", wants: []string{`"ok": false`, `"code": "not_found"`, `"status_code": 404`, "execution exec_x not found"}},
		{name: "exec-resume-conflict-with-code", wants: []string{`"ok": false`, `"code": "invalid_state"`, `"status_code": 409`}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := runAgentExecExitHelper(t, tc.name)
			require.Error(t, err)
			exitErr := &exec.ExitError{}
			require.ErrorAs(t, err, &exitErr)
			for _, want := range tc.wants {
				require.Contains(t, out, want)
			}
		})
	}
}

func runAgentExecExitHelper(t *testing.T, mode string) (string, error) {
	t.Helper()

	cmd := exec.Command(os.Args[0], "-test.run=TestAgentExecExitHelper", "--", mode)
	cmd.Env = append(os.Environ(), "GO_WANT_AGENT_EXEC_EXIT_HELPER=1")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestAgentExecExitHelper(t *testing.T) {
	if os.Getenv("GO_WANT_AGENT_EXEC_EXIT_HELPER") != "1" {
		return
	}

	runAgent := func(args ...string) {
		cmd := NewAgentCommand()
		cmd.SetArgs(args)
		_ = cmd.Execute()
	}

	mode := os.Args[len(os.Args)-1]
	switch mode {
	case "exec-pause-missing-id":
		runAgent("exec", "pause")
	case "exec-approve-missing-decision":
		runAgent("exec", "approve", "--id", "exec_x")
	case "exec-approve-invalid-decision":
		runAgent("exec", "approve", "--id", "exec_x", "--decision", "maybe")
	case "exec-restart-invalid-input":
		runAgent("exec", "restart", "--id", "exec_x", "--input", "{not-json")
	case "exec-pause-not-found":
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"execution exec_x not found"}`))
		}))
		defer server.Close()
		serverURL, requestTimeout = server.URL, 1
		runAgent("exec", "pause", "--id", "exec_x")
	case "exec-resume-conflict-with-code":
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":"invalid_state","message":"execution is in 'running' state; must be 'paused'"}`))
		}))
		defer server.Close()
		serverURL, requestTimeout = server.URL, 1
		runAgent("exec", "resume", "--id", "exec_x")
	}
}
