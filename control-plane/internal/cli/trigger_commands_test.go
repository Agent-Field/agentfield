package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func withTriggerTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	originalServerURL := serverURL
	originalAPIKey := apiKey
	originalTimeout := requestTimeout
	serverURL = server.URL
	apiKey = ""
	requestTimeout = 0
	t.Cleanup(func() {
		server.Close()
		serverURL = originalServerURL
		apiKey = originalAPIKey
		requestTimeout = originalTimeout
	})
	return server
}

func writeDiscoverySchema(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	_, err := w.Write([]byte(`{
		"capabilities": [{
			"reasoners": [{
				"id": "echo",
				"input_schema": {
					"type": "object",
					"required": ["message"],
					"properties": {
						"message": {"type": "string"},
						"count": {"type": "integer", "default": 1}
					}
				}
			}]
		}]
	}`))
	require.NoError(t, err)
}

func TestRunCallSyncSchemaAndAsync(t *testing.T) {
	var syncBody map[string]interface{}
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/discovery/capabilities":
			require.Equal(t, "node", r.URL.Query().Get("agent"))
			require.Equal(t, "echo", r.URL.Query().Get("reasoner"))
			writeDiscoverySchema(t, w)
		case r.URL.Path == "/api/v1/execute/node.echo":
			require.NoError(t, json.NewDecoder(r.Body).Decode(&syncBody))
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"exec-1","run_id":"run-1","status":"succeeded","result":{"echo":"hi","nested":{"score":9}}}`))
		case r.URL.Path == "/api/v1/execute/async/node.echo":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"exec-2","run_id":"run-2","status":"queued"}`))
		default:
			http.NotFound(w, r)
		}
	})

	var stdout bytes.Buffer
	err := runCall(context.Background(), "node.echo", &callOptions{
		inputSource:  `{"message":"hi"}`,
		outputFormat: "json",
		fieldPath:    ".nested.score",
		stdin:        bytes.NewBuffer(nil),
		stdout:       &stdout,
		stderr:       bytes.NewBuffer(nil),
		stdinTTY:     false,
		stdoutTTY:    false,
	})
	require.NoError(t, err)
	require.JSONEq(t, `9`, stdout.String())
	require.Equal(t, map[string]interface{}{"message": "hi"}, syncBody["input"])

	stdout.Reset()
	err = runCall(context.Background(), "node.echo", &callOptions{
		printSchema:  true,
		outputFormat: "json",
		stdin:        bytes.NewBuffer(nil),
		stdout:       &stdout,
		stderr:       bytes.NewBuffer(nil),
	})
	require.NoError(t, err)
	require.Contains(t, stdout.String(), `"message"`)

	// With -o json, async emits a structured envelope so parsers get valid JSON
	// (a harness driving the golden path pipes this straight into `af wait`).
	stdout.Reset()
	err = runCall(context.Background(), "node.echo", &callOptions{
		inputSource:  `{"message":"queued"}`,
		async:        true,
		outputFormat: "json",
		stdin:        bytes.NewBuffer(nil),
		stdout:       &stdout,
		stderr:       bytes.NewBuffer(nil),
		stdinTTY:     false,
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"run_id":"run-2","status":"accepted"}`, stdout.String())

	// The default/pretty path keeps the bare run-id line scripts capture with
	// $(af call --async) — this contract must not regress.
	stdout.Reset()
	err = runCall(context.Background(), "node.echo", &callOptions{
		inputSource:  `{"message":"queued"}`,
		async:        true,
		outputFormat: "pretty",
		stdin:        bytes.NewBuffer(nil),
		stdout:       &stdout,
		stderr:       bytes.NewBuffer(nil),
		stdinTTY:     false,
	})
	require.NoError(t, err)
	require.Equal(t, "run-2", strings.TrimSpace(stdout.String()))
}

func TestRunCallTTYStreamsAndFetchesFinalStatus(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/discovery/capabilities":
			writeDiscoverySchema(t, w)
		case r.URL.Path == "/api/v1/execute/async/node.echo":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"exec-tty","run_id":"run-tty","status":"queued"}`))
		case r.URL.Path == "/api/v1/executions/run-tty/events":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-tty","workflow_id":"run-tty","status":"running","type":"execution.updated"}`+"\n\n")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-tty","workflow_id":"run-tty","status":"succeeded","type":"execution.completed"}`+"\n\n")
		case r.URL.Path == "/api/v1/executions/exec-tty":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"execution_id":"exec-tty","run_id":"run-tty","status":"succeeded","result":{"echo":"done"}}`))
		default:
			http.NotFound(w, r)
		}
	})

	var stdout, stderr bytes.Buffer
	err := runCall(context.Background(), "node.echo", &callOptions{
		inputSource:  `{"message":"done"}`,
		outputFormat: "json",
		stdin:        bytes.NewBuffer(nil),
		stdout:       &stdout,
		stderr:       &stderr,
		stdoutTTY:    true,
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"echo":"done"}`, stdout.String())
	require.Contains(t, stderr.String(), "running")
	require.Contains(t, stderr.String(), "succeeded")
}

func TestRunCallClientErrors(t *testing.T) {
	tests := []struct {
		name   string
		target string
		opts   *callOptions
	}{
		{
			name:   "bad format",
			target: "node.echo",
			opts:   &callOptions{outputFormat: "xml"},
		},
		{
			name:   "conflicting interactive flags",
			target: "node.echo",
			opts:   &callOptions{interactive: true, noInteractive: true},
		},
		{
			name:   "bad target",
			target: "node",
			opts:   &callOptions{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := runCall(context.Background(), tt.target, tt.opts)
			var exitErr cliExitError
			require.ErrorAs(t, err, &exitErr)
			require.Equal(t, 2, exitErr.Code)
		})
	}
}

func TestNewTriggerCommandsExecute(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/discovery/capabilities":
			writeDiscoverySchema(t, w)
		case r.URL.Path == "/api/v1/reasoners":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"reasoners":[{"node":"node","reasoner":"echo","status":"live"}],"shown":1,"total":1}`))
		case r.URL.Path == "/api/v1/executions/run-command/events":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-command","status":"succeeded","type":"execution.completed"}`+"\n\n")
		default:
			http.NotFound(w, r)
		}
	})

	callCmd := NewCallCommand()
	callCmd.SetArgs([]string{"node.echo", "--schema", "-o", "json"})
	require.NoError(t, callCmd.Execute())

	lsCmd := NewReasonerListCommand()
	lsCmd.SetArgs([]string{"echo", "--node", "node", "-o", "json"})
	require.NoError(t, lsCmd.Execute())

	tailCmd := NewTailCommand()
	tailCmd.SetArgs([]string{"run-command", "-o", "json"})
	require.NoError(t, tailCmd.Execute())

	badTailCmd := NewTailCommand()
	badTailCmd.SetArgs([]string{"run-command", "-o", "yaml"})
	err := badTailCmd.Execute()
	var exitErr cliExitError
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)
}

func TestCallExecutionErrorBranches(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/v1/execute/failed-status":
			_, _ = w.Write([]byte(`{"status":"failed","error":"boom"}`))
		case "/api/v1/execute/failed-http":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"status":"failed","error":"boom"}`))
		case "/api/v1/execute/bad-request":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"bad"}`))
		case "/api/v1/executions/missing":
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"missing"}`))
		case "/api/v1/discovery/capabilities":
			if r.URL.Query().Get("reasoner") == "missing" {
				_, _ = w.Write([]byte(`{"capabilities":[{"reasoners":[]}]}`))
				return
			}
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"down"}`))
		default:
			http.NotFound(w, r)
		}
	})

	_, err := executeReasoner(context.Background(), "failed-status", map[string]interface{}{}, false)
	var exitErr cliExitError
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 1, exitErr.Code)

	_, err = executeReasoner(context.Background(), "failed-http", map[string]interface{}{}, false)
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 1, exitErr.Code)

	_, err = executeReasoner(context.Background(), "bad-request", map[string]interface{}{}, false)
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)

	_, err = fetchExecutionStatus(context.Background(), "missing")
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)

	_, err = fetchReasonerSchema(context.Background(), "node", "missing")
	require.ErrorContains(t, err, "not found")
	_, err = fetchReasonerSchema(context.Background(), "node", "down")
	require.ErrorContains(t, err, "schema request failed")
}

func TestCallResultPromptAndValidationHelpers(t *testing.T) {
	var stdout, stderr bytes.Buffer
	errText := "boom"
	err := printCallResult(&stdout, &stderr, &callResponse{Status: "failed", ErrorMessage: &errText}, "json", "")
	var exitErr cliExitError
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 1, exitErr.Code)
	require.Contains(t, stderr.String(), "boom")

	err = printCallResult(&stdout, &stderr, nil, "json", "")
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 3, exitErr.Code)

	err = printCallResult(&stdout, &stderr, &callResponse{Status: "succeeded", Result: map[string]interface{}{}}, "json", ".missing")
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)

	require.Equal(t, "fallback", firstNonEmptyString("", " fallback "))
	require.Equal(t, "value", pointerValue(ptrString("value")))
	require.Empty(t, pointerValue(nil))

	require.Equal(t, []string{"name"}, requiredFields(map[string]interface{}{"required": []interface{}{"name", ""}}))
	require.Equal(t, map[string]interface{}{"count": 3}, defaultsFromSchema(map[string]interface{}{
		"properties": map[string]interface{}{
			"count": map[string]interface{}{"default": 3},
		},
	}))

	value, err := parsePromptValue("", map[string]interface{}{"default": "fallback"})
	require.NoError(t, err)
	require.Equal(t, "fallback", value)
	_, err = parsePromptValue("", map[string]interface{}{})
	require.ErrorContains(t, err, "required")
	value, err = parsePromptValue("2.5", map[string]interface{}{"type": "number"})
	require.NoError(t, err)
	require.Equal(t, 2.5, value)
	value, err = parsePromptValue("true", map[string]interface{}{"type": "boolean"})
	require.NoError(t, err)
	require.Equal(t, true, value)
	value, err = parsePromptValue(`{"ok":true}`, map[string]interface{}{"type": "object"})
	require.NoError(t, err)
	require.Equal(t, map[string]interface{}{"ok": true}, value)
	_, err = parsePromptValue("{", map[string]interface{}{"type": "object"})
	require.Error(t, err)

	require.NoError(t, validateInputAgainstSchema(map[string]interface{}{"anything": "ok"}, nil))
	require.ErrorContains(t, validateSchemaType("name", 1, "string"), "must be a string")
	require.ErrorContains(t, validateSchemaType("count", 1.5, "integer"), "must be an integer")
	require.ErrorContains(t, validateSchemaType("score", "x", "number"), "must be a number")
	require.ErrorContains(t, validateSchemaType("ok", "true", "boolean"), "must be a boolean")
	// object/array types are not enforced client-side: the SDK emits
	// {"type": "object"} for Optional[...] params, so enforcing it would reject
	// valid scalars passed to optional fields. The control plane is authoritative.
	require.NoError(t, validateSchemaType("obj", "x", "object"))
	require.NoError(t, validateSchemaType("arr", "x", "array"))
}

func TestRunReasonerListFormats(t *testing.T) {
	now := time.Now().UTC().Format(time.RFC3339)
	var observedQuery url.Values
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/reasoners", r.URL.Path)
		observedQuery = r.URL.Query()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"reasoners":[{"node":"node","reasoner":"echo","last_run_at":%q,"status":"live"}],"shown":1,"total":2}`, now)
	})

	var stdout bytes.Buffer
	err := runReasonerList(context.Background(), "echo", &lsOptions{
		all:          true,
		node:         "node",
		live:         true,
		outputFormat: "json",
		stdout:       &stdout,
	})
	require.NoError(t, err)
	require.Equal(t, "echo", observedQuery.Get("query"))
	require.Equal(t, "node", observedQuery.Get("node"))
	require.Equal(t, "true", observedQuery.Get("all"))
	require.Equal(t, "true", observedQuery.Get("live"))
	require.Contains(t, stdout.String(), `"reasoners"`)

	stdout.Reset()
	err = runReasonerList(context.Background(), "", &lsOptions{
		outputFormat: "pretty",
		stdout:       &stdout,
	})
	require.NoError(t, err)
	require.Contains(t, stdout.String(), "RECENT")
	require.Contains(t, stdout.String(), "node.echo")
	require.Contains(t, stdout.String(), "1 more recent")

	err = runReasonerList(context.Background(), "", &lsOptions{outputFormat: "toml", stdout: &stdout})
	var exitErr cliExitError
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)
}

func ptrString(value string) *string {
	return &value
}

func TestStreamExecutionEventsFormatsAndFailures(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/executions/run-json/events":
			require.Equal(t, "2", r.URL.Query().Get("from"))
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-1","status":"running","type":"execution.updated"}`+"\n\n")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-1","status":"succeeded","type":"execution.completed"}`+"\n\n")
		case "/api/v1/executions/run-pretty/events":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, `data: {"execution_id":"exec-2","status":"failed","type":"execution.failed"}`+"\n\n")
		default:
			http.Error(w, `{"error":"missing"}`, http.StatusNotFound)
		}
	})

	var stdout bytes.Buffer
	err := streamExecutionEvents(context.Background(), "run-json", 2, "json", &stdout)
	require.NoError(t, err)
	require.Equal(t, 2, strings.Count(strings.TrimSpace(stdout.String()), "\n")+1)
	require.Contains(t, stdout.String(), `"status":"succeeded"`)

	stdout.Reset()
	err = streamExecutionEvents(context.Background(), "run-pretty", 0, "pretty", &stdout)
	var exitErr cliExitError
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 1, exitErr.Code)
	require.Contains(t, stdout.String(), "failed")

	err = streamExecutionEvents(context.Background(), "missing", 0, "json", &stdout)
	require.ErrorAs(t, err, &exitErr)
	require.Equal(t, 2, exitErr.Code)
}

func TestTriggerCommonHelpers(t *testing.T) {
	dir := t.TempDir()
	yamlPath := filepath.Join(dir, "payload.yaml")
	require.NoError(t, os.WriteFile(yamlPath, []byte("message: hello\nitems:\n  - 1\n"), 0o600))
	input, err := parseInputSource("@" + yamlPath)
	require.NoError(t, err)
	require.Equal(t, "hello", input["message"])

	input, err = parseInputSource(`{"message":"inline"}`)
	require.NoError(t, err)
	require.Equal(t, "inline", input["message"])

	_, err = parseInputSource("@")
	require.ErrorContains(t, err, "path cannot be empty")
	_, err = parseStructuredInput([]byte("[]"), "inline input")
	require.ErrorContains(t, err, "must decode to a JSON object")

	value := map[string]interface{}{"a": map[string]interface{}{"b": []interface{}{float64(1), float64(2)}}}
	extracted, err := extractField(value, ".a.b[1]")
	require.NoError(t, err)
	require.Equal(t, float64(2), extracted)
	_, err = extractField(value, "a.b")
	require.ErrorContains(t, err, "must start")
	_, err = extractField(value, ".a.b[bogus]")
	require.ErrorContains(t, err, "invalid array index")
	_, err = extractField(value, ".a.c")
	require.ErrorContains(t, err, "not found")

	require.True(t, terminalStatus("completed"))
	require.True(t, failedStatus("timed_out"))
	require.False(t, failedStatus("succeeded"))
	require.Equal(t, "/path?a=b", appendQuery("/path", url.Values{"a": []string{"b"}}))
}

func TestMakeRequestAddsAuthAndTimeoutPolicy(t *testing.T) {
	var gotAPIKey string
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAPIKey = r.Header.Get("X-API-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	})
	apiKey = "secret"

	resp, err := makeRequest(context.Background(), http.MethodPost, "api/test", map[string]string{"ok": "true"}, "application/json")
	require.NoError(t, err)
	body, err := readJSONResponse(resp, nil)
	require.NoError(t, err)
	require.JSONEq(t, `{"ok":true}`, string(body))
	require.Equal(t, "secret", gotAPIKey)
	require.Zero(t, triggerHTTPClient("text/event-stream").Timeout)
}
