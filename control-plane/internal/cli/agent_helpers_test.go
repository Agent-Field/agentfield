package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAgentHelpers(t *testing.T) {
	t.Run("output agent json supports pretty and compact", func(t *testing.T) {
		tests := []struct {
			name   string
			format string
			want   string
		}{
			{name: "pretty", format: "json", want: "\n  "},
			{name: "compact", format: "compact", want: `{"ok":true}`},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				oldFormat := outputFormat
				outputFormat = tt.format
				defer func() { outputFormat = oldFormat }()

				output := captureOutput(t, func() {
					require.NoError(t, outputAgentJSON(map[string]bool{"ok": true}))
				})
				require.Contains(t, output, tt.want)
			})
		}
	})

	t.Run("agent http covers success headers and failures", func(t *testing.T) {
		var gotAPIKey string
		var gotContentType string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAPIKey = r.Header.Get("X-API-Key")
			gotContentType = r.Header.Get("Content-Type")
			require.Equal(t, "/api/test", r.URL.Path)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()

		oldServer, oldKey, oldTimeout := serverURL, apiKey, requestTimeout
		serverURL, apiKey, requestTimeout = server.URL+"/", "api-secret", 1
		defer func() {
			serverURL, apiKey, requestTimeout = oldServer, oldKey, oldTimeout
		}()

		body, status, err := agentHTTP(http.MethodPost, "api/test", map[string]string{"name": "demo"})
		require.NoError(t, err)
		require.Equal(t, http.StatusOK, status)
		require.JSONEq(t, `{"ok":true}`, string(body))
		require.Equal(t, "api-secret", gotAPIKey)
		require.Equal(t, "application/json", gotContentType)

		_, _, err = agentHTTP(http.MethodPost, "/api/test", map[string]func(){"bad": func() {}})
		require.ErrorContains(t, err, "encode request body")

		serverURL = "://bad-url"
		_, _, err = agentHTTP(http.MethodGet, "/api/test", nil)
		require.ErrorContains(t, err, "build request")
	})

	t.Run("read batch input covers stdin and files", func(t *testing.T) {
		withStdin(t, `{"operations":[]}`, func() {
			data, err := readBatchInput("-")
			require.NoError(t, err)
			require.JSONEq(t, `{"operations":[]}`, string(data))
		})

		withStdin(t, "   \n", func() {
			_, err := readBatchInput("")
			require.ErrorContains(t, err, "stdin is empty")
		})

		path := filepath.Join(t.TempDir(), "ops.json")
		require.NoError(t, os.WriteFile(path, []byte(`{"operations":[1]}`), 0o644))
		data, err := readBatchInput(path)
		require.NoError(t, err)
		require.JSONEq(t, `{"operations":[1]}`, string(data))

		emptyPath := filepath.Join(t.TempDir(), "empty.json")
		require.NoError(t, os.WriteFile(emptyPath, []byte(" \n"), 0o644))
		_, err = readBatchInput(emptyPath)
		require.ErrorContains(t, err, "is empty")

		_, err = readBatchInput(filepath.Join(t.TempDir(), "missing.json"))
		require.ErrorContains(t, err, "read file")
	})
}

func TestDefaultHintForStatusAndEscaping(t *testing.T) {
	cases := []struct {
		status int
		want   string
	}{
		{status: http.StatusUnauthorized, want: "Provide a valid API key"},
		{status: http.StatusForbidden, want: "lacks required permissions"},
		{status: http.StatusNotFound, want: "Check the endpoint path"},
		{status: http.StatusBadRequest, want: "Review command flags"},
		{status: http.StatusInternalServerError, want: "Server error"},
		{status: http.StatusTeapot, want: "Request failed"},
	}

	for _, tc := range cases {
		require.Contains(t, defaultHintForStatus(tc.status), tc.want)
	}

	require.Equal(t, "agent/id/with%20spaces", escapePathSegments(" agent /id/with spaces "))
	require.Equal(t, "", escapePathSegments(" / / "))
}

func TestAgentHelpDataShape(t *testing.T) {
	data := agentHelpData()
	require.Equal(t, "af agent", data["command"])
	require.Equal(t, "v1", data["version"])

	globalFlags, ok := data["global_flags"].([]map[string]interface{})
	require.True(t, ok)
	require.Len(t, globalFlags, 4)

	subcommands, ok := data["subcommands"].([]map[string]interface{})
	require.True(t, ok)
	require.NotEmpty(t, subcommands)

	raw, err := json.Marshal(data)
	require.NoError(t, err)
	require.Contains(t, string(raw), `"quick_start"`)
	require.Contains(t, string(raw), `"response_schemas"`)
}

func TestListCommandAndLogViewer(t *testing.T) {
	t.Run("list command covers empty, parse error, and populated registry", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)

		output := captureOutput(t, func() {
			runListCommand(nil, nil)
		})
		require.Contains(t, output, "No agent node packages installed")

		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))
		cmd := NewListCommand()
		var stderr bytes.Buffer
		cmd.SetErr(&stderr)
		runListCommand(cmd, nil)
		require.Contains(t, stderr.String(), "failed to parse registry")

		port := 8123
		pid := 456
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    version: "1.2.3"
    description: example agent
    path: /tmp/demo
    status: running
    runtime:
      port: 8123
      pid: 456
`), 0o644))
		output = captureOutput(t, func() {
			runListCommand(cmd, nil)
		})
		require.Contains(t, output, "Installed Agent Node Packages (1 total)")
		require.Contains(t, output, "demo (v1.2.3)")
		require.Contains(t, output, "Running on port 8123 (PID: 456)")
		_ = port
		_ = pid
	})

	t.Run("log viewer handles errors missing logs and tailing", func(t *testing.T) {
		home := t.TempDir()
		lv := &LogViewer{AgentFieldHome: home, Tail: 5}

		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))
		err := lv.ViewLogs("demo")
		require.ErrorContains(t, err, "failed to parse registry")

		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: {}\n"), 0o644))
		err = lv.ViewLogs("demo")
		require.ErrorContains(t, err, "not installed")

		logPath := filepath.Join(home, "demo.log")
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    runtime:
      log_file: `+logPath+`
`), 0o644))
		err = lv.ViewLogs("demo")
		require.NoError(t, err)

		require.NoError(t, os.WriteFile(logPath, []byte("one\ntwo\nthree\n"), 0o644))
		output := captureOutput(t, func() {
			require.NoError(t, lv.ViewLogs("demo"))
		})
		require.Contains(t, output, "one")
		require.Contains(t, output, "three")

		err = lv.tailLogs(filepath.Join(home, "missing.log"), 2)
		require.Error(t, err)
	})
}

func TestAgentCommandSubcommands(t *testing.T) {
	type requestRecord struct {
		Method string
		Path   string
		Query  string
		Body   string
	}

	var records []requestRecord
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body bytes.Buffer
		_, _ = body.ReadFrom(r.Body)
		records = append(records, requestRecord{
			Method: r.Method,
			Path:   r.URL.Path,
			Query:  r.URL.RawQuery,
			Body:   body.String(),
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true,"data":{"id":"demo"}}`))
	}))
	defer server.Close()

	oldServer, oldFormat, oldTimeout := serverURL, outputFormat, requestTimeout
	serverURL, outputFormat, requestTimeout = server.URL, "json", 1
	defer func() {
		serverURL, outputFormat, requestTimeout = oldServer, oldFormat, oldTimeout
	}()

	tests := []struct {
		name         string
		args         []string
		wantMethod   string
		wantPath     string
		wantQuery    string
		wantBodyPart string
	}{
		{name: "status", args: []string{"status"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/status"},
		{name: "discover", args: []string{"discover", "--query", "runs", "--group", "agentic", "--method", "get", "--limit", "5"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/discover", wantQuery: "group=agentic&limit=5&method=GET&q=runs"},
		{name: "query", args: []string{"query", "--resource", "runs", "--status", "completed", "--agent-id", "agent-1", "--run-id", "run-1", "--since", "2026-01-01T00:00:00Z", "--until", "2026-01-02T00:00:00Z", "--limit", "5", "--offset", "2", "--include", "steps,metrics"}, wantMethod: http.MethodPost, wantPath: "/api/v1/agentic/query", wantBodyPart: `"resource":"runs"`},
		{name: "run", args: []string{"run", "--id", "run/1"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/run/run/1"},
		{name: "agent summary", args: []string{"agent-summary", "--id", "agent/1"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/agent/agent/1/summary"},
		{name: "kb topics", args: []string{"kb", "topics"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/kb/topics"},
		{name: "kb search", args: []string{"kb", "search", "--query", "test", "--topic", "agents", "--sdk", "go", "--difficulty", "advanced", "--limit", "3"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/kb/articles", wantQuery: "difficulty=advanced&limit=3&q=test&sdk=go&topic=agents"},
		{name: "kb read", args: []string{"kb", "read", "--id", "patterns/hunt prove"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/kb/articles/patterns/hunt prove"},
		{name: "kb guide", args: []string{"kb", "guide", "--goal", "build audit flow"}, wantMethod: http.MethodGet, wantPath: "/api/v1/agentic/kb/guide", wantQuery: "goal=build+audit+flow"},
		{name: "batch", args: []string{"batch", "--file", batchFile(t, `{"operations":[{"id":"op1"}]}`)}, wantMethod: http.MethodPost, wantPath: "/api/v1/agentic/batch", wantBodyPart: `"operations":[{"id":"op1"}]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			records = nil
			cmd := NewAgentCommand()
			cmd.SetArgs(tt.args)
			output := captureOutput(t, func() {
				require.NoError(t, cmd.Execute())
			})
			require.NotEmpty(t, records)
			require.Equal(t, tt.wantMethod, records[0].Method)
			require.Equal(t, tt.wantPath, records[0].Path)
			if tt.wantQuery != "" {
				require.Equal(t, tt.wantQuery, records[0].Query)
			}
			if tt.wantBodyPart != "" {
				require.Contains(t, records[0].Body, tt.wantBodyPart)
			}
			require.Contains(t, output, `"server": "`+server.URL+`"`)
		})
	}

	records = nil
	output := captureOutput(t, func() {
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{})
		require.NoError(t, cmd.Execute())
	})
	require.Empty(t, records)
	require.Contains(t, output, `"command": "af agent"`)

	output = captureOutput(t, func() {
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"kb"})
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, `"available": [`)
}

func TestSpinnerAndPrintHelpers(t *testing.T) {
	output := captureOutput(t, func() {
		spinner := NewSpinner("working")
		require.Equal(t, "working", spinner.message)
		spinner.Start()
		// Sleep is inherent to the test: let the spinner goroutine animate briefly before stopping.
		time.Sleep(50 * time.Millisecond)
		spinner.UpdateMessage("updated")
		spinner.Success("done")

		spinner = NewSpinner("working")
		spinner.Start()
		// Sleep is inherent to the test: let the spinner goroutine animate briefly before stopping.
		time.Sleep(50 * time.Millisecond)
		spinner.Error("failed")

		PrintSuccess("ok")
		PrintError("bad")
		PrintInfo("info")
		PrintWarning("warn")
		PrintHeader("header")
		PrintSubheader("subheader")
		PrintBullet("bullet")
	})

	require.Contains(t, output, "done")
	require.Contains(t, output, "failed")
	require.Contains(t, output, "ok")
	require.Contains(t, output, "bad")
	require.Contains(t, output, "info")
	require.Contains(t, output, "warn")
	require.Contains(t, output, "header")
	require.Contains(t, output, "subheader")
	require.Contains(t, output, "bullet")
}

func TestProxyToServerArrayResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"one"}]`))
	}))
	defer server.Close()

	oldServer, oldFormat, oldTimeout := serverURL, outputFormat, requestTimeout
	serverURL, outputFormat, requestTimeout = server.URL, "json", 1
	defer func() {
		serverURL, outputFormat, requestTimeout = oldServer, oldFormat, oldTimeout
	}()

	output := captureOutput(t, func() {
		proxyToServer(http.MethodGet, "/array", nil)
	})
	require.Contains(t, output, `"ok": true`)
	require.Contains(t, output, `"server": "`+server.URL+`"`)
}

func batchFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "batch.json")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}
