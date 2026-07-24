package cli

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"gopkg.in/yaml.v3"
)

func TestCommandAndAgentHelpers(t *testing.T) {
	t.Run("version command prints build and runtime details", func(t *testing.T) {
		output := captureOutput(t, func() {
			NewVersionCommand(VersionInfo{
				Version: "1.2.3",
				Commit:  "abc123",
				Date:    "2026-04-08",
			}).Run(&cobra.Command{}, nil)
		})
		require.Contains(t, output, "AgentField Control Plane")
		require.Contains(t, output, "Version:    1.2.3")
		require.Contains(t, output, "Commit:     abc123")
	})

	t.Run("list command covers parse and populated registry branches", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)

		badRegistry := filepath.Join(home, "installed.yaml")
		require.NoError(t, os.WriteFile(badRegistry, []byte("installed: ["), 0o644))
		cmd := &cobra.Command{}
		errBuf := &bytes.Buffer{}
		cmd.SetErr(errBuf)
		runListCommand(cmd, nil)
		require.Contains(t, errBuf.String(), "failed to parse registry")

		port := 8080
		pid := 1234
		registry := &packages.InstallationRegistry{
			Installed: map[string]packages.InstalledPackage{
				"demo": {
					Name:        "demo",
					Version:     "1.0.0",
					Description: "demo package",
					Path:        "/tmp/demo",
					Status:      "running",
					Runtime:     packages.RuntimeInfo{Port: &port, PID: &pid},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(badRegistry, data, 0o644))

		output := captureOutput(t, func() {
			runListCommand(&cobra.Command{}, nil)
		})
		require.Contains(t, output, "Installed agent nodes")
		require.Contains(t, output, "demo package")
		require.Contains(t, output, "8080")
	})

	t.Run("log viewer covers missing logs and tail output", func(t *testing.T) {
		home := t.TempDir()
		logFile := filepath.Join(home, "demo.log")
		registryPath := filepath.Join(home, "installed.yaml")

		registry := []byte(`
installed:
  demo:
    name: demo
    path: /tmp/demo
    status: running
    runtime:
      log_file: ` + logFile + `
`)
		require.NoError(t, os.WriteFile(registryPath, registry, 0o644))

		lv := &LogViewer{AgentFieldHome: home, Tail: 5}
		require.NoError(t, lv.ViewLogs("demo"))

		require.NoError(t, os.WriteFile(logFile, []byte("one\ntwo\n"), 0o644))
		output := captureOutput(t, func() {
			require.NoError(t, lv.tailLogs(logFile, 1))
		})
		require.Contains(t, output, "two")
	})

	t.Run("agent helper functions cover request paths and formatting", func(t *testing.T) {
		tests := []struct {
			status int
			want   string
		}{
			{status: http.StatusUnauthorized, want: "Provide a valid API key"},
			{status: http.StatusForbidden, want: "lacks required permissions"},
			{status: http.StatusNotFound, want: "Check the endpoint path"},
			{status: http.StatusBadRequest, want: "Review command flags"},
			{status: http.StatusBadGateway, want: "Server error"},
			{status: http.StatusTeapot, want: "Request failed"},
		}
		for _, tc := range tests {
			require.Contains(t, defaultHintForStatus(tc.status), tc.want)
		}

		require.Equal(t, "alpha/beta%20gamma", escapePathSegments(" alpha / beta gamma "))
		require.NotEmpty(t, agentHelpData()["subcommands"])

		outputFormat = "compact"
		output := captureOutput(t, func() {
			require.NoError(t, outputAgentJSON(map[string]string{"ok": "yes"}))
		})
		require.Equal(t, "{\"ok\":\"yes\"}\n", output)
		outputFormat = ""

		withStdin(t, "{\"operations\":[]}\n", func() {
			data, err := readBatchInput("-")
			require.NoError(t, err)
			require.JSONEq(t, `{"operations":[]}`, string(data))
		})

		file := filepath.Join(t.TempDir(), "batch.json")
		require.NoError(t, os.WriteFile(file, []byte(`{"operations":[{"id":"1"}]}`), 0o644))
		data, err := readBatchInput(file)
		require.NoError(t, err)
		require.JSONEq(t, `{"operations":[{"id":"1"}]}`, string(data))

		oldTransport := http.DefaultTransport
		http.DefaultTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			require.Equal(t, "application/json", req.Header.Get("Accept"))
			require.Equal(t, "secret", req.Header.Get("X-API-Key"))
			require.Equal(t, "/api/test", req.URL.Path)
			return &http.Response{
				StatusCode: http.StatusCreated,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(bytes.NewReader([]byte(`{"ok":true}`))),
				Request:    req,
			}, nil
		})
		defer func() { http.DefaultTransport = oldTransport }()

		oldServer, oldAPIKey, oldTimeout := serverURL, apiKey, requestTimeout
		serverURL, apiKey, requestTimeout = "http://agent.test", "secret", 1
		defer func() {
			serverURL, apiKey, requestTimeout = oldServer, oldAPIKey, oldTimeout
		}()

		body, status, err := agentHTTP(http.MethodPost, "api/test", map[string]string{"hello": "world"})
		require.NoError(t, err)
		require.Equal(t, http.StatusCreated, status)
		require.JSONEq(t, `{"ok":true}`, string(body))
	})

	t.Run("agent command help path returns structured output", func(t *testing.T) {
		oldServer := serverURL
		serverURL = "http://example.test"
		defer func() { serverURL = oldServer }()

		cmd := NewAgentCommand()
		cmd.SetArgs([]string{})
		output := captureOutput(t, func() {
			require.NoError(t, cmd.Execute())
		})
		require.Contains(t, output, `"ok": true`)
		require.Contains(t, output, `"command": "af agent"`)
		require.Contains(t, output, `"server": "http://example.test"`)
	})
}
