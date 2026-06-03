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

func TestNewInitCommandNonInteractive(t *testing.T) {
	t.Run("creates go project with flags", func(t *testing.T) {
		wd := t.TempDir()
		oldWD, err := os.Getwd()
		require.NoError(t, err)
		require.NoError(t, os.Chdir(wd))
		defer func() { _ = os.Chdir(oldWD) }()

		cmd := NewInitCommand()
		cmd.SetOut(&bytes.Buffer{})
		cmd.SetErr(&bytes.Buffer{})
		cmd.SetArgs([]string{"demo-agent", "--non-interactive", "--language", "go", "--author", "Jane Doe", "--email", "jane@example.com"})
		require.NoError(t, cmd.Execute())

		projectDir := filepath.Join(wd, "demo-agent")
		require.DirExists(t, projectDir)
		require.FileExists(t, filepath.Join(projectDir, "main.go"))
		require.FileExists(t, filepath.Join(projectDir, "go.mod"))
		readme, err := os.ReadFile(filepath.Join(projectDir, "README.md"))
		require.NoError(t, err)
		require.Contains(t, string(readme), "demo-agent")
	})

	t.Run("invalid project and unsupported language fail", func(t *testing.T) {
		wd := t.TempDir()
		oldWD, err := os.Getwd()
		require.NoError(t, err)
		require.NoError(t, os.Chdir(wd))
		defer func() { _ = os.Chdir(oldWD) }()

		cmd := NewInitCommand()
		cmd.SetArgs([]string{"Invalid Name", "--non-interactive", "--language", "go"})
		require.ErrorContains(t, cmd.Execute(), "invalid project name")

		cmd = NewInitCommand()
		cmd.SetArgs([]string{"demo-agent", "--non-interactive", "--language", "rust"})
		require.ErrorContains(t, cmd.Execute(), "unsupported language")
	})

	t.Run("defaults mode creates python project", func(t *testing.T) {
		wd := t.TempDir()
		oldWD, err := os.Getwd()
		require.NoError(t, err)
		require.NoError(t, os.Chdir(wd))
		defer func() { _ = os.Chdir(oldWD) }()

		cmd := NewInitCommand()
		cmd.SetArgs([]string{"demo-defaults", "--defaults"})
		require.NoError(t, cmd.Execute())
		require.FileExists(t, filepath.Join(wd, "demo-defaults", "main.py"))
	})
}

func TestNodesAndRootHelpers(t *testing.T) {
	t.Run("register serverless command success and errors", func(t *testing.T) {
		var gotAuth string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuth = r.Header.Get("Authorization")
			require.Equal(t, "/api/v1/nodes/register-serverless", r.URL.Path)
			var payload map[string]string
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			require.Equal(t, "https://fn.example/run", payload["invocation_url"])
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"node":{"id":"serverless-1"}}`))
		}))
		defer server.Close()

		cmd := NewNodesCommand()
		cmd.SetArgs([]string{"register-serverless", "--url", "https://fn.example/run", "--server", server.URL, "--token", "secret"})
		output := captureOutput(t, func() {
			require.NoError(t, cmd.Execute())
		})
		require.Contains(t, output, "Registered serverless agent: serverless-1")
		require.Equal(t, "Bearer secret", gotAuth)

		cmd = NewNodesCommand()
		cmd.SetArgs([]string{"register-serverless"})
		require.ErrorContains(t, cmd.Execute(), "--url is required")

		badJSONServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`not-json`))
		}))
		defer badJSONServer.Close()
		cmd = NewNodesCommand()
		cmd.SetArgs([]string{"register-serverless", "--url", "https://fn.example/run", "--server", badJSONServer.URL, "--timeout", time.Second.String()})
		require.ErrorContains(t, cmd.Execute(), "decode response")
	})

	t.Run("agent hint and root getters", func(t *testing.T) {
		hint := AgentHintJSON("bad command")
		require.Contains(t, hint, `"code":"invalid_command"`)
		require.Contains(t, hint, "bad command")

		oldServer, oldAPIKey := serverURL, apiKey
		serverURL, apiKey = "", ""
		defer func() {
			serverURL, apiKey = oldServer, oldAPIKey
		}()

		t.Setenv("AGENTFIELD_SERVER", "http://env-server")
		require.Equal(t, "http://env-server", GetServerURL())
		t.Setenv("AGENTFIELD_SERVER", "")
		t.Setenv("AGENTFIELD_SERVER_URL", "http://env-server-url")
		require.Equal(t, "http://env-server-url", GetServerURL())
		serverURL = "http://flag-server"
		require.Equal(t, "http://flag-server", GetServerURL())

		t.Setenv("AGENTFIELD_API_KEY", "env-key")
		require.Equal(t, "env-key", GetAPIKey())
		apiKey = "flag-key"
		require.Equal(t, "flag-key", GetAPIKey())
	})
}
