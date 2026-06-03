package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDoctorHelpersAndCommand(t *testing.T) {
	t.Run("build report honors env and recommendations", func(t *testing.T) {
		t.Setenv("OPENROUTER_API_KEY", "")
		t.Setenv("ANTHROPIC_API_KEY", "")
		t.Setenv("OPENAI_API_KEY", "openai-key")
		t.Setenv("GOOGLE_API_KEY", "")

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/api/v1/health", r.URL.Path)
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		report := buildDoctorReport(server.URL)
		require.Equal(t, server.URL, report.ControlPlane.URL)
		require.True(t, report.ControlPlane.Reachable)
		require.Equal(t, "openai", report.Recommendation.Provider)
		require.Equal(t, "gpt-4o", report.Recommendation.AIModel)
		require.True(t, report.ProviderKeys["openai"].Set)
		require.Equal(t, "OPENAI_API_KEY", report.ProviderKeys["openai"].EnvVar)
	})

	t.Run("tool and control plane checks cover success and failure", func(t *testing.T) {
		tool := checkTool("go", "version")
		require.True(t, tool.Available)
		require.NotEmpty(t, tool.Path)

		missing := checkTool("agentfield-missing-binary-for-tests", "--version")
		require.False(t, missing.Available)

		okServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`ok`))
		}))
		defer okServer.Close()

		status := checkControlPlane(okServer.URL)
		require.True(t, status.Reachable)
		require.Equal(t, "200 OK", status.HealthStatus)
		require.Equal(t, "agentfield/control-plane:latest", status.DockerImageName)

		failServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer failServer.Close()

		status = checkControlPlane(failServer.URL)
		require.False(t, status.Reachable)
		require.Equal(t, "503 Service Unavailable", status.HealthStatus)
	})

	t.Run("pretty report and helpers render expected text", func(t *testing.T) {
		report := DoctorReport{
			OS:   "linux",
			Arch: "amd64",
			Python: ToolStatus{
				Available: true,
				Path:      "/usr/bin/python3",
				Version:   "Python 3.12.0",
			},
			Node:   ToolStatus{},
			Docker: ToolStatus{Available: true, Path: "/usr/bin/docker"},
			HarnessProviders: map[string]ToolStatus{
				"claude-code": {Available: true, Path: "/usr/bin/claude"},
				"codex":       {},
				"gemini":      {},
				"opencode":    {},
			},
			ProviderKeys: map[string]ProviderKey{
				"openrouter": {EnvVar: "OPENROUTER_API_KEY", Set: true},
				"anthropic":  {EnvVar: "ANTHROPIC_API_KEY", Set: false},
				"openai":     {EnvVar: "OPENAI_API_KEY", Set: false},
				"google":     {EnvVar: "GOOGLE_API_KEY", Set: false},
			},
			ControlPlane: ControlPlaneStatus{
				URL:              "http://localhost:8080",
				Reachable:        true,
				HealthStatus:     "200 OK",
				DockerImageName:  "agentfield/control-plane:latest",
				DockerImageLocal: true,
			},
			Recommendation: Recommendation{
				Provider:         "openrouter",
				AIModel:          "openrouter/google/gemini-2.5-flash",
				HarnessUsable:    true,
				HarnessProviders: []string{"claude-code"},
				Notes:            []string{"note one"},
			},
		}

		output := captureOutput(t, func() {
			printDoctorReport(report)
		})
		require.Contains(t, output, "os/arch: linux/amd64")
		require.Contains(t, output, "Python 3.12.0")
		require.Contains(t, output, "note one")
		require.Contains(t, output, "af doctor --json | jq")

		require.Equal(t, "yes", ifThen(true, "yes", "no"))
		require.Equal(t, "no", ifThen(false, "yes", "no"))
	})

	t.Run("doctor command emits json", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		cmd := NewDoctorCommand()
		cmd.SetOut(&bytes.Buffer{})
		cmd.SetErr(&bytes.Buffer{})
		cmd.SetArgs([]string{"--json", "--server", server.URL})

		output := captureOutput(t, func() {
			require.NoError(t, cmd.Execute())
		})

		var report DoctorReport
		require.NoError(t, json.Unmarshal([]byte(output), &report))
		require.Equal(t, server.URL, report.ControlPlane.URL)
	})
}
