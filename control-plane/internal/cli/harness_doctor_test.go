package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHarnessDoctorJSONReportsRequestedProvider(t *testing.T) {
	binDir := t.TempDir()
	writeHarnessTestBinary(t, binDir, "codex", "codex-cli 1.2.3")
	t.Setenv("PATH", binDir)
	t.Setenv("OPENAI_API_KEY", "configured")

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "codex", "--json"})

	require.NoError(t, cmd.Execute())
	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.Equal(t, []HarnessProviderHealth{{
		Provider:       "codex",
		Binary:         filepath.Join(binDir, "codex"),
		Installed:      true,
		Version:        "codex-cli 1.2.3",
		Auth:           "configured",
		Usable:         true,
		InstallCommand: "npm install -g @openai/codex",
		AuthEnvVars:    []string{"OPENAI_API_KEY"},
		Issues:         []string{},
	}}, reports)
}

func TestHarnessDoctorReturnsErrorForRequestedMissingProvider(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "opencode", "--json"})

	err := cmd.Execute()
	require.ErrorContains(t, err, "requested harness provider is unavailable")

	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.False(t, reports[0].Installed)
	require.False(t, reports[0].Usable)
	require.Equal(t, []string{"binary_not_found"}, reports[0].Issues)
}

func TestHarnessDoctorClaudeCodeReportsInstalledWrapper(t *testing.T) {
	binDir := t.TempDir()
	// Stub interpreter standing in for `python3 -c <probe>`: prints "ok" as the
	// probe does when claude_agent_sdk is importable.
	writeHarnessTestBinary(t, binDir, "python3", "ok")
	t.Setenv("PATH", binDir)
	t.Setenv("ANTHROPIC_API_KEY", "configured")

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "claude-code", "--json"})

	require.NoError(t, cmd.Execute())
	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.Equal(t, []HarnessProviderHealth{{
		Provider:       "claude-code",
		Installed:      true,
		Auth:           "configured",
		Usable:         true,
		InstallCommand: "pip install 'agentfield[harness-claude]'",
		AuthEnvVars:    []string{"ANTHROPIC_API_KEY"},
		Issues:         []string{},
	}}, reports)
}

func TestHarnessDoctorClaudeCodeReportsMissingWrapper(t *testing.T) {
	binDir := t.TempDir()
	// Interpreter is present but claude_agent_sdk is not importable.
	writeHarnessTestBinary(t, binDir, "python3", "missing")
	t.Setenv("PATH", binDir)

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "claude-code", "--json"})

	err := cmd.Execute()
	require.ErrorContains(t, err, "requested harness provider is unavailable: claude-code")

	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.False(t, reports[0].Installed)
	require.False(t, reports[0].Usable)
	require.Equal(t, []string{"wrapper_not_installed"}, reports[0].Issues)
	require.Equal(t, "pip install 'agentfield[harness-claude]'", reports[0].InstallCommand)
}

func TestHarnessDoctorClaudeCodeReportsMissingPython(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "claude-code", "--json"})

	err := cmd.Execute()
	require.ErrorContains(t, err, "requested harness provider is unavailable: claude-code")

	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.False(t, reports[0].Installed)
	require.False(t, reports[0].Usable)
	require.Equal(t, []string{"python_not_found"}, reports[0].Issues)
}

func TestHarnessDoctorRejectsUnknownProvider(t *testing.T) {
	cmd := NewHarnessCommand()
	cmd.SetArgs([]string{"doctor", "--provider", "unknown"})
	require.ErrorContains(t, cmd.Execute(), "unknown harness provider")
}

func writeHarnessTestBinary(t *testing.T, dir, name, version string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}
	path := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\nprintf '%s\\n' '"+version+"'\n"), 0o755))
}
