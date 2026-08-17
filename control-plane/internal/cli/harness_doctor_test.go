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

func TestAforgeCommandHelp(t *testing.T) {
	cmd := NewAforgeCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"--help"})
	require.NoError(t, cmd.Execute())
	require.Contains(t, stdout.String(), "ensure")

	stdout.Reset()
	cmd = NewAforgeCommand()
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"ensure", "--help"})
	require.NoError(t, cmd.Execute())
	require.Contains(t, stdout.String(), "--force")
}

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

func TestProbeHarnessBinaryVersionBehavior(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	// Keep the managed-bin fallback out of these results: a developer machine
	// with a real ~/.agentfield/bin/aforge would otherwise answer differently
	// from CI.
	t.Setenv("AGENTFIELD_HOME", t.TempDir())

	// The pinned aforge build prints its whole usage banner and exits 1 for
	// every version-ish argument, so the fixture does exactly that — a probe
	// that only checked for output would file the banner as the version.
	const usageOnExitOne = "echo 'aforge — build and revise task graphs' >&2; exit 1"

	t.Run("optional version", func(t *testing.T) {
		writeHarnessTestScript(t, binDir, "optional", usageOnExitOne)
		reports := reportsForTestSpec(t, harnessProviderSpec{Name: "optional", Binary: "optional", VersionArgs: [][]string{{"version"}, {"--version"}}, VersionOptional: true})
		require.True(t, reports[0].Usable)
		require.Equal(t, "unknown", reports[0].Version)
		require.Equal(t, []string{"version_unavailable"}, reports[0].Issues)
	})

	t.Run("required version", func(t *testing.T) {
		writeHarnessTestScript(t, binDir, "required", usageOnExitOne)
		reports := reportsForTestSpec(t, harnessProviderSpec{Name: "required", Binary: "required"})
		require.False(t, reports[0].Usable)
		require.Empty(t, reports[0].Version)
		require.Equal(t, []string{"version_probe_failed"}, reports[0].Issues)
	})

	t.Run("managed bin fallback", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)
		managed := filepath.Join(home, "bin")
		require.NoError(t, os.MkdirAll(managed, 0o755))
		writeHarnessTestScript(t, managed, "offpath", "printf 'offpath 2.0\\n'")
		reports := reportsForTestSpec(t, harnessProviderSpec{Name: "offpath", Binary: "offpath"})
		require.True(t, reports[0].Installed)
		require.Equal(t, "offpath 2.0", reports[0].Version)
		require.Equal(t, filepath.Join(managed, "offpath"), reports[0].Binary)
	})

	t.Run("ordered arguments", func(t *testing.T) {
		writeHarnessTestScript(t, binDir, "ordered", "[ \"$1\" = --version ] && printf 'ordered 1.0\\n'")
		reports := reportsForTestSpec(t, harnessProviderSpec{Name: "ordered", Binary: "ordered", VersionArgs: [][]string{{"version"}, {"--version"}}})
		require.True(t, reports[0].Usable)
		require.Equal(t, "ordered 1.0", reports[0].Version)
	})
}

func TestHarnessDoctorAforgeSpec(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	reports, err := buildHarnessDoctorReports([]string{"aforge"})
	require.NoError(t, err)
	require.Len(t, reports, 1)
	require.Equal(t, "aforge", reports[0].Provider)
	require.Equal(t, "af aforge ensure", reports[0].InstallCommand)
	require.Equal(t, []string{"OPENROUTER_API_KEY"}, reports[0].AuthEnvVars)
}

func reportsForTestSpec(t *testing.T, spec harnessProviderSpec) []HarnessProviderHealth {
	t.Helper()
	original := harnessProviderSpecs
	harnessProviderSpecs = []harnessProviderSpec{spec}
	t.Cleanup(func() { harnessProviderSpecs = original })
	reports, err := buildHarnessDoctorReports([]string{spec.Name})
	require.NoError(t, err)
	return reports
}

func writeHarnessTestScript(t *testing.T, dir, name, body string) {
	t.Helper()
	require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\n"+body+"\n"), 0o755))
}

func writeHarnessTestBinary(t *testing.T, dir, name, version string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}
	path := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\nprintf '%s\\n' '"+version+"'\n"), 0o755))
}
