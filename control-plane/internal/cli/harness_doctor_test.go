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

func TestAforgeEnsureCommand(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "1")
	for _, args := range [][]string{{"ensure"}, {"ensure", "--force"}} {
		cmd := NewAforgeCommand()
		cmd.SetArgs(args)
		require.NoError(t, cmd.Execute())
	}
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

func TestHarnessDoctorSurveyReturnsSuccessWhenProvidersAreMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor"})

	require.NoError(t, cmd.Execute())
	require.Contains(t, stdout.String(), "aforge: unavailable")
}

func TestHarnessDoctorJSONSurveyReturnsSuccessWhenProvidersAreMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())

	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--json"})

	require.NoError(t, cmd.Execute())
	var reports []HarnessProviderHealth
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &reports))
	require.NotEmpty(t, reports)
	for _, report := range reports {
		require.False(t, report.Usable)
	}
}

func TestHarnessDoctorReturnsErrorForRequestedMissingProvider(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "opencode", "--json"})

	err := cmd.Execute()
	require.ErrorContains(t, err, "requested harness provider is unavailable")
	require.True(t, IsCLIExitError(err))
	require.Equal(t, 1, ExitCode(err))

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

	// A CLI that does not recognise the version argument prints its usage
	// banner and exits non-zero, so the fixture does exactly that — a probe
	// that only checked for output would file the banner as the version.
	const usageOnExitOne = "echo 'usage: build and revise task graphs' >&2; exit 1"

	t.Run("version is required", func(t *testing.T) {
		writeHarnessTestScript(t, binDir, "required", usageOnExitOne)
		reports := reportsForTestSpec(t, harnessProviderSpec{Name: "required", Binary: "required", VersionArgs: [][]string{{"version"}, {"--version"}}})
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

func TestHarnessDoctorGrokSpec(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	reports, err := buildHarnessDoctorReports([]string{"grok"})
	require.NoError(t, err)
	require.Len(t, reports, 1)
	require.Equal(t, "grok", reports[0].Provider)
	require.Equal(t, "Install the Grok Build CLI, then run: grok login", reports[0].InstallCommand)
	require.Equal(t, []string{"XAI_API_KEY"}, reports[0].AuthEnvVars)
}

func TestHarnessDoctorAcceptsGrokProvider(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	cmd := NewHarnessCommand()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetArgs([]string{"doctor", "--provider", "grok", "--json"})

	err := cmd.Execute()
	require.Error(t, err)
	require.NotContains(t, err.Error(), "unknown harness provider")
	require.Contains(t, err.Error(), "requested harness provider is unavailable: grok")
}

func TestHarnessDoctorGrokReportsUsable(t *testing.T) {
	binDir := t.TempDir()
	writeHarnessTestBinary(t, binDir, "grok", "grok 1.2.3")
	t.Setenv("PATH", binDir)
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("XAI_API_KEY", "configured")

	reports, err := buildHarnessDoctorReports([]string{"grok"})
	require.NoError(t, err)
	require.Len(t, reports, 1)
	require.True(t, reports[0].Usable)
	require.Equal(t, "grok 1.2.3", reports[0].Version)
	require.Equal(t, "configured", reports[0].Auth)
}

func TestHarnessProviderSpecsMatchPythonProviderNames(t *testing.T) {
	// Keep in sync with sdk/python/agentfield/harness/_availability.py.
	expected := []string{"aforge", "claude-code", "codex", "gemini", "opencode", "grok"}
	actual := make([]string, 0, len(harnessProviderSpecs))
	for _, spec := range harnessProviderSpecs {
		actual = append(actual, spec.Name)
	}
	require.ElementsMatch(t, expected, actual)
}

// The pinned aforge release answers `version`, so aforge is held to the same
// bar as every other provider: a binary that cannot name itself is unusable.
func TestHarnessDoctorAforgeRequiresRealVersion(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}

	t.Run("reports the version it prints", func(t *testing.T) {
		binDir := t.TempDir()
		writeHarnessTestScript(t, binDir, "aforge", "[ \"$1\" = version ] && printf 'aforge v0.1.0\\n'")
		t.Setenv("PATH", binDir)
		t.Setenv("AGENTFIELD_HOME", t.TempDir())

		reports, err := buildHarnessDoctorReports([]string{"aforge"})
		require.NoError(t, err)
		require.True(t, reports[0].Usable)
		require.Equal(t, "aforge v0.1.0", reports[0].Version)
		require.Empty(t, reports[0].Issues)
	})

	t.Run("unusable without a version", func(t *testing.T) {
		binDir := t.TempDir()
		writeHarnessTestScript(t, binDir, "aforge", "echo 'usage: aforge <command>' >&2; exit 1")
		t.Setenv("PATH", binDir)
		t.Setenv("AGENTFIELD_HOME", t.TempDir())

		reports, err := buildHarnessDoctorReports([]string{"aforge"})
		require.NoError(t, err)
		require.True(t, reports[0].Installed)
		require.False(t, reports[0].Usable)
		require.Empty(t, reports[0].Version)
		require.Equal(t, []string{"version_probe_failed"}, reports[0].Issues)
	})
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
