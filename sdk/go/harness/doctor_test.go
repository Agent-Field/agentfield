package harness

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeLookPath resolves only the binaries present in installed, so tests never
// depend on what happens to be on the developer's or CI runner's PATH.
func fakeLookPath(installed map[string]string) func(string) (string, error) {
	return func(binary string) (string, error) {
		if path, ok := installed[binary]; ok {
			return path, nil
		}
		return "", fmt.Errorf("exec: %q: executable file not found in $PATH", binary)
	}
}

func fakeEnv(values map[string]string) func(string) string {
	return func(name string) string { return values[name] }
}

func healthByProvider(reports []ProviderHealth) map[string]ProviderHealth {
	out := make(map[string]ProviderHealth, len(reports))
	for _, report := range reports {
		out[report.Provider] = report
	}
	return out
}

func TestSupportedProviderNames_CoversEveryBuildableProvider(t *testing.T) {
	names := SupportedProviderNames()

	// Sorted and stable so callers can print it directly.
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	assert.Equal(t, sorted, names)

	// Every provider BuildProvider accepts must have an availability spec,
	// otherwise Doctor silently cannot report on a usable provider.
	for _, name := range names {
		_, err := BuildProvider(name, "")
		assert.NoError(t, err, "provider %q has a spec but BuildProvider rejects it", name)
	}

	for _, name := range []string{
		ProviderAforge, ProviderClaudeCode, ProviderCodex, ProviderGemini,
		ProviderOpenCode, ProviderPi, ProviderOMP, ProviderCursor,
	} {
		_, ok := SpecForProvider(name)
		assert.True(t, ok, "missing availability spec for %q", name)
	}
}

func TestSpecForProvider_UnknownReportsMissing(t *testing.T) {
	_, ok := SpecForProvider("nope")
	assert.False(t, ok)
}

func TestDoctor_InstalledProviderIsUsable(t *testing.T) {
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex},
		LookPath:  fakeLookPath(map[string]string{"codex": "/usr/local/bin/codex"}),
		Env:       fakeEnv(map[string]string{"OPENAI_API_KEY": "sk-test"}),
	})
	require.NoError(t, err)
	require.Len(t, reports, 1)

	health := reports[0]
	assert.Equal(t, ProviderCodex, health.Provider)
	assert.Equal(t, "/usr/local/bin/codex", health.Binary)
	assert.True(t, health.Installed)
	assert.True(t, health.Usable)
	assert.Equal(t, AuthConfigured, health.Auth)
	assert.Empty(t, health.Issues)
	// No probe requested, so no version is reported.
	assert.Empty(t, health.Version)
	assert.Equal(t, "npm install -g @openai/codex", health.InstallCommand)
	assert.Equal(t, []string{"OPENAI_API_KEY"}, health.AuthEnvVars)
}

func TestDoctor_MissingBinaryIsUnusableWithInstallCommand(t *testing.T) {
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderGemini},
		LookPath:  fakeLookPath(nil),
		Env:       fakeEnv(nil),
	})
	require.NoError(t, err)
	require.Len(t, reports, 1)

	health := reports[0]
	assert.False(t, health.Installed)
	assert.False(t, health.Usable)
	assert.Empty(t, health.Binary)
	assert.Equal(t, []string{IssueBinaryNotFound}, health.Issues)
	assert.Equal(t, AuthUnknown, health.Auth)
	// The report still tells the user how to fix it.
	assert.Equal(t, "npm install -g @google/gemini-cli", health.InstallCommand)
	assert.Equal(t, []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}, health.AuthEnvVars)
}

func TestDoctor_AuthConfiguredWhenAnyEnvVarSet(t *testing.T) {
	// gemini accepts either GEMINI_API_KEY or GOOGLE_API_KEY; the second alone
	// must still count as configured.
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderGemini},
		LookPath:  fakeLookPath(map[string]string{"gemini": "/usr/bin/gemini"}),
		Env:       fakeEnv(map[string]string{"GOOGLE_API_KEY": "key"}),
	})
	require.NoError(t, err)
	assert.Equal(t, AuthConfigured, reports[0].Auth)
}

func TestDoctor_ProviderWithoutAuthVarsReportsConfigured(t *testing.T) {
	// opencode needs no credentials of its own, so it must never look blocked
	// on auth.
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderOpenCode},
		LookPath:  fakeLookPath(map[string]string{"opencode": "/usr/bin/opencode"}),
		Env:       fakeEnv(nil),
	})
	require.NoError(t, err)
	assert.Equal(t, AuthConfigured, reports[0].Auth)
	assert.True(t, reports[0].Usable)
	assert.Empty(t, reports[0].AuthEnvVars)
}

func TestDoctor_AuthUnknownDoesNotBlockUsable(t *testing.T) {
	// Missing credentials are reported, but usability is about the binary:
	// several CLIs authenticate interactively rather than by env var.
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex},
		LookPath:  fakeLookPath(map[string]string{"codex": "/usr/bin/codex"}),
		Env:       fakeEnv(nil),
	})
	require.NoError(t, err)
	assert.Equal(t, AuthUnknown, reports[0].Auth)
	assert.True(t, reports[0].Usable)
}

func TestDoctor_ProbeRecordsVersionAndPassesResolvedCommand(t *testing.T) {
	var gotCommand []string
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex},
		Probe:     true,
		LookPath:  fakeLookPath(map[string]string{"codex": "/usr/local/bin/codex"}),
		Env:       fakeEnv(nil),
		VersionProbe: func(_ context.Context, command []string) (string, error) {
			gotCommand = command
			return "codex 0.1.2", nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "codex 0.1.2", reports[0].Version)
	assert.True(t, reports[0].Usable)
	// The probe receives the resolved path plus the spec's version args.
	assert.Equal(t, []string{"/usr/local/bin/codex", "--version"}, gotCommand)
}

func TestDoctor_ProbeFailureMarksUnusable(t *testing.T) {
	// An installed but broken CLI is exactly what --probe is meant to catch.
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex},
		Probe:     true,
		LookPath:  fakeLookPath(map[string]string{"codex": "/usr/bin/codex"}),
		Env:       fakeEnv(nil),
		VersionProbe: func(_ context.Context, _ []string) (string, error) {
			return "", errors.New("broken install")
		},
	})
	require.NoError(t, err)
	assert.True(t, reports[0].Installed)
	assert.False(t, reports[0].Usable)
	assert.Equal(t, []string{IssueVersionProbeFailed}, reports[0].Issues)
	assert.Empty(t, reports[0].Version)
}

func TestDoctor_NoProbeWhenBinaryMissing(t *testing.T) {
	probed := false
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex},
		Probe:     true,
		LookPath:  fakeLookPath(nil),
		Env:       fakeEnv(nil),
		VersionProbe: func(_ context.Context, _ []string) (string, error) {
			probed = true
			return "", nil
		},
	})
	require.NoError(t, err)
	assert.False(t, probed, "must not probe a binary that was never found")
	assert.Equal(t, []string{IssueBinaryNotFound}, reports[0].Issues)
}

func TestDoctor_DefaultsToEveryProviderSortedByName(t *testing.T) {
	reports, err := Doctor(context.Background(), DoctorOptions{
		LookPath: fakeLookPath(nil),
		Env:      fakeEnv(nil),
	})
	require.NoError(t, err)
	require.Len(t, reports, len(SupportedProviderNames()))

	names := make([]string, 0, len(reports))
	for _, report := range reports {
		names = append(names, report.Provider)
	}
	assert.Equal(t, SupportedProviderNames(), names)
}

func TestDoctor_UnknownProviderIsAnError(t *testing.T) {
	// A typo must fail loudly rather than reporting nothing.
	_, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{"gpt-4"},
		LookPath:  fakeLookPath(nil),
		Env:       fakeEnv(nil),
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "gpt-4")
	assert.Contains(t, err.Error(), ProviderCodex, "error should list supported providers")
}

func TestDoctor_MixedSelectionReportsEachIndependently(t *testing.T) {
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex, ProviderGemini},
		LookPath:  fakeLookPath(map[string]string{"codex": "/usr/bin/codex"}),
		Env:       fakeEnv(nil),
	})
	require.NoError(t, err)
	byName := healthByProvider(reports)

	assert.True(t, byName[ProviderCodex].Usable)
	assert.False(t, byName[ProviderGemini].Usable)
}

func TestUnusable_FiltersToBlockingProviders(t *testing.T) {
	reports, err := Doctor(context.Background(), DoctorOptions{
		Providers: []string{ProviderCodex, ProviderGemini, ProviderOpenCode},
		LookPath: fakeLookPath(map[string]string{
			"codex":    "/usr/bin/codex",
			"opencode": "/usr/bin/opencode",
		}),
		Env: fakeEnv(nil),
	})
	require.NoError(t, err)

	unusable := Unusable(reports)
	require.Len(t, unusable, 1)
	assert.Equal(t, ProviderGemini, unusable[0].Provider)

	// Nothing unusable yields an empty result, so callers can exit 0 on len==0.
	assert.Empty(t, Unusable(nil))
}

func TestProviderUnavailableError_MessageCarriesInstallAndAuth(t *testing.T) {
	err := &ProviderUnavailableError{
		Provider:       ProviderCodex,
		Binary:         "codex",
		InstallCommand: "npm install -g @openai/codex",
		MissingAuthEnv: []string{"OPENAI_API_KEY"},
	}
	message := err.Error()
	assert.Contains(t, message, `"codex"`)
	assert.Contains(t, message, "npm install -g @openai/codex")
	assert.Contains(t, message, "OPENAI_API_KEY")

	// errors.Is against the sentinel, and errors.As to read the details.
	assert.True(t, errors.Is(err, ErrProviderUnavailable))
	var typed *ProviderUnavailableError
	require.True(t, errors.As(error(err), &typed))
	assert.Equal(t, ProviderCodex, typed.Provider)
}

func TestProviderUnavailableError_OmitsAuthClauseWhenNoneMissing(t *testing.T) {
	err := &ProviderUnavailableError{
		Provider:       ProviderOpenCode,
		Binary:         "opencode",
		InstallCommand: "curl -fsSL https://opencode.ai/install | bash",
	}
	assert.NotContains(t, err.Error(), "Configure one of")
}

func TestEnsureCLIAvailable_MissingBinaryYieldsTypedError(t *testing.T) {
	_, err := ensureCLIAvailable(ProviderCodex, "definitely-not-a-real-binary-xyz")
	require.Error(t, err)

	var unavailable *ProviderUnavailableError
	require.True(t, errors.As(err, &unavailable))
	assert.Equal(t, ProviderCodex, unavailable.Provider)
	assert.Equal(t, "npm install -g @openai/codex", unavailable.InstallCommand)
}

func TestEnsureCLIAvailable_EmptyBinaryYieldsTypedError(t *testing.T) {
	_, err := ensureCLIAvailable(ProviderCodex, "")
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderUnavailable))
}

func TestEnsureCLIAvailable_UnknownProviderFallsBackToDocsHint(t *testing.T) {
	// An unregistered provider must still produce actionable guidance rather
	// than an empty install command.
	_, err := ensureCLIAvailable("brand-new-provider", "definitely-not-a-real-binary-xyz")
	require.Error(t, err)

	var unavailable *ProviderUnavailableError
	require.True(t, errors.As(err, &unavailable))
	assert.Contains(t, unavailable.InstallCommand, "docs/harness-providers.md")
}

func TestEnsureCLIAvailable_ResolvesRealBinary(t *testing.T) {
	// Use the Go toolchain itself: it is guaranteed present wherever these
	// tests run, on any OS.
	goBin, lookErr := exec.LookPath("go")
	if lookErr != nil {
		t.Skip("go binary not on PATH")
	}
	resolved, err := ensureCLIAvailable(ProviderCodex, "go")
	require.NoError(t, err)
	assert.Equal(t, goBin, resolved)
}

func TestAnyEnvSet(t *testing.T) {
	env := fakeEnv(map[string]string{"SET": "value", "EMPTY": ""})
	assert.True(t, anyEnvSet([]string{"MISSING", "SET"}, env))
	assert.False(t, anyEnvSet([]string{"MISSING", "EMPTY"}, env))
	assert.False(t, anyEnvSet(nil, env))
}

func TestFirstLine(t *testing.T) {
	assert.Equal(t, "codex 1.2.3", firstLine("codex 1.2.3\nextra noise"))
	assert.Equal(t, "codex 1.2.3", firstLine("codex 1.2.3\r\nwindows"))
	assert.Equal(t, "single", firstLine("single"))
	assert.Equal(t, "", firstLine(""))
}

func TestExecVersionProbe_ReportsVersionFromRealProcess(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go binary not on PATH")
	}
	version, err := execVersionProbe(context.Background(), []string{"go", "version"})
	require.NoError(t, err)
	assert.Contains(t, version, "go")
}

func TestExecVersionProbe_EmptyCommandErrors(t *testing.T) {
	_, err := execVersionProbe(context.Background(), nil)
	require.Error(t, err)
}

func TestExecVersionProbe_FailingCommandErrors(t *testing.T) {
	_, err := execVersionProbe(context.Background(), []string{"definitely-not-a-real-binary-xyz", "--version"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version probe failed")
}
