package harness

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"
)

// versionProbeTimeout bounds a single provider's version probe so Doctor stays
// responsive when a CLI hangs.
const versionProbeTimeout = 2 * time.Second

// Issue codes reported in ProviderHealth.Issues. They match the Python SDK's
// strings so tooling can key off the same values across SDKs.
const (
	IssueBinaryNotFound     = "binary_not_found"
	IssueVersionProbeFailed = "version_probe_failed"
)

// Auth states reported in ProviderHealth.Auth.
const (
	AuthConfigured = "configured"
	AuthUnknown    = "unknown"
)

// ProviderHealth is the per-provider result of a Doctor run. Field names mirror
// the Python SDK's ProviderHealth dataclass and the TypeScript SDK's
// ProviderHealth interface so the three SDKs report the same shape.
type ProviderHealth struct {
	Provider string `json:"provider"`
	// Binary is the resolved absolute path, empty when the binary was not found.
	Binary    string `json:"binary,omitempty"`
	Installed bool   `json:"installed"`
	// Version is the first line of the CLI's version output, empty when not
	// probed or when the probe failed.
	Version string `json:"version,omitempty"`
	// Auth is AuthConfigured when any of the provider's auth variables is set,
	// otherwise AuthUnknown. Providers that need no credentials report
	// AuthConfigured.
	Auth string `json:"auth"`
	// Usable is true when the provider has no issues, i.e. the binary resolved
	// and any requested version probe succeeded.
	Usable         bool     `json:"usable"`
	InstallCommand string   `json:"install_command"`
	AuthEnvVars    []string `json:"auth_env_vars,omitempty"`
	Issues         []string `json:"issues,omitempty"`
}

// VersionProbe runs a provider's version command and returns its output. The
// command is the resolved binary path followed by the spec's VersionArgs.
type VersionProbe func(ctx context.Context, command []string) (string, error)

// DoctorOptions configures a Doctor run.
type DoctorOptions struct {
	// Providers limits the check to these provider names. Empty checks every
	// provider returned by SupportedProviderNames.
	Providers []string

	// Probe enables running each provider's version command. Static PATH and
	// environment checks are the default because they are fast and offline
	// safe; probing additionally catches an installed-but-broken CLI.
	Probe bool

	// Env overrides environment lookup for auth detection. Nil uses os.Getenv.
	Env func(string) string

	// VersionProbe overrides how a version is obtained when Probe is set. Nil
	// uses a bounded exec of the provider's version command. Tests inject this
	// so they never depend on a real CLI being installed.
	VersionProbe VersionProbe

	// LookPath overrides binary resolution. Nil uses exec.LookPath. Tests
	// inject this to simulate installed and missing binaries.
	LookPath func(string) (string, error)
}

// Doctor reports whether each requested harness provider is usable: its CLI on
// PATH, optionally its version, and whether credentials are configured. It is
// the preflight to run in a Dockerfile, a CI step or a container entrypoint so
// a missing harness fails the build instead of a real (paid) run.
//
// Results are ordered by provider name. An unknown provider name is an error,
// so a typo fails loudly rather than silently reporting nothing.
func Doctor(ctx context.Context, opts DoctorOptions) ([]ProviderHealth, error) {
	selected := opts.Providers
	if len(selected) == 0 {
		selected = SupportedProviderNames()
	} else {
		selected = append([]string(nil), selected...)
		sort.Strings(selected)
		var unknown []string
		for _, name := range selected {
			if _, ok := providerSpecs[name]; !ok {
				unknown = append(unknown, name)
			}
		}
		if len(unknown) > 0 {
			return nil, fmt.Errorf(
				"unknown harness provider: %q (supported: %s)",
				unknown[0], strings.Join(SupportedProviderNames(), ", "),
			)
		}
	}

	getenv := opts.Env
	if getenv == nil {
		getenv = os.Getenv
	}
	lookPath := opts.LookPath
	if lookPath == nil {
		lookPath = exec.LookPath
	}
	probe := opts.VersionProbe
	if probe == nil {
		probe = execVersionProbe
	}

	reports := make([]ProviderHealth, 0, len(selected))
	for _, provider := range selected {
		spec := providerSpecs[provider]

		health := ProviderHealth{
			Provider:       provider,
			Auth:           AuthUnknown,
			InstallCommand: spec.InstallCommand,
			AuthEnvVars:    append([]string(nil), spec.AuthEnvVars...),
		}

		if resolved, err := lookPath(spec.Binary); err == nil {
			health.Binary = resolved
			health.Installed = true
		} else {
			health.Issues = append(health.Issues, IssueBinaryNotFound)
		}

		if health.Installed && opts.Probe {
			command := append([]string{health.Binary}, spec.VersionArgs...)
			version, err := probe(ctx, command)
			if err != nil {
				health.Issues = append(health.Issues, IssueVersionProbeFailed)
			} else {
				health.Version = version
			}
		}

		// A provider that needs no credentials is never blocked on auth.
		if len(spec.AuthEnvVars) == 0 || anyEnvSet(spec.AuthEnvVars, getenv) {
			health.Auth = AuthConfigured
		}

		health.Usable = len(health.Issues) == 0
		reports = append(reports, health)
	}

	return reports, nil
}

// Unusable filters a Doctor result down to the providers that cannot run, so
// callers can exit non-zero on exactly those.
func Unusable(reports []ProviderHealth) []ProviderHealth {
	var out []ProviderHealth
	for _, report := range reports {
		if !report.Usable {
			out = append(out, report)
		}
	}
	return out
}

// execVersionProbe runs the provider's version command with a bounded timeout
// and returns its first output line. stderr is used when stdout is empty,
// because several CLIs print their version there.
func execVersionProbe(ctx context.Context, command []string) (string, error) {
	if len(command) == 0 {
		return "", fmt.Errorf("empty version command")
	}

	probeCtx, cancel := context.WithTimeout(ctx, versionProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(probeCtx, command[0], command[1:]...)
	output, err := cmd.Output()
	text := strings.TrimSpace(string(output))
	if text == "" {
		if exitErr, ok := err.(*exec.ExitError); ok {
			text = strings.TrimSpace(string(exitErr.Stderr))
		}
	}
	if err != nil {
		if probeCtx.Err() == context.DeadlineExceeded {
			return "", fmt.Errorf("version probe timed out after %s", versionProbeTimeout)
		}
		if text != "" {
			return "", fmt.Errorf("version probe failed: %s", firstLine(text))
		}
		return "", fmt.Errorf("version probe failed: %w", err)
	}
	if text == "" {
		return "unknown", nil
	}
	return firstLine(text), nil
}

func firstLine(s string) string {
	if idx := strings.IndexAny(s, "\r\n"); idx >= 0 {
		return s[:idx]
	}
	return s
}
