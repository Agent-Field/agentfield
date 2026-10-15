package harness

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// ProviderSpec describes how to find and authenticate a harness provider's CLI.
// It mirrors the Python SDK's PROVIDER_SPECS (harness/_availability.py) and the
// TypeScript SDK's PROVIDER_SPECS (harness/availability.ts) so all three SDKs
// report the same install commands and auth variables.
type ProviderSpec struct {
	// Binary is the executable name looked up on PATH.
	Binary string
	// VersionArgs are the arguments that make the CLI print its version.
	VersionArgs []string
	// InstallCommand is the exact command a user should run to install it.
	InstallCommand string
	// AuthEnvVars are the environment variables that can carry credentials.
	// Any one of them being set counts as "configured"; an empty list means the
	// provider needs no credentials of its own.
	AuthEnvVars []string
}

// providerSpecs is keyed by the provider names in provider.go.
//
// Divergence from the Python and TypeScript SDKs, on purpose: they treat
// claude-code as a language wrapper (claude_agent_sdk / @anthropic-ai/claude-agent-sdk)
// with no binary of its own, because that is how those SDKs invoke it. The Go
// provider shells out to the `claude` CLI (claudecode.go), so here it is a
// binary spec like every other provider. cursor is Go-only today and uses the
// Cursor CLI's `agent` binary (cursor.go).
var providerSpecs = map[string]ProviderSpec{
	ProviderAforge: {
		Binary:      "aforge",
		VersionArgs: []string{"version"},
		// `af` installs aforge beside itself in ~/.agentfield/bin; the curl
		// installer, the desktop app and the agent images all converge on this.
		InstallCommand: "af aforge ensure",
		AuthEnvVars:    []string{"OPENROUTER_API_KEY"},
	},
	ProviderClaudeCode: {
		Binary:         "claude",
		VersionArgs:    []string{"--version"},
		InstallCommand: "npm install -g @anthropic-ai/claude-code",
		AuthEnvVars:    []string{"ANTHROPIC_API_KEY"},
	},
	ProviderCodex: {
		Binary:         "codex",
		VersionArgs:    []string{"--version"},
		InstallCommand: "npm install -g @openai/codex",
		AuthEnvVars:    []string{"OPENAI_API_KEY"},
	},
	ProviderGemini: {
		Binary:         "gemini",
		VersionArgs:    []string{"--version"},
		InstallCommand: "npm install -g @google/gemini-cli",
		AuthEnvVars:    []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"},
	},
	ProviderOpenCode: {
		Binary:         "opencode",
		VersionArgs:    []string{"--version"},
		InstallCommand: "curl -fsSL https://opencode.ai/install | bash",
		AuthEnvVars:    nil,
	},
	ProviderPi: {
		Binary:         "pi",
		VersionArgs:    []string{"--version"},
		InstallCommand: "npm install -g --ignore-scripts @earendil-works/pi-coding-agent",
		AuthEnvVars: []string{
			"OPENROUTER_API_KEY",
			"ANTHROPIC_API_KEY",
			"OPENAI_API_KEY",
			"GEMINI_API_KEY",
			"GOOGLE_API_KEY",
		},
	},
	ProviderOMP: {
		Binary:         "omp",
		VersionArgs:    []string{"--version"},
		InstallCommand: "curl -fsSL https://omp.sh/install | sh",
		AuthEnvVars: []string{
			"OPENROUTER_API_KEY",
			"ANTHROPIC_API_KEY",
			"OPENAI_API_KEY",
			"GEMINI_API_KEY",
			"GOOGLE_API_KEY",
		},
	},
	ProviderCursor: {
		Binary:         "agent",
		VersionArgs:    []string{"--version"},
		InstallCommand: "install the Cursor CLI: https://docs.cursor.com/en/cli/overview",
		AuthEnvVars:    []string{"CURSOR_API_KEY"},
	},
}

// fallbackSpec keeps an unknown provider from producing a bare "not found"
// with no guidance, mirroring the Python SDK's _FALLBACK_SPEC.
var fallbackSpec = ProviderSpec{
	InstallCommand: "see docs/harness-providers.md for installation instructions",
}

// SupportedProviderNames returns every provider name that has an availability
// spec, sorted, so callers can enumerate what Doctor understands.
func SupportedProviderNames() []string {
	names := make([]string, 0, len(providerSpecs))
	for name := range providerSpecs {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// SpecForProvider returns the availability spec for a provider and whether one
// is registered.
func SpecForProvider(provider string) (ProviderSpec, bool) {
	spec, ok := providerSpecs[provider]
	return spec, ok
}

// ProviderUnavailableError reports that a harness provider cannot run because
// its CLI is missing, and carries the exact command that installs it. It is
// returned before the subprocess is spawned so callers fail fast with an
// actionable message instead of a late generic exec error.
//
// Mirrors the Python SDK's HarnessProviderUnavailable exception and the
// TypeScript SDK's HarnessProviderUnavailable error.
type ProviderUnavailableError struct {
	Provider       string
	Binary         string
	InstallCommand string
	// MissingAuthEnv lists the provider's auth variables when none of them are
	// set. It is empty when credentials are present or the provider needs none.
	MissingAuthEnv []string
}

func (e *ProviderUnavailableError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "harness provider %q is unavailable: binary %q was not found on PATH",
		e.Provider, e.Binary)
	if e.InstallCommand != "" {
		fmt.Fprintf(&b, ". Install it with: %s", e.InstallCommand)
	}
	if len(e.MissingAuthEnv) > 0 {
		fmt.Fprintf(&b, ". Configure one of: %s", strings.Join(e.MissingAuthEnv, ", "))
	}
	return b.String()
}

// ErrProviderUnavailable allows errors.Is checks against the sentinel without
// matching on message text.
var ErrProviderUnavailable = errors.New("harness provider unavailable")

func (e *ProviderUnavailableError) Is(target error) bool {
	return target == ErrProviderUnavailable
}

// newProviderUnavailable builds the typed error for a provider whose binary
// could not be resolved, filling in the install command and, when no
// credentials are present, the auth variables worth setting.
func newProviderUnavailable(provider, binary string) *ProviderUnavailableError {
	spec, ok := providerSpecs[provider]
	if !ok {
		spec = fallbackSpec
	}
	err := &ProviderUnavailableError{
		Provider:       provider,
		Binary:         binary,
		InstallCommand: spec.InstallCommand,
	}
	if len(spec.AuthEnvVars) > 0 && !anyEnvSet(spec.AuthEnvVars, os.Getenv) {
		err.MissingAuthEnv = append([]string(nil), spec.AuthEnvVars...)
	}
	return err
}

// ensureCLIAvailable resolves binary on PATH, returning a
// ProviderUnavailableError that names the exact install command when it is
// missing. Providers can call this before spawning to fail fast, and Doctor
// uses the same resolution so both report availability identically.
//
// A caller that injects its own runner (tests, or a custom transport) should
// not call this: resolution is intentionally PATH based, so an unresolvable
// path is only meaningful when a real subprocess is about to be spawned.
func ensureCLIAvailable(provider, binary string) (string, error) {
	if binary == "" {
		return "", newProviderUnavailable(provider, binary)
	}
	resolved, err := exec.LookPath(binary)
	if err != nil {
		return "", newProviderUnavailable(provider, binary)
	}
	return resolved, nil
}

// anyEnvSet reports whether any of names has a non-empty value in lookup.
func anyEnvSet(names []string, lookup func(string) string) bool {
	for _, name := range names {
		if lookup(name) != "" {
			return true
		}
	}
	return false
}
