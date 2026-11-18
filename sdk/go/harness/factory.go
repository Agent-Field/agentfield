package harness

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
)

// ResolveProviderName applies harness provider precedence: an explicit name
// wins, then AGENTFIELD_HARNESS_PROVIDER, then DefaultProvider ("codeaf").
// Blank / whitespace-only values are treated as unset.
func ResolveProviderName(name string) string {
	if trimmed := strings.TrimSpace(name); trimmed != "" {
		return trimmed
	}
	if envName := strings.TrimSpace(os.Getenv(ProviderEnvVar)); envName != "" {
		return envName
	}
	return DefaultProvider
}

var legacyPinOnce sync.Once

// resolveExecutionProvider applies harness provider precedence for actual
// execution: the legacy pin (AFORGE_BIN, the compatibility signal for aforge-era
// deployments) wins only for an implicit default, then an explicit name, then
// AGENTFIELD_HARNESS_PROVIDER, then DefaultProvider ("codeaf"). A generic
// binary override (BinPath) is not a legacy signal: it is honored by whatever
// provider is selected, so an implicit default with a custom binary runs CodeAF.
func resolveExecutionProvider(name string) string {
	if strings.TrimSpace(name) == "" && strings.TrimSpace(os.Getenv(ProviderEnvVar)) == "" &&
		strings.TrimSpace(os.Getenv("AFORGE_BIN")) != "" {
		legacyPinOnce.Do(func() {
			log.Print(`INFO Legacy aforge pin in effect; unset it or select provider "codeaf" to use CodeAF`)
		})
		return ProviderAforge
	}
	return ResolveProviderName(name)
}

// BuildProvider creates a Provider instance for the given provider name.
// Supported providers: "codeaf", "aforge", "claude-code", "codex", "gemini",
// "opencode", "pi", "omp", "cursor".
//
// binPath overrides the provider binary and never forces a provider. An
// explicit "codeaf" (or an implicit default) with a custom binary runs
// CodeAF with live watching/steering. The legacy aforge adapter remains
// explicitly selectable: "aforge" plus a binary override or AFORGE_BIN.
func BuildProvider(name string, binPath string) (Provider, error) {
	name = resolveExecutionProvider(name)
	switch name {
	case ProviderCodeAF:
		return NewCodeAFProvider(binPath), nil
	case ProviderAforge:
		if binPath != "" || strings.TrimSpace(os.Getenv("AFORGE_BIN")) != "" {
			return NewAforgeProvider(binPath), nil
		}
		provider := NewCodeAFProvider("")
		provider.alias = true
		return provider, nil
	case ProviderClaudeCode:
		return NewClaudeCodeProvider(binPath), nil
	case ProviderCodex:
		return NewCodexProvider(binPath), nil
	case ProviderGemini:
		return NewGeminiProvider(binPath), nil
	case ProviderOpenCode:
		return NewOpenCodeProvider(binPath, ""), nil
	case ProviderPi:
		return NewPiProvider(binPath), nil
	case ProviderOMP:
		return NewOMPProvider(binPath), nil
	case ProviderCursor:
		return NewCursorProvider(binPath), nil
	default:
		return nil, fmt.Errorf(
			"unknown harness provider: %q (supported: %s, %s, %s, %s, %s, %s, %s, %s, %s)",
			name, ProviderCodeAF, ProviderAforge, ProviderClaudeCode, ProviderCodex, ProviderGemini, ProviderOpenCode, ProviderPi, ProviderOMP, ProviderCursor,
		)
	}
}
