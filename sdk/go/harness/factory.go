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

func resolveExecutionProvider(name, binPath string) string {
	if strings.TrimSpace(name) == "" && strings.TrimSpace(os.Getenv(ProviderEnvVar)) == "" &&
		(binPath != "" || strings.TrimSpace(os.Getenv("AFORGE_BIN")) != "") {
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
func BuildProvider(name string, binPath string) (Provider, error) {
	name = resolveExecutionProvider(name, binPath)
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
