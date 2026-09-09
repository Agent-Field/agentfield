package harness

import (
	"fmt"
	"os"
	"strings"
)

// ResolveProviderName applies harness provider precedence: an explicit name
// wins, then AGENTFIELD_HARNESS_PROVIDER, then DefaultProvider ("aforge").
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

// BuildProvider creates a Provider instance for the given provider name.
// Supported providers: "aforge", "claude-code", "codex", "gemini",
// "opencode", "pi", "omp".
func BuildProvider(name string, binPath string) (Provider, error) {
	name = ResolveProviderName(name)
	switch name {
	case ProviderAforge:
		return NewAforgeProvider(binPath), nil
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
	default:
		return nil, fmt.Errorf(
			"unknown harness provider: %q (supported: %s, %s, %s, %s, %s, %s, %s)",
			name, ProviderAforge, ProviderClaudeCode, ProviderCodex, ProviderGemini, ProviderOpenCode, ProviderPi, ProviderOMP,
		)
	}
}
