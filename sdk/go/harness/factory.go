package harness

import "fmt"

// BuildProvider creates a Provider instance for the given provider name.
// An empty name selects DefaultProvider (OMP).
// Supported providers: "claude-code", "codex", "gemini", "opencode", "pi", "omp".
func BuildProvider(name string, binPath string) (Provider, error) {
	if name == "" {
		name = DefaultProvider
	}
	switch name {
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
			"unknown harness provider: %q (supported: %s, %s, %s, %s, %s, %s)",
			name, ProviderClaudeCode, ProviderCodex, ProviderGemini, ProviderOpenCode, ProviderPi, ProviderOMP,
		)
	}
}
