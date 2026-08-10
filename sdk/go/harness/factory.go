package harness

import "fmt"

// BuildProvider creates a Provider instance for the given provider name.
// Supported providers: "aforge", "claude-code", "codex", "gemini", "opencode".
func BuildProvider(name string, binPath string) (Provider, error) {
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
	default:
		return nil, fmt.Errorf(
			"unknown harness provider: %q (supported: %s, %s, %s, %s, %s)",
			name, ProviderAforge, ProviderClaudeCode, ProviderCodex, ProviderGemini, ProviderOpenCode,
		)
	}
}
