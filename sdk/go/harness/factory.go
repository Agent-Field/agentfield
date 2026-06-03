package harness

import "fmt"

// BuildProvider creates a Provider instance for the given provider name.
// Supported providers: "claude-code", "codex", "gemini", "opencode".
func BuildProvider(name string, binPath string) (Provider, error) {
	switch name {
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
			"unknown harness provider: %q (supported: %s, %s, %s, %s)",
			name, ProviderClaudeCode, ProviderCodex, ProviderGemini, ProviderOpenCode,
		)
	}
}
