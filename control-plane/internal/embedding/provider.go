package embedding

// ProviderConfig is the minimal config the embedding factory needs. It mirrors
// the knowledge/embedding section of the control-plane config without importing
// the config package (avoids an import cycle).
type ProviderConfig struct {
	// Provider selects the implementation: "openai" or "fake". Empty defaults to
	// auto: openai when an API key is present, otherwise fake.
	Provider string
	// APIKey is the OpenAI API key (OPENAI_API_KEY).
	APIKey string
	// Model overrides the OpenAI model; empty uses DefaultModel.
	Model string
}

// NewFromConfig builds an Embedder from config. It NEVER returns nil: when no
// OpenAI key is configured it falls back to the deterministic FakeEmbedder so
// the knowledge store works locally with zero external dependencies.
//
// The second return value reports whether the real OpenAI provider was selected
// (false means the FakeEmbedder fallback is in use).
func NewFromConfig(cfg ProviderConfig) (Embedder, bool) {
	switch cfg.Provider {
	case "fake":
		return NewFakeEmbedder(), false
	case "openai":
		if cfg.APIKey == "" {
			return NewFakeEmbedder(), false
		}
		return NewOpenAIEmbedder(cfg.APIKey, WithModel(cfg.Model)), true
	default:
		// Auto: prefer OpenAI when a key is available.
		if cfg.APIKey != "" {
			return NewOpenAIEmbedder(cfg.APIKey, WithModel(cfg.Model)), true
		}
		return NewFakeEmbedder(), false
	}
}
