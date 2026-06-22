package config

import (
	"testing"
)

// TestApplyEnvOverrides_KnowledgeEmbedding covers the knowledge/embedding env
// override block: OPENAI_API_KEY only fills an empty key, the explicit
// AGENTFIELD_KNOWLEDGE_OPENAI_API_KEY always wins, and provider/model/enabled
// are applied and trimmed.
func TestApplyEnvOverrides_KnowledgeEmbedding(t *testing.T) {
	cfg := &Config{}

	t.Setenv("OPENAI_API_KEY", "sk-standard")
	t.Setenv("AGENTFIELD_KNOWLEDGE_OPENAI_API_KEY", "  sk-explicit  ")
	t.Setenv("AGENTFIELD_KNOWLEDGE_PROVIDER", "  openai  ")
	t.Setenv("AGENTFIELD_KNOWLEDGE_OPENAI_MODEL", "  text-embedding-3-large  ")
	t.Setenv("AGENTFIELD_KNOWLEDGE_ENABLED", "false")

	ApplyEnvOverrides(cfg)

	k := cfg.Features.Knowledge
	// Explicit knowledge key must win over the standard OPENAI_API_KEY and be trimmed.
	if k.OpenAI.APIKey != "sk-explicit" {
		t.Errorf("APIKey = %q, want sk-explicit (explicit override wins, trimmed)", k.OpenAI.APIKey)
	}
	if k.Provider != "openai" {
		t.Errorf("Provider = %q, want openai (trimmed)", k.Provider)
	}
	if k.OpenAI.Model != "text-embedding-3-large" {
		t.Errorf("Model = %q, want text-embedding-3-large (trimmed)", k.OpenAI.Model)
	}
	if k.Enabled == nil || *k.Enabled {
		t.Errorf("Enabled = %v, want false", k.Enabled)
	}
}

// TestApplyEnvOverrides_KnowledgeStandardOpenAIKeyFallback covers the branch
// where only the standard OPENAI_API_KEY is set and the knowledge key is empty:
// it must be adopted (and trimmed).
func TestApplyEnvOverrides_KnowledgeStandardOpenAIKeyFallback(t *testing.T) {
	cfg := &Config{}

	t.Setenv("OPENAI_API_KEY", "  sk-from-standard  ")

	ApplyEnvOverrides(cfg)

	if cfg.Features.Knowledge.OpenAI.APIKey != "sk-from-standard" {
		t.Errorf("APIKey = %q, want sk-from-standard (adopted from OPENAI_API_KEY, trimmed)",
			cfg.Features.Knowledge.OpenAI.APIKey)
	}
}

// TestApplyEnvOverrides_KnowledgeStandardKeyDoesNotOverrideExisting covers the
// guard: when a knowledge key is already configured, the standard OPENAI_API_KEY
// must NOT overwrite it.
func TestApplyEnvOverrides_KnowledgeStandardKeyDoesNotOverrideExisting(t *testing.T) {
	cfg := &Config{}
	cfg.Features.Knowledge.OpenAI.APIKey = "sk-preconfigured"

	t.Setenv("OPENAI_API_KEY", "sk-standard")

	ApplyEnvOverrides(cfg)

	if cfg.Features.Knowledge.OpenAI.APIKey != "sk-preconfigured" {
		t.Errorf("APIKey = %q, want sk-preconfigured (existing key must not be overridden by OPENAI_API_KEY)",
			cfg.Features.Knowledge.OpenAI.APIKey)
	}
}

// TestKnowledgeConfig_IsEnabledDefault covers the IsEnabled default (nil => true)
// and explicit false.
func TestKnowledgeConfig_IsEnabledDefault(t *testing.T) {
	var nilCfg KnowledgeConfig
	if !nilCfg.IsEnabled() {
		t.Error("IsEnabled() with nil Enabled must default to true")
	}

	tru := true
	if !(KnowledgeConfig{Enabled: &tru}).IsEnabled() {
		t.Error("IsEnabled() with *true must be true")
	}

	fls := false
	if (KnowledgeConfig{Enabled: &fls}).IsEnabled() {
		t.Error("IsEnabled() with *false must be false")
	}
}
