package harness

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSplitModelVariant pins the Python split_model_variant matrix from
// sdk/python/tests/test_harness_cli.py: split on the FIRST "#", trim
// whitespace on both parts, empty parts become "".
func TestSplitModelVariant(t *testing.T) {
	cases := []struct {
		name        string
		model       string
		wantBase    string
		wantVariant string
	}{
		{"suffix parsed", "openrouter/z-ai/glm-5.2#high", "openrouter/z-ai/glm-5.2", "high"},
		{"bare model passes through", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash", ""},
		{"empty model", "", "", ""},
		{"whitespace-only model", "  ", "", ""},
		{"trailing separator drops empty variant", "model#", "model", ""},
		{"leading separator drops empty base", "#high", "", "high"},
		{"whitespace trimmed on both parts", " openai/gpt-5 # low ", "openai/gpt-5", "low"},
		{"split on first separator only", "model#high#extra", "model", "high#extra"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base, variant := SplitModelVariant(tc.model)
			assert.Equal(t, tc.wantBase, base)
			assert.Equal(t, tc.wantVariant, variant)
		})
	}
}

// TestOptionsResolveModelAndVariant pins the Python resolve_model_and_variant
// semantics: an explicit Options.Variant wins over a "#variant" suffix, and
// the returned model never carries the suffix.
func TestOptionsResolveModelAndVariant(t *testing.T) {
	t.Run("suffix used when no explicit variant", func(t *testing.T) {
		model, variant := Options{Model: "openai/gpt-5#minimal"}.resolveModelAndVariant()
		assert.Equal(t, "openai/gpt-5", model)
		assert.Equal(t, "minimal", variant)
	})

	t.Run("explicit variant wins over suffix", func(t *testing.T) {
		model, variant := Options{Model: "openai/gpt-5#low", Variant: "max"}.resolveModelAndVariant()
		assert.Equal(t, "openai/gpt-5", model)
		assert.Equal(t, "max", variant)
	})

	t.Run("explicit variant with bare model", func(t *testing.T) {
		model, variant := Options{Model: "gpt-5.3-codex", Variant: "high"}.resolveModelAndVariant()
		assert.Equal(t, "gpt-5.3-codex", model)
		assert.Equal(t, "high", variant)
	})

	t.Run("whitespace-only explicit variant falls back to suffix", func(t *testing.T) {
		model, variant := Options{Model: "openai/gpt-5#low", Variant: "  "}.resolveModelAndVariant()
		assert.Equal(t, "openai/gpt-5", model)
		assert.Equal(t, "low", variant)
	})

	t.Run("no model and no variant", func(t *testing.T) {
		model, variant := Options{}.resolveModelAndVariant()
		assert.Equal(t, "", model)
		assert.Equal(t, "", variant)
	})
}

// TestMergeOptions_Variant verifies Variant flows through the default/override
// merge like every other option field.
func TestMergeOptions_Variant(t *testing.T) {
	r := NewRunner(Options{Provider: ProviderOpenCode, Variant: "high"})
	assert.Equal(t, "high", r.mergeOptions(Options{}).Variant, "default carried through")
	assert.Equal(t, "max", r.mergeOptions(Options{Variant: "max"}).Variant, "override wins")
}

// TestGeminiProvider_StripsModelVariantSuffix mirrors the Python
// test_gemini_strips_model_variant_suffix: gemini has no effort control, so
// the CLI receives the bare base model and the variant is dropped.
func TestGeminiProvider_StripsModelVariantSuffix(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\necho \"args: $@\"\n")

	p := NewGeminiProvider(script)
	raw, err := p.Execute(context.Background(), "hello", Options{Model: "gemini-2.5-pro#high"})
	require.NoError(t, err)
	require.False(t, raw.IsError)

	assert.Contains(t, raw.Result, "-m gemini-2.5-pro")
	assert.NotContains(t, raw.Result, "#high")
	assert.NotContains(t, raw.Result, "--variant")
}
