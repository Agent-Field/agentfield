package harness

import "strings"

// ModelVariantSep separates the base model id from a reasoning-effort variant
// in the "provider/model#variant" model-string syntax. "#" is safe because
// provider model ids never contain it (":" is taken by OpenRouter suffixes
// like ":free", "@" by Vertex-style ids).
const ModelVariantSep = "#"

// SplitModelVariant splits a "provider/model#variant" string into its base
// model and variant parts.
//
// The "#variant" suffix carries a provider-specific reasoning-effort variant
// (e.g. "high", "minimal") through config surfaces that only hold a model
// string. The split happens on the FIRST separator and both parts are
// whitespace-trimmed. A bare model string passes through as (model, "");
// an empty or blank input yields ("", ""). Mirrors the Python SDK's
// split_model_variant (harness/_cli.py).
func SplitModelVariant(model string) (base, variant string) {
	if strings.TrimSpace(model) == "" {
		return "", ""
	}
	base, variant, _ = strings.Cut(model, ModelVariantSep)
	return strings.TrimSpace(base), strings.TrimSpace(variant)
}

// resolveModelAndVariant resolves the (model, variant) pair for a harness
// invocation. An explicit Options.Variant wins over a "#variant" suffix on
// Options.Model; the returned model never carries the suffix. Mirrors the
// Python SDK's resolve_model_and_variant (harness/_cli.py).
func (o Options) resolveModelAndVariant() (model, variant string) {
	model, variant = SplitModelVariant(o.Model)
	if explicit := strings.TrimSpace(o.Variant); explicit != "" {
		variant = explicit
	}
	return model, variant
}
