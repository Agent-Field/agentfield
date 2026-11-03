package ai

import "strings"

// runcrateModelPrefix marks a model as Runcrate-routed. It is stripped before
// the request goes out; see stripRuncratePrefix.
const runcrateModelPrefix = "runcrate/"

// defaultRuncrateModel is the model DefaultConfig falls back to when the
// configuration selects Runcrate and AI_MODEL is unset. Runcrate serves
// open-weight models only (its ids are Hugging Face-style org/name, e.g.
// zai-org/GLM-5.3), so the global gpt-4o fallback would 404 there.
const defaultRuncrateModel = "deepseek-ai/DeepSeek-V4.1-Flash"

// stripRuncratePrefix removes the routing-only "runcrate/" prefix from a model
// name. Runcrate serves bare Hugging Face-style `org/name` ids, so the marker
// must not reach the wire.
func stripRuncratePrefix(model string) string {
	if len(model) >= len(runcrateModelPrefix) &&
		strings.EqualFold(model[:len(runcrateModelPrefix)], runcrateModelPrefix) {
		return model[len(runcrateModelPrefix):]
	}
	return model
}
