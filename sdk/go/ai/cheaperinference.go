package ai

import "strings"

// cheaperInferenceModelPrefix marks a model as routed through Cheaper
// Inference. It is stripped before the request goes out; see
// stripCheaperInferencePrefix.
const cheaperInferenceModelPrefix = "cheaperinference/"

// stripCheaperInferencePrefix removes the routing-only "cheaperinference/"
// prefix from a model name. The gateway serves bare model ids, so the prefix
// must not reach the wire.
func stripCheaperInferencePrefix(model string) string {
	if len(model) >= len(cheaperInferenceModelPrefix) &&
		strings.EqualFold(model[:len(cheaperInferenceModelPrefix)], cheaperInferenceModelPrefix) {
		return model[len(cheaperInferenceModelPrefix):]
	}
	return model
}
