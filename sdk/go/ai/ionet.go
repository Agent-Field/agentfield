package ai

import "strings"

// ionetModelPrefix marks a model as IO Intelligence (io.net)-routed. It is
// stripped before the request goes out; see stripIonetPrefix.
const ionetModelPrefix = "ionet/"

// defaultIonetModel is the model DefaultConfig falls back to when the
// configuration selects IO Intelligence and AI_MODEL is unset. io.net serves
// no gpt-4o (its ids are Hugging Face-style org/name, e.g.
// openai/gpt-oss-120b), so the global fallback would 404 there.
const defaultIonetModel = "meta-llama/Llama-3.3-70B-Instruct"

// stripIonetPrefix removes the routing-only "ionet/" prefix from a model
// name. IO Intelligence serves bare Hugging Face-style `org/name` ids, so
// the marker must not reach the wire.
func stripIonetPrefix(model string) string {
	if len(model) >= len(ionetModelPrefix) &&
		strings.EqualFold(model[:len(ionetModelPrefix)], ionetModelPrefix) {
		return model[len(ionetModelPrefix):]
	}
	return model
}
