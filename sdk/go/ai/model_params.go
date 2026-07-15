package ai

import (
	"encoding/json"
	"net/url"
	"strings"
)

// needsMaxCompletionTokens returns true if the model requires
// max_completion_tokens instead of max_tokens.
//
// Every OpenAI chat model from gpt-4o onwards (gpt-4o, gpt-4.1, gpt-4.5,
// gpt-5, ...) and the whole o-series of reasoning models (o1, o3, o4, and
// any future oN) uses max_completion_tokens. Rather than allowlisting model
// names — which goes stale every time OpenAI ships a new family — we
// denylist the known-legacy families that still take max_tokens (gpt-3.x
// and the original gpt-4 line). Those lines are closed and can never grow.
//
// Reference: https://platform.openai.com/docs/api-reference/chat/create
func needsMaxCompletionTokens(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))

	// Strip provider prefix if present (e.g. "openai/gpt-4o" → "gpt-4o")
	if idx := strings.LastIndex(m, "/"); idx >= 0 {
		m = m[idx+1:]
	}

	// Known-legacy families that still use max_tokens: gpt-3.5-*, gpt-4,
	// gpt-4-turbo, gpt-4-32k, gpt-4-0613, etc.
	if m == "gpt-4" || strings.HasPrefix(m, "gpt-4-") || strings.HasPrefix(m, "gpt-3") {
		return false
	}

	// Any other gpt-* model (gpt-4o*, gpt-4.1, gpt-5, ...) is newer than the
	// legacy set and uses max_completion_tokens.
	if strings.HasPrefix(m, "gpt-") {
		return true
	}

	// o-series reasoning models: "o" followed by a digit (o1, o3, o4, o5, ...).
	// Requiring the digit keeps names like "omni-*" or "openchat" out.
	if len(m) >= 2 && m[0] == 'o' && m[1] >= '0' && m[1] <= '9' {
		return true
	}

	return false
}

// vouchedRewriteDomains lists the hosts known to accept (and, for newer
// models, require) max_completion_tokens. Matching is by exact host or any
// subdomain (e.g. api.openai.com, <resource>.openai.azure.com).
var vouchedRewriteDomains = []string{
	"openai.com",       // api.openai.com
	"openai.azure.com", // <resource>.openai.azure.com
	"openrouter.ai",
}

// isVouchedRewriteEndpoint reports whether baseURL points at an endpoint we
// can vouch for accepting max_completion_tokens: OpenAI, Azure OpenAI, or
// OpenRouter. For any other host — including self-hosted OpenAI-compatible
// servers (Ollama, LM Studio, llama.cpp, older vLLM) — we keep the legacy
// max_tokens field: several of those servers silently drop unknown fields,
// so rewriting would strip the output cap with no error (the silent failure
// #441 is about). If such a host actually proxies a newer OpenAI model, it
// rejects max_tokens loudly, which is the safer failure mode.
func isVouchedRewriteEndpoint(baseURL string) bool {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return false
	}

	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}

	for _, domain := range vouchedRewriteDomains {
		if host == domain || strings.HasSuffix(host, "."+domain) {
			return true
		}
	}
	return false
}

// marshalRequest serializes the request, applying provider-specific parameter
// rewrites. For vouched OpenAI endpoints with newer models, max_tokens is
// rewritten to max_completion_tokens.
func (c *Client) marshalRequest(req *Request) ([]byte, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}

	// If the model needs max_completion_tokens and we have a max_tokens value,
	// serialize with the rewritten field name — but only for endpoints known
	// to understand it.
	if req.MaxTokens != nil && needsMaxCompletionTokens(model) && isVouchedRewriteEndpoint(c.config.BaseURL) {
		return marshalWithMaxCompletionTokens(req)
	}

	return json.Marshal(req)
}

// marshalWithMaxCompletionTokens serializes the request with max_completion_tokens
// instead of max_tokens. We use a shadow struct to avoid modifying the original.
func marshalWithMaxCompletionTokens(req *Request) ([]byte, error) {
	type requestAlias Request

	wire := struct {
		*requestAlias
		MaxTokens           *int `json:"max_tokens,omitempty"`
		MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`
	}{
		requestAlias:        (*requestAlias)(req),
		MaxTokens:           nil, // suppress max_tokens
		MaxCompletionTokens: req.MaxTokens,
	}

	return json.Marshal(wire)
}
