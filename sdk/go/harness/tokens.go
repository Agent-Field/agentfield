package harness

// tokenUsage aggregates provider-reported token counts for one or more
// executions. All-zero means "the provider reported nothing", never data.
type tokenUsage struct {
	inputTokens         int
	outputTokens        int
	cacheReadTokens     int
	cacheCreationTokens int
}

// applyTo copies the aggregated counts onto a Result. TotalTokens is
// input+output — cache tokens are accounted separately, mirroring the Python
// runner's _token_result_kwargs.
func (t tokenUsage) applyTo(res *Result) {
	res.InputTokens = t.inputTokens
	res.OutputTokens = t.outputTokens
	res.CacheReadTokens = t.cacheReadTokens
	res.CacheCreationTokens = t.cacheCreationTokens
	res.TotalTokens = t.inputTokens + t.outputTokens
}

// usageFromMap normalizes a provider "usage" object into token counts. It
// tolerates the common shapes: OpenAI/Codex (input_tokens / output_tokens /
// cached_input_tokens, or prompt_tokens / completion_tokens) and
// Anthropic-native (cache_read_input_tokens / cache_creation_input_tokens).
// Mirrors the Python SDK's _cli.extract_token_usage field handling.
func usageFromMap(usage map[string]any) tokenUsage {
	return tokenUsage{
		inputTokens:         intField(usage, "input_tokens", "prompt_tokens"),
		outputTokens:        intField(usage, "output_tokens", "completion_tokens"),
		cacheReadTokens:     intField(usage, "cache_read_input_tokens", "cached_input_tokens"),
		cacheCreationTokens: intField(usage, "cache_creation_input_tokens"),
	}
}

// extractTokenUsage scans JSONL events for a "usage" object — top-level or
// nested under "item"/"turn" payloads (Codex nests it on some versions) — and
// returns the normalized counts. The LAST usage object seen wins, matching the
// Python SDK's extract_token_usage: Codex emits a cumulative usage on
// turn.completed, so the final one is the authoritative total. Returns
// all-zero when no usage is present.
func extractTokenUsage(events []map[string]any) tokenUsage {
	var result tokenUsage
	for _, event := range events {
		usage, ok := event["usage"].(map[string]any)
		if !ok {
			nested, nestedOK := event["item"].(map[string]any)
			if !nestedOK {
				nested, nestedOK = event["turn"].(map[string]any)
			}
			if !nestedOK {
				continue
			}
			usage, ok = nested["usage"].(map[string]any)
			if !ok {
				continue
			}
		}
		result = usageFromMap(usage)
	}
	return result
}

// intField returns the integer value of the first key present in m,
// tolerating JSON's float64 decoding. A present-but-non-numeric value yields
// 0 without falling through to later names (Python _int parity).
func intField(m map[string]any, names ...string) int {
	for _, name := range names {
		v, ok := m[name]
		if !ok || v == nil {
			continue
		}
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		case int64:
			return int(n)
		default:
			return 0
		}
	}
	return 0
}
