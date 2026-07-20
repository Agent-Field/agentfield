package harness

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExtractTokenUsage maps to the contract: token counts are pulled from
// JSONL events wherever the provider reports them — top-level or nested usage
// objects, OpenAI/Codex or Anthropic field names — with the last usage object
// winning (Codex emits a cumulative usage on turn.completed).
func TestExtractTokenUsage(t *testing.T) {
	tests := []struct {
		name   string
		events []map[string]any
		want   tokenUsage
	}{
		{
			name:   "no usage anywhere yields all-zero",
			events: []map[string]any{{"type": "text", "text": "hi"}},
			want:   tokenUsage{},
		},
		{
			name: "openai-style top-level usage",
			events: []map[string]any{
				{"type": "turn.completed", "usage": map[string]any{
					"input_tokens": float64(100), "output_tokens": float64(40), "cached_input_tokens": float64(30),
				}},
			},
			want: tokenUsage{inputTokens: 100, outputTokens: 40, cacheReadTokens: 30},
		},
		{
			name: "prompt/completion aliases",
			events: []map[string]any{
				{"usage": map[string]any{"prompt_tokens": float64(11), "completion_tokens": float64(7)}},
			},
			want: tokenUsage{inputTokens: 11, outputTokens: 7},
		},
		{
			name: "anthropic-native cache fields",
			events: []map[string]any{
				{"usage": map[string]any{
					"input_tokens": float64(5), "output_tokens": float64(3),
					"cache_read_input_tokens": float64(2048), "cache_creation_input_tokens": float64(64),
				}},
			},
			want: tokenUsage{inputTokens: 5, outputTokens: 3, cacheReadTokens: 2048, cacheCreationTokens: 64},
		},
		{
			name: "usage nested under item",
			events: []map[string]any{
				{"type": "item.completed", "item": map[string]any{"usage": map[string]any{"input_tokens": float64(9)}}},
			},
			want: tokenUsage{inputTokens: 9},
		},
		{
			name: "usage nested under turn",
			events: []map[string]any{
				{"type": "turn.completed", "turn": map[string]any{"usage": map[string]any{"output_tokens": float64(4)}}},
			},
			want: tokenUsage{outputTokens: 4},
		},
		{
			name: "last usage object wins (cumulative totals)",
			events: []map[string]any{
				{"usage": map[string]any{"input_tokens": float64(10), "output_tokens": float64(1)}},
				{"usage": map[string]any{"input_tokens": float64(50), "output_tokens": float64(9)}},
			},
			want: tokenUsage{inputTokens: 50, outputTokens: 9},
		},
		{
			name: "non-numeric values yield zero",
			events: []map[string]any{
				{"usage": map[string]any{"input_tokens": "many", "output_tokens": float64(2)}},
			},
			want: tokenUsage{outputTokens: 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractTokenUsage(tt.events))
		})
	}
}

// TestClaudeCodeParsesResultUsage maps to the contract: the Claude Code
// result event's Anthropic-native usage object is parsed into Metrics token
// fields alongside the existing cost extraction.
func TestClaudeCodeParsesResultUsage(t *testing.T) {
	p := NewClaudeCodeProvider("")
	raw := &RawResult{}
	stdout := `{"type":"system","subtype":"init"}
{"type":"result","result":"done","session_id":"sess-1","total_cost_usd":0.5,"num_turns":3,"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":2048,"cache_creation_input_tokens":64}}`

	p.parseJSONOutput(stdout, raw)

	assert.Equal(t, 100, raw.Metrics.InputTokens)
	assert.Equal(t, 50, raw.Metrics.OutputTokens)
	assert.Equal(t, 2048, raw.Metrics.CacheReadTokens)
	assert.Equal(t, 64, raw.Metrics.CacheCreationTokens)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.InDelta(t, 0.5, *raw.Metrics.CostUSD, 1e-12)
}

// TestClaudeCodeNoUsageLeavesZeroTokens maps to the contract: a result event
// without a usage object reports zero tokens ("unknown"), never fabricated
// counts.
func TestClaudeCodeNoUsageLeavesZeroTokens(t *testing.T) {
	p := NewClaudeCodeProvider("")
	raw := &RawResult{}
	p.parseJSONOutput(`{"type":"result","result":"done","session_id":"s"}`, raw)

	assert.Zero(t, raw.Metrics.InputTokens)
	assert.Zero(t, raw.Metrics.OutputTokens)
	assert.Zero(t, raw.Metrics.CacheReadTokens)
	assert.Zero(t, raw.Metrics.CacheCreationTokens)
}

// TestCodexParsesEventUsage maps to the contract: Codex JSONL events carrying
// usage (cumulative on turn.completed) populate Metrics token fields.
func TestCodexParsesEventUsage(t *testing.T) {
	p := NewCodexProvider("")
	raw := &RawResult{}
	stdout := `{"type":"thread.started","thread_id":"th-1"}
{"type":"item.completed","item":{"type":"agent_message","text":"done"}}
{"type":"turn.completed","usage":{"input_tokens":500,"output_tokens":200,"cached_input_tokens":123}}`

	p.parseJSONLOutput(stdout, raw)

	assert.Equal(t, 500, raw.Metrics.InputTokens)
	assert.Equal(t, 200, raw.Metrics.OutputTokens)
	assert.Equal(t, 123, raw.Metrics.CacheReadTokens)
}

// TestAccumulateMetricsSumsTokens maps to the contract: token counts sum
// across every provider execution that contributed to a result, including
// failed retry attempts, and Result.TotalTokens is input+output.
func TestAccumulateMetricsSumsTokens(t *testing.T) {
	_, _, _, _, tok := accumulateMetrics([]*RawResult{
		{Metrics: Metrics{InputTokens: 100, OutputTokens: 40, CacheReadTokens: 10, CacheCreationTokens: 5}},
		{Metrics: Metrics{InputTokens: 50, OutputTokens: 10, CacheReadTokens: 3}},
	})
	assert.Equal(t, tokenUsage{inputTokens: 150, outputTokens: 50, cacheReadTokens: 13, cacheCreationTokens: 5}, tok)

	var res Result
	tok.applyTo(&res)
	assert.Equal(t, 150, res.InputTokens)
	assert.Equal(t, 50, res.OutputTokens)
	assert.Equal(t, 13, res.CacheReadTokens)
	assert.Equal(t, 5, res.CacheCreationTokens)
	assert.Equal(t, 200, res.TotalTokens)
}
