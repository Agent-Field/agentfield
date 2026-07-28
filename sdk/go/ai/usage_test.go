package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestUsageParsing maps to the contract: ai.Usage tolerates both OpenAI-style
// and OpenRouter/Anthropic-style usage payloads, exposing cache token counts
// and the provider-native cost when present.
func TestUsageParsing(t *testing.T) {
	tests := []struct {
		name              string
		body              string
		wantPrompt        int
		wantCompletion    int
		wantTotal         int
		wantCacheRead     int
		wantCacheCreation int
		wantCost          *float64
	}{
		{
			name:           "openai shape with prompt_tokens_details.cached_tokens",
			body:           `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":7}}`,
			wantPrompt:     10,
			wantCompletion: 5,
			wantTotal:      15,
			wantCacheRead:  7,
		},
		{
			name:              "openrouter anthropic-native cache fields and cost",
			body:              `{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"cache_read_input_tokens":2048,"cache_creation_input_tokens":64,"cost":0.0123}`,
			wantPrompt:        100,
			wantCompletion:    50,
			wantTotal:         150,
			wantCacheRead:     2048,
			wantCacheCreation: 64,
			wantCost:          float64Ptr(0.0123),
		},
		{
			name:          "anthropic-native cache read wins over openai nesting",
			body:          `{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cache_read_input_tokens":100,"prompt_tokens_details":{"cached_tokens":7}}`,
			wantCacheRead: 100, wantPrompt: 1, wantCompletion: 1, wantTotal: 2,
		},
		{
			name:       "bare usage without cache or cost",
			body:       `{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}`,
			wantPrompt: 3, wantCompletion: 4, wantTotal: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var u Usage
			require.NoError(t, json.Unmarshal([]byte(tt.body), &u))
			assert.Equal(t, tt.wantPrompt, u.PromptTokens)
			assert.Equal(t, tt.wantCompletion, u.CompletionTokens)
			assert.Equal(t, tt.wantTotal, u.TotalTokens)
			assert.Equal(t, tt.wantCacheRead, u.CacheReadTokens())
			assert.Equal(t, tt.wantCacheCreation, u.CacheCreationTokens())
			if tt.wantCost == nil {
				assert.Nil(t, u.Cost)
			} else {
				require.NotNil(t, u.Cost)
				assert.InDelta(t, *tt.wantCost, *u.Cost, 1e-12)
			}
		})
	}
}

func float64Ptr(f float64) *float64 { return &f }

// TestNilUsageAccessors maps to the contract: cache accessors are safe on a
// response that carried no usage object.
func TestNilUsageAccessors(t *testing.T) {
	var u *Usage
	assert.Equal(t, 0, u.CacheReadTokens())
	assert.Equal(t, 0, u.CacheCreationTokens())
}

// TestOpenRouterRequestsIncludeUsageAccounting maps to the contract: a
// request targeting OpenRouter carries {"usage": {"include": true}} so the
// provider returns its native cost; a non-OpenRouter request must not have a
// usage key injected into its body.
func TestOpenRouterRequestsIncludeUsageAccounting(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		wantInclude bool
	}{
		{name: "openrouter model opts into usage accounting", model: "openrouter/qwen/qwen3-coder", wantInclude: true},
		{name: "non-openrouter model body has no usage key", model: "gpt-4o", wantInclude: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var captured map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.NoError(t, json.NewDecoder(r.Body).Decode(&captured))
				_ = json.NewEncoder(w).Encode(Response{
					Model: tt.model,
					Choices: []Choice{{
						Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "ok"}}},
						FinishReason: "stop",
					}},
				})
			}))
			defer server.Close()

			client, err := NewClient(&Config{APIKey: "test-key", BaseURL: server.URL, Model: tt.model})
			require.NoError(t, err)

			_, err = client.Complete(context.Background(), "hello")
			require.NoError(t, err)
			require.NotNil(t, captured)

			usageRaw, present := captured["usage"]
			if !tt.wantInclude {
				assert.False(t, present, "non-OpenRouter request must not carry a usage key")
				return
			}
			require.True(t, present, "OpenRouter request must carry the usage accounting opt-in")
			usage, ok := usageRaw.(map[string]any)
			require.True(t, ok)
			assert.Equal(t, true, usage["include"])
		})
	}
}

// TestOpenRouterStreamRequestIncludesUsageAccounting maps to the contract:
// the streaming path shares the opt-in, so streamed OpenRouter calls also get
// a terminal usage chunk.
func TestOpenRouterStreamRequestIncludesUsageAccounting(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&captured))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	client, err := NewClient(&Config{APIKey: "test-key", BaseURL: server.URL, Model: "openrouter/some/model"})
	require.NoError(t, err)

	chunks, errs := client.StreamComplete(context.Background(), "hello")
	for range chunks {
	}
	require.NoError(t, <-errs)

	usage, ok := captured["usage"].(map[string]any)
	require.True(t, ok, "streaming OpenRouter request must carry the usage opt-in")
	assert.Equal(t, true, usage["include"])
}

// TestStreamChunkParsesUsage maps to the contract: a terminal SSE chunk
// carrying a usage object is surfaced on StreamChunk.Usage without disturbing
// content chunks.
func TestStreamChunkParsesUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"1","model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.002}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	client, err := NewClient(&Config{APIKey: "test-key", BaseURL: server.URL, Model: "gpt-4o"})
	require.NoError(t, err)

	chunks, errs := client.StreamComplete(context.Background(), "hello")
	var collected []StreamChunk
	for chunk := range chunks {
		collected = append(collected, chunk)
	}
	require.NoError(t, <-errs)

	require.Len(t, collected, 2)
	assert.Nil(t, collected[0].Usage)
	assert.Equal(t, "hi", collected[0].Choices[0].Delta.Content)
	require.NotNil(t, collected[1].Usage)
	assert.Equal(t, 10, collected[1].Usage.PromptTokens)
	assert.Equal(t, 5, collected[1].Usage.CompletionTokens)
	require.NotNil(t, collected[1].Usage.Cost)
	assert.InDelta(t, 0.002, *collected[1].Usage.Cost, 1e-12)
}

// TestToolCallLoopTraceCarriesPerTurnUsage maps to the contract: every LLM
// call made during a tool-call loop — intermediate tool-calling turns and the
// final response — contributes its usage to the trace, in call order.
func TestToolCallLoopTraceCarriesPerTurnUsage(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch requestCount.Add(1) {
		case 1:
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Model: "gpt-4o",
				Choices: []Choice{{
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{{
							ID:       "call-1",
							Type:     "function",
							Function: ToolCallFunction{Name: "lookup", Arguments: `{}`},
						}},
					},
					FinishReason: "tool_calls",
				}},
				Usage: &Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120},
			}))
		default:
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Model: "gpt-4o",
				Choices: []Choice{{
					Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "done"}}},
					FinishReason: "stop",
				}},
				Usage: &Usage{PromptTokens: 150, CompletionTokens: 30, TotalTokens: 180},
			}))
		}
	})

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
	)
	require.NoError(t, err)
	require.NotNil(t, trace)

	require.Len(t, trace.Usage, 2, "both LLM calls must contribute usage")
	assert.Equal(t, "gpt-4o", trace.Usage[0].Model)
	assert.Equal(t, 100, trace.Usage[0].Usage.PromptTokens)
	assert.Equal(t, 20, trace.Usage[0].Usage.CompletionTokens)
	assert.Equal(t, 150, trace.Usage[1].Usage.PromptTokens)
	assert.Equal(t, 30, trace.Usage[1].Usage.CompletionTokens)
}

// TestToolCallLoopTraceSkipsMissingUsage maps to the contract: responses
// without usage objects contribute nothing (no phantom entries).
func TestToolCallLoopTraceSkipsMissingUsage(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "done"}}},
				FinishReason: "stop",
			}},
		}))
	})

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 2, MaxToolCalls: 2},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return nil, nil
		},
	)
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Empty(t, trace.Usage)
}
