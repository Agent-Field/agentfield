package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCapabilityToToolDefinition_AdditionalCases(t *testing.T) {
	t.Run("schema without explicit type gets wrapped", func(t *testing.T) {
		tool := CapabilityToToolDefinition(types.SkillCapability{
			InvocationTarget: "agent.lookup",
			InputSchema: map[string]interface{}{
				"ticket": map[string]interface{}{"type": "string"},
			},
		})
		assert.Equal(t, "object", tool.Function.Parameters["type"])
		_, ok := tool.Function.Parameters["properties"]
		assert.True(t, ok)
	})

	t.Run("unsupported capability returns zero value", func(t *testing.T) {
		tool := CapabilityToToolDefinition("invalid")
		assert.Equal(t, ToolDefinition{}, tool)
	})
}

func TestExecuteToolCallLoop_LimitAndFinalizationPaths(t *testing.T) {
	t.Run("tool call limit triggers final call without tools", func(t *testing.T) {
		var requestCount atomic.Int32
		client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
			count := requestCount.Add(1)
			var req Request
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

			switch count {
			case 1:
				require.Len(t, req.Tools, 1)
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{
						Message: Message{
							Role: "assistant",
							ToolCalls: []ToolCall{
								{
									ID:   "call-1",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "lookup",
										Arguments: `{"id":"1"}`,
									},
								},
								{
									ID:   "call-2",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "lookup",
										Arguments: `{"id":"2"}`,
									},
								},
							},
						},
					}},
				}))
			case 2:
				assert.Nil(t, req.Tools)
				assert.Nil(t, req.ToolChoice)
				require.Len(t, req.Messages, 4)
				assert.Contains(t, req.Messages[3].Content[0].Text, "Tool call limit reached")
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{
						Message: Message{
							Role:    "assistant",
							Content: []ContentPart{{Type: "text", Text: "final after limit"}},
						},
					}},
				}))
			default:
				t.Fatalf("unexpected request %d", count)
			}
		})

		var callCount int
		resp, trace, err := client.ExecuteToolCallLoop(
			context.Background(),
			[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "lookup"}}}},
			[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
			ToolCallConfig{MaxTurns: 3, MaxToolCalls: 1},
			func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
				callCount++
				assert.Equal(t, "lookup", target)
				return map[string]interface{}{"ok": true}, nil
			},
		)

		require.NoError(t, err)
		assert.Equal(t, 1, callCount)
		assert.Equal(t, "final after limit", resp.Text())
		assert.Equal(t, "final after limit", trace.FinalResponse)
		assert.Equal(t, 1, trace.TotalToolCalls)
		require.Len(t, trace.Calls, 1)
	})

	t.Run("max turns triggers final call without tools", func(t *testing.T) {
		var requestCount atomic.Int32
		client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
			count := requestCount.Add(1)
			var req Request
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

			switch count {
			case 1:
				require.Len(t, req.Tools, 1)
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{
						Message: Message{
							Role: "assistant",
							ToolCalls: []ToolCall{{
								ID:   "call-1",
								Type: "function",
								Function: ToolCallFunction{
									Name:      "lookup",
									Arguments: `{"id":"1"}`,
								},
							}},
						},
					}},
				}))
			case 2:
				assert.Nil(t, req.Tools)
				require.Len(t, req.Messages, 3)
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{
						Message: Message{
							Role:    "assistant",
							Content: []ContentPart{{Type: "text", Text: "final after max turns"}},
						},
					}},
				}))
			default:
				t.Fatalf("unexpected request %d", count)
			}
		})

		resp, trace, err := client.ExecuteToolCallLoop(
			context.Background(),
			[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "lookup"}}}},
			[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
			ToolCallConfig{MaxTurns: 1, MaxToolCalls: 5},
			func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
				return map[string]interface{}{"ok": true}, nil
			},
		)

		require.NoError(t, err)
		assert.Equal(t, "final after max turns", resp.Text())
		assert.Equal(t, 1, trace.TotalTurns)
		assert.Equal(t, "final after max turns", trace.FinalResponse)
	})
}

func TestExecuteToolCallLoop_ErrorPaths(t *testing.T) {
	t.Run("option error", func(t *testing.T) {
		client, err := NewClient(&Config{
			APIKey:  "test-key",
			BaseURL: "https://example.com",
			Model:   "gpt-4o",
		})
		require.NoError(t, err)

		_, trace, err := client.ExecuteToolCallLoop(
			context.Background(),
			nil,
			nil,
			ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1},
			func(context.Context, string, map[string]interface{}) (map[string]interface{}, error) { return nil, nil },
			func(*Request) error { return errors.New("bad option") },
		)

		require.Error(t, err)
		assert.Contains(t, err.Error(), "apply option: bad option")
		assert.NotNil(t, trace)
	})

	t.Run("llm call failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
		}))
		defer server.Close()

		client, err := NewClient(&Config{
			APIKey:  "test-key",
			BaseURL: server.URL,
			Model:   "gpt-4o",
		})
		require.NoError(t, err)

		_, trace, err := client.ExecuteToolCallLoop(
			context.Background(),
			[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}}},
			nil,
			ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1},
			func(context.Context, string, map[string]interface{}) (map[string]interface{}, error) { return nil, nil },
		)

		require.Error(t, err)
		assert.Contains(t, err.Error(), "LLM call failed")
		assert.NotNil(t, trace)
	})
}
