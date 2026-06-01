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

func newToolLoopClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	client, err := NewClient(&Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "gpt-4o",
	})
	require.NoError(t, err)
	return client
}

func TestExecuteToolCallLoop_CompletesAfterToolResponse(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

		switch count {
		case 1:
			require.Len(t, req.Tools, 1)
			require.Equal(t, "auto", req.ToolChoice)
			require.Len(t, req.Messages, 1)
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{{
							ID:   "call-1",
							Type: "function",
							Function: ToolCallFunction{
								Name:      "lookup",
								Arguments: `{"ticket":"123"}`,
							},
						}},
					},
					FinishReason: "tool_calls",
				}},
			}))
		case 2:
			require.Len(t, req.Messages, 3)
			assert.Equal(t, "tool", req.Messages[2].Role)
			assert.Equal(t, "call-1", req.Messages[2].ToolCallID)
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{
					Message: Message{
						Role:    "assistant",
						Content: []ContentPart{{Type: "text", Text: "resolved"}},
					},
					FinishReason: "stop",
				}},
			}))
		default:
			t.Fatalf("unexpected request %d", count)
		}
	})

	resp, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "Find ticket 123"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2},
		func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
			assert.Equal(t, "lookup", target)
			assert.Equal(t, map[string]interface{}{"ticket": "123"}, input)
			return map[string]interface{}{"status": "open"}, nil
		},
	)

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, trace)
	assert.Equal(t, "resolved", resp.Text())
	assert.Equal(t, 2, trace.TotalTurns)
	assert.Equal(t, 1, trace.TotalToolCalls)
	assert.Equal(t, "resolved", trace.FinalResponse)
	require.Len(t, trace.Calls, 1)
	assert.Equal(t, "lookup", trace.Calls[0].ToolName)
	assert.Equal(t, map[string]interface{}{"status": "open"}, trace.Calls[0].Result)
}

func TestExecuteToolCallLoop_RecordsToolErrorsAndMalformedArguments(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		if count == 1 {
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{{
							ID:   "call-2",
							Type: "function",
							Function: ToolCallFunction{Name: "lookup", Arguments: `{bad json`},
						}},
					},
					FinishReason: "tool_calls",
				}},
			}))
			return
		}

		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		require.Len(t, req.Messages, 3)
		assert.Equal(t, "tool", req.Messages[2].Role)
		assert.Contains(t, req.Messages[2].Content[0].Text, "assert.AnError")
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message: Message{
					Role:    "assistant",
					Content: []ContentPart{{Type: "text", Text: "done after error"}},
				},
				FinishReason: "stop",
			}},
		}))
	})

	resp, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "Lookup anyway"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2},
		func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
			assert.Equal(t, "lookup", target)
			assert.Empty(t, input)
			return nil, assert.AnError
		},
	)

	require.NoError(t, err)
	assert.Equal(t, "done after error", resp.Text())
	require.Len(t, trace.Calls, 1)
	assert.Equal(t, assert.AnError.Error(), trace.Calls[0].Error)
	assert.Empty(t, trace.Calls[0].Arguments)
}

func TestSimpleAndStructuredAIAndResponseHelpers(t *testing.T) {
	var requestCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.Equal(t, "Bearer env-key", r.Header.Get("Authorization"))

		switch count {
		case 1:
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{Message: Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "plain text"}}}}},
			}))
		case 2:
			require.NotNil(t, req.ResponseFormat)
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{Message: Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: `{"status":"ok"}`}}}}},
			}))
		default:
			t.Fatalf("unexpected request %d", count)
		}
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("AI_BASE_URL", server.URL)

	text, err := SimpleAI(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, "plain text", text)

	var dest struct {
		Status string `json:"status"`
	}
	require.NoError(t, StructuredAI(context.Background(), "hello", struct {
		Status string `json:"status"`
	}{}, &dest))
	assert.Equal(t, "ok", dest.Status)

	response := &Response{Choices: []Choice{{Message: Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "call", Type: "function"}}, Content: []ContentPart{{Type: "text", Text: "body"}}}}}}
	assert.True(t, response.HasToolCalls())
	assert.Len(t, response.ToolCalls(), 1)

	req := &Request{}
	require.NoError(t, WithTools([]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}})(req))
	assert.Equal(t, "auto", req.ToolChoice)
	assert.Len(t, req.Tools, 1)
}
