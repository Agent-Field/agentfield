package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallLoop_ZeroTurnsAppliesOptionsOnce(t *testing.T) {
	var sent Request
	requests := 0
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		require.NoError(t, json.NewDecoder(r.Body).Decode(&sent))
		require.NoError(t, json.NewEncoder(w).Encode(Response{Choices: []Choice{{Message: Message{Role: "assistant"}}}}))
	})
	client.config.MaxTokens = 100
	applied := 0
	_, trace, err := client.ExecuteToolCallLoop(context.Background(), []Message{{Role: "user"}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
		ToolCallConfig{MaxTurns: 0}, nil, func(r *Request) error {
			applied++
			require.NotNil(t, r.MaxTokens, "options must see initialized defaults")
			assert.Empty(t, r.Tools, "zero-turn request must not include tools")
			assert.Nil(t, r.ToolChoice)
			*r.MaxTokens++
			return nil
		})
	require.NoError(t, err)
	assert.Equal(t, 1, requests)
	assert.Equal(t, requests, applied)
	assert.Equal(t, 101, *sent.MaxTokens)
	assert.Equal(t, 0, trace.TotalTurns)
}

func TestToolCallLoop_FirstRequestMessageEditPersists(t *testing.T) {
	for _, maxTurns := range []int{1, 3} {
		t.Run(string(rune('0'+maxTurns)), func(t *testing.T) {
			var sent []Request
			client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
				var req Request
				require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
				sent = append(sent, req)
				msg := Message{Role: "assistant"}
				if len(sent) == 1 {
					msg.ToolCalls = []ToolCall{{ID: "c1", Type: "function", Function: ToolCallFunction{Name: "lookup", Arguments: `{}`}}}
				}
				require.NoError(t, json.NewEncoder(w).Encode(Response{Choices: []Choice{{Message: msg}}}))
			})
			applied := 0
			_, _, err := client.ExecuteToolCallLoop(context.Background(),
				[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "original"}}}}, nil,
				ToolCallConfig{MaxTurns: maxTurns, MaxToolCalls: 3},
				func(context.Context, string, map[string]interface{}) (map[string]interface{}, error) {
					return map[string]interface{}{"ok": true}, nil
				}, func(r *Request) error {
					applied++
					if applied == 1 {
						r.Messages[0].Content = []ContentPart{{Type: "text", Text: "revised"}}
					}
					return nil
				})
			require.NoError(t, err)
			require.Len(t, sent, 2)
			assert.Equal(t, len(sent), applied)
			for _, req := range sent {
				assert.Equal(t, "revised", req.Messages[0].Content[0].Text)
			}
			// Pin the default conversation through the public HTTP path too:
			// no injected system text, unchanged assistant turn and raw JSON result.
			wire, err := json.Marshal(sent[1].Messages)
			require.NoError(t, err)
			assert.Equal(t, `[{"role":"user","content":"revised"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","content":"{\"ok\":true}","tool_call_id":"c1"}]`, string(wire))
		})
	}
}

func TestToolCallTrace_OptionMessageOrder(t *testing.T) {
	message := func(role, text string) Message {
		return Message{Role: role, Content: []ContentPart{{Type: "text", Text: text}}}
	}
	for _, mode := range []string{"append", "insert", "reorder", "remove"} {
		t.Run(mode, func(t *testing.T) {
			var sent Request
			client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
				require.NoError(t, json.NewDecoder(r.Body).Decode(&sent))
				require.NoError(t, json.NewEncoder(w).Encode(Response{Choices: []Choice{{Message: Message{Role: "assistant"}}}}))
			})
			applied := 0
			_, trace, err := client.ExecuteToolCallLoop(context.Background(),
				[]Message{message("user", "go")}, nil, ToolCallConfig{MaxTurns: 1, SystemPrompt: "SHARED"}, nil,
				func(r *Request) error {
					applied++
					caller := message("system", "SHARED")
					switch mode {
					case "append":
						r.Messages = append(r.Messages, caller)
					case "insert":
						r.Messages = append(r.Messages, Message{})
						copy(r.Messages[2:], r.Messages[1:])
						r.Messages[1] = caller
					case "reorder":
						r.Messages[0], r.Messages[1] = r.Messages[1], r.Messages[0]
					case "remove":
						r.Messages = r.Messages[1:]
					}
					return nil
				}, WithSystem("CALLER"))
			require.NoError(t, err)
			assert.Equal(t, 1, applied)
			require.Len(t, trace.Messages, len(sent.Messages))
			for i, tagged := range trace.Messages {
				assert.Equal(t, sent.Messages[i], tagged.Message)
			}
			expected := map[string][]string{
				"append":  {TraceSourceUser, TraceSourceToolSystemPrompt, TraceSourceUser, TraceSourceUser},
				"insert":  {TraceSourceUser, TraceSourceToolSystemPrompt, TraceSourceUser, TraceSourceUser},
				"reorder": {TraceSourceUser, TraceSourceUser, TraceSourceToolSystemPrompt},
				"remove":  {TraceSourceUser, TraceSourceUser},
			}
			var sources []string
			for _, tagged := range trace.Messages {
				sources = append(sources, tagged.Source)
			}
			assert.Equal(t, expected[mode], sources)
		})
	}
}
