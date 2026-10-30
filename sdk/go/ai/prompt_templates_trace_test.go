package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToolCallTrace_TagsMessageSources verifies the TracedMessage source tags
// cover the user prompt, the assistant tool-call turn, and the tool result,
// using the cross-SDK taxonomy (issue #229).
func TestToolCallTrace_TagsMessageSources(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		if count == 1 {
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{{
							ID:       "call-1",
							Type:     "function",
							Function: ToolCallFunction{Name: "lookup", Arguments: `{"id":"1"}`},
						}},
					},
					FinishReason: "tool_calls",
				}},
			}))
			return
		}
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
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"status": "open"}, nil
		},
	)
	require.NoError(t, err)
	require.NotNil(t, trace)

	sources := make([]string, 0, len(trace.Messages))
	for _, m := range trace.Messages {
		sources = append(sources, m.Source)
	}
	// user prompt, assistant tool-call turn, tool result.
	assert.Equal(t, []string{
		TraceSourceUser,
		TraceSourceAssistant,
		TraceSourceToolResult,
	}, sources)

	// Messages hold references to the real wire messages, not reshaped copies.
	assert.Equal(t, "lookup", trace.Messages[1].Message.ToolCalls[0].Function.Name)
	assert.Equal(t, "tool", trace.Messages[2].Message.Role)
}

func TestToolCallTrace_TagsToolErrorSource(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		if count == 1 {
			require.NoError(t, json.NewEncoder(w).Encode(Response{
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
			}))
			return
		}
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "handled"}}},
				FinishReason: "stop",
			}},
		}))
	})

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return nil, assert.AnError
		},
	)
	require.NoError(t, err)

	var sawError bool
	for _, m := range trace.Messages {
		if m.Source == TraceSourceToolError {
			sawError = true
		}
	}
	assert.True(t, sawError, "a failed tool call must be tagged sdk.tool_error")
}

func TestToolCallTrace_TagsToolSystemPrompt(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "ok"}}},
				FinishReason: "stop",
			}},
		}))
	})

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 2, MaxToolCalls: 2, SystemPrompt: "Use tools wisely."},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
	)
	require.NoError(t, err)

	var sysMsg *TracedMessage
	for i := range trace.Messages {
		if trace.Messages[i].Source == TraceSourceToolSystemPrompt {
			sysMsg = &trace.Messages[i]
		}
	}
	require.NotNil(t, sysMsg, "tool system prompt must be tagged")
	assert.Equal(t, "Use tools wisely.", sysMsg.Message.Content[0].Text)
}

// TestToolCallTrace_OmitsEmptyMessagesInJSON guards the omitempty tag: an empty
// trace must not marshal "Messages":null, which would surprise existing callers
// that serialize a trace. (Calls/Usage marshalling null is pre-existing and out
// of scope here.)
func TestToolCallTrace_OmitsEmptyMessagesInJSON(t *testing.T) {
	data, err := json.Marshal(ToolCallTrace{})
	require.NoError(t, err)
	assert.NotContains(t, string(data), "\"Messages\"")
}

// TestToolCallTrace_TagsCallerSystemFromOptions covers the review gap: a caller
// that supplies their own system message via WithSystem reaches the model but
// was absent from the trace. It must now be captured as user/caller content.
func TestToolCallTrace_TagsCallerSystemFromOptions(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "ok"}}},
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
			return map[string]interface{}{"ok": true}, nil
		},
		WithSystem("Caller system instructions."),
	)
	require.NoError(t, err)

	var callerSystem *TracedMessage
	for i := range trace.Messages {
		if trace.Messages[i].Source == TraceSourceUser &&
			trace.Messages[i].Message.Role == "system" {
			callerSystem = &trace.Messages[i]
		}
	}
	require.NotNil(t, callerSystem, "caller WithSystem text must be traced")
	assert.Equal(t, "Caller system instructions.", callerSystem.Message.Content[0].Text)
}

// TestToolCallTrace_DoesNotReplayOptions guards the review finding: provenance
// capture must read the real request, not re-execute caller options (which runs
// their side effects twice). A counting option must fire exactly once per turn.
func TestToolCallTrace_DoesNotReplayOptions(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "ok"}}},
				FinishReason: "stop",
			}},
		}))
	})

	var applied int
	countingOption := func(r *Request) error {
		applied++
		return nil
	}

	_, _, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 1, MaxToolCalls: 2},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		countingOption,
	)
	require.NoError(t, err)
	// One turn builds one request, so the option applies exactly once; the
	// trace must not have re-run it for provenance capture.
	assert.Equal(t, 1, applied, "caller option must not be executed twice")
}

// TestToolCallTrace_CallerAndSDKSystemProvenance covers the review checklist:
// with both a caller WithSystem("CALLER") and config.SystemPrompt="SDK", the
// request order is CALLER, SDK, user (WithSystem prepends). Each system occurs
// once with its correct source, and SDK is not recorded twice.
func TestToolCallTrace_CallerAndSDKSystemProvenance(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
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
		ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1, SystemPrompt: "SDK"},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		WithSystem("CALLER"),
	)
	require.NoError(t, err)

	type st struct{ text, source string }
	var systems []st
	for _, m := range trace.Messages {
		if m.Message.Role == "system" {
			systems = append(systems, st{m.Message.Content[0].Text, m.Source})
		}
	}
	// CALLER (caller content) and SDK (tool system prompt), each exactly once,
	// in request order.
	assert.Equal(t, []st{
		{"CALLER", TraceSourceUser},
		{"SDK", TraceSourceToolSystemPrompt},
	}, systems)
}

// TestToolCallTrace_InitialSystemNotDoubleTraced: a system message already in
// the initial list is traced once, not duplicated.
func TestToolCallTrace_InitialSystemNotDoubleTraced(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "done"}}},
				FinishReason: "stop",
			}},
		}))
	})

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{
			{Role: "system", Content: []ContentPart{{Type: "text", Text: "INITIAL"}}},
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}},
		},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
	)
	require.NoError(t, err)

	count := 0
	for _, m := range trace.Messages {
		if m.Message.Role == "system" && m.Message.Content[0].Text == "INITIAL" {
			count++
			assert.Equal(t, TraceSourceUser, m.Source)
		}
	}
	assert.Equal(t, 1, count, "initial system message must be traced exactly once")
}

// TestToolCallTrace_MaxTurnsZeroStillTracesCallerSystem: MaxTurns:0 still sends
// the final request, and caller system text must be in its trace.
func TestToolCallTrace_MaxTurnsZeroStillTracesCallerSystem(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
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
		ToolCallConfig{MaxTurns: 0, MaxToolCalls: 1},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		WithSystem("CALLER"),
	)
	require.NoError(t, err)

	var found bool
	for _, m := range trace.Messages {
		if m.Message.Role == "system" && m.Message.Content[0].Text == "CALLER" {
			found = true
			assert.Equal(t, TraceSourceUser, m.Source)
		}
	}
	assert.True(t, found, "caller system must be traced even when MaxTurns is 0")
}

// TestToolCallTrace_MultipleAndIdenticalCallerSystems: multiple caller systems,
// including text identical to the SDK prompt, are each attributed correctly
// without positional or text-based dedup.
func TestToolCallTrace_MultipleAndIdenticalCallerSystems(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
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
		ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1, SystemPrompt: "SHARED"},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		WithSystem("SHARED"), // identical text to the SDK prompt
		WithSystem("OTHER"),
	)
	require.NoError(t, err)

	// One SDK-sourced "SHARED", plus two caller systems ("SHARED" and "OTHER").
	sdkShared, callerShared, callerOther := 0, 0, 0
	for _, m := range trace.Messages {
		if m.Message.Role != "system" {
			continue
		}
		text := m.Message.Content[0].Text
		switch {
		case text == "SHARED" && m.Source == TraceSourceToolSystemPrompt:
			sdkShared++
		case text == "SHARED" && m.Source == TraceSourceUser:
			callerShared++
		case text == "OTHER" && m.Source == TraceSourceUser:
			callerOther++
		}
	}
	assert.Equal(t, 1, sdkShared, "exactly one SHARED from the SDK prompt")
	assert.Equal(t, 1, callerShared, "exactly one SHARED from the caller")
	assert.Equal(t, 1, callerOther, "exactly one OTHER from the caller")
}

// TestToolCallTrace_AppendingOptionProvenance covers the review case where a
// caller option appends (not prepends) a system message. Wire order is
// SDK, user, CALLER, and each must keep its correct source rather than swapping
// by position.
func TestToolCallTrace_AppendingOptionProvenance(t *testing.T) {
	client := newToolLoopClient(t, func(w http.ResponseWriter, _ *http.Request) {
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "done"}}},
				FinishReason: "stop",
			}},
		}))
	})

	appendSystem := func(text string) Option {
		return func(r *Request) error {
			r.Messages = append(r.Messages, Message{
				Role:    "system",
				Content: []ContentPart{{Type: "text", Text: text}},
			})
			return nil
		}
	}

	_, trace, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1, SystemPrompt: "SDK"},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		appendSystem("CALLER"),
	)
	require.NoError(t, err)

	byText := map[string]string{}
	for _, m := range trace.Messages {
		if m.Message.Role == "system" {
			byText[m.Message.Content[0].Text] = m.Source
		}
	}
	assert.Equal(t, TraceSourceToolSystemPrompt, byText["SDK"])
	assert.Equal(t, TraceSourceUser, byText["CALLER"])
}

// TestToolCallTrace_MaxTurnsZeroDoesNotReplayOptions: the zero-turn path sends
// exactly one request, so an option that mutates state (MaxTokens++) must be
// applied once, not twice.
func TestToolCallTrace_MaxTurnsZeroDoesNotReplayOptions(t *testing.T) {
	var sawMaxTokens int
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		if req.MaxTokens != nil {
			sawMaxTokens = *req.MaxTokens
		}
		require.NoError(t, json.NewEncoder(w).Encode(Response{
			Choices: []Choice{{
				Message:      Message{Role: "assistant", Content: []ContentPart{{Type: "text", Text: "done"}}},
				FinishReason: "stop",
			}},
		}))
	})

	incMaxTokens := func(r *Request) error {
		if r.MaxTokens != nil {
			v := *r.MaxTokens + 1
			r.MaxTokens = &v
		}
		return nil
	}

	_, _, err := client.ExecuteToolCallLoop(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "go"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 0, MaxToolCalls: 1},
		func(_ context.Context, _ string, _ map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
		incMaxTokens,
	)
	require.NoError(t, err)
	// The option ran once: base MaxTokens + 1. Running twice would be +2.
	defaultMax := 0 // client default MaxTokens is 0 unless configured
	assert.Equal(t, defaultMax+1, sawMaxTokens, "option must apply exactly once for the single request")
}
