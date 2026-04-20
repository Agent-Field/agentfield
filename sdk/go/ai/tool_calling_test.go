package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCapabilityToToolDefinition_Reasoner(t *testing.T) {
	desc := "Analyze text sentiment"
	r := types.ReasonerCapability{
		ID:          "analyze",
		Description: &desc,
		Tags:        []string{"nlp"},
		InputSchema: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"text": map[string]interface{}{"type": "string"}},
		},
		InvocationTarget: "sentiment_agent.analyze",
	}

	tool := CapabilityToToolDefinition(r)
	if tool.Type != "function" {
		t.Errorf("expected type 'function', got %q", tool.Type)
	}
	if tool.Function.Name != "sentiment_agent.analyze" {
		t.Errorf("expected name 'sentiment_agent.analyze', got %q", tool.Function.Name)
	}
	if tool.Function.Description != "Analyze text sentiment" {
		t.Errorf("expected description, got %q", tool.Function.Description)
	}
}

func TestCapabilityToToolDefinition_Skill(t *testing.T) {
	desc := "Send an email"
	s := types.SkillCapability{
		ID:          "send_email",
		Description: &desc,
		InputSchema: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"to": map[string]interface{}{"type": "string"}},
		},
		InvocationTarget: "notif_agent.send_email",
	}

	tool := CapabilityToToolDefinition(s)
	if tool.Function.Name != "notif_agent.send_email" {
		t.Errorf("expected name 'notif_agent.send_email', got %q", tool.Function.Name)
	}
}

func TestCapabilityToToolDefinition_SanitizesColonTargets(t *testing.T) {
	desc := "Lookup weather"
	s := types.SkillCapability{
		ID:               "lookup_weather",
		Description:      &desc,
		InvocationTarget: "weather-agent:skill:lookup_weather",
	}

	tool := CapabilityToToolDefinition(s)
	if tool.Function.Name != "weather-agent__skill__lookup_weather" {
		t.Errorf("expected sanitized name, got %q", tool.Function.Name)
	}
}

func TestCapabilityToToolDefinition_NilSchema(t *testing.T) {
	r := types.ReasonerCapability{
		ID:               "test",
		InvocationTarget: "agent.test",
	}
	tool := CapabilityToToolDefinition(r)
	if tool.Function.Parameters == nil {
		t.Error("expected non-nil parameters")
	}
	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("expected type 'object', got %v", tool.Function.Parameters["type"])
	}
}

func TestCapabilityToToolDefinition_NilDescription(t *testing.T) {
	r := types.ReasonerCapability{
		ID:               "test",
		InvocationTarget: "agent.test",
	}
	tool := CapabilityToToolDefinition(r)
	if tool.Function.Description != "Call agent.test" {
		t.Errorf("expected fallback description, got %q", tool.Function.Description)
	}
}

func TestCapabilitiesToToolDefinitions(t *testing.T) {
	desc1 := "Analyze"
	desc2 := "Send"
	caps := []types.AgentCapability{
		{
			AgentID: "test-agent",
			Reasoners: []types.ReasonerCapability{
				{ID: "analyze", Description: &desc1, InvocationTarget: "agent.analyze"},
			},
			Skills: []types.SkillCapability{
				{ID: "send", Description: &desc2, InvocationTarget: "agent.send"},
			},
		},
	}

	tools := CapabilitiesToToolDefinitions(caps)
	if len(tools) != 2 {
		t.Errorf("expected 2 tools, got %d", len(tools))
	}
}

func TestToolCallConfig_Defaults(t *testing.T) {
	cfg := DefaultToolCallConfig()
	if cfg.MaxTurns != 10 {
		t.Errorf("expected MaxTurns 10, got %d", cfg.MaxTurns)
	}
	if cfg.MaxToolCalls != 25 {
		t.Errorf("expected MaxToolCalls 25, got %d", cfg.MaxToolCalls)
	}
	if cfg.PromptConfig == nil {
		t.Fatal("expected PromptConfig to be initialized")
	}
	if cfg.PromptConfig.ToolCallLimitReached != "Tool call limit reached. Please provide a final response." {
		t.Errorf("expected default prompt config, got %q", cfg.PromptConfig.ToolCallLimitReached)
	}
}

func TestToolDefinition_JSONRoundTrip(t *testing.T) {
	tool := ToolDefinition{
		Type: "function",
		Function: ToolFunction{
			Name:        "test.fn",
			Description: "A test function",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{"x": map[string]interface{}{"type": "string"}},
			},
		},
	}

	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded ToolDefinition
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Function.Name != "test.fn" {
		t.Errorf("expected name 'test.fn', got %q", decoded.Function.Name)
	}
}

func TestToolCallTrace(t *testing.T) {
	trace := ToolCallTrace{
		Calls: []ToolCallRecord{
			{ToolName: "agent.fn", Arguments: map[string]interface{}{"x": 1}, LatencyMs: 42.5, Turn: 0},
		},
		TotalTurns:     1,
		TotalToolCalls: 1,
		FinalResponse:  "done",
	}

	if len(trace.Calls) != 1 {
		t.Errorf("expected 1 call, got %d", len(trace.Calls))
	}
	if trace.Calls[0].Error != "" {
		t.Errorf("expected no error, got %q", trace.Calls[0].Error)
	}
}

func TestToolCallResult(t *testing.T) {
	resp := &Response{
		Choices: []Choice{{
			Message: Message{
				Role:    "assistant",
				Content: []ContentPart{{Type: "text", Text: "from response"}},
			},
		}},
	}
	trace := &ToolCallTrace{
		FinalResponse: "from trace",
	}

	result := &ToolCallResult{Response: resp, Trace: trace}
	if result.Text() != "from trace" {
		t.Fatalf("expected trace text, got %q", result.Text())
	}
}

func TestSanitizeToolNameRoundTrip(t *testing.T) {
	name := "worker:skill:lookup"
	if got := sanitizeToolName(name); got != "worker__skill__lookup" {
		t.Fatalf("expected sanitized name, got %q", got)
	}
	if got := unsanitizeToolName("worker__skill__lookup"); got != name {
		t.Fatalf("expected unsanitized name %q, got %q", name, got)
	}
}

func TestExecuteToolCallLoopResult_UsesSystemPromptAndUnsanitizesToolNames(t *testing.T) {
	var requestCount atomic.Int32
	client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)
		var req Request
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

		switch count {
		case 1:
			require.Len(t, req.Messages, 2)
			assert.Equal(t, "system", req.Messages[0].Role)
			assert.Equal(t, "Use tools carefully.", req.Messages[0].Content[0].Text)
			require.Len(t, req.Tools, 1)
			assert.Equal(t, "worker__skill__lookup", req.Tools[0].Function.Name)
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{{
							ID:   "call-1",
							Type: "function",
							Function: ToolCallFunction{
								Name:      "worker__skill__lookup",
								Arguments: `{"ticket":"123"}`,
							},
						}},
					},
					FinishReason: "tool_calls",
				}},
			}))
		case 2:
			require.Len(t, req.Messages, 4)
			assert.Equal(t, "system", req.Messages[0].Role)
			assert.Equal(t, "tool", req.Messages[3].Role)
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

	result, err := client.ExecuteToolCallLoopResult(
		context.Background(),
		[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "Find ticket 123"}}}},
		[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "worker__skill__lookup", Parameters: map[string]interface{}{"type": "object"}}}},
		ToolCallConfig{MaxTurns: 3, MaxToolCalls: 2, SystemPrompt: "Use tools carefully."},
		func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
			assert.Equal(t, "worker:skill:lookup", target)
			assert.Equal(t, map[string]interface{}{"ticket": "123"}, input)
			return map[string]interface{}{"status": "open"}, nil
		},
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "resolved", result.Text())
	require.NotNil(t, result.Response)
	require.NotNil(t, result.Trace)
	assert.Equal(t, "worker:skill:lookup", result.Trace.Calls[0].ToolName)
}

func TestExecuteToolCallLoopResult_AppliesPromptConfig(t *testing.T) {
	t.Run("custom limit message", func(t *testing.T) {
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
									ID:       "call-1",
									Type:     "function",
									Function: ToolCallFunction{Name: "lookup", Arguments: `{"id":"1"}`},
								},
								{
									ID:       "call-2",
									Type:     "function",
									Function: ToolCallFunction{Name: "lookup", Arguments: `{"id":"2"}`},
								},
							},
						},
					}},
				}))
			case 2:
				require.Len(t, req.Messages, 4)
				var toolMessage map[string]string
				require.NoError(t, json.Unmarshal([]byte(req.Messages[3].Content[0].Text), &toolMessage))
				assert.Equal(t, "custom limit reached", toolMessage["error"])
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{Message: Message{
						Role:    "assistant",
						Content: []ContentPart{{Type: "text", Text: "final after limit"}},
					}}},
				}))
			default:
				t.Fatalf("unexpected request %d", count)
			}
		})

		resp, trace, err := client.ExecuteToolCallLoop(
			context.Background(),
			[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "lookup"}}}},
			[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
			ToolCallConfig{
				MaxTurns:     3,
				MaxToolCalls: 1,
				PromptConfig: &PromptConfig{ToolCallLimitReached: "custom limit reached"},
			},
			func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
				assert.Equal(t, "lookup", target)
				return map[string]interface{}{"ok": true}, nil
			},
		)

		require.NoError(t, err)
		assert.Equal(t, "final after limit", resp.Text())
		assert.Equal(t, "final after limit", trace.FinalResponse)
	})

	t.Run("custom result formatter", func(t *testing.T) {
		var requestCount atomic.Int32
		client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
			count := requestCount.Add(1)
			if count == 1 {
				require.NoError(t, json.NewEncoder(w).Encode(Response{
					Choices: []Choice{{
						Message: Message{
							Role: "assistant",
							ToolCalls: []ToolCall{{
								ID:   "call-1",
								Type: "function",
								Function: ToolCallFunction{
									Name:      "worker__skill__lookup",
									Arguments: `{"id":"1"}`,
								},
							}},
						},
					}},
				}))
				return
			}

			var req Request
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			require.Len(t, req.Messages, 3)
			assert.Equal(t, "tool=worker:skill:lookup status=open", req.Messages[2].Content[0].Text)
			require.NoError(t, json.NewEncoder(w).Encode(Response{
				Choices: []Choice{{Message: Message{
					Role:    "assistant",
					Content: []ContentPart{{Type: "text", Text: "done"}},
				}}},
			}))
		})

		result, err := client.ExecuteToolCallLoopResult(
			context.Background(),
			[]Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "lookup"}}}},
			[]ToolDefinition{{Type: "function", Function: ToolFunction{Name: "worker__skill__lookup"}}},
			ToolCallConfig{
				MaxTurns:     2,
				MaxToolCalls: 2,
				PromptConfig: &PromptConfig{
					ToolResultFormatter: func(toolName string, result map[string]interface{}) interface{} {
						return "tool=" + toolName + " status=" + result["status"].(string)
					},
				},
			},
			func(_ context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
				assert.Equal(t, "worker:skill:lookup", target)
				assert.Equal(t, map[string]interface{}{"id": "1"}, input)
				return map[string]interface{}{"status": "open"}, nil
			},
		)

		require.NoError(t, err)
		assert.Equal(t, "done", result.Text())
	})
}

func TestEncodeToolContent_AllCases(t *testing.T) {
    // string
    assert.Equal(t, "hello", encodeToolContent("hello"))

    // []byte
    assert.Equal(t, "hi", encodeToolContent([]byte("hi")))

    // JSON
    out := encodeToolContent(map[string]string{"a": "b"})
    assert.Contains(t, out, `"a":"b"`)

    // invalid (marshal error)
    ch := make(chan int)
    assert.Equal(t, "{}", encodeToolContent(ch))
}

func TestNormalizeToolParameters_AllCases(t *testing.T) {
    // nil
    out := normalizeToolParameters(nil)
    assert.Equal(t, "object", out["type"])

    // already valid
    in := map[string]interface{}{"type": "object"}
    out = normalizeToolParameters(in)
    assert.Equal(t, in, out)

    // missing type
    in = map[string]interface{}{"field": "value"}
    out = normalizeToolParameters(in)
    assert.Equal(t, "object", out["type"])
}

func TestExecuteToolCallLoopResult_ErrorFormatter(t *testing.T) {
    var requestCount atomic.Int32

    client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
        count := requestCount.Add(1)

        if count == 1 {
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
            return
        }

        var req Request
        require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

        assert.Contains(t, req.Messages[2].Content[0].Text, "custom-error")

        require.NoError(t, json.NewEncoder(w).Encode(Response{
            Choices: []Choice{{Message: Message{
                Role:    "assistant",
                Content: []ContentPart{{Type: "text", Text: "done"}},
            }}},
        }))
    })

    _, err := client.ExecuteToolCallLoopResult(
        context.Background(),
        []Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "lookup"}}}},
        []ToolDefinition{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
        ToolCallConfig{
            MaxTurns:     2,
            MaxToolCalls: 2,
            PromptConfig: &PromptConfig{
                ToolErrorFormatter: func(tool string, err error) interface{} {
                    return map[string]string{"error": "custom-error"}
                },
            },
        },
        func(context.Context, string, map[string]interface{}) (map[string]interface{}, error) {
            return nil, assert.AnError
        },
    )

    require.NoError(t, err)
}

func TestExecuteToolCallLoopResult_NoToolCalls(t *testing.T) {
    client := newToolLoopClient(t, func(w http.ResponseWriter, r *http.Request) {
        require.NoError(t, json.NewEncoder(w).Encode(Response{
            Choices: []Choice{{
                Message: Message{
                    Role:    "assistant",
                    Content: []ContentPart{{Type: "text", Text: "direct"}},
                },
                FinishReason: "stop",
            }},
        }))
    })

    result, err := client.ExecuteToolCallLoopResult(
        context.Background(),
        []Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "hi"}}}},
        nil,
        DefaultToolCallConfig(),
        nil,
    )

    require.NoError(t, err)
    assert.Equal(t, "direct", result.Text())
}

func TestSanitizeToolName_NoChange(t *testing.T) {
    name := "simpletool"
    assert.Equal(t, name, sanitizeToolName(name))
}

func TestUnsanitizeToolName_Standalone(t *testing.T) {
    assert.Equal(t, "worker:skill:lookup", unsanitizeToolName("worker__skill__lookup"))
}
func TestSanitizeToolName_MultipleColons(t *testing.T) {
    name := "a:b:c:d"
    sanitized := sanitizeToolName(name)
    assert.Equal(t, "a__b__c__d", sanitized)
    assert.Equal(t, name, unsanitizeToolName(sanitized))
}

func TestResolvePromptConfig_Nil(t *testing.T) {
    cfg := resolvePromptConfig(nil)
    assert.NotNil(t, cfg)
    assert.NotEmpty(t, cfg.ToolCallLimitReached)
}