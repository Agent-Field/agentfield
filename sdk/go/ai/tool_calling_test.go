package ai

import (
	"encoding/json"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
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
