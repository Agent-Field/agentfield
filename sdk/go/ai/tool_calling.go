package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// ToolCallConfig configures the tool-call execution loop.
type ToolCallConfig struct {
	MaxTurns     int
	MaxToolCalls int
}

// DefaultToolCallConfig returns default configuration for the tool-call loop.
func DefaultToolCallConfig() ToolCallConfig {
	return ToolCallConfig{
		MaxTurns:     10,
		MaxToolCalls: 25,
	}
}

// ToolCallRecord records a single tool call for observability.
type ToolCallRecord struct {
	ToolName  string
	Arguments map[string]interface{}
	Result    map[string]interface{}
	Error     string
	LatencyMs float64
	Turn      int
}

// ToolCallTrace records the full trace of a tool-call loop.
type ToolCallTrace struct {
	Calls          []ToolCallRecord
	TotalTurns     int
	TotalToolCalls int
	FinalResponse  string
}

// CallFunc is the function signature for dispatching tool calls.
// It maps to agent.Call(ctx, target, input).
type CallFunc func(ctx context.Context, target string, input map[string]interface{}) (map[string]interface{}, error)

// CapabilityToToolDefinition converts a ReasonerCapability to a ToolDefinition.
func CapabilityToToolDefinition(cap interface{}) ToolDefinition {
	switch c := cap.(type) {
	case types.ReasonerCapability:
		desc := ""
		if c.Description != nil {
			desc = *c.Description
		}
		if desc == "" {
			desc = "Call " + c.InvocationTarget
		}
		params := c.InputSchema
		if params == nil {
			params = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		}
		if _, ok := params["type"]; !ok {
			params = map[string]interface{}{"type": "object", "properties": params}
		}
		return ToolDefinition{
			Type: "function",
			Function: ToolFunction{
				Name:        c.InvocationTarget,
				Description: desc,
				Parameters:  params,
			},
		}
	case types.SkillCapability:
		desc := ""
		if c.Description != nil {
			desc = *c.Description
		}
		if desc == "" {
			desc = "Call " + c.InvocationTarget
		}
		params := c.InputSchema
		if params == nil {
			params = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		}
		if _, ok := params["type"]; !ok {
			params = map[string]interface{}{"type": "object", "properties": params}
		}
		return ToolDefinition{
			Type: "function",
			Function: ToolFunction{
				Name:        c.InvocationTarget,
				Description: desc,
				Parameters:  params,
			},
		}
	default:
		return ToolDefinition{}
	}
}

// CapabilitiesToToolDefinitions converts discovery capabilities to tool definitions.
func CapabilitiesToToolDefinitions(capabilities []types.AgentCapability) []ToolDefinition {
	var tools []ToolDefinition
	for _, agent := range capabilities {
		for _, r := range agent.Reasoners {
			tools = append(tools, CapabilityToToolDefinition(r))
		}
		for _, s := range agent.Skills {
			tools = append(tools, CapabilityToToolDefinition(s))
		}
	}
	return tools
}

// ExecuteToolCallLoop runs the LLM tool-call loop: send messages with tools,
// dispatch any tool calls via callFn, feed results back, repeat until the LLM
// produces a final text response or limits are reached.
func (c *Client) ExecuteToolCallLoop(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	config ToolCallConfig,
	callFn CallFunc,
	opts ...Option,
) (*Response, *ToolCallTrace, error) {
	trace := &ToolCallTrace{}
	totalCalls := 0

	for turn := 0; turn < config.MaxTurns; turn++ {
		trace.TotalTurns = turn + 1

		// Build request
		req := &Request{
			Messages:    messages,
			Model:       c.config.Model,
			Temperature: &c.config.Temperature,
			MaxTokens:   &c.config.MaxTokens,
			Tools:       tools,
			ToolChoice:  "auto",
		}

		for _, opt := range opts {
			if err := opt(req); err != nil {
				return nil, trace, fmt.Errorf("apply option: %w", err)
			}
		}

		resp, err := c.doRequest(ctx, req)
		if err != nil {
			return nil, trace, fmt.Errorf("LLM call failed: %w", err)
		}

		if !resp.HasToolCalls() {
			trace.FinalResponse = resp.Text()
			return resp, trace, nil
		}

		// Append assistant message with tool calls
		messages = append(messages, resp.Choices[0].Message)

		// Execute each tool call
		for _, tc := range resp.ToolCalls() {
			if totalCalls >= config.MaxToolCalls {
				messages = append(messages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: `{"error": "Tool call limit reached. Please provide a final response."}`}},
					ToolCallID: tc.ID,
				})
				continue
			}

			totalCalls++
			trace.TotalToolCalls = totalCalls

			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = map[string]interface{}{}
			}

			record := ToolCallRecord{
				ToolName:  tc.Function.Name,
				Arguments: args,
				Turn:      turn,
			}

			start := time.Now()
			result, err := callFn(ctx, tc.Function.Name, args)
			record.LatencyMs = float64(time.Since(start).Milliseconds())

			if err != nil {
				record.Error = err.Error()
				errJSON, _ := json.Marshal(map[string]string{
					"error": err.Error(),
					"tool":  tc.Function.Name,
				})
				messages = append(messages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: string(errJSON)}},
					ToolCallID: tc.ID,
				})
			} else {
				record.Result = result
				resultJSON, _ := json.Marshal(result)
				messages = append(messages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: string(resultJSON)}},
					ToolCallID: tc.ID,
				})
			}

			trace.Calls = append(trace.Calls, record)
		}

		// If tool call limit reached, make one final call without tools
		if totalCalls >= config.MaxToolCalls {
			req := &Request{
				Messages:    messages,
				Model:       c.config.Model,
				Temperature: &c.config.Temperature,
				MaxTokens:   &c.config.MaxTokens,
			}
			for _, opt := range opts {
				if err := opt(req); err != nil {
					return nil, trace, fmt.Errorf("apply option: %w", err)
				}
			}
			resp, err := c.doRequest(ctx, req)
			if err != nil {
				return nil, trace, fmt.Errorf("final LLM call failed: %w", err)
			}
			trace.FinalResponse = resp.Text()
			return resp, trace, nil
		}
	}

	// Max turns reached - make final call without tools
	req := &Request{
		Messages:    messages,
		Model:       c.config.Model,
		Temperature: &c.config.Temperature,
		MaxTokens:   &c.config.MaxTokens,
	}
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, trace, fmt.Errorf("apply option: %w", err)
		}
	}
	resp, err := c.doRequest(ctx, req)
	if err != nil {
		return nil, trace, fmt.Errorf("final LLM call failed: %w", err)
	}
	trace.FinalResponse = resp.Text()
	trace.TotalTurns = config.MaxTurns
	return resp, trace, nil
}
