package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// ToolCallConfig configures the tool-call execution loop.
type ToolCallConfig struct {
	MaxTurns     int
	MaxToolCalls int
	SystemPrompt string
	PromptConfig *PromptConfig
}

// DefaultToolCallConfig returns default configuration for the tool-call loop.
func DefaultToolCallConfig() ToolCallConfig {
	pc := DefaultPromptConfig()
	return ToolCallConfig{
		MaxTurns:     10,
		MaxToolCalls: 25,
		PromptConfig: &pc,
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

// PromptConfig customizes the tool-call loop's tool-facing prompt content.
type PromptConfig struct {
	// ToolCallLimitReached is sent back to the model when it asks for more tool
	// calls than MaxToolCalls allows.
	ToolCallLimitReached string
	// ToolErrorFormatter formats tool execution failures before they are sent
	// back to the model. It may return a raw string or any JSON-marshable value.
	ToolErrorFormatter func(toolName string, err error) interface{}
	// ToolResultFormatter formats successful tool results before they are sent
	// back to the model. It may return a raw string or any JSON-marshable value.
	ToolResultFormatter func(toolName string, result map[string]interface{}) interface{}
}

// ToolCallTrace records the full trace of a tool-call loop.
type ToolCallTrace struct {
	Calls          []ToolCallRecord
	TotalTurns     int
	TotalToolCalls int
	FinalResponse  string
}

// ToolCallResult wraps a tool-call response and its execution trace.
type ToolCallResult struct {
	Response *Response
	Trace    *ToolCallTrace
}

// Text returns the final text response from the tool-call loop.
func (r *ToolCallResult) Text() string {
	if r == nil || r.Trace == nil {
		return ""
	}
	return r.Trace.FinalResponse
}

// CallFunc is the function signature for dispatching tool calls.
// It maps to agent.Call(ctx, target, input).
type CallFunc func(ctx context.Context, target string, input map[string]interface{}) (map[string]interface{}, error)

// DefaultPromptConfig returns the default prompt content for tool execution.
func DefaultPromptConfig() PromptConfig {
	return PromptConfig{
		ToolCallLimitReached: "Tool call limit reached. Please provide a final response.",
		ToolErrorFormatter: func(toolName string, err error) interface{} {
			return map[string]string{
				"error": err.Error(),
				"tool":  toolName,
			}
		},
		ToolResultFormatter: func(_ string, result map[string]interface{}) interface{} {
			return result
		},
	}
}

// CapabilityToToolDefinition converts a ReasonerCapability or SkillCapability
// to a ToolDefinition.
func CapabilityToToolDefinition(cap interface{}) ToolDefinition {
	switch c := cap.(type) {
	case types.ReasonerCapability:
		return capabilityToTool(c.InvocationTarget, c.Description, c.InputSchema)
	case types.SkillCapability:
		return capabilityToTool(c.InvocationTarget, c.Description, c.InputSchema)
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
	result, err := c.ExecuteToolCallLoopResult(ctx, messages, tools, config, callFn, opts...)
	if result == nil {
		return nil, nil, err
	}
	return result.Response, result.Trace, err
}

// ExecuteToolCallLoopResult runs the tool-call loop and returns a wrapped
// response plus the full execution trace.
func (c *Client) ExecuteToolCallLoopResult(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	config ToolCallConfig,
	callFn CallFunc,
	opts ...Option,
) (*ToolCallResult, error) {
	trace := &ToolCallTrace{}
	result := &ToolCallResult{Trace: trace}
	totalCalls := 0
	promptConfig := resolvePromptConfig(config.PromptConfig)
	effectiveOpts := opts
	if strings.TrimSpace(config.SystemPrompt) != "" {
		effectiveOpts = append([]Option{WithSystem(config.SystemPrompt)}, opts...)
	}
	loopMessages := append([]Message(nil), messages...)

	for turn := 0; turn < config.MaxTurns; turn++ {
		trace.TotalTurns = turn + 1

		req, err := c.buildToolCallRequest(loopMessages, tools, true, effectiveOpts)
		if err != nil {
			return result, err
		}

		resp, err := c.doRequest(ctx, req)
		if err != nil {
			return result, fmt.Errorf("LLM call failed: %w", err)
		}

		if !resp.HasToolCalls() {
			result.Response = resp
			trace.FinalResponse = resp.Text()
			return result, nil
		}

		// Append assistant message with tool calls
		loopMessages = append(loopMessages, resp.Choices[0].Message)

		// Execute each tool call
		for _, tc := range resp.ToolCalls() {
			if totalCalls >= config.MaxToolCalls {
				loopMessages = append(loopMessages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: encodeToolContent(map[string]string{"error": promptConfig.ToolCallLimitReached})}},
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
			toolName := unsanitizeToolName(tc.Function.Name)

			record := ToolCallRecord{
				ToolName:  toolName,
				Arguments: args,
				Turn:      turn,
			}

			start := time.Now()
			toolResult, err := callFn(ctx, toolName, args)
			record.LatencyMs = float64(time.Since(start).Milliseconds())

			if err != nil {
				record.Error = err.Error()
				loopMessages = append(loopMessages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: encodeToolContent(promptConfig.ToolErrorFormatter(toolName, err))}},
					ToolCallID: tc.ID,
				})
			} else {
				record.Result = toolResult
				loopMessages = append(loopMessages, Message{
					Role:       "tool",
					Content:    []ContentPart{{Type: "text", Text: encodeToolContent(promptConfig.ToolResultFormatter(toolName, toolResult))}},
					ToolCallID: tc.ID,
				})
			}

			trace.Calls = append(trace.Calls, record)
		}

		// If tool call limit reached, make one final call without tools
		if totalCalls >= config.MaxToolCalls {
			req, err := c.buildToolCallRequest(loopMessages, nil, false, effectiveOpts)
			if err != nil {
				return result, err
			}
			resp, err := c.doRequest(ctx, req)
			if err != nil {
				return result, fmt.Errorf("final LLM call failed: %w", err)
			}
			result.Response = resp
			trace.FinalResponse = resp.Text()
			return result, nil
		}
	}

	// Max turns reached - make final call without tools
	req, err := c.buildToolCallRequest(loopMessages, nil, false, effectiveOpts)
	if err != nil {
		return result, err
	}
	resp, err := c.doRequest(ctx, req)
	if err != nil {
		return result, fmt.Errorf("final LLM call failed: %w", err)
	}
	result.Response = resp
	trace.FinalResponse = resp.Text()
	trace.TotalTurns = config.MaxTurns
	return result, nil
}

func capabilityToTool(invocationTarget string, description *string, inputSchema map[string]interface{}) ToolDefinition {
	desc := ""
	if description != nil {
		desc = *description
	}
	if desc == "" {
		desc = "Call " + invocationTarget
	}

	return ToolDefinition{
		Type: "function",
		Function: ToolFunction{
			Name:        sanitizeToolName(invocationTarget),
			Description: desc,
			Parameters:  normalizeToolParameters(inputSchema),
		},
	}
}

func normalizeToolParameters(inputSchema map[string]interface{}) map[string]interface{} {
	if inputSchema == nil {
		return map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
	}
	if _, ok := inputSchema["type"]; ok {
		return inputSchema
	}
	return map[string]interface{}{"type": "object", "properties": inputSchema}
}

func sanitizeToolName(name string) string {
	return strings.ReplaceAll(name, ":", "__")
}

func unsanitizeToolName(name string) string {
	return strings.ReplaceAll(name, "__", ":")
}

func resolvePromptConfig(config *PromptConfig) PromptConfig {
	resolved := DefaultPromptConfig()
	if config == nil {
		return resolved
	}
	if strings.TrimSpace(config.ToolCallLimitReached) != "" {
		resolved.ToolCallLimitReached = config.ToolCallLimitReached
	}
	if config.ToolErrorFormatter != nil {
		resolved.ToolErrorFormatter = config.ToolErrorFormatter
	}
	if config.ToolResultFormatter != nil {
		resolved.ToolResultFormatter = config.ToolResultFormatter
	}
	return resolved
}

func encodeToolContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return "{}"
		}
		return string(data)
	}
}

func (c *Client) buildToolCallRequest(messages []Message, tools []ToolDefinition, includeTools bool, opts []Option) (*Request, error) {
	req := &Request{
		Messages:    messages,
		Model:       c.config.Model,
		Temperature: &c.config.Temperature,
		MaxTokens:   &c.config.MaxTokens,
	}
	if includeTools {
		req.Tools = tools
		req.ToolChoice = "auto"
	}

	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, fmt.Errorf("apply option: %w", err)
		}
	}

	return req, nil
}
