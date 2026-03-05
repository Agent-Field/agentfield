package ai

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Response represents the API response from OpenAI/OpenRouter.
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk represents a streaming response chunk.
type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []StreamDelta `json:"choices"`
}

// StreamDelta represents a delta in a streaming response.
type StreamDelta struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// MessageDelta represents the incremental message content.
type MessageDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ErrorResponse represents an error from the API.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// HasToolCalls returns true if the response contains tool calls.
func (r *Response) HasToolCalls() bool {
	if len(r.Choices) == 0 {
		return false
	}
	return len(r.Choices[0].Message.ToolCalls) > 0
}

// ToolCalls returns the tool calls from the first choice, or nil.
func (r *Response) ToolCalls() []ToolCall {
	if len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].Message.ToolCalls
}

// Text returns the text content from the first choice.
func (r *Response) Text() string {
	if len(r.Choices) == 0 || len(r.Choices[0].Message.Content) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, part := range r.Choices[0].Message.Content {
		if part.Type == "text" {
			sb.WriteString(part.Text)
		}
	}

	return sb.String()
}

// JSON parses the response content as JSON into the provided destination.
func (r *Response) JSON(dest interface{}) error {
	content := r.Text()
	if content == "" {
		return fmt.Errorf("empty response content")
	}
	return json.Unmarshal([]byte(content), dest)
}

// Into is an alias for JSON for ergonomic usage.
func (r *Response) Into(dest interface{}) error {
	return r.JSON(dest)
}
