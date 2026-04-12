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

// APIError represents a structured API error with full context for retry/fallback decisions.
// Formerly all non-2xx responses were flattened into fmt.Errorf, losing HTTP status code,
// provider error type, and structured error code — preventing callers from distinguishing
// auth (401) vs rate-limit (429) vs content filter (400) vs server error (5xx).
type APIError struct {
	HTTPStatus int
	Type       string // e.g. "invalid_request_error", "authentication_error", "rate_limit_error"
	Code       string // provider-specific error code
	Message    string // human-readable message
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API error (%d): [%s] %s", e.HTTPStatus, e.Type, e.Message)
}

// IsRetryable returns true if the error represents a transient condition worth retrying.
// 429 Rate Limited and 5xx Server Errors are retryable; 400, 401, 403, 404 are not.
func (e *APIError) IsRetryable() bool {
	return e.HTTPStatus == 429 || e.HTTPStatus >= 500
}

// IsAuthError returns true if this is an authentication error (401).
func (e *APIError) IsAuthError() bool {
	return e.HTTPStatus == 401
}

// IsRateLimited returns true if this is a rate limit error (429).
func (e *APIError) IsRateLimited() bool {
	return e.HTTPStatus == 429
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
