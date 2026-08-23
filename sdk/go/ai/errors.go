package ai

import (
	"encoding/json"
	"fmt"
)

const maxAPIErrorBody = 4 << 10

// APIError describes a non-success response from an AI provider.
//
// Body contains the raw response when the provider did not return the
// standard ErrorResponse shape. It is capped to keep proxy or provider error
// pages from being retained without a bound.
type APIError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
	Body       []byte
}

// Error returns a human-readable description including the HTTP status.
func (e *APIError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("API error (%d): %s", e.StatusCode, e.Message)
	}
	if len(e.Body) > 0 {
		return fmt.Sprintf("API error (%d): %s", e.StatusCode, e.Body)
	}
	return fmt.Sprintf("API error (%d)", e.StatusCode)
}

// Is matches API errors by HTTP status code.
func (e *APIError) Is(target error) bool {
	t, ok := target.(*APIError)
	return ok && e.StatusCode == t.StatusCode
}

func newAPIError(statusCode int, body []byte) *APIError {
	apiErr := &APIError{StatusCode: statusCode}

	var response ErrorResponse
	if err := json.Unmarshal(body, &response); err == nil &&
		(response.Error.Message != "" || response.Error.Type != "" || response.Error.Code != "") {
		apiErr.Message = response.Error.Message
		apiErr.Type = response.Error.Type
		apiErr.Code = response.Error.Code
		return apiErr
	}

	if len(body) > maxAPIErrorBody {
		body = body[:maxAPIErrorBody]
	}
	apiErr.Body = append([]byte(nil), body...)
	return apiErr
}
