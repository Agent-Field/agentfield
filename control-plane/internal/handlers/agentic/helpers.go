package agentic

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// AgenticResponse is the standard envelope for all agentic API responses.
type AgenticResponse struct {
	OK    bool        `json:"ok"`
	Data  interface{} `json:"data,omitempty"`
	Error *ErrorInfo  `json:"error,omitempty"`
	Meta  *MetaInfo   `json:"meta,omitempty"`
}

// ErrorInfo provides structured error information.
type ErrorInfo struct {
	Code    string      `json:"code"`
	Message string      `json:"message"`
	Details interface{} `json:"details,omitempty"`
}

// MetaInfo provides response metadata.
type MetaInfo struct {
	TokenEstimate int    `json:"token_estimate,omitempty"`
	Hint          string `json:"hint,omitempty"`
}

// respondOK sends a successful agentic response.
func respondOK(c *gin.Context, data interface{}) {
	resp := AgenticResponse{OK: true, Data: data}
	tokens := estimateTokens(data)
	if tokens > 0 {
		c.Header("X-Token-Estimate", fmt.Sprintf("%d", tokens))
	}
	c.JSON(http.StatusOK, resp)
}

// respondError sends an error agentic response.
func respondError(c *gin.Context, status int, code, message string) {
	resp := AgenticResponse{
		OK:    false,
		Error: &ErrorInfo{Code: code, Message: message},
	}
	c.JSON(status, resp)
}

// respondErrorWithDetails sends an error with additional details.
func respondErrorWithDetails(c *gin.Context, status int, code, message string, details interface{}) {
	resp := AgenticResponse{
		OK:    false,
		Error: &ErrorInfo{Code: code, Message: message, Details: details},
	}
	c.JSON(status, resp)
}

// estimateTokens gives a rough token count for response sizing.
func estimateTokens(v interface{}) int {
	// Rough estimate: JSON marshal length / 4
	s := fmt.Sprintf("%v", v)
	return len(s) / 4
}

// getAuthLevel extracts the auth level from the gin context.
// Falls back to "public" if not set.
func getAuthLevel(c *gin.Context) string {
	level, exists := c.Get("auth_level")
	if !exists {
		// Check if API key header is present as a fallback
		if c.GetHeader("X-API-Key") != "" || strings.HasPrefix(c.GetHeader("Authorization"), "Bearer ") || c.Query("api_key") != "" {
			return "api_key"
		}
		return "public"
	}
	if s, ok := level.(string); ok {
		return s
	}
	return "public"
}

// getIntQuery gets an integer query parameter with a default value.
func getIntQuery(c *gin.Context, key string, defaultVal int) int {
	val := c.Query(key)
	if val == "" {
		return defaultVal
	}
	var result int
	if _, err := fmt.Sscanf(val, "%d", &result); err != nil {
		return defaultVal
	}
	if result < 0 {
		return defaultVal
	}
	return result
}
