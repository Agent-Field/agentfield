package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// ErrorResponse defines the shared structure for handler error responses.
type ErrorResponse struct {
	Error   string      `json:"error"`
	Code    int         `json:"code,omitempty"`
	Details interface{} `json:"details,omitempty"`
}

// RespondError writes a basic error response without changing existing payload shapes.
func RespondError(c *gin.Context, status int, message string) {
	c.JSON(status, ErrorResponse{Error: message})
}

// RespondBadRequest writes a 400 error response.
func RespondBadRequest(c *gin.Context, message string) {
	RespondError(c, http.StatusBadRequest, message)
}

// RespondNotFound writes a 404 error response.
func RespondNotFound(c *gin.Context, message string) {
	RespondError(c, http.StatusNotFound, message)
}

// RespondInternalError writes a 500 error response.
func RespondInternalError(c *gin.Context, message string) {
	RespondError(c, http.StatusInternalServerError, message)
}
