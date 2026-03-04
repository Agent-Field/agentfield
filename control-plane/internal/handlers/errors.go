package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// ErrorResponse is the standard error response format.
type ErrorResponse struct {
	Error string `json:"error"`
}

// RespondError sends a JSON error response with the given status code.
func RespondError(c *gin.Context, status int, message string) {
	c.JSON(status, ErrorResponse{Error: message})
}

// RespondBadRequest sends a 400 Bad Request error response.
func RespondBadRequest(c *gin.Context, message string) {
	RespondError(c, http.StatusBadRequest, message)
}

// RespondNotFound sends a 404 Not Found error response.
func RespondNotFound(c *gin.Context, message string) {
	RespondError(c, http.StatusNotFound, message)
}

// RespondInternalError sends a 500 Internal Server Error response.
func RespondInternalError(c *gin.Context, message string) {
	RespondError(c, http.StatusInternalServerError, message)
}
