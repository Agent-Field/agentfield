package ui

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/gin-gonic/gin"
)

type ErrorResponse = handlers.ErrorResponse

func RespondError(c *gin.Context, status int, message string) {
	handlers.RespondError(c, status, message)
}

func RespondBadRequest(c *gin.Context, message string) {
	handlers.RespondBadRequest(c, message)
}

func RespondNotFound(c *gin.Context, message string) {
	handlers.RespondNotFound(c, message)
}

func RespondInternalError(c *gin.Context, message string) {
	handlers.RespondInternalError(c, message)
}
