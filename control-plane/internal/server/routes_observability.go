package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/ui"

	"github.com/gin-gonic/gin"
)

// registerObservabilityRoutes installs the settings/observability-webhook/*
// tree used by the UI to configure and operate the external observability
// forwarder (get/set/delete config, status, redrive, and dead-letter queue).
func (s *AgentFieldServer) registerObservabilityRoutes(agentAPI *gin.RouterGroup) {
	settings := agentAPI.Group("/settings")
	{
		obsHandler := ui.NewObservabilityWebhookHandler(s.storage, s.observabilityForwarder)
		settings.GET("/observability-webhook", obsHandler.GetWebhookHandler)
		settings.POST("/observability-webhook", obsHandler.SetWebhookHandler)
		settings.DELETE("/observability-webhook", obsHandler.DeleteWebhookHandler)
		settings.GET("/observability-webhook/status", obsHandler.GetStatusHandler)
		settings.POST("/observability-webhook/redrive", obsHandler.RedriveHandler)
		settings.GET("/observability-webhook/dlq", obsHandler.GetDeadLetterQueueHandler)
		settings.DELETE("/observability-webhook/dlq", obsHandler.ClearDeadLetterQueueHandler)
	}
}
