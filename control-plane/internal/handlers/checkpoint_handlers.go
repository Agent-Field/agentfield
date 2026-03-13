package handlers

import (
	"fmt"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
)

func GetCheckpointHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		executionID := c.Param("execution_id")
		if executionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
			return
		}

		checkpoint, err := store.GetCheckpoint(c.Request.Context(), executionID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to get checkpoint")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to look up checkpoint"})
			return
		}

		if checkpoint == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("checkpoint for execution %s not found", executionID)})
			return
		}

		c.JSON(http.StatusOK, checkpoint)
	}
}
