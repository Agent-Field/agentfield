package agentic

import (
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// AgentSummaryHandler returns agent info plus recent executions and metrics.
func AgentSummaryHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		agentID := c.Param("agent_id")
		if agentID == "" {
			respondError(c, http.StatusBadRequest, "missing_agent_id", "agent_id path parameter is required")
			return
		}

		ctx := c.Request.Context()

		// Get agent info
		agent, err := store.GetAgent(ctx, agentID)
		if err != nil {
			respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
			return
		}
		if agent == nil {
			respondError(c, http.StatusNotFound, "agent_not_found", "agent "+agentID+" not found")
			return
		}

		// Get recent executions (last 24h)
		since := time.Now().Add(-24 * time.Hour)
		filter := types.ExecutionFilter{
			AgentNodeID: &agentID,
			StartTime:   &since,
		}
		recentExecs, _ := store.QueryExecutionRecords(ctx, filter)

		// Compute metrics
		statusCounts := make(map[string]int)
		var totalDurationMs int64
		completedCount := 0
		for _, e := range recentExecs {
			statusCounts[e.Status]++
			if e.Status == "completed" && e.CompletedAt != nil {
				if e.DurationMS != nil {
					totalDurationMs += *e.DurationMS
					completedCount++
				}
			}
		}

		var avgDurationMs int64
		if completedCount > 0 {
			avgDurationMs = totalDurationMs / int64(completedCount)
		}

		respondOK(c, gin.H{
			"agent":             agent,
			"recent_executions": recentExecs,
			"metrics_24h": gin.H{
				"total_executions": len(recentExecs),
				"status_counts":    statusCounts,
				"avg_duration_ms":  avgDurationMs,
				"completed_count":  completedCount,
			},
		})
	}
}
