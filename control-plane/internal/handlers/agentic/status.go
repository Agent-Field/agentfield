package agentic

import (
	"context"
	"runtime"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// StatusHandler returns system status overview.
func StatusHandler(store storage.StorageProvider) gin.HandlerFunc {
	startTime := time.Now()

	return func(c *gin.Context) {
		ctx := c.Request.Context()

		// Get agent counts
		agents, _ := store.ListAgents(ctx, types.AgentFilters{})
		totalAgents := len(agents)
		activeAgents := 0
		for _, a := range agents {
			// Check agent lifecycle status
			if a.LifecycleStatus == types.AgentStatusReady || a.LifecycleStatus == types.AgentStatusStarting {
				activeAgents++
			}
		}

		// Get execution stats (last 24h)
		stats := getExecutionStats(ctx, store)

		// Health check
		healthOK := true
		if err := store.HealthCheck(ctx); err != nil {
			healthOK = false
		}

		respondOK(c, gin.H{
			"health": gin.H{
				"status":  boolToStatus(healthOK),
				"storage": boolToStatus(healthOK),
			},
			"agents": gin.H{
				"total":  totalAgents,
				"active": activeAgents,
			},
			"executions_24h": stats,
			"server": gin.H{
				"uptime_seconds": int(time.Since(startTime).Seconds()),
				"go_version":     runtime.Version(),
				"goroutines":     runtime.NumGoroutine(),
			},
		})
	}
}

func getExecutionStats(ctx context.Context, store storage.StorageProvider) gin.H {
	now := time.Now()
	since := now.Add(-24 * time.Hour)

	filter := types.ExecutionFilter{
		StartTime: &since,
	}

	execs, _ := store.QueryExecutionRecords(ctx, filter)
	total := len(execs)
	statusCounts := make(map[string]int)
	for _, e := range execs {
		statusCounts[e.Status]++
	}

	return gin.H{
		"total":    total,
		"statuses": statusCounts,
		"since":    since.Format(time.RFC3339),
	}
}

func boolToStatus(ok bool) string {
	if ok {
		return "healthy"
	}
	return "unhealthy"
}
