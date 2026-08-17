package agentic

import (
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

const (
	// agentSummaryRecentExecutionsLimit caps how many recent executions the
	// response carries. Unbounded, this endpoint inlines every execution of the
	// last 24h — a multi-megabyte reply from what callers use as a cheap
	// orientation call.
	agentSummaryRecentExecutionsLimit = 20

	// agentSummaryMetricsQueryLimit bounds the query itself. Metrics are computed
	// over every row it returns, so it sits far above the serialized list; it is
	// only a guardrail against pathological execution volume in the window.
	// Payload-free rows are cheap, which is what makes the wide read affordable.
	agentSummaryMetricsQueryLimit = 500
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

		// Get executions from the last 24h, newest first. Payloads are dropped:
		// this is an orientation surface, and the full input/result of a given
		// execution is one POST /api/v1/agentic/query away. Dropping them is what
		// lets the window stay wide enough for the metrics below to be exact
		// while only the newest few rows are serialized.
		since := time.Now().Add(-24 * time.Hour)
		filter := types.ExecutionFilter{
			AgentNodeID:     &agentID,
			StartTime:       &since,
			Limit:           agentSummaryMetricsQueryLimit,
			SortBy:          "started_at",
			SortDescending:  true,
			ExcludePayloads: true,
		}
		windowExecs, _ := store.QueryExecutionRecords(ctx, filter)

		// Compute metrics over the whole window, not just the rows we return.
		statusCounts := make(map[string]int)
		var totalDurationMs int64
		completedCount := 0
		for _, e := range windowExecs {
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

		// Only the newest few executions are serialized; the metrics above still
		// describe every row in the window.
		recentExecs := windowExecs
		if len(recentExecs) > agentSummaryRecentExecutionsLimit {
			recentExecs = recentExecs[:agentSummaryRecentExecutionsLimit]
		}

		respondOK(c, gin.H{
			"agent":             withResolvedReasonerDescriptions(agent),
			"recent_executions": recentExecs,
			"metrics_24h": gin.H{
				"total_executions": len(windowExecs),
				"status_counts":    statusCounts,
				"avg_duration_ms":  avgDurationMs,
				"completed_count":  completedCount,
			},
		})
	}
}

// withResolvedReasonerDescriptions returns a copy of the agent whose reasoners
// carry a resolved description (registered field, else the legacy agent-metadata
// map). Serializing the stored record directly skips that fallback, so an agent
// registered by an older SDK reads as undocumented here while
// /api/v1/discovery/capabilities shows its descriptions. The copy leaves the
// stored record — which storage may hand out from a cache — untouched.
func withResolvedReasonerDescriptions(agent *types.AgentNode) *types.AgentNode {
	if agent == nil || len(agent.Reasoners) == 0 {
		return agent
	}
	resolved := *agent
	resolved.Reasoners = make([]types.ReasonerDefinition, len(agent.Reasoners))
	copy(resolved.Reasoners, agent.Reasoners)
	for i := range resolved.Reasoners {
		resolved.Reasoners[i].Description = reasonerDescription(agent, resolved.Reasoners[i])
	}
	return &resolved
}
