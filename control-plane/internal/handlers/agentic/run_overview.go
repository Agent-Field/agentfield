package agentic

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// RunOverviewHandler returns everything about a workflow run in one call.
func RunOverviewHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		runID := c.Param("run_id")
		if runID == "" {
			respondError(c, http.StatusBadRequest, "missing_run_id", "run_id path parameter is required")
			return
		}

		ctx := c.Request.Context()

		// Get all executions for this run
		filter := types.ExecutionFilter{
			RunID: &runID,
		}
		executions, err := store.QueryExecutionRecords(ctx, filter)
		if err != nil {
			respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
			return
		}

		if len(executions) == 0 {
			respondError(c, http.StatusNotFound, "run_not_found", "no executions found for run "+runID)
			return
		}

		// Collect unique agents
		agentSet := make(map[string]bool)
		statusCounts := make(map[string]int)
		allNotes := make([]types.ExecutionNote, 0)

		for _, e := range executions {
			if e.AgentNodeID != "" {
				agentSet[e.AgentNodeID] = true
			}
			statusCounts[e.Status]++
			// Collect notes from each execution
			if len(e.Notes) > 0 {
				allNotes = append(allNotes, e.Notes...)
			}
		}

		agents := make([]string, 0, len(agentSet))
		for a := range agentSet {
			agents = append(agents, a)
		}

		respondOK(c, gin.H{
			"run_id":     runID,
			"executions": executions,
			"agents":     agents,
			"summary": gin.H{
				"total_executions": len(executions),
				"status_counts":    statusCounts,
				"unique_agents":    len(agents),
			},
			"notes": allNotes,
		})
	}
}
