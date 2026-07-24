package handlers

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ListNodesHandler handles listing all registered agent nodes
func ListNodesHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		// Parse query parameters for filtering
		filters := types.AgentFilters{}

		// Check for health_status filter parameter
		if healthStatusParam := c.Query("health_status"); healthStatusParam != "" {
			healthStatus := types.HealthStatus(healthStatusParam)
			filters.HealthStatus = &healthStatus
		} else {
			// Default to showing only active nodes unless explicitly requested otherwise
			activeStatus := types.HealthStatusActive
			filters.HealthStatus = &activeStatus
		}

		// Check for team_id filter parameter
		if teamID := c.Query("team_id"); teamID != "" {
			filters.TeamID = &teamID
		}

		// Check for group_id filter parameter
		if groupID := c.Query("group_id"); groupID != "" {
			filters.GroupID = &groupID
		}

		// Check for show_all parameter to override default active filter
		if showAll := c.Query("show_all"); showAll == "true" {
			filters.HealthStatus = nil // Remove health status filter to show all nodes
		}

		// Get filtered nodes from storage
		nodes, err := storageProvider.ListAgents(ctx, filters)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get nodes"})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"nodes":   nodes,
			"count":   len(nodes),
			"filters": filters,
		})
	}
}
