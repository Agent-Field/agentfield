package handlers

import (
	"fmt"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// UpdateLifecycleStatusHandler handles lifecycle status updates from agent nodes
// Now integrates with the unified status management system
func UpdateLifecycleStatusHandler(storageProvider storage.StorageProvider, uiService *services.UIService, statusManager *services.StatusManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		var statusUpdate struct {
			LifecycleStatus string `json:"lifecycle_status" binding:"required"`
		}

		if err := c.ShouldBindJSON(&statusUpdate); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format: " + err.Error()})
			return
		}

		// Validate lifecycle status
		validStatuses := map[string]bool{
			string(types.AgentStatusStarting): true,
			string(types.AgentStatusReady):    true,
			string(types.AgentStatusDegraded): true,
			string(types.AgentStatusOffline):  true,
		}

		if !validStatuses[statusUpdate.LifecycleStatus] {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid lifecycle status"})
			return
		}

		// Verify node exists
		existingNode, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		// Protect pending_approval: only admin tag approval can transition out of this state
		newLifecycleStatus := types.AgentLifecycleStatus(statusUpdate.LifecycleStatus)
		if existingNode.LifecycleStatus == types.AgentStatusPendingApproval {
			logger.Logger.Debug().Msgf("⏸️ Rejecting lifecycle status update for node %s: agent is pending_approval (admin action required)", nodeID)
			c.JSON(http.StatusConflict, gin.H{
				"error":   "agent_pending_approval",
				"message": "Cannot update lifecycle status: agent is awaiting tag approval. Use admin approval endpoint instead.",
			})
			return
		}

		// Update through unified status system if available
		if statusManager != nil {
			if err := statusManager.UpdateFromHeartbeat(ctx, nodeID, &newLifecycleStatus, ""); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update unified status for node %s", nodeID)
				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update status"})
				return
			}
		} else {
			// Fallback to legacy update for backward compatibility
			if err := storageProvider.UpdateAgentLifecycleStatus(ctx, nodeID, newLifecycleStatus); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update lifecycle status for node %s", nodeID)
				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update lifecycle status"})
				return
			}
		}

		logger.Logger.Debug().Msgf("🔄 Lifecycle status updated for node %s: %s", nodeID, statusUpdate.LifecycleStatus)

		// Note: Status change events are now handled by the unified status system
		// The StatusManager will detect status changes and emit appropriate events

		c.JSON(http.StatusOK, gin.H{
			"success":          true,
			"message":          "Lifecycle status updated successfully",
			"lifecycle_status": statusUpdate.LifecycleStatus,
			"timestamp":        time.Now().UTC().Format(time.RFC3339),
		})
	}
}

// GetNodeStatusHandler handles getting the unified status for a specific node.
// Uses the snapshot (no live probe) so that status is controlled by the
// background HealthMonitor which has proper consecutive-failure debouncing.
// For a live health check, use the POST .../status/refresh endpoint instead.
func GetNodeStatusHandler(statusManager *services.StatusManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "node_id is required",
				"code":  "MISSING_NODE_ID",
			})
			return
		}

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		status, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
		if err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to get node status")
			c.JSON(http.StatusNotFound, gin.H{
				"error":   "Node not found or status unavailable",
				"code":    "NODE_NOT_FOUND",
				"details": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"node_id": nodeID,
			"status":  status,
		})
	}
}

// RefreshNodeStatusHandler handles manual refresh of a node's status
func RefreshNodeStatusHandler(statusManager *services.StatusManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "node_id is required",
				"code":  "MISSING_NODE_ID",
			})
			return
		}

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		// Refresh the status
		if err := statusManager.RefreshAgentStatus(ctx, nodeID); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to refresh node status")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to refresh node status",
				"code":    "REFRESH_FAILED",
				"details": err.Error(),
			})
			return
		}

		// Get the refreshed status
		status, err := statusManager.GetAgentStatus(ctx, nodeID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to get refreshed node status")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to get refreshed status",
				"code":    "STATUS_RETRIEVAL_FAILED",
				"details": err.Error(),
			})
			return
		}

		logger.Logger.Debug().Str("node_id", nodeID).Msg("🔄 Node status refreshed successfully")

		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": "Node status refreshed successfully",
			"node_id": nodeID,
			"status":  status,
		})
	}
}

// BulkNodeStatusHandler handles bulk status queries for multiple nodes
func BulkNodeStatusHandler(statusManager *services.StatusManager, storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()

		// Parse request body for node IDs
		var request struct {
			NodeIDs []string `json:"node_ids" binding:"required"`
		}

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"code":    "INVALID_REQUEST",
				"details": err.Error(),
			})
			return
		}

		// Validate node IDs limit
		if len(request.NodeIDs) > 50 {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Too many node IDs requested (max 50)",
				"code":  "TOO_MANY_NODES",
			})
			return
		}

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		// Get status for each node
		results := make(map[string]interface{})
		var errors []string

		for _, nodeID := range request.NodeIDs {
			status, err := statusManager.GetAgentStatusSnapshot(ctx, nodeID, nil)
			if err != nil {
				logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("⚠️ Failed to get status for node in bulk request")
				results[nodeID] = gin.H{
					"error":   "Status unavailable",
					"details": err.Error(),
				}
				errors = append(errors, fmt.Sprintf("Node %s: %v", nodeID, err))
			} else {
				results[nodeID] = status
			}
		}

		response := gin.H{
			"success":         len(errors) == 0,
			"results":         results,
			"total_requested": len(request.NodeIDs),
			"successful":      len(request.NodeIDs) - len(errors),
			"failed":          len(errors),
		}

		if len(errors) > 0 {
			response["errors"] = errors
		}

		// Return 207 Multi-Status if some requests failed
		statusCode := http.StatusOK
		if len(errors) > 0 && len(errors) < len(request.NodeIDs) {
			statusCode = 207 // Multi-Status
		} else if len(errors) == len(request.NodeIDs) {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, response)
	}
}

// RefreshAllNodeStatusHandler handles manual refresh of all nodes' status
func RefreshAllNodeStatusHandler(statusManager *services.StatusManager, storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		// Get all nodes
		nodes, err := storageProvider.ListAgents(ctx, types.AgentFilters{})
		if err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Failed to list agents for bulk refresh")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to list agents",
				"code":    "LIST_AGENTS_FAILED",
				"details": err.Error(),
			})
			return
		}

		// Refresh status for each node
		var successful, failed int
		var errors []string

		for _, node := range nodes {
			if err := statusManager.RefreshAgentStatus(ctx, node.ID); err != nil {
				logger.Logger.Warn().Err(err).Str("node_id", node.ID).Msg("⚠️ Failed to refresh status for node")
				failed++
				errors = append(errors, fmt.Sprintf("Node %s: %v", node.ID, err))
			} else {
				successful++
			}
		}

		logger.Logger.Debug().
			Int("total", len(nodes)).
			Int("successful", successful).
			Int("failed", failed).
			Msg("🔄 Bulk node status refresh completed")

		response := gin.H{
			"success":     failed == 0,
			"message":     "Bulk node status refresh completed",
			"total_nodes": len(nodes),
			"successful":  successful,
			"failed":      failed,
		}

		if len(errors) > 0 {
			response["errors"] = errors
		}

		// Return appropriate status code
		statusCode := http.StatusOK
		if failed > 0 && successful > 0 {
			statusCode = 207 // Multi-Status
		} else if failed == len(nodes) && len(nodes) > 0 {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, response)
	}
}

// StartNodeHandler handles starting a node (lifecycle management)
func StartNodeHandler(statusManager *services.StatusManager, storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "node_id is required",
				"code":  "MISSING_NODE_ID",
			})
			return
		}

		// Verify node exists
		_, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Node not found",
				"code":  "NODE_NOT_FOUND",
			})
			return
		}

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		// Update status to starting
		startingState := types.AgentStateStarting
		update := &types.AgentStatusUpdate{
			State:  &startingState,
			Source: types.StatusSourceManual,
			Reason: "manual start request",
		}

		if err := statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to start node")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to start node",
				"code":    "START_FAILED",
				"details": err.Error(),
			})
			return
		}

		logger.Logger.Debug().Str("node_id", nodeID).Msg("🚀 Node start initiated")

		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": "Node start initiated",
			"node_id": nodeID,
			"status":  "starting",
		})
	}
}

// StopNodeHandler handles stopping a node (lifecycle management)
func StopNodeHandler(statusManager *services.StatusManager, storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "node_id is required",
				"code":  "MISSING_NODE_ID",
			})
			return
		}

		// Verify node exists
		_, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Node not found",
				"code":  "NODE_NOT_FOUND",
			})
			return
		}

		if statusManager == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "Status manager not available",
				"code":  "SERVICE_UNAVAILABLE",
			})
			return
		}

		// Update status to stopping
		stoppingState := types.AgentStateStopping
		update := &types.AgentStatusUpdate{
			State:  &stoppingState,
			Source: types.StatusSourceManual,
			Reason: "manual stop request",
		}

		if err := statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to stop node")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to stop node",
				"code":    "STOP_FAILED",
				"details": err.Error(),
			})
			return
		}

		logger.Logger.Debug().Str("node_id", nodeID).Msg("🛑 Node stop initiated")

		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": "Node stop initiated",
			"node_id": nodeID,
			"status":  "stopping",
		})
	}
}
