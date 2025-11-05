package handlers

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/services"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// DefaultLeaseTTL is the lease duration returned to agents when explicit configuration is not yet available.
const DefaultLeaseTTL = 5 * time.Minute

// NodeStatusLeaseHandler processes lease-based status updates from agents.
func NodeStatusLeaseHandler(storageProvider storage.StorageProvider, statusManager *services.StatusManager, presenceManager *services.PresenceManager, leaseTTL time.Duration) gin.HandlerFunc {
	if leaseTTL <= 0 {
		leaseTTL = DefaultLeaseTTL
	}

	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		var payload struct {
			Phase       string `json:"phase"`
			HealthScore *int   `json:"health_score"`
			// Conditions are accepted for future use but currently ignored by the control plane.
			Conditions []map[string]interface{} `json:"conditions"`
		}

		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid payload", "details": err.Error()})
			return
		}

		agent, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil || agent == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		update := &types.AgentStatusUpdate{
			Source: types.StatusSourceManual,
		}

		if payload.HealthScore != nil {
			if *payload.HealthScore < 0 || *payload.HealthScore > 100 {
				c.JSON(http.StatusBadRequest, gin.H{"error": "health_score must be between 0 and 100"})
				return
			}
			update.HealthScore = payload.HealthScore
		}

		if payload.Phase != "" {
			state, lifecycle, err := normalizePhase(payload.Phase)
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}
			update.State = state
			update.LifecycleStatus = lifecycle
		}

		if statusManager != nil {
			if err := statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
				logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("failed to update agent status from lease handler")
				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to update status"})
				return
			}
		}

		now := time.Now().UTC()
		if err := storageProvider.UpdateAgentHeartbeat(ctx, nodeID, now); err != nil {
			logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("failed to persist heartbeat during status update")
		}

		if presenceManager != nil {
			presenceManager.Touch(nodeID, now)
		}

		c.JSON(http.StatusOK, gin.H{
			"lease_seconds":      int(leaseTTL.Seconds()),
			"next_lease_renewal": now.Add(leaseTTL).Format(time.RFC3339),
		})
	}
}

// NodeActionAckHandler acknowledges actions in push mode. Currently it only renews the lease and logs the payload.
func NodeActionAckHandler(storageProvider storage.StorageProvider, presenceManager *services.PresenceManager, leaseTTL time.Duration) gin.HandlerFunc {
	if leaseTTL <= 0 {
		leaseTTL = DefaultLeaseTTL
	}

	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		var payload struct {
			ActionID   string                 `json:"action_id"`
			Status     string                 `json:"status"`
			DurationMS *int                   `json:"duration_ms"`
			ResultRef  string                 `json:"result_ref"`
			Result     map[string]interface{} `json:"result"`
			Error      string                 `json:"error_message"`
			Notes      []string               `json:"notes"`
		}

		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid payload", "details": err.Error()})
			return
		}

		if payload.ActionID == "" || payload.Status == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "action_id and status are required"})
			return
		}

		if _, err := storageProvider.GetAgent(ctx, nodeID); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		canonicalStatus := types.NormalizeExecutionStatus(payload.Status)
		logger.Logger.Debug().
			Str("node_id", nodeID).
			Str("action_id", payload.ActionID).
			Str("status", canonicalStatus).
			Msg("action acknowledgement received")

		now := time.Now().UTC()
		if err := storageProvider.UpdateAgentHeartbeat(ctx, nodeID, now); err != nil {
			logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("failed to persist heartbeat during action ack")
		}
		if presenceManager != nil {
			presenceManager.Touch(nodeID, now)
		}

		c.JSON(http.StatusOK, gin.H{
			"lease_seconds":      int(leaseTTL.Seconds()),
			"next_lease_renewal": now.Add(leaseTTL).Format(time.RFC3339),
		})
	}
}

// ClaimActionsHandler returns pending actions for poll-mode agents.
// Currently the scheduler backend is under construction, so this returns an empty queue but still renews leases.
func ClaimActionsHandler(storageProvider storage.StorageProvider, presenceManager *services.PresenceManager, leaseTTL time.Duration) gin.HandlerFunc {
	if leaseTTL <= 0 {
		leaseTTL = DefaultLeaseTTL
	}

	return func(c *gin.Context) {
		ctx := c.Request.Context()

		var payload struct {
			NodeID      string `json:"node_id"`
			MaxItems    int    `json:"max_items"`
			WaitSeconds int    `json:"wait_seconds"`
		}

		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid payload", "details": err.Error()})
			return
		}

		if payload.NodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		if payload.MaxItems <= 0 {
			payload.MaxItems = 1
		}

		if _, err := storageProvider.GetAgent(ctx, payload.NodeID); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		now := time.Now().UTC()
		if err := storageProvider.UpdateAgentHeartbeat(ctx, payload.NodeID, now); err != nil {
			logger.Logger.Warn().Err(err).Str("node_id", payload.NodeID).Msg("failed to persist heartbeat during claim")
		}
		if presenceManager != nil {
			presenceManager.Touch(payload.NodeID, now)
		}

		nextPoll := payload.WaitSeconds
		if nextPoll <= 0 {
			nextPoll = 5
		}

		c.JSON(http.StatusOK, gin.H{
			"items":              []interface{}{},
			"lease_seconds":      int(leaseTTL.Seconds()),
			"next_poll_after":    nextPoll,
			"next_lease_renewal": now.Add(leaseTTL).Format(time.RFC3339),
		})
	}
}

// NodeShutdownHandler processes graceful shutdown notifications from agents.
func NodeShutdownHandler(storageProvider storage.StorageProvider, statusManager *services.StatusManager, presenceManager *services.PresenceManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		if _, err := storageProvider.GetAgent(ctx, nodeID); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		var payload struct {
			Reason          string `json:"reason"`
			ExpectedRestart string `json:"expected_restart"`
		}
		_ = c.ShouldBindJSON(&payload) // best-effort parse; optional fields

		now := time.Now().UTC()
		if presenceManager != nil {
			presenceManager.Forget(nodeID)
		}
		if err := storageProvider.UpdateAgentHeartbeat(ctx, nodeID, now); err != nil {
			logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("failed to persist heartbeat during shutdown")
		}

		if statusManager != nil {
			inactive := types.AgentStateStopping
			lifecycle := types.AgentStatusOffline
			update := &types.AgentStatusUpdate{
				State:           &inactive,
				LifecycleStatus: &lifecycle,
				Source:          types.StatusSourceManual,
				Reason:          "agent shutdown",
			}
			if err := statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
				logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("failed to update status during shutdown")
			}
		}

		c.JSON(http.StatusAccepted, gin.H{
			"lease_seconds":      0,
			"next_lease_renewal": now.Format(time.RFC3339),
			"message":            "shutdown acknowledged",
		})
	}
}

func normalizePhase(phase string) (*types.AgentState, *types.AgentLifecycleStatus, error) {
	if phase == "" {
		return nil, nil, nil
	}

	switch strings.ToLower(strings.TrimSpace(phase)) {
	case "starting":
		state := types.AgentStateStarting
		lifecycle := types.AgentStatusStarting
		return &state, &lifecycle, nil
	case "ready":
		state := types.AgentStateActive
		lifecycle := types.AgentStatusReady
		return &state, &lifecycle, nil
	case "degraded":
		state := types.AgentStateActive
		lifecycle := types.AgentStatusDegraded
		return &state, &lifecycle, nil
	case "offline":
		state := types.AgentStateInactive
		lifecycle := types.AgentStatusOffline
		return &state, &lifecycle, nil
	default:
		return nil, nil, fmt.Errorf("unsupported phase: %s", phase)
	}
}
