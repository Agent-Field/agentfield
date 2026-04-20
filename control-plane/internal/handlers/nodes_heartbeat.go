package handlers

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// CachedNodeData holds cached heartbeat data for a node
type CachedNodeData struct {
	LastDBUpdate    time.Time
	LastCacheUpdate time.Time
	Status          string
}

// HeartbeatCache manages cached heartbeat data to reduce database writes
type HeartbeatCache struct {
	nodes map[string]*CachedNodeData
	mutex sync.RWMutex
}

var (
	heartbeatCache = &HeartbeatCache{
		nodes: make(map[string]*CachedNodeData),
	}
	// Only write to DB if heartbeat is older than this threshold.
	// Reduced from 8s to 2s to keep DB timestamps fresh and prevent
	// other systems (reconciliation, health monitor) from seeing stale data.
	dbUpdateThreshold = 2 * time.Second
)

// shouldUpdateDatabase determines if a heartbeat should trigger a database update
func (hc *HeartbeatCache) shouldUpdateDatabase(nodeID string, now time.Time, status string) (bool, *CachedNodeData) {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()

	cached, exists := hc.nodes[nodeID]
	if !exists {
		// First heartbeat for this node
		cached = &CachedNodeData{
			LastDBUpdate:    now,
			LastCacheUpdate: now,
			Status:          status,
		}
		hc.nodes[nodeID] = cached
		return true, cached
	}

	// Update cache timestamp
	cached.LastCacheUpdate = now
	cached.Status = status

	// Check if enough time has passed since last DB update
	timeSinceDBUpdate := now.Sub(cached.LastDBUpdate)
	if timeSinceDBUpdate >= dbUpdateThreshold {
		cached.LastDBUpdate = now
		return true, cached
	}

	return false, cached
}

// processHeartbeatAsync processes heartbeat database updates asynchronously
func processHeartbeatAsync(storageProvider storage.StorageProvider, uiService *services.UIService, nodeID string, version string, cached *CachedNodeData) {
	go func() {
		ctx := context.Background()

		// Verify node exists using the resolved version
		if version != "" {
			if _, err := storageProvider.GetAgentVersion(ctx, nodeID, version); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Node %s version '%s' not found during heartbeat update", nodeID, version)
				return
			}
		} else {
			if _, err := storageProvider.GetAgent(ctx, nodeID); err != nil {
				// If not found as default, try finding any version
				versions, listErr := storageProvider.ListAgentVersions(ctx, nodeID)
				if listErr != nil || len(versions) == 0 {
					logger.Logger.Error().Err(err).Msgf("❌ Node %s not found during heartbeat update", nodeID)
					return
				}
			}
		}

		// Update heartbeat in database
		if err := storageProvider.UpdateAgentHeartbeat(ctx, nodeID, version, cached.LastDBUpdate); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ HEARTBEAT_CONTENTION: Failed to update heartbeat for node %s version '%s' - %v", nodeID, version, err)
			return
		}

		logger.Logger.Debug().Msgf("💓 HEARTBEAT_CONTENTION: Async DB update completed for node %s version '%s'", nodeID, version)
	}()
}

// HeartbeatHandler handles heartbeat requests from agent nodes
// Supports both simple heartbeats and enhanced heartbeats with status updates
// Now integrates with the unified status management system
func HeartbeatHandler(storageProvider storage.StorageProvider, uiService *services.UIService, healthMonitor *services.HealthMonitor, statusManager *services.StatusManager, presenceManager *services.PresenceManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		// We'll verify node exists conditionally during heartbeat caching
		var existingNode *types.AgentNode

		// Try to parse enhanced heartbeat data (optional)
		var enhancedHeartbeat struct {
			Version     string `json:"version,omitempty"`
			Status      string `json:"status,omitempty"`
			Timestamp   string `json:"timestamp,omitempty"`
			HealthScore *int   `json:"health_score,omitempty"`
		}

		// Read the request body if present
		if c.Request.ContentLength > 0 {
			if err := c.ShouldBindJSON(&enhancedHeartbeat); err != nil {
				// If JSON parsing fails, treat as simple heartbeat
				logger.Logger.Debug().Msgf("💓 Simple heartbeat from node: %s", nodeID)
			} else {
				logger.Logger.Debug().Msgf("💓 Enhanced heartbeat from node: %s with status: %s", nodeID, enhancedHeartbeat.Status)
			}
		}

		// Check if database update is needed using caching
		now := time.Now().UTC()
		if presenceManager != nil && presenceManager.HasLease(nodeID) {
			presenceManager.Touch(nodeID, enhancedHeartbeat.Version, now)
		}
		needsDBUpdate, cached := heartbeatCache.shouldUpdateDatabase(nodeID, now, enhancedHeartbeat.Status)

		if needsDBUpdate {
			// Verify node exists only when we need to update DB.
			// Use the outer-scoped existingNode so it's available for status processing below.
			var err error
			if enhancedHeartbeat.Version != "" {
				existingNode, err = storageProvider.GetAgentVersion(ctx, nodeID, enhancedHeartbeat.Version)
			} else {
				existingNode, err = storageProvider.GetAgent(ctx, nodeID)
			}
			if err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Node %s not found during heartbeat update", nodeID)
				c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
				return
			}

			// Check for nil node (can happen when database returns no error but also no rows)
			if existingNode == nil {
				logger.Logger.Error().Msgf("❌ Node %s returned nil from storage during heartbeat update", nodeID)
				c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
				return
			}

			// Register agent with health monitor for HTTP-based monitoring
			if healthMonitor != nil {
				healthMonitor.RegisterAgent(nodeID, existingNode.BaseURL)
			}

			if presenceManager != nil {
				presenceManager.Touch(nodeID, existingNode.Version, now)
			}

			// Process heartbeat asynchronously to avoid blocking the response.
			// Use existingNode.Version (resolved from DB) instead of the heartbeat payload version
			// to handle old SDKs that may not send version in heartbeats.
			processHeartbeatAsync(storageProvider, uiService, nodeID, existingNode.Version, cached)

			logger.Logger.Debug().Msgf("💓 Heartbeat DB update queued for node: %s at %s", nodeID, now.Format(time.RFC3339))
		} else {
			logger.Logger.Debug().Msgf("💓 Heartbeat cached for node: %s (no DB update needed)", nodeID)
		}

		// Process enhanced heartbeat data through unified status system
		if statusManager != nil && (enhancedHeartbeat.Status != "" || enhancedHeartbeat.HealthScore != nil) {
			// Prepare lifecycle status
			var lifecycleStatus *types.AgentLifecycleStatus
			if enhancedHeartbeat.Status != "" {
				// Validate status
				validStatuses := map[string]bool{
					"starting": true,
					"ready":    true,
					"degraded": true,
					"offline":  true,
				}

				if validStatuses[enhancedHeartbeat.Status] {
					// Protect pending_approval: heartbeats cannot override admin-controlled state
					if existingNode == nil {
						var err error
						if enhancedHeartbeat.Version != "" {
							existingNode, err = storageProvider.GetAgentVersion(ctx, nodeID, enhancedHeartbeat.Version)
						} else {
							existingNode, err = storageProvider.GetAgent(ctx, nodeID)
						}
						if err != nil {
							logger.Logger.Error().Err(err).Msgf("❌ Failed to get node %s for pending_approval check", nodeID)
						}
					}
					if existingNode != nil && existingNode.LifecycleStatus == types.AgentStatusPendingApproval {
						logger.Logger.Debug().Msgf("⏸️ Ignoring heartbeat status update for node %s: agent is pending_approval (admin action required)", nodeID)
					} else {
						status := types.AgentLifecycleStatus(enhancedHeartbeat.Status)
						lifecycleStatus = &status
					}
				}
			}

			// Resolve version from DB record when available, fall back to heartbeat payload
			resolvedVersion := enhancedHeartbeat.Version
			if existingNode != nil {
				resolvedVersion = existingNode.Version
			}

			// Update status through unified system
			if err := statusManager.UpdateFromHeartbeat(ctx, nodeID, lifecycleStatus, resolvedVersion); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update unified status for node %s", nodeID)
				// Continue processing - don't fail the heartbeat
			}

			// Handle health score if provided
			if enhancedHeartbeat.HealthScore != nil {
				update := &types.AgentStatusUpdate{
					HealthScore: enhancedHeartbeat.HealthScore,
					Source:      types.StatusSourceHeartbeat,
					Reason:      "health score from heartbeat",
					Version:     resolvedVersion,
				}

				if err := statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
					logger.Logger.Error().Err(err).Msgf("❌ Failed to update health score for node %s", nodeID)
				}
			}
		} else {
			// Fallback to legacy status update for backward compatibility
			if enhancedHeartbeat.Status != "" {
				// Validate status
				validStatuses := map[string]bool{
					"starting": true,
					"ready":    true,
					"degraded": true,
					"offline":  true,
				}

				if validStatuses[enhancedHeartbeat.Status] {
					newStatus := types.AgentLifecycleStatus(enhancedHeartbeat.Status)

					// Get existing node to check current status
					if existingNode == nil {
						var err error
						existingNode, err = storageProvider.GetAgent(ctx, nodeID)
						if (err != nil || existingNode == nil) && enhancedHeartbeat.Version != "" {
							existingNode, err = storageProvider.GetAgentVersion(ctx, nodeID, enhancedHeartbeat.Version)
						}
						if err != nil || existingNode == nil {
							logger.Logger.Error().Err(err).Msgf("❌ Failed to get node %s for lifecycle status update", nodeID)
							c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
							return
						}
					}

					// Protect pending_approval: heartbeats cannot override admin-controlled state
					if existingNode.LifecycleStatus == types.AgentStatusPendingApproval {
						logger.Logger.Debug().Msgf("⏸️ Ignoring legacy heartbeat status for node %s: agent is pending_approval", nodeID)
					} else if existingNode.LifecycleStatus != newStatus {
						if err := storageProvider.UpdateAgentLifecycleStatus(ctx, nodeID, newStatus); err != nil {
							logger.Logger.Error().Err(err).Msgf("❌ Failed to update lifecycle status for node %s", nodeID)
						} else {
							logger.Logger.Debug().Msgf("🔄 Lifecycle status updated for node %s: %s", nodeID, enhancedHeartbeat.Status)
						}
					}
				}
			}
		}

		// Note: Status change events are now handled by the unified status system
		// The StatusManager will detect status changes and emit appropriate events

		logger.Logger.Debug().Msgf("💓 Heartbeat received from node: %s at %s", nodeID, now.Format(time.RFC3339))

		// Return immediate acknowledgment
		c.JSON(http.StatusOK, gin.H{
			"success":   true,
			"message":   "heartbeat received",
			"timestamp": now.Format(time.RFC3339),
		})
	}
}
