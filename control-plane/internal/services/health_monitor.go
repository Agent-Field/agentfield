package services

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// Health score constants for status updates.
const (
	// healthScoreActive is the score assigned when an HTTP health check passes.
	// Below 100 to leave room for "excellent" states.
	healthScoreActive = 85

	// healthScoreInactive is the score when an agent fails consecutive health checks.
	healthScoreInactive = 0
)

// HealthMonitorConfig holds configuration for the health monitor service
type HealthMonitorConfig struct {
	CheckInterval       time.Duration // How often to check node health via HTTP
	CheckTimeout        time.Duration // Timeout for individual HTTP health checks
	ConsecutiveFailures int           // Number of consecutive failures before marking inactive
	RecoveryDebounce    time.Duration // Time to wait before allowing inactive->active recovery
}

// ActiveAgent represents an agent currently being monitored
type ActiveAgent struct {
	NodeID              string
	BaseURL             string
	LastStatus          types.HealthStatus
	LastChecked         time.Time
	ConsecutiveFailures int       // Track consecutive HTTP check failures
	LastTransition      time.Time // When the health status last changed
}

// HealthMonitor monitors the health of actively registered agent nodes
// Uses HTTP /status endpoint as single source of truth
// Now integrates with the unified status management system
type HealthMonitor struct {
	storage       storage.StorageProvider
	config        HealthMonitorConfig
	uiService     *UIService
	agentClient   interfaces.AgentClient
	statusManager *StatusManager
	presence      *PresenceManager
	stopCh        chan struct{}
	stopOnce      sync.Once

	// Active agents registry - only agents currently running
	activeAgents map[string]*ActiveAgent
	agentsMutex  sync.RWMutex

}

// NewHealthMonitor creates a new HTTP-first health monitor service
func NewHealthMonitor(storage storage.StorageProvider, config HealthMonitorConfig, uiService *UIService, agentClient interfaces.AgentClient, statusManager *StatusManager, presence *PresenceManager) *HealthMonitor {
	// Set default values
	if config.CheckInterval == 0 {
		config.CheckInterval = 10 * time.Second
	}
	if config.CheckTimeout == 0 {
		config.CheckTimeout = 5 * time.Second
	}
	if config.ConsecutiveFailures == 0 {
		config.ConsecutiveFailures = 3 // Require 3 failures before marking inactive
	}
	if config.RecoveryDebounce == 0 {
		config.RecoveryDebounce = 5 * time.Second // Reduced from 30s for faster recovery
	}

	return &HealthMonitor{
		storage:        storage,
		config:         config,
		uiService:      uiService,
		agentClient:    agentClient,
		statusManager:  statusManager,
		presence:       presence,
		stopCh:         make(chan struct{}),
		activeAgents:   make(map[string]*ActiveAgent),
		agentsMutex:    sync.RWMutex{},
	}
}

// RegisterAgent adds an agent to the active monitoring list
func (hm *HealthMonitor) RegisterAgent(nodeID, baseURL string) {
	hm.agentsMutex.Lock()
	defer hm.agentsMutex.Unlock()

	seenAt := time.Now()

	hm.activeAgents[nodeID] = &ActiveAgent{
		NodeID:         nodeID,
		BaseURL:        baseURL,
		LastStatus:     types.HealthStatusUnknown,
		LastChecked:    seenAt,
		LastTransition: seenAt, // Initialize so debounce checks have a valid baseline
	}

	if hm.presence != nil {
		hm.presence.Touch(nodeID, "", seenAt)
	}

	logger.Logger.Debug().Msgf("🏥 Registered agent %s for HTTP health monitoring", nodeID)
}

// UnregisterAgent removes an agent from the active monitoring list
func (hm *HealthMonitor) UnregisterAgent(nodeID string) {
	hm.agentsMutex.Lock()
	defer hm.agentsMutex.Unlock()

	if _, exists := hm.activeAgents[nodeID]; exists {
		delete(hm.activeAgents, nodeID)
		logger.Logger.Debug().Msgf("🏥 Unregistered agent %s from health monitoring", nodeID)

		if hm.presence != nil {
			hm.presence.Forget(nodeID)
		}

		// Update status to inactive through unified system
		ctx := context.Background()
		if hm.statusManager != nil {
			// Use unified status system
			inactiveState := types.AgentStateInactive
			healthScore := 0
			update := &types.AgentStatusUpdate{
				State:       &inactiveState,
				HealthScore: &healthScore,
				Source:      types.StatusSourceHealthCheck,
				Reason:      "agent unregistered from health monitoring",
			}

			if err := hm.statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update unified status for unregistered agent %s", nodeID)
				// Fallback to legacy update
				if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusInactive); err != nil {
					logger.Logger.Error().Err(err).Msgf("❌ Failed to update agent %s status to inactive", nodeID)
				}
			}
		} else {
			// Fallback to legacy system
			if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusInactive); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update agent %s status to inactive", nodeID)
			}
			// Also update lifecycle status to offline for consistency
			if err := hm.storage.UpdateAgentLifecycleStatus(ctx, nodeID, types.AgentStatusOffline); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update agent %s lifecycle status to offline", nodeID)
			}

			// Broadcast offline event (legacy)
			if hm.uiService != nil {
				if agent, err := hm.storage.GetAgent(ctx, nodeID); err == nil {
					events.PublishNodeOffline(nodeID, agent)
					events.PublishNodeHealthChanged(nodeID, string(types.HealthStatusInactive), agent)
					hm.uiService.OnNodeStatusChanged(agent)
				}
			}
		}
	}
}

// RecoverFromDatabase loads previously registered nodes from the database
// and performs initial health checks. This should be called on startup to
// recover state after a control plane restart.
func (hm *HealthMonitor) RecoverFromDatabase(ctx context.Context) error {
	nodes, err := hm.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return err
	}

	if len(nodes) == 0 {
		logger.Logger.Debug().Msg("🏥 No nodes to recover from database")
		return nil
	}

	logger.Logger.Info().Int("count", len(nodes)).Msg("🏥 Recovering nodes from database for health monitoring")

	// Register all nodes with the health monitor
	for _, node := range nodes {
		if node == nil || node.BaseURL == "" {
			continue // Skip nodes without callback URL
		}

		hm.RegisterAgent(node.ID, node.BaseURL)
	}

	// Perform health checks asynchronously to avoid blocking startup
	// The regular health monitor loop will also check these nodes
	go func() {
		logger.Logger.Debug().Msg("🏥 Starting async health checks for recovered nodes")
		hm.checkActiveAgents()
		logger.Logger.Debug().Msg("🏥 Async health checks complete")
	}()

	logger.Logger.Info().Msg("🏥 Node recovery complete")
	return nil
}

// Start begins the HTTP-based health monitoring process
func (hm *HealthMonitor) Start() {
	logger.Logger.Debug().Msgf("🏥 Starting HTTP-first health monitor service (check interval: %v)",
		hm.config.CheckInterval)

	ticker := time.NewTicker(hm.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hm.checkActiveAgents()
		case <-hm.stopCh:
			logger.Logger.Debug().Msg("🏥 Health monitor service stopped")
			return
		}
	}
}

// Stop stops the health monitoring process. Safe to call multiple times.
func (hm *HealthMonitor) Stop() {
	hm.stopOnce.Do(func() {
		close(hm.stopCh)
	})
}

// checkActiveAgents performs HTTP health checks on all actively registered agents
func (hm *HealthMonitor) checkActiveAgents() {
	hm.agentsMutex.RLock()
	nodeIDs := make([]string, 0, len(hm.activeAgents))
	for id := range hm.activeAgents {
		nodeIDs = append(nodeIDs, id)
	}
	hm.agentsMutex.RUnlock()

	if len(nodeIDs) == 0 {
		logger.Logger.Debug().Msg("🏥 No active agents to monitor")
		return
	}

	logger.Logger.Debug().Msgf("🏥 Checking health of %d active agents via HTTP", len(nodeIDs))

	for _, nodeID := range nodeIDs {
		hm.checkAgentHealth(nodeID)
	}
}

// checkAgentHealth performs HTTP health check for a single agent identified by nodeID.
// Uses consecutive failure tracking to prevent flapping from transient network issues.
// Accepts nodeID rather than *ActiveAgent to avoid holding stale pointers across the
// HTTP call boundary — the canonical state is always re-read from hm.activeAgents.
func (hm *HealthMonitor) checkAgentHealth(nodeID string) {
	// Early check: ensure agent is still in active registry before making HTTP call
	hm.agentsMutex.RLock()
	_, exists := hm.activeAgents[nodeID]
	hm.agentsMutex.RUnlock()

	if !exists {
		logger.Logger.Debug().Msgf("🏥 Skipping health check for %s - agent no longer in active registry", nodeID)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), hm.config.CheckTimeout)
	defer cancel()

	// Perform HTTP health check
	status, err := hm.agentClient.GetAgentStatus(ctx, nodeID)

	var checkPassed bool
	if err != nil {
		checkPassed = false
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check failed: %v", nodeID, err)
	} else if status.Status == "running" {
		checkPassed = true
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check successful: %s", nodeID, status.Status)
	} else {
		checkPassed = false
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check shows not running: %s", nodeID, status.Status)
	}

	// Update agent state with consecutive failure tracking.
	// Re-fetch from the map since the agent may have been unregistered/re-registered
	// during the HTTP call above.
	hm.agentsMutex.Lock()
	activeAgent, exists := hm.activeAgents[nodeID]
	if !exists {
		hm.agentsMutex.Unlock()
		return
	}

	activeAgent.LastChecked = time.Now()

	if checkPassed {
		// SUCCESS: Reset failure counter
		previousFailures := activeAgent.ConsecutiveFailures
		activeAgent.ConsecutiveFailures = 0

		if previousFailures > 0 {
			logger.Logger.Debug().Msgf("🏥 Agent %s check passed, reset failure counter from %d", nodeID, previousFailures)
		}

		if activeAgent.LastStatus == types.HealthStatusInactive {
			// Recovery from inactive: apply debounce
			if time.Since(activeAgent.LastTransition) < hm.config.RecoveryDebounce {
				hm.agentsMutex.Unlock()
				logger.Logger.Debug().Msgf("🏥 Agent %s recovery debounce active, waiting", nodeID)
				return
			}
			activeAgent.LastStatus = types.HealthStatusActive
			activeAgent.LastTransition = time.Now()
			hm.agentsMutex.Unlock()
			logger.Logger.Info().Msgf("✅ Agent %s recovered to active", nodeID)
			hm.markAgentActive(nodeID)
			return
		} else if activeAgent.LastStatus != types.HealthStatusActive {
			// First time becoming active (e.g. from unknown)
			activeAgent.LastStatus = types.HealthStatusActive
			activeAgent.LastTransition = time.Now()
			hm.agentsMutex.Unlock()
			hm.markAgentActive(nodeID)
			return
		}
		// Already active, no status change needed
		hm.agentsMutex.Unlock()
	} else {
		// FAILURE: Increment consecutive failure counter (capped to prevent unbounded growth)
		if activeAgent.ConsecutiveFailures < hm.config.ConsecutiveFailures+1 {
			activeAgent.ConsecutiveFailures++
		}

		logger.Logger.Debug().Msgf("🏥 Agent %s failure %d/%d",
			nodeID, activeAgent.ConsecutiveFailures, hm.config.ConsecutiveFailures)

		// Only mark inactive after reaching the consecutive failure threshold
		if activeAgent.ConsecutiveFailures >= hm.config.ConsecutiveFailures {
			// HEARTBEAT GATE: Before marking inactive, check if the agent has sent
			// a recent heartbeat. Heartbeats are direct proof of agent liveness —
			// if the agent is sending heartbeats, HTTP check failures are transient
			// and should not trigger an inactive transition. We check the storage
			// heartbeat timestamp rather than the presence lease because the presence
			// lease is also set by RegisterAgent (not just heartbeats).
			if hm.statusManager != nil {
				staleThreshold := hm.statusManager.config.HeartbeatStaleThreshold
				if staleThreshold == 0 {
					staleThreshold = 60 * time.Second
				}
				if agent, err := hm.storage.GetAgent(context.Background(), nodeID); err == nil && agent != nil {
					if time.Since(agent.LastHeartbeat) < staleThreshold {
						logger.Logger.Debug().Msgf("🏥 Agent %s has %d HTTP failures but heartbeat is fresh (%v ago) — not marking inactive",
							nodeID, activeAgent.ConsecutiveFailures, time.Since(agent.LastHeartbeat))
						hm.agentsMutex.Unlock()
						return
					}
				}
			}

			if activeAgent.LastStatus != types.HealthStatusInactive {
				activeAgent.LastStatus = types.HealthStatusInactive
				activeAgent.LastTransition = time.Now()
				failCount := activeAgent.ConsecutiveFailures
				hm.agentsMutex.Unlock()
				logger.Logger.Warn().Msgf("Agent %s marked inactive after %d consecutive failures", nodeID, failCount)
				hm.markAgentInactive(nodeID, failCount)
				return
			}
		}
		hm.agentsMutex.Unlock()
	}
}

// markAgentActive marks an agent as active through the unified status system
func (hm *HealthMonitor) markAgentActive(nodeID string) {
	ctx := context.Background()

	if hm.statusManager != nil {
		activeState := types.AgentStateActive
		healthScore := healthScoreActive
		update := &types.AgentStatusUpdate{
			State:       &activeState,
			HealthScore: &healthScore,
			Source:      types.StatusSourceHealthCheck,
			Reason:      "HTTP health check passed",
		}

		if err := hm.statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to mark agent %s active via status manager", nodeID)
			// Fallback to direct storage update
			if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusActive); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", nodeID)
			}
			return
		}

		if hm.presence != nil {
			hm.presence.Touch(nodeID, "", time.Now())
		}

	} else {
		// Legacy fallback
		if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusActive); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", nodeID)
			return
		}
		if err := hm.storage.UpdateAgentLifecycleStatus(ctx, nodeID, types.AgentStatusReady); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to update lifecycle status for agent %s", nodeID)
		}
		if updatedAgent, err := hm.storage.GetAgent(ctx, nodeID); err == nil {
			events.PublishNodeOnline(nodeID, updatedAgent)
			if hm.presence != nil {
				hm.presence.Touch(nodeID, "", time.Now())
			}
			events.PublishNodeHealthChanged(nodeID, string(types.HealthStatusActive), updatedAgent)
			if hm.uiService != nil {
				hm.uiService.OnNodeStatusChanged(updatedAgent)
			}
		}
	}
}

// markAgentInactive marks an agent as inactive through the unified status system
func (hm *HealthMonitor) markAgentInactive(nodeID string, failCount int) {
	ctx := context.Background()

	if hm.statusManager != nil {
		inactiveState := types.AgentStateInactive
		healthScore := healthScoreInactive
		update := &types.AgentStatusUpdate{
			State:       &inactiveState,
			HealthScore: &healthScore,
			Source:      types.StatusSourceHealthCheck,
			Reason:      fmt.Sprintf("%d consecutive health check failures", failCount),
		}

		if err := hm.statusManager.UpdateAgentStatus(ctx, nodeID, update); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to mark agent %s inactive via status manager", nodeID)
			if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusInactive); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", nodeID)
			}
		}
	} else {
		// Legacy fallback
		if err := hm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusInactive); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", nodeID)
			return
		}
		if err := hm.storage.UpdateAgentLifecycleStatus(ctx, nodeID, types.AgentStatusOffline); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to update lifecycle status for agent %s", nodeID)
		}
		if updatedAgent, err := hm.storage.GetAgent(ctx, nodeID); err == nil {
			events.PublishNodeOffline(nodeID, updatedAgent)
			events.PublishNodeHealthChanged(nodeID, string(types.HealthStatusInactive), updatedAgent)
			if hm.uiService != nil {
				hm.uiService.OnNodeStatusChanged(updatedAgent)
			}
		}
	}
}

