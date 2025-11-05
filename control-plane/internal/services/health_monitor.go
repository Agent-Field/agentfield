package services

import (
	"context"
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/events"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"
	"sync"
	"time"
)

// HealthMonitorConfig holds configuration for the health monitor service
type HealthMonitorConfig struct {
	CheckInterval time.Duration // How often to check node health via HTTP
}

// ActiveAgent represents an agent currently being monitored
type ActiveAgent struct {
	NodeID      string
	BaseURL     string
	LastStatus  types.HealthStatus
	LastChecked time.Time
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

	// Active agents registry - only agents currently running
	activeAgents map[string]*ActiveAgent
	agentsMutex  sync.RWMutex

	// MCP health tracking
	mcpHealthCache map[string]*domain.MCPSummaryData
	mcpCacheMutex  sync.RWMutex
}

// NewHealthMonitor creates a new HTTP-first health monitor service
func NewHealthMonitor(storage storage.StorageProvider, config HealthMonitorConfig, uiService *UIService, agentClient interfaces.AgentClient, statusManager *StatusManager, presence *PresenceManager) *HealthMonitor {
	// Set default values - using efficient 10s intervals
	if config.CheckInterval == 0 {
		config.CheckInterval = 10 * time.Second // HTTP health check every 10 seconds
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
		mcpHealthCache: make(map[string]*domain.MCPSummaryData),
		mcpCacheMutex:  sync.RWMutex{},
	}
}

// RegisterAgent adds an agent to the active monitoring list
func (hm *HealthMonitor) RegisterAgent(nodeID, baseURL string) {
	hm.agentsMutex.Lock()
	defer hm.agentsMutex.Unlock()

	seenAt := time.Now()

	hm.activeAgents[nodeID] = &ActiveAgent{
		NodeID:      nodeID,
		BaseURL:     baseURL,
		LastStatus:  types.HealthStatusUnknown,
		LastChecked: seenAt,
	}

	if hm.presence != nil {
		hm.presence.Touch(nodeID, seenAt)
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

// Stop stops the health monitoring process
func (hm *HealthMonitor) Stop() {
	close(hm.stopCh)
}

// checkActiveAgents performs HTTP health checks on all actively registered agents
func (hm *HealthMonitor) checkActiveAgents() {
	hm.agentsMutex.RLock()
	agents := make([]*ActiveAgent, 0, len(hm.activeAgents))
	for _, agent := range hm.activeAgents {
		agents = append(agents, agent)
	}
	hm.agentsMutex.RUnlock()

	if len(agents) == 0 {
		logger.Logger.Debug().Msg("🏥 No active agents to monitor")
		return
	}

	logger.Logger.Debug().Msgf("🏥 Checking health of %d active agents via HTTP", len(agents))

	for _, agent := range agents {
		hm.checkAgentHealth(agent)
	}
}

// checkAgentHealth performs HTTP health check for a single agent
func (hm *HealthMonitor) checkAgentHealth(agent *ActiveAgent) {
	// Early check: ensure agent is still in active registry before making HTTP call
	hm.agentsMutex.RLock()
	_, exists := hm.activeAgents[agent.NodeID]
	hm.agentsMutex.RUnlock()

	if !exists {
		logger.Logger.Debug().Msgf("🏥 Skipping health check for %s - agent no longer in active registry", agent.NodeID)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Use HTTP /status endpoint as single source of truth
	status, err := hm.agentClient.GetAgentStatus(ctx, agent.NodeID)

	var newStatus types.HealthStatus
	if err != nil {
		// HTTP request failed - agent is offline
		newStatus = types.HealthStatusInactive
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check failed: %v", agent.NodeID, err)
	} else if status.Status == "running" {
		// FIXED: More conservative health detection to prevent false positives
		// Only consider agent active if it's actually running and responsive
		newStatus = types.HealthStatusActive
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check successful: %s", agent.NodeID, status.Status)
	} else {
		// Agent responded but not running
		newStatus = types.HealthStatusInactive
		logger.Logger.Debug().Msgf("🏥 Agent %s HTTP check shows not running: %s", agent.NodeID, status.Status)
	}

	// Update agent's last checked time
	hm.agentsMutex.Lock()
	if activeAgent, exists := hm.activeAgents[agent.NodeID]; exists {
		activeAgent.LastChecked = time.Now()
		statusChanged := activeAgent.LastStatus != newStatus
		activeAgent.LastStatus = newStatus
		hm.agentsMutex.Unlock()

		// Only update database and broadcast events if status actually changed
		if statusChanged {
			logger.Logger.Debug().Msgf("🔄 Agent %s status changed to %s via HTTP check", agent.NodeID, newStatus)

			// FIXED: Add debouncing to prevent rapid oscillation
			// If agent just changed to inactive, wait before allowing it to become active again
			if newStatus == types.HealthStatusInactive {
				// Mark agent as recently failed to prevent immediate reactivation
				activeAgent.LastChecked = time.Now()
			} else if newStatus == types.HealthStatusActive && activeAgent.LastStatus == types.HealthStatusInactive {
				// If agent was recently inactive, require a longer period of stability
				if time.Since(activeAgent.LastChecked) < 30*time.Second {
					logger.Logger.Debug().Msgf("🏥 Agent %s status change to active too soon after inactive, skipping", agent.NodeID)
					return
				}
			}

			// Update through unified status system if available
			if hm.statusManager != nil {
				// Determine the new agent state based on health status
				var newState types.AgentState
				var healthScore int

				switch newStatus {
				case types.HealthStatusActive:
					newState = types.AgentStateActive
					// FIXED: More conservative health score to prevent oscillation
					// Only give high score if agent is consistently responsive
					healthScore = 75 // Moderate health from HTTP check
				case types.HealthStatusInactive:
					newState = types.AgentStateInactive
					healthScore = 0 // No health
				default:
					newState = types.AgentStateInactive
					healthScore = 0
				}

				// Create status update
				update := &types.AgentStatusUpdate{
					State:       &newState,
					HealthScore: &healthScore,
					Source:      types.StatusSourceHealthCheck,
					Reason:      "HTTP health check result",
				}

				// Update through unified system
				ctx := context.Background()
				if err := hm.statusManager.UpdateAgentStatus(ctx, agent.NodeID, update); err != nil {
					logger.Logger.Error().Err(err).Msgf("❌ Failed to update unified status for agent %s", agent.NodeID)
					// Fallback to legacy update
					if err := hm.storage.UpdateAgentHealth(ctx, agent.NodeID, newStatus); err != nil {
						logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", agent.NodeID)
					}
				} else if hm.presence != nil && newStatus == types.HealthStatusActive {
					hm.presence.Touch(agent.NodeID, time.Now())
				}

				// Check MCP health for active agents
				if newStatus == types.HealthStatusActive {
					hm.checkMCPHealthForNode(agent.NodeID)
				}
			} else {
				// Fallback to legacy system for backward compatibility
				ctx := context.Background()
				if err := hm.storage.UpdateAgentHealth(ctx, agent.NodeID, newStatus); err != nil {
					logger.Logger.Error().Err(err).Msgf("❌ Failed to update health status for agent %s", agent.NodeID)
					return
				}

				// Broadcast status change events (legacy)
				if updatedAgent, err := hm.storage.GetAgent(ctx, agent.NodeID); err == nil {
					// Broadcast health-specific events
					if newStatus == types.HealthStatusActive {
						events.PublishNodeOnline(agent.NodeID, updatedAgent)
						if hm.presence != nil {
							hm.presence.Touch(agent.NodeID, time.Now())
						}
					} else {
						events.PublishNodeOffline(agent.NodeID, updatedAgent)
					}
					events.PublishNodeHealthChanged(agent.NodeID, string(newStatus), updatedAgent)

					// Send to UI service
					if hm.uiService != nil {
						hm.uiService.OnNodeStatusChanged(updatedAgent)
					}

					// Check MCP health for active agents
					if newStatus == types.HealthStatusActive {
						hm.checkMCPHealthForNode(agent.NodeID)
					}
				}
			}
		}
	} else {
		hm.agentsMutex.Unlock()
	}
}

// checkMCPHealthForNode checks MCP health for a specific node
func (hm *HealthMonitor) checkMCPHealthForNode(nodeID string) {
	if hm.agentClient == nil {
		return
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Fetch MCP health from agent
	healthResponse, err := hm.agentClient.GetMCPHealth(ctx, nodeID)
	if err != nil {
		// Silently continue - agent might not support MCP
		return
	}

	// Convert to domain model
	newMCPSummary := &domain.MCPSummaryData{
		TotalServers:   healthResponse.Summary.TotalServers,
		RunningServers: healthResponse.Summary.RunningServers,
		TotalTools:     healthResponse.Summary.TotalTools,
		OverallHealth:  healthResponse.Summary.OverallHealth,
	}

	// Check if MCP health has changed
	if hm.hasMCPHealthChanged(nodeID, newMCPSummary) {
		// Update cache
		hm.updateMCPHealthCache(nodeID, newMCPSummary)

		// Transform for UI
		uiSummary := &domain.MCPSummaryForUI{
			TotalServers:          newMCPSummary.TotalServers,
			RunningServers:        newMCPSummary.RunningServers,
			TotalTools:            newMCPSummary.TotalTools,
			OverallHealth:         newMCPSummary.OverallHealth,
			HasIssues:             newMCPSummary.RunningServers < newMCPSummary.TotalServers || newMCPSummary.OverallHealth < 0.8,
			CapabilitiesAvailable: newMCPSummary.RunningServers > 0,
		}

		// Set service status for user mode
		if newMCPSummary.OverallHealth >= 0.9 {
			uiSummary.ServiceStatus = string(domain.MCPServiceStatusReady)
		} else if newMCPSummary.OverallHealth >= 0.5 {
			uiSummary.ServiceStatus = string(domain.MCPServiceStatusDegraded)
		} else {
			uiSummary.ServiceStatus = string(domain.MCPServiceStatusUnavailable)
		}

		// Broadcast MCP health change event
		if hm.uiService != nil {
			hm.uiService.OnMCPHealthChanged(nodeID, uiSummary)
		}

		logger.Logger.Debug().Msgf("🔧 MCP health changed for node %s: %d/%d servers running, health: %.2f",
			nodeID, newMCPSummary.RunningServers, newMCPSummary.TotalServers, newMCPSummary.OverallHealth)
	}
}

// hasMCPHealthChanged checks if MCP health has changed for a node
func (hm *HealthMonitor) hasMCPHealthChanged(nodeID string, newSummary *domain.MCPSummaryData) bool {
	hm.mcpCacheMutex.RLock()
	defer hm.mcpCacheMutex.RUnlock()

	cached, exists := hm.mcpHealthCache[nodeID]
	if !exists {
		return true // First time checking this node
	}

	// Compare key metrics
	return cached.TotalServers != newSummary.TotalServers ||
		cached.RunningServers != newSummary.RunningServers ||
		cached.TotalTools != newSummary.TotalTools ||
		cached.OverallHealth != newSummary.OverallHealth
}

// updateMCPHealthCache updates the cached MCP health data for a node
func (hm *HealthMonitor) updateMCPHealthCache(nodeID string, summary *domain.MCPSummaryData) {
	hm.mcpCacheMutex.Lock()
	defer hm.mcpCacheMutex.Unlock()

	hm.mcpHealthCache[nodeID] = summary
}

// GetMCPHealthCache returns the current MCP health cache (for debugging/monitoring)
func (hm *HealthMonitor) GetMCPHealthCache() map[string]*domain.MCPSummaryData {
	hm.mcpCacheMutex.RLock()
	defer hm.mcpCacheMutex.RUnlock()

	// Return a copy to avoid race conditions
	cache := make(map[string]*domain.MCPSummaryData)
	for nodeID, summary := range hm.mcpHealthCache {
		cache[nodeID] = summary
	}
	return cache
}
