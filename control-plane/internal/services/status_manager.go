package services

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/events"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"
)

// StatusManagerConfig holds configuration for the status manager
type StatusManagerConfig struct {
	ReconcileInterval time.Duration // How often to reconcile status
	StatusCacheTTL    time.Duration // How long to cache status
	MaxTransitionTime time.Duration // Max time for state transitions
}

// StatusManager provides a single source of truth for agent status
// It reconciles between different status sources and manages status persistence
type StatusManager struct {
	storage     storage.StorageProvider
	config      StatusManagerConfig
	uiService   *UIService
	agentClient interfaces.AgentClient

	// Status cache for fast access (short-term, 5-second TTL)
	statusCache map[string]*cachedAgentStatus
	cacheMutex  sync.RWMutex

	// Transition tracking
	activeTransitions map[string]*types.StateTransition
	transitionMutex   sync.RWMutex

	// Control channels
	stopCh chan struct{}

	// Event handlers
	eventHandlers []StatusEventHandler
}

// cachedAgentStatus represents a cached status with timestamp
type cachedAgentStatus struct {
	Status    *types.AgentStatus
	Timestamp time.Time
}

func cloneAgentStatus(status *types.AgentStatus) *types.AgentStatus {
	if status == nil {
		return nil
	}

	clone := *status

	if status.MCPStatus != nil {
		mcpCopy := *status.MCPStatus
		clone.MCPStatus = &mcpCopy
	}

	if status.StateTransition != nil {
		transitionCopy := *status.StateTransition
		clone.StateTransition = &transitionCopy
	}

	if status.LastVerified != nil {
		lastVerifiedCopy := *status.LastVerified
		clone.LastVerified = &lastVerifiedCopy
	}

	return &clone
}

// StatusEventHandler defines the interface for status event handlers
type StatusEventHandler interface {
	OnStatusChanged(nodeID string, oldStatus, newStatus *types.AgentStatus)
}

// NewStatusManager creates a new status reconciliation service
func NewStatusManager(storage storage.StorageProvider, config StatusManagerConfig, uiService *UIService, agentClient interfaces.AgentClient) *StatusManager {
	// Set default values
	if config.ReconcileInterval == 0 {
		config.ReconcileInterval = 30 * time.Second
	}
	if config.StatusCacheTTL == 0 {
		config.StatusCacheTTL = 5 * time.Minute
	}
	if config.MaxTransitionTime == 0 {
		config.MaxTransitionTime = 2 * time.Minute
	}

	return &StatusManager{
		storage:           storage,
		config:            config,
		uiService:         uiService,
		agentClient:       agentClient,
		statusCache:       make(map[string]*cachedAgentStatus),
		activeTransitions: make(map[string]*types.StateTransition),
		stopCh:            make(chan struct{}),
		eventHandlers:     make([]StatusEventHandler, 0),
	}
}

// Start begins the status manager background processes
func (sm *StatusManager) Start() {
	logger.Logger.Debug().Msg("🔄 Starting status manager")

	// Start reconciliation loop
	go sm.reconcileLoop()

	// Start transition timeout checker
	go sm.transitionTimeoutLoop()
}

// Stop gracefully shuts down the status manager
func (sm *StatusManager) Stop() {
	logger.Logger.Debug().Msg("🔄 Stopping status manager")
	close(sm.stopCh)
}

// GetAgentStatus retrieves the current unified status for an agent using live health checks
func (sm *StatusManager) GetAgentStatus(ctx context.Context, nodeID string) (*types.AgentStatus, error) {
	// Check short-term cache with intelligent logic
	sm.cacheMutex.RLock()
	if cached, exists := sm.statusCache[nodeID]; exists {
		cacheAge := time.Since(cached.Timestamp)

		// For agents marked as inactive/offline, use cache for up to 5 seconds
		if cached.Status.State == types.AgentStateInactive && cacheAge < 5*time.Second {
			sm.cacheMutex.RUnlock()
			// Return cached status with preserved source attribution
			return cached.Status, nil
		}

		// For agents marked as active, only use very fresh cache (1 second) to ensure responsiveness
		// This prevents serving stale heartbeat data when agents go offline
		if cached.Status.State == types.AgentStateActive && cacheAge < 1*time.Second {
			sm.cacheMutex.RUnlock()
			// Return cached status with preserved source attribution
			return cached.Status, nil
		}

		// For all other cases or expired cache, proceed with live health check
	}
	sm.cacheMutex.RUnlock()

	// Perform live health check via HTTP
	var status *types.AgentStatus
	var healthCheckSuccessful bool

	if sm.agentClient != nil {
		// Create a timeout context for the health check (2-3 seconds)
		healthCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()

		agentStatusResp, err := sm.agentClient.GetAgentStatus(healthCtx, nodeID)
		if err != nil {
			logger.Logger.Debug().Err(err).Str("node_id", nodeID).Msg("🏥 Live health check failed, marking agent as inactive")
			// Health check failed - agent is offline/inactive
			healthCheckSuccessful = false

			// Invalidate cache when health check fails to ensure fresh checks on subsequent requests
			sm.cacheMutex.Lock()
			delete(sm.statusCache, nodeID)
			sm.cacheMutex.Unlock()
		} else {
			logger.Logger.Debug().Str("node_id", nodeID).Str("status", agentStatusResp.Status).Msg("🏥 Live health check successful")
			healthCheckSuccessful = true
		}

		// Create status based on health check result
		now := time.Now()
		if healthCheckSuccessful && agentStatusResp.Status == "running" {
			// Agent is active and running
			status = &types.AgentStatus{
				State:           types.AgentStateActive,
				HealthScore:     85, // Good health from live verification
				LastSeen:        now,
				LifecycleStatus: types.AgentStatusReady,
				HealthStatus:    types.HealthStatusActive,
				LastUpdated:     now,
				LastVerified:    &now, // Set when live health check was performed
				Source:          types.StatusSourceHealthCheck,
			}
		} else {
			// Agent is inactive or not responding
			status = &types.AgentStatus{
				State:           types.AgentStateInactive,
				HealthScore:     0, // No health
				LastSeen:        now,
				LifecycleStatus: types.AgentStatusOffline,
				HealthStatus:    types.HealthStatusInactive,
				LastUpdated:     now,
				LastVerified:    &now, // Set when live health check was performed
				Source:          types.StatusSourceHealthCheck,
			}
		}
	} else {
		// Fallback to storage-based status if no agent client available
		agent, err := sm.storage.GetAgent(ctx, nodeID)
		if err != nil {
			return nil, fmt.Errorf("failed to get agent: %w", err)
		}
		status = types.FromLegacyStatus(agent.HealthStatus, agent.LifecycleStatus, agent.LastHeartbeat)
	}

	// Update storage with live verification result
	if healthCheckSuccessful {
		if err := sm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusActive); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to update agent health status in storage")
		}
	} else {
		if err := sm.storage.UpdateAgentHealth(ctx, nodeID, types.HealthStatusInactive); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to update agent health status in storage")
		}
	}

	// Cache the status with timestamp
	sm.cacheMutex.Lock()
	sm.statusCache[nodeID] = &cachedAgentStatus{
		Status:    status,
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	// Emit SSE events if status changed
	if sm.uiService != nil {
		// Get the agent for event emission
		if agent, err := sm.storage.GetAgent(ctx, nodeID); err == nil {
			sm.uiService.OnNodeStatusChanged(agent)
		}
	}

	return status, nil
}

// GetAgentStatusSnapshot returns the best-known status without performing live health checks.
// This is optimized for UI summaries where fast responses are preferred over live verification.
func (sm *StatusManager) GetAgentStatusSnapshot(ctx context.Context, nodeID string, cachedNode *types.AgentNode) (*types.AgentStatus, error) {
	// Prefer cached status if available
	sm.cacheMutex.RLock()
	if cached, exists := sm.statusCache[nodeID]; exists && cached.Status != nil {
		statusCopy := cloneAgentStatus(cached.Status)
		sm.cacheMutex.RUnlock()
		return statusCopy, nil
	}
	sm.cacheMutex.RUnlock()

	// Use provided node data or pull from storage without hitting agent HTTP endpoints
	var agent *types.AgentNode
	var err error
	if cachedNode != nil {
		agent = cachedNode
	} else {
		agent, err = sm.storage.GetAgent(ctx, nodeID)
		if err != nil {
			return nil, fmt.Errorf("failed to get agent: %w", err)
		}
	}

	status := types.FromLegacyStatus(agent.HealthStatus, agent.LifecycleStatus, agent.LastHeartbeat)
	status.LastSeen = agent.LastHeartbeat
	status.LastUpdated = agent.LastHeartbeat
	status.HealthStatus = agent.HealthStatus
	status.LifecycleStatus = agent.LifecycleStatus
	status.Source = types.StatusSourceReconcile

	sm.cacheMutex.Lock()
	sm.statusCache[nodeID] = &cachedAgentStatus{
		Status:    status,
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	return cloneAgentStatus(status), nil
}

// UpdateAgentStatus updates the agent status with reconciliation
func (sm *StatusManager) UpdateAgentStatus(ctx context.Context, nodeID string, update *types.AgentStatusUpdate) error {
	// Get current status
	currentStatus, err := sm.GetAgentStatus(ctx, nodeID)
	if err != nil {
		return fmt.Errorf("failed to get current status: %w", err)
	}

	// Create a copy for the new status
	newStatus := *currentStatus
	oldStatus := *currentStatus

	// Apply updates
	if update.State != nil {
		if newStatus.State != *update.State {
			// Handle state transition
			if err := sm.handleStateTransition(nodeID, &newStatus, *update.State, update.Reason); err != nil {
				return fmt.Errorf("failed to handle state transition: %w", err)
			}
		}
	}

	if update.HealthScore != nil {
		newStatus.HealthScore = *update.HealthScore
	}

	if update.LifecycleStatus != nil {
		newStatus.LifecycleStatus = *update.LifecycleStatus
	}

	if update.MCPStatus != nil {
		newStatus.MCPStatus = update.MCPStatus
	}

	// Update metadata
	newStatus.LastUpdated = time.Now()
	newStatus.Source = update.Source

	// Update backward compatibility fields
	newStatus.HealthStatus = newStatus.ToLegacyHealthStatus()
	if newStatus.LifecycleStatus == "" {
		newStatus.LifecycleStatus = newStatus.ToLegacyLifecycleStatus()
	}

	// Persist to storage
	if err := sm.persistStatus(ctx, nodeID, &newStatus); err != nil {
		return fmt.Errorf("failed to persist status: %w", err)
	}

	// Update cache
	sm.cacheMutex.Lock()
	sm.statusCache[nodeID] = &cachedAgentStatus{
		Status:    &newStatus,
		Timestamp: time.Now(),
	}
	sm.cacheMutex.Unlock()

	// Notify event handlers
	sm.notifyStatusChanged(nodeID, &oldStatus, &newStatus)

	// Broadcast events
	sm.broadcastStatusEvents(nodeID, &oldStatus, &newStatus)

	logger.Logger.Debug().
		Str("node_id", nodeID).
		Str("old_state", string(oldStatus.State)).
		Str("new_state", string(newStatus.State)).
		Int("health_score", newStatus.HealthScore).
		Str("source", string(update.Source)).
		Msg("🔄 Agent status updated")

	return nil
}

// UpdateFromHeartbeat updates status based on heartbeat data
func (sm *StatusManager) UpdateFromHeartbeat(ctx context.Context, nodeID string, lifecycleStatus *types.AgentLifecycleStatus, mcpStatus *types.MCPStatusInfo) error {
	currentStatus, err := sm.GetAgentStatus(ctx, nodeID)
	if err != nil {
		// If agent doesn't exist, create new status
		currentStatus = types.NewAgentStatus(types.AgentStateStarting, types.StatusSourceHeartbeat)
	}

	// INTELLIGENT HEARTBEAT PROCESSING:
	// If we recently performed a live health check that determined the agent is offline,
	// don't override that with heartbeat data (which could be stale/delayed)
	if currentStatus.Source == types.StatusSourceHealthCheck &&
		currentStatus.State == types.AgentStateInactive &&
		currentStatus.LastVerified != nil &&
		time.Since(*currentStatus.LastVerified) < 10*time.Second {

		logger.Logger.Debug().
			Str("node_id", nodeID).
			Str("current_source", string(currentStatus.Source)).
			Str("current_state", string(currentStatus.State)).
			Msg("🚫 Ignoring heartbeat update - recent health check determined agent is offline")

		// Don't process this heartbeat as it conflicts with recent live health check
		return nil
	}

	// Update from heartbeat
	currentStatus.UpdateFromHeartbeat(lifecycleStatus, mcpStatus)

	// Persist changes
	update := &types.AgentStatusUpdate{
		LifecycleStatus: lifecycleStatus,
		MCPStatus:       mcpStatus,
		Source:          types.StatusSourceHeartbeat,
		Reason:          "heartbeat update",
	}

	return sm.UpdateAgentStatus(ctx, nodeID, update)
}

// RefreshAgentStatus manually refreshes an agent's status
func (sm *StatusManager) RefreshAgentStatus(ctx context.Context, nodeID string) error {
	// Clear cache to force reload
	sm.cacheMutex.Lock()
	delete(sm.statusCache, nodeID)
	sm.cacheMutex.Unlock()

	// Reload status
	refreshedStatus, err := sm.GetAgentStatus(ctx, nodeID)
	if err != nil {
		return fmt.Errorf("failed to refresh status: %w", err)
	}

	// Broadcast refresh event
	events.PublishNodeStatusRefreshed(nodeID, refreshedStatus)

	logger.Logger.Debug().Str("node_id", nodeID).Msg("🔄 Agent status refreshed")
	return nil
}

// AddEventHandler adds a status event handler
func (sm *StatusManager) AddEventHandler(handler StatusEventHandler) {
	sm.eventHandlers = append(sm.eventHandlers, handler)
}

// handleStateTransition manages state transitions
func (sm *StatusManager) handleStateTransition(nodeID string, status *types.AgentStatus, newState types.AgentState, reason string) error {
	// Check if transition is valid
	if !sm.isValidTransition(status.State, newState) {
		return fmt.Errorf("invalid state transition from %s to %s", status.State, newState)
	}

	// Start transition
	status.StartTransition(newState, reason)

	// Track active transition
	sm.transitionMutex.Lock()
	sm.activeTransitions[nodeID] = status.StateTransition
	sm.transitionMutex.Unlock()

	// For immediate transitions, complete right away
	if sm.isImmediateTransition(status.State, newState) {
		status.CompleteTransition()

		// Remove from active transitions
		sm.transitionMutex.Lock()
		delete(sm.activeTransitions, nodeID)
		sm.transitionMutex.Unlock()
	}

	return nil
}

// isValidTransition checks if a state transition is valid
func (sm *StatusManager) isValidTransition(from, to types.AgentState) bool {
	validTransitions := map[types.AgentState][]types.AgentState{
		types.AgentStateInactive: {types.AgentStateStarting, types.AgentStateActive},
		types.AgentStateStarting: {types.AgentStateActive, types.AgentStateInactive},
		types.AgentStateActive:   {types.AgentStateInactive, types.AgentStateStopping},
		types.AgentStateStopping: {types.AgentStateInactive},
	}

	allowed, exists := validTransitions[from]
	if !exists {
		return false
	}

	for _, allowedState := range allowed {
		if allowedState == to {
			return true
		}
	}

	return false
}

// isImmediateTransition checks if a transition should complete immediately
func (sm *StatusManager) isImmediateTransition(from, to types.AgentState) bool {
	// Most transitions are immediate except starting->active which may take time
	return !(from == types.AgentStateStarting && to == types.AgentStateActive)
}

// persistStatus persists the status to storage
func (sm *StatusManager) persistStatus(ctx context.Context, nodeID string, status *types.AgentStatus) error {
	// Update health status
	if err := sm.storage.UpdateAgentHealth(ctx, nodeID, status.HealthStatus); err != nil {
		return fmt.Errorf("failed to update health status: %w", err)
	}

	// Update lifecycle status
	if err := sm.storage.UpdateAgentLifecycleStatus(ctx, nodeID, status.LifecycleStatus); err != nil {
		return fmt.Errorf("failed to update lifecycle status: %w", err)
	}

	// Update heartbeat timestamp
	if err := sm.storage.UpdateAgentHeartbeat(ctx, nodeID, status.LastSeen); err != nil {
		return fmt.Errorf("failed to update heartbeat: %w", err)
	}

	return nil
}

// notifyStatusChanged notifies all event handlers of status changes
func (sm *StatusManager) notifyStatusChanged(nodeID string, oldStatus, newStatus *types.AgentStatus) {
	for _, handler := range sm.eventHandlers {
		go func(h StatusEventHandler) {
			defer func() {
				if r := recover(); r != nil {
					logger.Logger.Error().
						Interface("panic", r).
						Str("node_id", nodeID).
						Msg("❌ Panic in status event handler")
				}
			}()
			h.OnStatusChanged(nodeID, oldStatus, newStatus)
		}(handler)
	}
}

// broadcastStatusEvents broadcasts status change events using enhanced event system
func (sm *StatusManager) broadcastStatusEvents(nodeID string, oldStatus, newStatus *types.AgentStatus) {
	// Get updated agent for events
	ctx := context.Background()
	agent, err := sm.storage.GetAgent(ctx, nodeID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("❌ Failed to get agent for event broadcasting")
		return
	}

	// FIXED: Only broadcast ONE unified status change event to prevent spam
	// This single event contains all the necessary information for the frontend
	events.PublishNodeUnifiedStatusChanged(nodeID, oldStatus, newStatus, string(newStatus.Source), "status update")

	// FIXED: Only broadcast legacy events if specifically needed for backward compatibility
	// and only if state actually changed to prevent duplicate events
	if oldStatus.State != newStatus.State {
		switch newStatus.State {
		case types.AgentStateActive:
			events.PublishNodeOnline(nodeID, agent)
		case types.AgentStateInactive:
			events.PublishNodeOffline(nodeID, agent)
		}
	}

	// FIXED: Removed duplicate event publishing:
	// - Removed PublishNodeStateTransition (redundant with unified event)
	// - Removed PublishNodeHealthChangedEnhanced (redundant with unified event)
	// - Removed PublishNodeStatusUpdatedEnhanced (was calling PublishNodeUnifiedStatusChanged again!)

	// Notify UI service for SSE broadcasting (this goes through deduplication)
	if sm.uiService != nil {
		sm.uiService.OnNodeStatusChanged(agent)
	}
}

// reconcileLoop periodically reconciles status across all agents
func (sm *StatusManager) reconcileLoop() {
	ticker := time.NewTicker(sm.config.ReconcileInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sm.performReconciliation()
		case <-sm.stopCh:
			return
		}
	}
}

// performReconciliation reconciles status for all agents
func (sm *StatusManager) performReconciliation() {
	ctx := context.Background()

	// Get all agents
	agents, err := sm.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("❌ Failed to list agents for reconciliation")
		return
	}

	logger.Logger.Debug().Int("agent_count", len(agents)).Msg("🔄 Starting status reconciliation")

	for _, agent := range agents {
		// Check if status needs reconciliation
		if sm.needsReconciliation(agent) {
			if err := sm.reconcileAgentStatus(ctx, agent); err != nil {
				logger.Logger.Error().
					Err(err).
					Str("node_id", agent.ID).
					Msg("❌ Failed to reconcile agent status")
			}
		}
	}
}

// needsReconciliation checks if an agent needs status reconciliation
func (sm *StatusManager) needsReconciliation(agent *types.AgentNode) bool {
	// Check if last heartbeat is too old
	timeSinceHeartbeat := time.Since(agent.LastHeartbeat)
	if timeSinceHeartbeat > 30*time.Second && agent.HealthStatus == types.HealthStatusActive {
		return true
	}

	// Check for inconsistent status
	if agent.HealthStatus == types.HealthStatusActive && agent.LifecycleStatus == types.AgentStatusOffline {
		return true
	}

	return false
}

// reconcileAgentStatus reconciles status for a specific agent
func (sm *StatusManager) reconcileAgentStatus(ctx context.Context, agent *types.AgentNode) error {
	// Determine correct status based on heartbeat age
	timeSinceHeartbeat := time.Since(agent.LastHeartbeat)

	var newHealthStatus types.HealthStatus
	var newLifecycleStatus types.AgentLifecycleStatus

	if timeSinceHeartbeat > 30*time.Second {
		newHealthStatus = types.HealthStatusInactive
		newLifecycleStatus = types.AgentStatusOffline
	} else {
		newHealthStatus = types.HealthStatusActive
		if agent.LifecycleStatus == "" || agent.LifecycleStatus == types.AgentStatusOffline {
			newLifecycleStatus = types.AgentStatusReady
		} else {
			newLifecycleStatus = agent.LifecycleStatus
		}
	}

	// Update if changed
	if agent.HealthStatus != newHealthStatus || agent.LifecycleStatus != newLifecycleStatus {
		update := &types.AgentStatusUpdate{
			Source: types.StatusSourceReconcile,
			Reason: "periodic reconciliation",
		}

		if agent.HealthStatus != newHealthStatus {
			newState := types.AgentStateInactive
			if newHealthStatus == types.HealthStatusActive {
				newState = types.AgentStateActive
			}
			update.State = &newState
		}

		if agent.LifecycleStatus != newLifecycleStatus {
			update.LifecycleStatus = &newLifecycleStatus
		}

		return sm.UpdateAgentStatus(ctx, agent.ID, update)
	}

	return nil
}

// transitionTimeoutLoop checks for stuck transitions
func (sm *StatusManager) transitionTimeoutLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sm.checkTransitionTimeouts()
		case <-sm.stopCh:
			return
		}
	}
}

// checkTransitionTimeouts checks for and handles stuck transitions
func (sm *StatusManager) checkTransitionTimeouts() {
	sm.transitionMutex.Lock()
	defer sm.transitionMutex.Unlock()

	now := time.Now()
	for nodeID, transition := range sm.activeTransitions {
		if now.Sub(transition.StartedAt) > sm.config.MaxTransitionTime {
			logger.Logger.Warn().
				Str("node_id", nodeID).
				Str("from", string(transition.From)).
				Str("to", string(transition.To)).
				Dur("duration", now.Sub(transition.StartedAt)).
				Msg("🔄 Transition timeout, forcing completion")

			// Force complete the transition
			ctx := context.Background()
			if status, err := sm.GetAgentStatus(ctx, nodeID); err == nil {
				status.CompleteTransition()
				if err := sm.persistStatus(ctx, nodeID, status); err != nil {
					logger.Logger.Warn().
						Err(err).
						Str("node_id", nodeID).
						Msg("failed to persist status during transition timeout")
				}
			}

			delete(sm.activeTransitions, nodeID)
		}
	}
}
