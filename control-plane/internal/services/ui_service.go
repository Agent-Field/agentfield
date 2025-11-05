package services

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/events"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"
	"sync"
	"time"
)

// NodeEvent represents a real-time event related to an agent node.
type NodeEvent struct {
	Type      string      `json:"type"` // e.g., "node_registered", "node_health_changed", "node_removed"
	Node      interface{} `json:"node"` // Can be AgentNodeSummaryForUI or full AgentNode
	Timestamp time.Time   `json:"timestamp"`
}

// UIService provides data optimized for the UI and manages SSE clients.
type UIService struct {
	storage       storage.StorageProvider
	agentClient   interfaces.AgentClient
	agentService  interfaces.AgentService // Add AgentService for robust status checking
	statusManager *StatusManager          // Unified status management
	// clients map[chan NodeEvent]bool // Deprecated: Use sync.Map for concurrent access
	clients sync.Map // Map of chan NodeEvent to bool (true if active)

	// Event deduplication
	lastEventCache  map[string]NodeEvent
	eventCacheMutex sync.RWMutex

	// Connection management
	heartbeatTicker *time.Ticker
	stopHeartbeat   chan struct{}
}

// NewUIService creates a new UIService.
func NewUIService(storageProvider storage.StorageProvider, agentClient interfaces.AgentClient, agentService interfaces.AgentService, statusManager *StatusManager) *UIService {
	service := &UIService{
		storage:        storageProvider,
		agentClient:    agentClient,
		agentService:   agentService,
		statusManager:  statusManager,
		clients:        sync.Map{},
		lastEventCache: make(map[string]NodeEvent),
		stopHeartbeat:  make(chan struct{}),
	}

	// Start heartbeat mechanism to keep connections alive
	service.startHeartbeat()

	return service
}

// AgentNodeSummaryForUI is a subset of types.AgentNode for summary display.
type AgentNodeSummaryForUI struct {
	ID              string                     `json:"id"`
	TeamID          string                     `json:"team_id"`
	Version         string                     `json:"version"`
	HealthStatus    types.HealthStatus         `json:"health_status"`
	LifecycleStatus types.AgentLifecycleStatus `json:"lifecycle_status"`
	ReasonerCount   int                        `json:"reasoner_count"`
	SkillCount      int                        `json:"skill_count"`
	LastHeartbeat   time.Time                  `json:"last_heartbeat"`

	// New MCP fields
	MCPSummary *domain.MCPSummaryForUI `json:"mcp_summary,omitempty"`
}

// GetNodesSummary retrieves a list of node summaries with robust status checking.
// This method ensures consistency by using the same reconciliation logic as the detailed status endpoint.
func (s *UIService) GetNodesSummary(ctx context.Context) ([]AgentNodeSummaryForUI, int, error) {
	nodes, err := s.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Error listing agents")
		return nil, 0, err
	}

	logger.Logger.Debug().Msgf("📊 Found %d registered nodes", len(nodes))
	for i, node := range nodes {
		logger.Logger.Debug().Msgf("  Node %d: ID=%s, TeamID=%s, Version=%s, Status=%s, LastHeartbeat=%s",
			i+1, node.ID, node.TeamID, node.Version, node.HealthStatus, node.LastHeartbeat.Format(time.RFC3339))
	}

	summaries := make([]AgentNodeSummaryForUI, len(nodes))
	for i, node := range nodes {
		// Use the robust status reconciliation from AgentService as single source of truth
		lifecycleStatus, healthStatus := s.getReconciledNodeStatus(node.ID, node)

		summaries[i] = AgentNodeSummaryForUI{
			ID:              node.ID,
			TeamID:          node.TeamID,
			Version:         node.Version,
			HealthStatus:    healthStatus,
			LifecycleStatus: lifecycleStatus,
			ReasonerCount:   len(node.Reasoners),
			SkillCount:      len(node.Skills),
			LastHeartbeat:   node.LastHeartbeat,
		}

		// Enhance with MCP health data
		s.enhanceNodeSummaryWithMCP(&summaries[i])
	}
	return summaries, len(summaries), nil
}

// getReconciledNodeStatus provides a single source of truth for node status by using
// the unified status management system.
func (s *UIService) getReconciledNodeStatus(nodeID string, node *types.AgentNode) (types.AgentLifecycleStatus, types.HealthStatus) {
	// Use StatusManager snapshot as the primary source of truth without triggering live probes
	if s.statusManager != nil {
		ctx := context.Background()
		unifiedStatus, err := s.statusManager.GetAgentStatusSnapshot(ctx, nodeID, node)
		if err == nil && unifiedStatus != nil {
			logger.Logger.Debug().Msgf("🔧 UNIFIED: Using cached status for node %s: state=%s, health=%d",
				nodeID, unifiedStatus.State, unifiedStatus.HealthScore)
			return unifiedStatus.LifecycleStatus, unifiedStatus.HealthStatus
		}
		logger.Logger.Warn().Err(err).Msgf("⚠️  Failed to get cached status for node %s, using fallback", nodeID)
	}

	// Fallback to AgentService if StatusManager is not available
	if s.agentService != nil {
		agentStatus, err := s.agentService.GetAgentStatus(nodeID)
		if err == nil && agentStatus != nil {
			// AgentService provides the authoritative running state
			if agentStatus.IsRunning {
				// If agent is actually running, set appropriate lifecycle status
				if node.LifecycleStatus == "" || node.LifecycleStatus == "offline" {
					logger.Logger.Debug().Msgf("🔧 RECONCILE: Node %s is running but lifecycle was %s, setting to ready", nodeID, node.LifecycleStatus)
					return "ready", "active"
				}
				// Keep existing lifecycle status if it's already a running state
				if node.LifecycleStatus == "ready" || node.LifecycleStatus == "degraded" {
					return node.LifecycleStatus, "active"
				}
				return "ready", "active"
			} else {
				// Agent is not running according to process reconciliation
				logger.Logger.Debug().Msgf("🔧 RECONCILE: Node %s is not running, setting to offline", nodeID)
				return "offline", "inactive"
			}
		}
		// If AgentService call failed, log warning but continue with fallback
		logger.Logger.Warn().Err(err).Msgf("⚠️  Failed to get reconciled status for node %s, using fallback logic", nodeID)
	}

	// Final fallback: Ensure consistent state - fix the inconsistent "inactive + ready" issue
	lifecycleStatus := node.LifecycleStatus
	healthStatus := node.HealthStatus

	// CONSISTENCY FIX: Ensure health and lifecycle status are consistent
	if healthStatus == "inactive" {
		// If health is inactive, lifecycle should be offline
		if lifecycleStatus == "ready" || lifecycleStatus == "degraded" {
			logger.Logger.Warn().Msgf("🔧 CONSISTENCY: Node %s has inactive health but %s lifecycle, correcting to offline", nodeID, lifecycleStatus)
			lifecycleStatus = "offline"
		}
	} else if healthStatus == "active" {
		// If health is active, lifecycle should not be offline
		if lifecycleStatus == "" || lifecycleStatus == "offline" {
			logger.Logger.Warn().Msgf("🔧 CONSISTENCY: Node %s has active health but %s lifecycle, correcting to ready", nodeID, lifecycleStatus)
			lifecycleStatus = "ready"
		}
	}

	return lifecycleStatus, healthStatus
}

// NodeDetailsWithPackageInfo represents node details enhanced with package information
type NodeDetailsWithPackageInfo struct {
	*types.AgentNode
	PackageInfo *PackageInfo `json:"package_info,omitempty"`
}

// PackageInfo represents package information for the node details response
type PackageInfo struct {
	PackageID string `json:"package_id"`
	Version   string `json:"version"`
	Status    string `json:"status"`
}

// GetNodeDetails retrieves full details for a specific node.
// For now, it's the same as storage.GetAgent, but can be optimized later.
func (s *UIService) GetNodeDetails(ctx context.Context, nodeID string) (*types.AgentNode, error) {
	return s.storage.GetAgent(ctx, nodeID)
}

// GetNodeDetailsWithPackageInfo retrieves full details for a specific node including package information.
func (s *UIService) GetNodeDetailsWithPackageInfo(ctx context.Context, nodeID string) (*NodeDetailsWithPackageInfo, error) {
	// Get base node details
	node, err := s.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, err
	}

	// Create response with node details
	response := &NodeDetailsWithPackageInfo{
		AgentNode: node,
	}

	// Find the package that corresponds to this node by searching through package configurations
	// The relationship is defined in the package's configuration_schema: agent_node.node_id
	agentPackage, err := s.findPackageByNodeID(ctx, nodeID)
	if err != nil {
		// Log the error but don't fail the request - package info is optional
		logger.Logger.Warn().Err(err).Msgf("Failed to find package for node %s", nodeID)
	} else {
		// Add package information to response
		response.PackageInfo = &PackageInfo{
			PackageID: agentPackage.ID,
			Version:   agentPackage.Version,
			Status:    string(agentPackage.Status),
		}
	}

	return response, nil
}

// findPackageByNodeID searches for the package that contains the given node_id in its configuration
func (s *UIService) findPackageByNodeID(ctx context.Context, nodeID string) (*types.AgentPackage, error) {
	// Query all packages to find the one with matching node_id in configuration
	packages, err := s.storage.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		return nil, err
	}

	for _, pkg := range packages {
		if pkg.ConfigurationSchema != nil {
			// Parse the configuration schema to find agent_node.node_id
			var config map[string]interface{}
			if err := json.Unmarshal(pkg.ConfigurationSchema, &config); err != nil {
				continue // Skip packages with invalid schema
			}

			// Check if this package's configuration contains our node_id
			if agentNode, ok := config["agent_node"].(map[string]interface{}); ok {
				if configNodeID, ok := agentNode["node_id"].(string); ok && configNodeID == nodeID {
					return pkg, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("no package found for node_id: %s", nodeID)
}

// RegisterClient registers a new SSE client and returns a channel for events.
func (s *UIService) RegisterClient() chan NodeEvent {
	clientChan := make(chan NodeEvent)
	s.clients.Store(clientChan, true)
	logger.Logger.Debug().Msgf("➕ SSE client registered. Total clients: %d", s.countClients())
	return clientChan
}

// DeregisterClient removes an SSE client.
func (s *UIService) DeregisterClient(clientChan chan NodeEvent) {
	if _, exists := s.clients.LoadAndDelete(clientChan); exists {
		// Only close the channel if it was actually in our map
		// Use a safe close approach
		defer func() {
			if r := recover(); r != nil {
				// Channel was already closed, ignore the panic
			}
		}()
		close(clientChan)
		logger.Logger.Debug().Msgf("➖ SSE client deregistered. Total clients: %d", s.countClients())
	}
}

// BroadcastEvent sends an event to all registered SSE clients with deduplication.
func (s *UIService) BroadcastEvent(eventType string, node interface{}) {
	event := NodeEvent{
		Type:      eventType,
		Node:      node,
		Timestamp: time.Now(),
	}

	// Check for event deduplication
	if s.isDuplicateEvent(event) {
		logger.Logger.Debug().Msgf("🔄 Skipping duplicate event: %s", eventType)
		return
	}

	// Cache the event for deduplication
	s.cacheEvent(event)

	// Broadcast to all clients with improved error handling
	var failedClients []chan NodeEvent
	clientCount := 0

	s.clients.Range(func(key, value interface{}) bool {
		clientChan, ok := key.(chan NodeEvent)
		if !ok {
			return true // Continue iteration
		}
		clientCount++

		select {
		case clientChan <- event:
			// Event sent successfully
		case <-time.After(100 * time.Millisecond):
			// Client channel is blocked or slow, mark for removal
			failedClients = append(failedClients, clientChan)
			logger.Logger.Warn().Msgf("⚠️ SSE client timeout, marking for removal")
		}
		return true // Continue iteration
	})

	// Remove failed clients
	for _, clientChan := range failedClients {
		s.DeregisterClient(clientChan)
	}

	logger.Logger.Debug().Msgf("📡 Broadcasted %s event to %d clients (%d failed)", eventType, clientCount-len(failedClients), len(failedClients))
}

// countClients returns the number of active SSE clients.
func (s *UIService) countClients() int {
	count := 0
	s.clients.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

// OnAgentRegistered is a callback for when an agent is registered.
func (s *UIService) OnAgentRegistered(node *types.AgentNode) {
	summary := AgentNodeSummaryForUI{
		ID:              node.ID,
		TeamID:          node.TeamID,
		Version:         node.Version,
		HealthStatus:    node.HealthStatus,
		LifecycleStatus: node.LifecycleStatus,
		ReasonerCount:   len(node.Reasoners),
		SkillCount:      len(node.Skills),
		LastHeartbeat:   node.LastHeartbeat,
	}

	// Only keep the new SSE broadcast - health monitor will handle status events
	s.BroadcastEvent("node_registered", summary)
}

// OnNodeStatusChanged is a callback for when an agent's status (health or lifecycle) changes.
// It sends a single, consolidated event to the frontend.
func (s *UIService) OnNodeStatusChanged(node *types.AgentNode) {
	summary := AgentNodeSummaryForUI{
		ID:              node.ID,
		TeamID:          node.TeamID,
		Version:         node.Version,
		HealthStatus:    node.HealthStatus,
		LifecycleStatus: node.LifecycleStatus,
		ReasonerCount:   len(node.Reasoners),
		SkillCount:      len(node.Skills),
		LastHeartbeat:   node.LastHeartbeat,
	}
	s.BroadcastEvent("node_status_changed", summary)

	// CRITICAL FIX: Also broadcast reasoner-specific events for immediate UI updates
	s.OnReasonerStatusChanged(node)
}

// OnReasonerStatusChanged broadcasts reasoner-specific status change events
// This ensures the reasoners UI gets immediate updates when node status changes
func (s *UIService) OnReasonerStatusChanged(node *types.AgentNode) {
	// Determine effective reasoner status based on node health and lifecycle
	reasonerStatus := "online"
	if node.HealthStatus != types.HealthStatusActive || node.LifecycleStatus == types.AgentStatusOffline {
		reasonerStatus = "offline"
	}

	// Broadcast individual reasoner status events
	for _, reasoner := range node.Reasoners {
		reasonerEvent := map[string]interface{}{
			"reasoner_id": reasoner.ID,
			"node_id":     node.ID,
			"status":      reasonerStatus,
			"timestamp":   node.LastHeartbeat,
		}
		s.BroadcastEvent("reasoner_status_changed", reasonerEvent)
	}

	// Also broadcast a general reasoners refresh event
	s.BroadcastEvent("reasoners_refresh", map[string]interface{}{
		"node_id":        node.ID,
		"reasoner_count": len(node.Reasoners),
		"status":         reasonerStatus,
	})
}

// OnAgentRemoved is a callback for when an agent is removed.
func (s *UIService) OnAgentRemoved(nodeID string) {
	// Only keep the new SSE broadcast - health monitor will handle status events
	s.BroadcastEvent("node_removed", map[string]string{"id": nodeID})
}

// fetchMCPHealthForNode retrieves MCP health data for a specific node
func (s *UIService) fetchMCPHealthForNode(nodeID string, mode domain.MCPHealthMode) (*domain.MCPSummaryForUI, []domain.MCPServerHealthForUI, error) {
	if s.agentClient == nil {
		// Agent client not available, return empty data
		return nil, nil, nil
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Fetch MCP health from agent
	healthResponse, err := s.agentClient.GetMCPHealth(ctx, nodeID)
	if err != nil {
		// Log error but don't fail - agent might not support MCP
		logger.Logger.Warn().Err(err).Msgf("Failed to fetch MCP health for node %s", nodeID)
		return nil, nil, nil
	}

	// Transform to domain models
	healthData := &domain.MCPHealthResponseData{
		Servers: make([]domain.MCPServerHealthData, len(healthResponse.Servers)),
		Summary: domain.MCPSummaryData{
			TotalServers:   healthResponse.Summary.TotalServers,
			RunningServers: healthResponse.Summary.RunningServers,
			TotalTools:     healthResponse.Summary.TotalTools,
			OverallHealth:  healthResponse.Summary.OverallHealth,
		},
	}

	for i, server := range healthResponse.Servers {
		var startedAt, lastHealthCheck *time.Time

		// Convert FlexibleTime to *time.Time
		if server.StartedAt != nil {
			t := server.StartedAt.Time
			startedAt = &t
		}
		if server.LastHealthCheck != nil {
			t := server.LastHealthCheck.Time
			lastHealthCheck = &t
		}

		healthData.Servers[i] = domain.MCPServerHealthData{
			Alias:           server.Alias,
			Status:          server.Status,
			ToolCount:       server.ToolCount,
			StartedAt:       startedAt,
			LastHealthCheck: lastHealthCheck,
			ErrorMessage:    server.ErrorMessage,
			Port:            server.Port,
			ProcessID:       server.ProcessID,
			SuccessRate:     server.SuccessRate,
			AvgResponseTime: server.AvgResponseTime,
		}
	}

	// Transform based on mode
	summary, servers := domain.TransformMCPHealthForMode(healthData, mode)
	return summary, servers, nil
}

// GetNodeDetailsWithMCP retrieves full details for a specific node including MCP data
func (s *UIService) GetNodeDetailsWithMCP(ctx context.Context, nodeID string, mode domain.MCPHealthMode) (*domain.AgentNodeDetailsForUI, error) {
	// Get base node details
	node, err := s.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, err
	}

	// Create base details
	details := &domain.AgentNodeDetailsForUI{
		ID:            node.ID,
		TeamID:        node.TeamID,
		BaseURL:       node.BaseURL,
		Version:       node.Version,
		HealthStatus:  string(node.HealthStatus),
		LastHeartbeat: node.LastHeartbeat,
		RegisteredAt:  node.RegisteredAt,
	}

	// Fetch MCP health data
	mcpSummary, mcpServers, err := s.fetchMCPHealthForNode(nodeID, mode)
	if err != nil {
		// Log error but continue without MCP data
		logger.Logger.Warn().Err(err).Msgf("Failed to fetch MCP health for node details %s", nodeID)
	} else {
		details.MCPSummary = mcpSummary
		if mode == domain.MCPHealthModeDeveloper {
			details.MCPServers = mcpServers
		}
	}

	return details, nil
}

// enhanceNodeSummaryWithMCP adds MCP health data to a node summary
func (s *UIService) enhanceNodeSummaryWithMCP(summary *AgentNodeSummaryForUI) {
	if s.agentClient == nil {
		return
	}

	// Fetch MCP health in user mode for summary
	mcpSummary, _, err := s.fetchMCPHealthForNode(summary.ID, domain.MCPHealthModeUser)
	if err != nil {
		// Silently continue without MCP data
		return
	}

	summary.MCPSummary = mcpSummary
}

// OnMCPHealthChanged is a callback for when MCP health status changes
func (s *UIService) OnMCPHealthChanged(nodeID string, mcpSummary *domain.MCPSummaryForUI) {
	mcpData := map[string]interface{}{
		"node_id":     nodeID,
		"mcp_summary": mcpSummary,
		"timestamp":   time.Now(),
	}

	// Publish to dedicated node event bus for immediate updates
	events.PublishNodeMCPHealthChanged(nodeID, mcpData)

	// Keep legacy broadcast for backward compatibility
	s.BroadcastEvent("mcp_health_changed", mcpData)
}

// startHeartbeat starts the SSE heartbeat mechanism to keep connections alive
func (s *UIService) startHeartbeat() {
	s.heartbeatTicker = time.NewTicker(30 * time.Second) // Send heartbeat every 30 seconds

	go func() {
		defer s.heartbeatTicker.Stop()

		for {
			select {
			case <-s.heartbeatTicker.C:
				if s.countClients() > 0 {
					s.BroadcastEvent("heartbeat", map[string]interface{}{
						"timestamp": time.Now(),
						"clients":   s.countClients(),
					})
				}
			case <-s.stopHeartbeat:
				return
			}
		}
	}()

	logger.Logger.Debug().Msg("🫀 SSE heartbeat mechanism started")
}

// StopHeartbeat stops the SSE heartbeat mechanism
func (s *UIService) StopHeartbeat() {
	if s.stopHeartbeat != nil {
		close(s.stopHeartbeat)
	}
	if s.heartbeatTicker != nil {
		s.heartbeatTicker.Stop()
	}
	logger.Logger.Debug().Msg("🫀 SSE heartbeat mechanism stopped")
}

// isDuplicateEvent checks if an event is a duplicate of the last cached event
func (s *UIService) isDuplicateEvent(event NodeEvent) bool {
	s.eventCacheMutex.RLock()
	defer s.eventCacheMutex.RUnlock()

	// Create a cache key based on event type and node data
	cacheKey := s.getEventCacheKey(event)
	if cacheKey == "" {
		return false // Can't determine, allow the event
	}

	lastEvent, exists := s.lastEventCache[cacheKey]
	if !exists {
		return false
	}

	// Check if events are too close in time (within 1 second)
	if time.Since(lastEvent.Timestamp) < 1*time.Second {
		// For status events, also check if the actual status changed
		if event.Type == "node_status_changed" || event.Type == "node_health_changed" {
			return s.compareStatusEvents(lastEvent, event)
		}
		return true
	}

	return false
}

// cacheEvent caches an event for deduplication
func (s *UIService) cacheEvent(event NodeEvent) {
	s.eventCacheMutex.Lock()
	defer s.eventCacheMutex.Unlock()

	cacheKey := s.getEventCacheKey(event)
	if cacheKey != "" {
		s.lastEventCache[cacheKey] = event

		// Clean up old cache entries (keep only last 100)
		if len(s.lastEventCache) > 100 {
			// Remove oldest entries
			oldestTime := time.Now()
			oldestKey := ""
			for key, cachedEvent := range s.lastEventCache {
				if cachedEvent.Timestamp.Before(oldestTime) {
					oldestTime = cachedEvent.Timestamp
					oldestKey = key
				}
			}
			if oldestKey != "" {
				delete(s.lastEventCache, oldestKey)
			}
		}
	}
}

// getEventCacheKey generates a cache key for an event
func (s *UIService) getEventCacheKey(event NodeEvent) string {
	switch event.Type {
	case "node_status_changed", "node_health_changed", "node_registered":
		if summary, ok := event.Node.(AgentNodeSummaryForUI); ok {
			return fmt.Sprintf("%s:%s", event.Type, summary.ID)
		}
	case "mcp_health_changed":
		if data, ok := event.Node.(map[string]interface{}); ok {
			if nodeID, exists := data["node_id"]; exists {
				return fmt.Sprintf("%s:%v", event.Type, nodeID)
			}
		}
	case "node_removed":
		if data, ok := event.Node.(map[string]string); ok {
			if nodeID, exists := data["id"]; exists {
				return fmt.Sprintf("%s:%s", event.Type, nodeID)
			}
		}
	}
	return ""
}

// compareStatusEvents compares two status events to see if they represent the same status
func (s *UIService) compareStatusEvents(lastEvent, newEvent NodeEvent) bool {
	lastSummary, lastOk := lastEvent.Node.(AgentNodeSummaryForUI)
	newSummary, newOk := newEvent.Node.(AgentNodeSummaryForUI)

	if !lastOk || !newOk {
		return false // Can't compare, allow the event
	}

	// Compare relevant status fields
	return lastSummary.HealthStatus == newSummary.HealthStatus &&
		lastSummary.LifecycleStatus == newSummary.LifecycleStatus
}

// RefreshNodeStatus manually refreshes a node's status through the unified system
func (s *UIService) RefreshNodeStatus(ctx context.Context, nodeID string) error {
	if s.statusManager == nil {
		return fmt.Errorf("status manager not available")
	}

	return s.statusManager.RefreshAgentStatus(ctx, nodeID)
}

// GetUnifiedNodeStatus gets the unified status for a node
func (s *UIService) GetUnifiedNodeStatus(ctx context.Context, nodeID string) (*types.AgentStatus, error) {
	if s.statusManager == nil {
		return nil, fmt.Errorf("status manager not available")
	}

	return s.statusManager.GetAgentStatus(ctx, nodeID)
}

// GetNodeUnifiedStatus gets the unified status for a node (alias for GetUnifiedNodeStatus)
func (s *UIService) GetNodeUnifiedStatus(ctx context.Context, nodeID string) (*types.AgentStatus, error) {
	return s.GetUnifiedNodeStatus(ctx, nodeID)
}

// BulkNodeStatus gets unified status for multiple nodes
func (s *UIService) BulkNodeStatus(ctx context.Context, nodeIDs []string) (map[string]*types.AgentStatus, error) {
	if s.statusManager == nil {
		return nil, fmt.Errorf("status manager not available")
	}

	statuses := make(map[string]*types.AgentStatus)
	for _, nodeID := range nodeIDs {
		status, err := s.statusManager.GetAgentStatus(ctx, nodeID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("node_id", nodeID).Msg("Failed to get status for node")
			continue
		}
		statuses[nodeID] = status
	}

	return statuses, nil
}

// RefreshAllNodeStatus refreshes status for all registered nodes
func (s *UIService) RefreshAllNodeStatus(ctx context.Context) (map[string]*types.AgentStatus, error) {
	if s.statusManager == nil {
		return nil, fmt.Errorf("status manager not available")
	}

	// Get all registered nodes
	nodes, err := s.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, fmt.Errorf("failed to list agents: %w", err)
	}

	statuses := make(map[string]*types.AgentStatus)
	for _, node := range nodes {
		// Refresh status for each node
		if err := s.statusManager.RefreshAgentStatus(ctx, node.ID); err != nil {
			logger.Logger.Error().Err(err).Str("node_id", node.ID).Msg("Failed to refresh status for node")
			continue
		}

		// Get the refreshed status
		status, err := s.statusManager.GetAgentStatus(ctx, node.ID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("node_id", node.ID).Msg("Failed to get refreshed status for node")
			continue
		}

		statuses[node.ID] = status
	}

	return statuses, nil
}
