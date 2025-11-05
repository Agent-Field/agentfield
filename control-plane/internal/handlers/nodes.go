package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services" // Import services package
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
)

// readCloser wraps a reader to implement io.ReadCloser
type readCloser struct {
	io.Reader
}

func (rc *readCloser) Close() error {
	return nil
}

var validate = validator.New()

// validateCallbackURL validates that a callback URL is properly formatted and reachable
func validateCallbackURL(baseURL string) error {
	if baseURL == "" {
		return fmt.Errorf("base_url cannot be empty")
	}

	// Parse the URL to ensure it's valid
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid URL format: %w", err)
	}

	// Ensure it's an HTTP or HTTPS URL
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("URL must use http or https scheme, got: %s", parsedURL.Scheme)
	}

	// Ensure host is present
	if parsedURL.Host == "" {
		return fmt.Errorf("URL must include a host")
	}

	// Basic reachability test with timeout
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	healthURL := strings.TrimSuffix(baseURL, "/") + "/health"
	req, err := http.NewRequest("GET", healthURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		// Log the error but don't fail registration - agent might not be fully started yet
		logger.Logger.Warn().Err(err).Msgf("⚠️ Callback URL health check failed for %s (agent may still be starting)", healthURL)
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logger.Logger.Warn().Msgf("⚠️ Callback URL health check returned status %d for %s", resp.StatusCode, healthURL)
	} else {
		logger.Logger.Debug().Msgf("✅ Callback URL validated successfully: %s", baseURL)
	}

	return nil
}

func extractPortFromURL(rawURL string) string {
	if rawURL == "" {
		return ""
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}

	if parsed.Port() != "" {
		return parsed.Port()
	}

	switch parsed.Scheme {
	case "https":
		return "443"
	case "http":
		return "80"
	default:
		return ""
	}
}

func gatherCallbackCandidates(baseURL string, discovery *types.CallbackDiscoveryInfo, clientIP string) ([]string, string) {
	seen := make(map[string]struct{})
	candidates := make([]string, 0)

	defaultPort := extractPortFromURL(baseURL)

	add := func(candidate string) {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			return
		}
		if _, exists := seen[candidate]; exists {
			return
		}
		seen[candidate] = struct{}{}
		candidates = append(candidates, candidate)
	}

	add(baseURL)

	if discovery != nil {
		if discovery.Preferred != "" {
			add(discovery.Preferred)
			if defaultPort == "" {
				defaultPort = extractPortFromURL(discovery.Preferred)
			}
		}
		for _, candidate := range discovery.Candidates {
			add(candidate)
			if defaultPort == "" {
				defaultPort = extractPortFromURL(candidate)
			}
		}
	}

	if clientIP != "" && defaultPort != "" {
		add(fmt.Sprintf("http://%s:%s", clientIP, defaultPort))
	}

	return candidates, defaultPort
}

func normalizeCandidate(raw string, defaultPort string) (string, error) {
	if raw == "" {
		return "", fmt.Errorf("empty candidate")
	}

	trimmed := strings.TrimSpace(raw)
	if !strings.Contains(trimmed, "://") {
		trimmed = "http://" + trimmed
	}

	parsed, err := url.Parse(trimmed)
	if err != nil {
		return "", err
	}

	scheme := parsed.Scheme
	if scheme == "" {
		scheme = "http"
	}

	host := parsed.Host
	if host == "" {
		host = parsed.Path
		parsed.Path = ""
	}

	if host == "" {
		return "", fmt.Errorf("missing host")
	}

	port := parsed.Port()
	if port == "" {
		port = defaultPort
	}

	hostname := parsed.Hostname()
	if hostname == "" {
		hostname = host
	}

	if strings.Contains(hostname, ":") && !strings.HasPrefix(hostname, "[") {
		hostname = fmt.Sprintf("[%s]", hostname)
	}

	var netloc string
	if port != "" {
		netloc = fmt.Sprintf("%s:%s", hostname, port)
	} else {
		netloc = hostname
	}

	return fmt.Sprintf("%s://%s", scheme, netloc), nil
}

func probeCandidate(ctx context.Context, client *http.Client, baseURL string) types.CallbackTestResult {
	result := types.CallbackTestResult{URL: baseURL}
	trimmedBase := strings.TrimSuffix(baseURL, "/")
	endpoints := []string{"/health/mcp", "/health"}

	for _, endpoint := range endpoints {
		start := time.Now()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, trimmedBase+endpoint, nil)
		if err != nil {
			result.Error = err.Error()
			continue
		}

		resp, err := client.Do(req)
		latency := time.Since(start).Milliseconds()
		if err != nil {
			result.Error = err.Error()
			continue
		}

		result.Status = resp.StatusCode
		result.Endpoint = endpoint
		result.LatencyMS = latency
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 400 {
			result.Success = true
			result.Error = ""
			return result
		}
	}

	return result
}

func resolveCallbackCandidates(ctx context.Context, rawCandidates []string, defaultPort string) (string, []string, []types.CallbackTestResult) {
	if len(rawCandidates) == 0 {
		return "", nil, nil
	}

	client := &http.Client{Timeout: 2 * time.Second}
	normalized := make([]string, 0, len(rawCandidates))
	results := make([]types.CallbackTestResult, 0, len(rawCandidates))
	seen := make(map[string]struct{})

	for _, candidate := range rawCandidates {
		normalizedURL, err := normalizeCandidate(candidate, defaultPort)
		if err != nil {
			logger.Logger.Debug().Msgf("⚠️ Skipping invalid callback candidate '%s': %v", candidate, err)
			continue
		}

		if _, exists := seen[normalizedURL]; exists {
			continue
		}
		seen[normalizedURL] = struct{}{}
		normalized = append(normalized, normalizedURL)

		probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		result := probeCandidate(probeCtx, client, normalizedURL)
		cancel()

		results = append(results, result)
		if result.Success {
			return normalizedURL, normalized, results
		}
	}

	return "", normalized, results
}

// CachedNodeData holds cached heartbeat data for a node
type CachedNodeData struct {
	LastDBUpdate    time.Time
	LastCacheUpdate time.Time
	Status          string
	MCPServers      []struct {
		Alias     string `json:"alias"`
		Status    string `json:"status"`
		ToolCount int    `json:"tool_count"`
	}
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
	// Only write to DB if heartbeat is older than this threshold
	// Must be less than health monitor grace period (12s) to prevent false inactivity
	dbUpdateThreshold = 8 * time.Second
)

// shouldUpdateDatabase determines if a heartbeat should trigger a database update
func (hc *HeartbeatCache) shouldUpdateDatabase(nodeID string, now time.Time, status string, mcpServers []struct {
	Alias     string `json:"alias"`
	Status    string `json:"status"`
	ToolCount int    `json:"tool_count"`
}) (bool, *CachedNodeData) {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()

	cached, exists := hc.nodes[nodeID]
	if !exists {
		// First heartbeat for this node
		cached = &CachedNodeData{
			LastDBUpdate:    now,
			LastCacheUpdate: now,
			Status:          status,
			MCPServers:      mcpServers,
		}
		hc.nodes[nodeID] = cached
		return true, cached
	}

	// Update cache timestamp
	cached.LastCacheUpdate = now
	cached.Status = status
	cached.MCPServers = mcpServers

	// Check if enough time has passed since last DB update
	timeSinceDBUpdate := now.Sub(cached.LastDBUpdate)
	if timeSinceDBUpdate >= dbUpdateThreshold {
		cached.LastDBUpdate = now
		return true, cached
	}

	return false, cached
}

// processHeartbeatAsync processes heartbeat database updates asynchronously
func processHeartbeatAsync(storageProvider storage.StorageProvider, uiService *services.UIService, nodeID string, cached *CachedNodeData) {
	go func() {
		ctx := context.Background()

		// Verify node exists only when we need to update DB
		if _, err := storageProvider.GetAgent(ctx, nodeID); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Node %s not found during heartbeat update", nodeID)
			return
		}

		// Update heartbeat in database
		if err := storageProvider.UpdateAgentHeartbeat(ctx, nodeID, cached.LastDBUpdate); err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ HEARTBEAT_CONTENTION: Failed to update heartbeat for node %s - %v", nodeID, err)
			return
		}

		logger.Logger.Debug().Msgf("💓 HEARTBEAT_CONTENTION: Async DB update completed for node %s", nodeID)
	}()
}

// RegisterNodeHandler handles the registration of a new agent node.
func RegisterNodeHandler(storageProvider storage.StorageProvider, uiService *services.UIService, didService *services.DIDService, presenceManager *services.PresenceManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		var newNode types.AgentNode

		// Log the incoming request
		body, _ := c.GetRawData()
		c.Request.Body = http.NoBody // Reset body for ShouldBindJSON
		c.Request.Body = &readCloser{bytes.NewReader(body)}

		logger.Logger.Debug().Msgf("🔍 Received registration request: %s", string(body))

		if err := c.ShouldBindJSON(&newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ JSON binding error")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format: " + err.Error()})
			return
		}

		logger.Logger.Debug().Msgf("✅ Successfully parsed node data for ID: %s", newNode.ID)

		// Validate the incoming node data
		if err := validate.Struct(newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Validation error")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Validation failed: " + err.Error()})
			return
		}

		logger.Logger.Debug().Msgf("✅ Node validation passed for ID: %s", newNode.ID)

		candidateList, defaultPort := gatherCallbackCandidates(newNode.BaseURL, newNode.CallbackDiscovery, c.ClientIP())
		resolvedBaseURL := ""
		var normalizedCandidates []string
		var probeResults []types.CallbackTestResult

		if len(candidateList) > 0 {
			probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			resolvedBaseURL, normalizedCandidates, probeResults = resolveCallbackCandidates(probeCtx, candidateList, defaultPort)
			cancel()

			if resolvedBaseURL != "" && resolvedBaseURL != newNode.BaseURL {
				logger.Logger.Info().Msgf("🔗 Auto-discovered callback URL for %s: %s (was %s)", newNode.ID, resolvedBaseURL, newNode.BaseURL)
				newNode.BaseURL = resolvedBaseURL
			}
		}

		// Validate callback URL if provided
		if newNode.BaseURL != "" {
			if err := validateCallbackURL(newNode.BaseURL); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Callback URL validation failed for node %s", newNode.ID)
				c.JSON(http.StatusBadRequest, gin.H{
					"error":   "Invalid callback URL: " + err.Error(),
					"details": "The provided base_url must be a valid, reachable HTTP/HTTPS endpoint",
				})
				return
			}
		}

		// Persist discovery metadata for observability
		if newNode.CallbackDiscovery == nil {
			newNode.CallbackDiscovery = &types.CallbackDiscoveryInfo{}
		}

		if newNode.CallbackDiscovery.Mode == "" {
			newNode.CallbackDiscovery.Mode = "auto"
		}

		if newNode.CallbackDiscovery.Preferred == "" {
			newNode.CallbackDiscovery.Preferred = newNode.BaseURL
		}

		if resolvedBaseURL != "" {
			newNode.CallbackDiscovery.Resolved = resolvedBaseURL
		} else {
			newNode.CallbackDiscovery.Resolved = newNode.BaseURL
		}

		if len(normalizedCandidates) > 0 {
			newNode.CallbackDiscovery.Candidates = normalizedCandidates
		}

		if len(probeResults) > 0 {
			newNode.CallbackDiscovery.Tests = probeResults
		}

		newNode.CallbackDiscovery.SubmittedAt = time.Now().UTC().Format(time.RFC3339)

		// Check if node with the same ID already exists
		existingNode, err := storageProvider.GetAgent(ctx, newNode.ID)
		isReRegistration := false
		if err == nil && existingNode != nil {
			isReRegistration = true
		}

		// Set initial health status to UNKNOWN for new registrations
		// The health monitor will determine the actual status based on heartbeats
		newNode.HealthStatus = types.HealthStatusUnknown

		// Handle lifecycle status for re-registrations vs new registrations
		if isReRegistration {
			// For re-registrations, preserve existing lifecycle status if the new one is empty
			// This prevents status resets that cause oscillation
			if newNode.LifecycleStatus == "" && existingNode.LifecycleStatus != "" {
				newNode.LifecycleStatus = existingNode.LifecycleStatus
			} else if newNode.LifecycleStatus == "" {
				newNode.LifecycleStatus = types.AgentStatusStarting
			}
		} else {
			// For new registrations, use provided status or default to starting
			if newNode.LifecycleStatus == "" {
				newNode.LifecycleStatus = types.AgentStatusStarting
			}
		}

		newNode.RegisteredAt = time.Now().UTC()
		newNode.LastHeartbeat = time.Now().UTC() // Set initial heartbeat to registration time

		if newNode.Metadata.Custom == nil {
			newNode.Metadata.Custom = map[string]interface{}{}
		}
		newNode.Metadata.Custom["callback_discovery"] = newNode.CallbackDiscovery

		// Store the new node
		if err := storageProvider.RegisterAgent(ctx, &newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Storage error")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to store node: " + err.Error()})
			return
		}

		logger.Logger.Debug().Msgf("✅ Successfully registered node: %s", newNode.ID)

		// Enhanced DID registration integration
		// The enhanced DID service handles all scenarios automatically (new, re-registration, partial updates)
		if didService != nil {
			// Create DID registration request from node data
			didReq := &types.DIDRegistrationRequest{
				AgentNodeID: newNode.ID,
				Reasoners:   newNode.Reasoners,
				Skills:      newNode.Skills,
			}

			// Enhanced DID service handles differential analysis and routing automatically
			didResponse, err := didService.RegisterAgent(didReq)
			if err != nil {
				// DID registration failure is now a critical error
				logger.Logger.Error().Err(err).Msgf("❌ DID registration failed for node %s", newNode.ID)
				c.JSON(http.StatusInternalServerError, gin.H{
					"success": false,
					"error":   "DID registration failed",
					"details": fmt.Sprintf("Failed to register node %s with DID system: %v", newNode.ID, err),
				})
				return
			}

			if !didResponse.Success {
				// DID registration unsuccessful is now a critical error
				logger.Logger.Error().Msgf("❌ DID registration unsuccessful for node %s: %s", newNode.ID, didResponse.Error)
				c.JSON(http.StatusInternalServerError, gin.H{
					"success": false,
					"error":   "DID registration unsuccessful",
					"details": fmt.Sprintf("DID registration failed for node %s: %s", newNode.ID, didResponse.Error),
				})
				return
			}

			// Log appropriate message based on registration type
			if isReRegistration {
				logger.Logger.Debug().Msgf("✅ Node %s re-registered with DID service: %s", newNode.ID, didResponse.Message)
			} else {
				logger.Logger.Debug().Msgf("✅ Node %s registered with DID: %s", newNode.ID, didResponse.IdentityPackage.AgentDID.DID)
			}
		}

		// Note: Node registration events are now handled by the health monitor
		// The health monitor will detect the new node and emit appropriate events

		if presenceManager != nil {
			presenceManager.Touch(newNode.ID, time.Now().UTC())
		}

		responsePayload := gin.H{
			"success": true,
			"message": "Node registered successfully",
			"node_id": newNode.ID,
		}

		if newNode.BaseURL != "" {
			responsePayload["resolved_base_url"] = newNode.BaseURL
		}

		if newNode.CallbackDiscovery != nil {
			responsePayload["callback_discovery"] = newNode.CallbackDiscovery
		}

		c.JSON(http.StatusCreated, responsePayload)
	}
}

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

// GetNodeHandler handles getting a specific node by ID
func GetNodeHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		node, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		c.JSON(http.StatusOK, node)
	}
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
			Status     string `json:"status,omitempty"`
			MCPServers []struct {
				Alias     string `json:"alias"`
				Status    string `json:"status"`
				ToolCount int    `json:"tool_count"`
			} `json:"mcp_servers,omitempty"`
			Timestamp   string `json:"timestamp,omitempty"`
			HealthScore *int   `json:"health_score,omitempty"` // New: allow agents to report health score
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
			presenceManager.Touch(nodeID, now)
		}
		needsDBUpdate, cached := heartbeatCache.shouldUpdateDatabase(nodeID, now, enhancedHeartbeat.Status, enhancedHeartbeat.MCPServers)

		if needsDBUpdate {
			// Verify node exists only when we need to update DB
			existingNode, err := storageProvider.GetAgent(ctx, nodeID)
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
				presenceManager.Touch(nodeID, now)
			}

			// Process heartbeat asynchronously to avoid blocking the response
			processHeartbeatAsync(storageProvider, uiService, nodeID, cached)

			logger.Logger.Debug().Msgf("💓 Heartbeat DB update queued for node: %s at %s", nodeID, now.Format(time.RFC3339))
		} else {
			logger.Logger.Debug().Msgf("💓 Heartbeat cached for node: %s (no DB update needed)", nodeID)
		}

		// Process enhanced heartbeat data through unified status system
		if statusManager != nil && (enhancedHeartbeat.Status != "" || len(enhancedHeartbeat.MCPServers) > 0 || enhancedHeartbeat.HealthScore != nil) {
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
					status := types.AgentLifecycleStatus(enhancedHeartbeat.Status)
					lifecycleStatus = &status
				}
			}

			// Prepare MCP status
			var mcpStatus *types.MCPStatusInfo
			if len(enhancedHeartbeat.MCPServers) > 0 {
				totalServers := len(enhancedHeartbeat.MCPServers)
				runningServers := 0
				totalTools := 0

				for _, server := range enhancedHeartbeat.MCPServers {
					if server.Status == "running" {
						runningServers++
					}
					totalTools += server.ToolCount
				}

				// Calculate overall health based on running servers
				overallHealth := 0.0
				if totalServers > 0 {
					overallHealth = float64(runningServers) / float64(totalServers)
				}

				// Determine service status
				serviceStatus := "unavailable"
				if overallHealth >= 0.9 {
					serviceStatus = "ready"
				} else if overallHealth >= 0.5 {
					serviceStatus = "degraded"
				}

				mcpStatus = &types.MCPStatusInfo{
					TotalServers:   totalServers,
					RunningServers: runningServers,
					TotalTools:     totalTools,
					OverallHealth:  overallHealth,
					ServiceStatus:  serviceStatus,
					LastChecked:    now,
				}
			}

			// Update status through unified system
			if err := statusManager.UpdateFromHeartbeat(ctx, nodeID, lifecycleStatus, mcpStatus); err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ Failed to update unified status for node %s", nodeID)
				// Continue processing - don't fail the heartbeat
			}

			// Handle health score if provided
			if enhancedHeartbeat.HealthScore != nil {
				update := &types.AgentStatusUpdate{
					HealthScore: enhancedHeartbeat.HealthScore,
					Source:      types.StatusSourceHeartbeat,
					Reason:      "health score from heartbeat",
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
						if err != nil {
							logger.Logger.Error().Err(err).Msgf("❌ Failed to get node %s for lifecycle status update", nodeID)
							c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
							return
						}
					}

					if existingNode.LifecycleStatus != newStatus {
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
			MCPServers      *struct {
				Total   int `json:"total"`
				Running int `json:"running"`
			} `json:"mcp_servers,omitempty"`
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
		_, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		// Prepare status update for unified system
		newLifecycleStatus := types.AgentLifecycleStatus(statusUpdate.LifecycleStatus)

		// Prepare MCP status if provided
		var mcpStatus *types.MCPStatusInfo
		if statusUpdate.MCPServers != nil {
			overallHealth := 0.0
			serviceStatus := "unavailable"

			if statusUpdate.MCPServers.Total > 0 {
				overallHealth = float64(statusUpdate.MCPServers.Running) / float64(statusUpdate.MCPServers.Total)
				if overallHealth >= 0.9 {
					serviceStatus = "ready"
				} else if overallHealth >= 0.5 {
					serviceStatus = "degraded"
				}
			}

			mcpStatus = &types.MCPStatusInfo{
				TotalServers:   statusUpdate.MCPServers.Total,
				RunningServers: statusUpdate.MCPServers.Running,
				TotalTools:     0, // Not provided in this endpoint
				OverallHealth:  overallHealth,
				ServiceStatus:  serviceStatus,
				LastChecked:    time.Now(),
			}
		}

		// Update through unified status system if available
		if statusManager != nil {
			if err := statusManager.UpdateFromHeartbeat(ctx, nodeID, &newLifecycleStatus, mcpStatus); err != nil {
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

// GetNodeStatusHandler handles getting the unified status for a specific node
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

		status, err := statusManager.GetAgentStatus(ctx, nodeID)
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
			status, err := statusManager.GetAgentStatus(ctx, nodeID)
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

// RegisterServerlessAgentHandler handles the registration of a serverless agent node
// by discovering its capabilities via the /discover endpoint
func RegisterServerlessAgentHandler(storageProvider storage.StorageProvider, uiService *services.UIService, didService *services.DIDService, presenceManager *services.PresenceManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()

		var req struct {
			InvocationURL string `json:"invocation_url" binding:"required"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Invalid request body")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request: " + err.Error()})
			return
		}

		// Validate URL format
		if req.InvocationURL == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invocation_url is required"})
			return
		}

		logger.Logger.Info().Msgf("🔍 Discovering serverless agent at: %s", req.InvocationURL)

		// Normalize 0.0.0.0 to localhost for discovery
		// 0.0.0.0 is not a valid address for making HTTP requests
		normalizedURL := req.InvocationURL
		if strings.Contains(normalizedURL, "://0.0.0.0") {
			normalizedURL = strings.Replace(normalizedURL, "://0.0.0.0", "://localhost", 1)
			logger.Logger.Info().Msgf("🔄 Normalized invocation URL from %s to %s for discovery", req.InvocationURL, normalizedURL)
		}

		// Call the discovery endpoint using normalized URL
		discoveryURL := strings.TrimSuffix(normalizedURL, "/") + "/discover"

		client := &http.Client{
			Timeout: 10 * time.Second,
		}

		resp, err := client.Get(discoveryURL)
		if err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to call discovery endpoint: %s", discoveryURL)
			c.JSON(http.StatusBadGateway, gin.H{
				"error":   "Failed to discover serverless agent",
				"details": fmt.Sprintf("Could not reach discovery endpoint: %v", err),
			})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			logger.Logger.Error().Msgf("❌ Discovery endpoint returned status %d", resp.StatusCode)
			c.JSON(http.StatusBadGateway, gin.H{
				"error":   "Discovery endpoint failed",
				"details": fmt.Sprintf("Discovery endpoint returned status %d", resp.StatusCode),
			})
			return
		}

		// Parse discovery response
		var discoveryData struct {
			NodeID    string `json:"node_id"`
			Version   string `json:"version"`
			Reasoners []struct {
				ID           string                 `json:"id"`
				Name         string                 `json:"name"`
				Description  string                 `json:"description"`
				InputSchema  map[string]interface{} `json:"input_schema"`
				OutputSchema map[string]interface{} `json:"output_schema"`
			} `json:"reasoners"`
			Skills []struct {
				ID           string                 `json:"id"`
				Name         string                 `json:"name"`
				Description  string                 `json:"description"`
				InputSchema  map[string]interface{} `json:"input_schema"`
				OutputSchema map[string]interface{} `json:"output_schema"`
			} `json:"skills"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&discoveryData); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Failed to parse discovery response")
			c.JSON(http.StatusBadGateway, gin.H{
				"error":   "Invalid discovery response",
				"details": fmt.Sprintf("Could not parse discovery data: %v", err),
			})
			return
		}

		// Validate required fields
		if discoveryData.NodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid discovery response",
				"details": "node_id is missing from discovery response",
			})
			return
		}

		logger.Logger.Info().Msgf("✅ Discovered serverless agent: %s (version: %s)", discoveryData.NodeID, discoveryData.Version)

		// Convert discovered reasoners to AgentNode format
		reasoners := make([]types.ReasonerDefinition, len(discoveryData.Reasoners))
		for i, r := range discoveryData.Reasoners {
			inputSchemaBytes, _ := json.Marshal(r.InputSchema)
			outputSchemaBytes, _ := json.Marshal(r.OutputSchema)
			reasoners[i] = types.ReasonerDefinition{
				ID:           r.ID,
				InputSchema:  json.RawMessage(inputSchemaBytes),
				OutputSchema: json.RawMessage(outputSchemaBytes),
			}
		}

		// Convert discovered skills to AgentNode format
		skills := make([]types.SkillDefinition, len(discoveryData.Skills))
		for i, s := range discoveryData.Skills {
			inputSchemaBytes, _ := json.Marshal(s.InputSchema)
			skills[i] = types.SkillDefinition{
				ID:          s.ID,
				InputSchema: json.RawMessage(inputSchemaBytes),
			}
		}

		// Create the agent node
		executionURL := strings.TrimSuffix(req.InvocationURL, "/") + "/execute"

		newNode := types.AgentNode{
			ID:              discoveryData.NodeID,
			TeamID:          "default", // Default team for serverless agents
			BaseURL:         req.InvocationURL,
			Version:         discoveryData.Version,
			DeploymentType:  "serverless",
			InvocationURL:   &executionURL,
			Reasoners:       reasoners,
			Skills:          skills,
			RegisteredAt:    time.Now().UTC(),
			LastHeartbeat:   time.Now().UTC(),
			HealthStatus:    types.HealthStatusUnknown, // Serverless agents don't have persistent health
			LifecycleStatus: types.AgentStatusReady,    // Serverless agents are always ready
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{
					"serverless":    true,
					"discovery_url": discoveryURL,
				},
			},
		}

		// Check if node already exists
		existingNode, err := storageProvider.GetAgent(ctx, newNode.ID)
		if err == nil && existingNode != nil {
			logger.Logger.Warn().Msgf("⚠️ Serverless agent %s already registered, updating...", newNode.ID)
		}

		// Register the node
		if err := storageProvider.RegisterAgent(ctx, &newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Failed to register serverless agent")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to register serverless agent",
				"details": err.Error(),
			})
			return
		}

		logger.Logger.Info().Msgf("✅ Successfully registered serverless agent: %s", newNode.ID)

		// Register with DID service if available
		if didService != nil {
			didReq := &types.DIDRegistrationRequest{
				AgentNodeID: newNode.ID,
				Reasoners:   newNode.Reasoners,
				Skills:      newNode.Skills,
			}

			didResponse, err := didService.RegisterAgent(didReq)
			if err != nil {
				logger.Logger.Error().Err(err).Msgf("❌ DID registration failed for serverless agent %s", newNode.ID)
				// Don't fail the registration, just log the error
			} else if didResponse.Success {
				logger.Logger.Info().Msgf("✅ Serverless agent %s registered with DID service", newNode.ID)
			}
		}

		// Touch presence manager
		if presenceManager != nil {
			presenceManager.Touch(newNode.ID, time.Now().UTC())
		}

		c.JSON(http.StatusCreated, gin.H{
			"success": true,
			"message": "Serverless agent registered successfully",
			"node": gin.H{
				"id":              newNode.ID,
				"version":         newNode.Version,
				"deployment_type": newNode.DeploymentType,
				"invocation_url":  newNode.InvocationURL,
				"reasoners_count": len(newNode.Reasoners),
				"skills_count":    len(newNode.Skills),
			},
		})
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
