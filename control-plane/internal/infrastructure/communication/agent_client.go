package communication

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/your-org/haxen/control-plane/internal/core/interfaces"
	"github.com/your-org/haxen/control-plane/internal/storage"
)

// HTTPAgentClient implements the AgentClient interface using HTTP communication
type HTTPAgentClient struct {
	httpClient *http.Client
	storage    storage.StorageProvider
	timeout    time.Duration
	
	// Cache for MCP health data (30-second TTL)
	cache      map[string]*CachedMCPHealth
	cacheMutex sync.RWMutex
}

// CachedMCPHealth represents cached MCP health data
type CachedMCPHealth struct {
	Data      *interfaces.MCPHealthResponse
	Timestamp time.Time
}

// NewHTTPAgentClient creates a new HTTP-based agent client
func NewHTTPAgentClient(storage storage.StorageProvider, timeout time.Duration) *HTTPAgentClient {
	if timeout == 0 {
		timeout = 5 * time.Second // Default 5-second timeout
	}

	return &HTTPAgentClient{
		httpClient: &http.Client{
			Timeout: timeout,
		},
		storage:    storage,
		timeout:    timeout,
		cache:      make(map[string]*CachedMCPHealth),
		cacheMutex: sync.RWMutex{},
	}
}

// GetMCPHealth retrieves MCP health information from an agent node
func (c *HTTPAgentClient) GetMCPHealth(ctx context.Context, nodeID string) (*interfaces.MCPHealthResponse, error) {
	// Check cache first
	if cached := c.getCachedHealth(nodeID); cached != nil {
		return cached, nil
	}

	// Get agent node details
	agent, err := c.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get agent node %s: %w", nodeID, err)
	}

	// Construct health endpoint URL
	healthURL := fmt.Sprintf("%s/health/mcp", agent.BaseURL)

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Haxen-Server/1.0")

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call agent health endpoint: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode == http.StatusNotFound {
		// Agent doesn't support MCP health endpoint - return empty response
		return &interfaces.MCPHealthResponse{
			Servers: []interfaces.MCPServerHealth{},
			Summary: interfaces.MCPSummary{
				TotalServers:  0,
				RunningServers: 0,
				TotalTools:    0,
				OverallHealth: 1.0, // Consider healthy if no MCP servers
			},
		}, nil
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("agent returned status %d", resp.StatusCode)
	}

	// Parse response
	var healthResponse interfaces.MCPHealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&healthResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Cache the result
	c.cacheHealth(nodeID, &healthResponse)

	return &healthResponse, nil
}

// RestartMCPServer restarts a specific MCP server on an agent node
func (c *HTTPAgentClient) RestartMCPServer(ctx context.Context, nodeID, alias string) error {
	// Get agent node details
	agent, err := c.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return fmt.Errorf("failed to get agent node %s: %w", nodeID, err)
	}

	// Construct restart endpoint URL
	restartURL := fmt.Sprintf("%s/mcp/servers/%s/restart", agent.BaseURL, alias)

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "POST", restartURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Haxen-Server/1.0")

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call agent restart endpoint: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("agent does not support MCP server restart or server %s not found", alias)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("agent returned status %d", resp.StatusCode)
	}

	// Parse response
	var restartResponse interfaces.MCPRestartResponse
	if err := json.NewDecoder(resp.Body).Decode(&restartResponse); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	if !restartResponse.Success {
		return fmt.Errorf("restart failed: %s", restartResponse.Message)
	}

	// Invalidate cache for this node
	c.invalidateCache(nodeID)

	return nil
}

// GetMCPTools retrieves the list of tools from a specific MCP server
func (c *HTTPAgentClient) GetMCPTools(ctx context.Context, nodeID, alias string) (*interfaces.MCPToolsResponse, error) {
	// Get agent node details
	agent, err := c.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get agent node %s: %w", nodeID, err)
	}

	// Construct tools endpoint URL
	toolsURL := fmt.Sprintf("%s/mcp/servers/%s/tools", agent.BaseURL, alias)

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "GET", toolsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Haxen-Server/1.0")

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call agent tools endpoint: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("agent does not support MCP tools endpoint or server %s not found", alias)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("agent returned status %d", resp.StatusCode)
	}

	// Parse response
	var toolsResponse interfaces.MCPToolsResponse
	if err := json.NewDecoder(resp.Body).Decode(&toolsResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &toolsResponse, nil
}

// ShutdownAgent requests graceful shutdown of an agent node via HTTP
func (c *HTTPAgentClient) ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
	// Get agent node details
	agent, err := c.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get agent node %s: %w", nodeID, err)
	}

	// Construct shutdown endpoint URL
	shutdownURL := fmt.Sprintf("%s/shutdown", agent.BaseURL)

	// Prepare request body
	requestBody := map[string]interface{}{
		"graceful":        graceful,
		"timeout_seconds": timeoutSeconds,
	}

	// Marshal request body
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "POST", shutdownURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Haxen-Server/1.0")

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call agent shutdown endpoint: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("agent does not support HTTP shutdown endpoint")
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("agent returned status %d", resp.StatusCode)
	}

	// Parse response
	var shutdownResponse interfaces.AgentShutdownResponse
	if err := json.NewDecoder(resp.Body).Decode(&shutdownResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &shutdownResponse, nil
}

// GetAgentStatus retrieves detailed status information from an agent node with timeout and retry logic
func (c *HTTPAgentClient) GetAgentStatus(ctx context.Context, nodeID string) (*interfaces.AgentStatusResponse, error) {
	// Get agent node details
	agent, err := c.storage.GetAgent(ctx, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to get agent node %s: %w", nodeID, err)
	}
	
	// Check for nil agent (can happen when database returns no error but also no rows)
	if agent == nil {
		return nil, fmt.Errorf("agent node %s not found in storage", nodeID)
	}

	// Construct status endpoint URL
	statusURL := fmt.Sprintf("%s/status", agent.BaseURL)

	// Implement retry logic (1 retry for transient network failures)
	maxRetries := 1
	var lastErr error
	
	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Create timeout context for each attempt (2-3 seconds)
		timeoutCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		
		// Create HTTP request with timeout context
		req, err := http.NewRequestWithContext(timeoutCtx, "GET", statusURL, nil)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "Haxen-Server/1.0")

		// Make the request
		resp, err := c.httpClient.Do(req)
		cancel() // Always cancel the timeout context
		
		if err != nil {
			lastErr = err
			// Check if this is a transient network error that might benefit from retry
			if attempt < maxRetries && isRetryableError(err) {
				// Brief delay before retry (100ms)
				time.Sleep(100 * time.Millisecond)
				continue
			}
			// Network error - distinguish from agent-reported status
			return nil, fmt.Errorf("network failure calling agent status endpoint: %w", err)
		}
		defer resp.Body.Close()

		// Check status code
		if resp.StatusCode == http.StatusNotFound {
			return nil, fmt.Errorf("agent does not support status endpoint")
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("agent returned status %d", resp.StatusCode)
		}

		// Parse response
		var statusResponse interfaces.AgentStatusResponse
		if err := json.NewDecoder(resp.Body).Decode(&statusResponse); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}

		return &statusResponse, nil
	}
	
	// All retries exhausted
	return nil, fmt.Errorf("failed after %d retries, last error: %w", maxRetries+1, lastErr)
}

// getCachedHealth retrieves cached MCP health data if still valid
func (c *HTTPAgentClient) getCachedHealth(nodeID string) *interfaces.MCPHealthResponse {
	c.cacheMutex.RLock()
	defer c.cacheMutex.RUnlock()

	cached, exists := c.cache[nodeID]
	if !exists {
		return nil
	}

	// Check if cache is still valid (30 seconds)
	if time.Since(cached.Timestamp) > 30*time.Second {
		// Cache expired, remove it
		delete(c.cache, nodeID)
		return nil
	}

	return cached.Data
}

// cacheHealth stores MCP health data in cache
func (c *HTTPAgentClient) cacheHealth(nodeID string, data *interfaces.MCPHealthResponse) {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()

	c.cache[nodeID] = &CachedMCPHealth{
		Data:      data,
		Timestamp: time.Now(),
	}
}

// invalidateCache removes cached data for a specific node
func (c *HTTPAgentClient) invalidateCache(nodeID string) {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()

	delete(c.cache, nodeID)
}

// InvalidateAllCache removes all cached data (useful for testing or manual refresh)
func (c *HTTPAgentClient) InvalidateAllCache() {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()

	c.cache = make(map[string]*CachedMCPHealth)
}

// GetCacheStats returns cache statistics for monitoring
func (c *HTTPAgentClient) GetCacheStats() map[string]interface{} {
	c.cacheMutex.RLock()
	defer c.cacheMutex.RUnlock()

	stats := map[string]interface{}{
		"total_entries": len(c.cache),
		"entries":       make([]map[string]interface{}, 0, len(c.cache)),
	}

	for nodeID, cached := range c.cache {
		entry := map[string]interface{}{
			"node_id":   nodeID,
			"timestamp": cached.Timestamp,
			"age_seconds": time.Since(cached.Timestamp).Seconds(),
		}
		stats["entries"] = append(stats["entries"].([]map[string]interface{}), entry)
	}

	return stats
}

// CleanupExpiredCache removes expired cache entries (should be called periodically)
func (c *HTTPAgentClient) CleanupExpiredCache() {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()

	now := time.Now()
	for nodeID, cached := range c.cache {
		if now.Sub(cached.Timestamp) > 30*time.Second {
			delete(c.cache, nodeID)
		}
	}
}

// isRetryableError determines if an error is worth retrying
func isRetryableError(err error) bool {
	// Check for common transient network errors
	if err == nil {
		return false
	}
	
	errStr := err.Error()
	// Common transient errors that might benefit from retry
	transientErrors := []string{
		"connection refused",
		"connection reset",
		"timeout",
		"temporary failure",
		"network is unreachable",
	}
	
	for _, transient := range transientErrors {
		if strings.Contains(strings.ToLower(errStr), transient) {
			return true
		}
	}
	
	return false
}