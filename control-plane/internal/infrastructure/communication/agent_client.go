package communication

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

// HTTPAgentClient implements the AgentClient interface using HTTP communication
type HTTPAgentClient struct {
	httpClient *http.Client
	storage    storage.StorageProvider
	timeout    time.Duration
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
		storage: storage,
		timeout: timeout,
	}
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
	req.Header.Set("User-Agent", "AgentField-Server/1.0")

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
		req.Header.Set("User-Agent", "AgentField-Server/1.0")

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

		// Safety check: ensure the responding agent matches the expected node ID.
		// If node_id is missing, allow it (legacy agents) but log a warning.
		if statusResponse.NodeID == "" {
			logger.Logger.Warn().Str("node_id", nodeID).Msg("agent status response missing node_id; skipping identity verification")
		} else if statusResponse.NodeID != nodeID {
			return nil, fmt.Errorf("agent ID mismatch: expected %s, got %s", nodeID, statusResponse.NodeID)
		}

		return &statusResponse, nil
	}

	// All retries exhausted
	return nil, fmt.Errorf("failed after %d retries, last error: %w", maxRetries+1, lastErr)
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
