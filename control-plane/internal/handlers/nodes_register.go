package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
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

// validateCallbackURL validates that a callback URL is properly formatted.
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

	if parsedURL.User != nil {
		return fmt.Errorf("URL must not include user info")
	}

	// Ensure host is present
	if parsedURL.Host == "" {
		return fmt.Errorf("URL must include a host")
	}

	logger.Logger.Debug().Msgf("✅ Callback URL validated successfully: %s", baseURL)
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

func resolveCallbackCandidates(rawCandidates []string, defaultPort string) (string, []string, []types.CallbackTestResult) {
	if len(rawCandidates) == 0 {
		return "", nil, nil
	}

	normalized := make([]string, 0, len(rawCandidates))
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
	}

	if len(normalized) == 0 {
		return "", nil, nil
	}

	return normalized[0], normalized, nil
}

func normalizeServerlessDiscoveryURL(rawURL string, allowedHosts []string) (string, error) {
	parsedURL, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return "", fmt.Errorf("invalid invocation_url format: %w", err)
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return "", fmt.Errorf("invocation_url must use http or https")
	}

	if parsedURL.User != nil {
		return "", fmt.Errorf("invocation_url must not include user info")
	}

	if parsedURL.Host == "" {
		return "", fmt.Errorf("invocation_url must include a host")
	}

	if parsedURL.RawQuery != "" || parsedURL.Fragment != "" {
		return "", fmt.Errorf("invocation_url must not include query parameters or fragments")
	}

	hostname := parsedURL.Hostname()
	if hostname == "" {
		return "", fmt.Errorf("invocation_url must include a host")
	}

	if hostname == "0.0.0.0" || hostname == "::" {
		hostname = "localhost"
		if port := parsedURL.Port(); port != "" {
			parsedURL.Host = net.JoinHostPort(hostname, port)
		} else {
			parsedURL.Host = hostname
		}
	}

	if !isServerlessDiscoveryHostAllowed(hostname, allowedHosts) {
		return "", fmt.Errorf("invocation_url host %q is not allowlisted for server-side discovery; configure agentfield.registration.serverless_discovery_allowed_hosts or AGENTFIELD_REGISTRATION_SERVERLESS_DISCOVERY_ALLOWED_HOSTS", hostname)
	}

	return strings.TrimSuffix(parsedURL.String(), "/"), nil
}

func isServerlessDiscoveryHostAllowed(host string, allowedHosts []string) bool {
	if host == "" {
		return false
	}

	if strings.EqualFold(host, "localhost") {
		return true
	}

	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return true
	}

	hostLower := strings.ToLower(host)
	for _, entry := range allowedHosts {
		candidate := strings.ToLower(strings.TrimSpace(entry))
		if candidate == "" {
			continue
		}

		if _, network, err := net.ParseCIDR(candidate); err == nil {
			if ip := net.ParseIP(host); ip != nil && network.Contains(ip) {
				return true
			}
			continue
		}

		if strings.HasPrefix(candidate, "*.") {
			suffix := strings.TrimPrefix(candidate, "*")
			if strings.HasSuffix(hostLower, suffix) && hostLower != strings.TrimPrefix(suffix, ".") {
				return true
			}
			continue
		}

		if hostLower == candidate {
			return true
		}
	}

	return false
}

// RegisterNodeHandler handles the registration of a new agent node.
func RegisterNodeHandler(storageProvider storage.StorageProvider, uiService *services.UIService, didService *services.DIDService, presenceManager *services.PresenceManager, didWebService *services.DIDWebService, tagApprovalService *services.TagApprovalService) gin.HandlerFunc {
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

		// Default group_id to agent id for backward compatibility
		if newNode.GroupID == "" {
			newNode.GroupID = newNode.ID
		}

		// Normalize proposed_tags → tags for backward compatibility.
		// If a skill/reasoner has proposed_tags but no tags, copy proposed_tags to tags.
		for i := range newNode.Reasoners {
			if len(newNode.Reasoners[i].ProposedTags) > 0 && len(newNode.Reasoners[i].Tags) == 0 {
				newNode.Reasoners[i].Tags = newNode.Reasoners[i].ProposedTags
			}
			if len(newNode.Reasoners[i].Tags) > 0 && len(newNode.Reasoners[i].ProposedTags) == 0 {
				newNode.Reasoners[i].ProposedTags = newNode.Reasoners[i].Tags
			}
		}
		for i := range newNode.Skills {
			if len(newNode.Skills[i].ProposedTags) > 0 && len(newNode.Skills[i].Tags) == 0 {
				newNode.Skills[i].Tags = newNode.Skills[i].ProposedTags
			}
			if len(newNode.Skills[i].Tags) > 0 && len(newNode.Skills[i].ProposedTags) == 0 {
				newNode.Skills[i].ProposedTags = newNode.Skills[i].Tags
			}
		}

		candidateList, defaultPort := gatherCallbackCandidates(newNode.BaseURL, newNode.CallbackDiscovery, c.ClientIP())
		resolvedBaseURL := ""
		var normalizedCandidates []string
		var probeResults []types.CallbackTestResult

		// Determine if auto-discovery should be skipped
		// Skip auto-discovery if:
		// 1. An explicit BaseURL was provided by the agent AND
		// 2. Either no discovery mode is set OR mode is explicitly "manual"/"explicit"
		skipAutoDiscovery := false
		if newNode.BaseURL != "" {
			// If callback discovery mode is explicitly set to manual/explicit, respect it
			if newNode.CallbackDiscovery != nil &&
				(newNode.CallbackDiscovery.Mode == "manual" || newNode.CallbackDiscovery.Mode == "explicit") {
				skipAutoDiscovery = true
				logger.Logger.Info().Msgf("✅ Using explicit callback URL for %s (mode=%s): %s",
					newNode.ID, newNode.CallbackDiscovery.Mode, newNode.BaseURL)
			} else if newNode.CallbackDiscovery == nil || newNode.CallbackDiscovery.Mode == "" {
				// No discovery info provided - treat BaseURL as explicit
				skipAutoDiscovery = true
				logger.Logger.Info().Msgf("✅ Using explicit callback URL for %s (no discovery mode): %s",
					newNode.ID, newNode.BaseURL)
			}
		}

		if len(candidateList) > 0 && !skipAutoDiscovery {
			logger.Logger.Debug().Msgf("🔍 Auto-discovering callback URL for %s from %d candidates", newNode.ID, len(candidateList))
			resolvedBaseURL, normalizedCandidates, probeResults = resolveCallbackCandidates(candidateList, defaultPort)

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

		// Check if node with the same ID and version already exists
		var existingNode *types.AgentNode
		if newNode.Version != "" {
			existingNode, _ = storageProvider.GetAgentVersion(ctx, newNode.ID, newNode.Version)

			// Clean up stale empty-version row if the agent is now registering with a proper version.
			// This handles upgrades from older SDKs that didn't send version during registration.
			if stale, _ := storageProvider.GetAgentVersion(ctx, newNode.ID, ""); stale != nil {
				if err := storageProvider.DeleteAgentVersion(ctx, newNode.ID, ""); err != nil {
					logger.Logger.Warn().Err(err).Msgf("⚠️ Failed to clean up stale empty-version row for agent %s", newNode.ID)
				} else {
					logger.Logger.Info().Msgf("🧹 Cleaned up stale empty-version row for agent %s (now registering as %s)", newNode.ID, newNode.Version)
				}
			}
		} else {
			existingNode, _ = storageProvider.GetAgent(ctx, newNode.ID)
		}
		isReRegistration := existingNode != nil

		// Set initial health status to UNKNOWN for new registrations
		// The health monitor will determine the actual status based on heartbeats
		newNode.HealthStatus = types.HealthStatusUnknown

		// Handle lifecycle status for re-registrations vs new registrations.
		if isReRegistration {
			// Detect admin revocation: pending_approval with nil/empty approved tags
			// means an admin explicitly revoked this agent's tags. In that case,
			// force the agent to stay in pending_approval until re-approved.
			adminRevoked := existingNode.LifecycleStatus == types.AgentStatusPendingApproval &&
				len(existingNode.ApprovedTags) == 0

			if adminRevoked {
				newNode.LifecycleStatus = types.AgentStatusPendingApproval
			} else {
				// Preserve existing approval state from the database.
				// The SDK never sends approved_tags (only proposed_tags), so without
				// this the UPSERT would overwrite approved_tags with an empty array,
				// forcing re-approval after every CP restart or re-registration.
				//
				// IMPORTANT: We deliberately do NOT preserve existingNode.LifecycleStatus
				// here. The lifecycle state machine owns that field — a stale terminal
				// status (stopping/offline) from a previous shutdown would otherwise
				// leak into the fresh registration, causing the re-registered agent to
				// appear mid-shutdown, which breaks downstream status inference
				// (e.g. webhook event type determination, health monitoring, and the
				// docs-quick-start execution webhook contract test). The fallback
				// below resets empty/offline to AgentStatusStarting, and the state
				// machine takes it from there.
				newNode.ApprovedTags = existingNode.ApprovedTags

				// Carry over per-reasoner and per-skill approved tags.
				if len(existingNode.ApprovedTags) > 0 {
					approvedSet := make(map[string]struct{})
					for _, t := range existingNode.ApprovedTags {
						approvedSet[strings.ToLower(strings.TrimSpace(t))] = struct{}{}
					}
					for i := range newNode.Reasoners {
						var approved []string
						proposed := newNode.Reasoners[i].ProposedTags
						if len(proposed) == 0 {
							proposed = newNode.Reasoners[i].Tags
						}
						for _, t := range proposed {
							if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
								approved = append(approved, t)
							}
						}
						newNode.Reasoners[i].ApprovedTags = approved
					}
					for i := range newNode.Skills {
						var approved []string
						proposed := newNode.Skills[i].ProposedTags
						if len(proposed) == 0 {
							proposed = newNode.Skills[i].Tags
						}
						for _, t := range proposed {
							if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
								approved = append(approved, t)
							}
						}
						newNode.Skills[i].ApprovedTags = approved
					}
				}

				// If lifecycle was offline or empty, reset to starting so the
				// agent can go through normal startup.
				if newNode.LifecycleStatus == "" || newNode.LifecycleStatus == types.AgentStatusOffline {
					newNode.LifecycleStatus = types.AgentStatusStarting
				}
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

		// Evaluate tag approval rules if the service is available and enabled.
		// With default_mode=auto and no rules, this is a no-op (all tags auto-approved).
		var tagApprovalResult *services.TagApprovalResult
		if tagApprovalService != nil && tagApprovalService.IsEnabled() {
			result := tagApprovalService.ProcessRegistrationTags(&newNode)
			tagApprovalResult = &result
			if len(result.Forbidden) > 0 {
				c.JSON(http.StatusForbidden, gin.H{
					"error":          "forbidden_tags",
					"message":        "Registration rejected: agent proposes forbidden tags",
					"forbidden_tags": result.Forbidden,
				})
				return
			}
			if !result.AllAutoApproved {
				logger.Logger.Info().
					Str("agent_id", newNode.ID).
					Strs("pending_tags", result.ManualReview).
					Strs("auto_approved", result.AutoApproved).
					Msg("Agent registration requires tag approval")
			}
		}

		// Store the new node
		if err := storageProvider.RegisterAgent(ctx, &newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Storage error")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to store node: " + err.Error()})
			return
		}
		InvalidateDiscoveryCache()

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

		// Create DID:web document so the DID auth middleware can verify this agent.
		// This is non-fatal — DID:key registration above is the critical path.
		if didWebService != nil {
			if _, _, err := didWebService.GetOrCreateDIDDocument(ctx, newNode.ID); err != nil {
				logger.Logger.Warn().Err(err).Msgf("⚠️ DID:web document creation failed for node %s (non-fatal)", newNode.ID)
			} else {
				logger.Logger.Debug().Msgf("✅ DID:web document ensured for node %s", newNode.ID)
			}
		}

		// Issue Tag VC for auto-approved agents now that agent + DID are stored.
		// This must happen AFTER RegisterAgent + DID registration so that
		// issueTagVC can look up the agent's DID from storage.
		if tagApprovalResult != nil && tagApprovalResult.AllAutoApproved && len(tagApprovalResult.AutoApproved) > 0 && tagApprovalService != nil {
			tagApprovalService.IssueAutoApprovedTagsVC(ctx, newNode.ID, tagApprovalResult.AutoApproved)
		}

		// Note: Node registration events are now handled by the health monitor
		// The health monitor will detect the new node and emit appropriate events

		if presenceManager != nil {
			presenceManager.Touch(newNode.ID, newNode.Version, time.Now().UTC())
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

		// Include tag approval status in response when agent is pending
		if newNode.LifecycleStatus == types.AgentStatusPendingApproval && tagApprovalResult != nil {
			responsePayload["status"] = "pending_approval"
			responsePayload["message"] = "Node registered but awaiting tag approval"
			responsePayload["proposed_tags"] = newNode.ProposedTags
			responsePayload["pending_tags"] = tagApprovalResult.ManualReview
			responsePayload["auto_approved_tags"] = tagApprovalResult.AutoApproved
		}

		c.JSON(http.StatusCreated, responsePayload)
	}
}

// RegisterServerlessAgentHandler handles the registration of a serverless agent node
// by discovering its capabilities via the /discover endpoint
func RegisterServerlessAgentHandler(storageProvider storage.StorageProvider, uiService *services.UIService, didService *services.DIDService, presenceManager *services.PresenceManager, didWebService *services.DIDWebService, allowedDiscoveryHosts []string) gin.HandlerFunc {
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

		normalizedURL, err := normalizeServerlessDiscoveryURL(req.InvocationURL, allowedDiscoveryHosts)
		if err != nil {
			logger.Logger.Warn().Err(err).Msgf("❌ Rejected serverless discovery URL: %s", req.InvocationURL)
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid invocation_url",
				"details": err.Error(),
			})
			return
		}

		logger.Logger.Info().Msgf("🔍 Discovering serverless agent at: %s", normalizedURL)

		// Call the discovery endpoint using normalized URL
		discoveryURL := strings.TrimSuffix(normalizedURL, "/") + "/discover"

		client := &http.Client{
			Timeout: 10 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}

		discoveryCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		discoveryReq, err := http.NewRequestWithContext(discoveryCtx, "GET", discoveryURL, nil)
		if err != nil {
			logger.Logger.Error().Err(err).Msgf("❌ Failed to create discovery request: %s", discoveryURL)
			c.JSON(http.StatusBadGateway, gin.H{
				"error":   "Failed to discover serverless agent",
				"details": fmt.Sprintf("Could not create discovery request: %v", err),
			})
			return
		}

		resp, err := client.Do(discoveryReq)
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
				Tags         []string               `json:"tags"`
			} `json:"reasoners"`
			Skills []struct {
				ID           string                 `json:"id"`
				Name         string                 `json:"name"`
				Description  string                 `json:"description"`
				InputSchema  map[string]interface{} `json:"input_schema"`
				OutputSchema map[string]interface{} `json:"output_schema"`
				Tags         []string               `json:"tags"`
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
				Tags:         r.Tags,
			}
		}

		// Convert discovered skills to AgentNode format. Tags must be copied
		// here so the re-registration preservation path below (which filters
		// Skills[].Tags against existingNode.ApprovedTags) can actually do
		// its job; without this, skills carried no tags in production and
		// the preservation loop was silently a no-op for the Skills slice.
		skills := make([]types.SkillDefinition, len(discoveryData.Skills))
		for i, s := range discoveryData.Skills {
			inputSchemaBytes, _ := json.Marshal(s.InputSchema)
			skills[i] = types.SkillDefinition{
				ID:          s.ID,
				InputSchema: json.RawMessage(inputSchemaBytes),
				Tags:        s.Tags,
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

		// Validate the constructed node before persisting
		if err := validate.Struct(newNode); err != nil {
			logger.Logger.Error().Err(err).Msg("❌ Serverless agent validation error")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Validation failed: " + err.Error()})
			return
		}

		// Check if node already exists
		existingNode, err := storageProvider.GetAgent(ctx, newNode.ID)
		if err == nil && existingNode != nil {
			logger.Logger.Warn().Msgf("⚠️ Serverless agent %s already registered, updating...", newNode.ID)

			adminRevoked := existingNode.LifecycleStatus == types.AgentStatusPendingApproval &&
				len(existingNode.ApprovedTags) == 0
			if adminRevoked {
				logger.Logger.Warn().Msgf("⏸️ Rejecting serverless re-registration for node %s: agent is pending_approval (admin action required)", newNode.ID)
				c.JSON(http.StatusServiceUnavailable, gin.H{
					"error":   "agent_pending_approval",
					"message": fmt.Sprintf("agent node '%s' is awaiting tag approval and cannot re-register", existingNode.ID),
				})
				return
			}

			// Preserve existing approval state so re-registration does not clear
			// approved tags during the UPSERT.
			//
			// IMPORTANT: We deliberately do NOT preserve existingNode.LifecycleStatus
			// here. The serverless node is constructed above with
			// LifecycleStatus = AgentStatusReady, which is the correct state for a
			// serverless agent that just completed discovery. Overwriting with a
			// stale terminal status from a previous row would break downstream
			// status inference (webhook event type, health monitoring, etc.).
			newNode.ApprovedTags = existingNode.ApprovedTags

			if len(existingNode.ApprovedTags) > 0 {
				approvedSet := make(map[string]struct{})
				for _, t := range existingNode.ApprovedTags {
					approvedSet[strings.ToLower(strings.TrimSpace(t))] = struct{}{}
				}
				for i := range newNode.Reasoners {
					var approved []string
					for _, t := range newNode.Reasoners[i].Tags {
						if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
							approved = append(approved, t)
						}
					}
					newNode.Reasoners[i].ApprovedTags = approved
				}
				for i := range newNode.Skills {
					var approved []string
					for _, t := range newNode.Skills[i].Tags {
						if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
							approved = append(approved, t)
						}
					}
					newNode.Skills[i].ApprovedTags = approved
				}
			}
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
		InvalidateDiscoveryCache()

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

		// Create DID:web document so the DID auth middleware can verify this agent.
		// This is non-fatal — DID:key registration above is the critical path.
		if didWebService != nil {
			if _, _, err := didWebService.GetOrCreateDIDDocument(ctx, newNode.ID); err != nil {
				logger.Logger.Warn().Err(err).Msgf("⚠️ DID:web document creation failed for serverless agent %s (non-fatal)", newNode.ID)
			} else {
				logger.Logger.Debug().Msgf("✅ DID:web document ensured for serverless agent %s", newNode.ID)
			}
		}

		// Touch presence manager
		if presenceManager != nil {
			presenceManager.Touch(newNode.ID, newNode.Version, time.Now().UTC())
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
