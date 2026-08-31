package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// registerPublicRoutes installs endpoints that are available without
// authentication: Prometheus metrics scraper and the load-balancer health probe.
func (s *AgentFieldServer) registerPublicRoutes() {
	s.Router.GET("/metrics", gin.WrapH(promhttp.Handler()))
	s.Router.GET("/health", s.healthCheckHandler)
	s.Router.GET("/readyz", s.readinessHandler)
}

// registerCoreRoutes installs the core agent-facing REST surface under
// /api/v1: node lifecycle, discovery, execution, approvals, and notes.
func (s *AgentFieldServer) registerCoreRoutes(agentAPI *gin.RouterGroup) {
	// Health check endpoint for container orchestration
	agentAPI.GET("/health", s.healthCheckHandler)
	agentAPI.GET("/health/ready", s.readinessHandler)
	agentAPI.GET("/version", s.versionHandler)

	// Apply global rate limiting if enabled
	if s.rateLimitGlobal != nil {
		agentAPI.Use(middleware.RateLimit(s.rateLimitGlobal))
	}

	// Discovery endpoints
	discovery := agentAPI.Group("/discovery")
	if s.rateLimitDiscovery != nil {
		discovery.Use(middleware.RateLimit(s.rateLimitDiscovery))
	}
	{
		discovery.GET("/capabilities", handlers.DiscoveryCapabilitiesHandler(s.storage))
	}
	agentAPI.GET("/reasoners", handlers.ListReasonersHandler(s.storage))

	// Node management endpoints
	agentAPI.POST("/nodes/register", handlers.RegisterNodeHandler(s.storage, s.uiService, s.didService, s.presenceManager, s.didWebService, s.tagApprovalService))
	agentAPI.POST("/nodes", handlers.RegisterNodeHandler(s.storage, s.uiService, s.didService, s.presenceManager, s.didWebService, s.tagApprovalService))
	agentAPI.POST("/nodes/register-serverless", handlers.RegisterServerlessAgentHandler(s.storage, s.uiService, s.didService, s.presenceManager, s.didWebService, s.config.AgentField.Registration.ServerlessDiscoveryAllowedHosts))
	agentAPI.GET("/nodes", handlers.ListNodesHandler(s.storage))
	agentAPI.GET("/nodes/:node_id", handlers.GetNodeHandler(s.storage))
	agentAPI.POST("/nodes/:node_id/heartbeat", handlers.HeartbeatHandler(s.storage, s.uiService, s.healthMonitor, s.statusManager, s.presenceManager))
	agentAPI.DELETE("/nodes/:node_id/monitoring", s.unregisterAgentFromMonitoring)

	// New unified status API endpoints
	agentAPI.GET("/nodes/:node_id/status", handlers.GetNodeStatusHandler(s.statusManager))
	agentAPI.POST("/nodes/:node_id/status/refresh", handlers.RefreshNodeStatusHandler(s.statusManager))
	agentAPI.POST("/nodes/status/bulk", s.withBulkStatusRateLimit(handlers.BulkNodeStatusHandler(s.statusManager, s.storage)))
	agentAPI.POST("/nodes/status/refresh", handlers.RefreshAllNodeStatusHandler(s.statusManager, s.storage))

	// Enhanced lifecycle management endpoints
	agentAPI.POST("/nodes/:node_id/start", handlers.StartNodeHandler(s.statusManager, s.storage))
	agentAPI.POST("/nodes/:node_id/stop", handlers.StopNodeHandler(s.statusManager, s.storage))
	agentAPI.POST("/nodes/:node_id/lifecycle/status", handlers.UpdateLifecycleStatusHandler(s.storage, s.uiService, s.statusManager))
	agentAPI.PATCH("/nodes/:node_id/status", handlers.NodeStatusLeaseHandler(s.storage, s.statusManager, s.healthMonitor, s.presenceManager, handlers.DefaultLeaseTTL))
	agentAPI.POST("/nodes/:node_id/actions/ack", handlers.NodeActionAckHandler(s.storage, s.presenceManager, handlers.DefaultLeaseTTL))
	agentAPI.POST("/nodes/:node_id/shutdown", handlers.NodeShutdownHandler(s.storage, s.statusManager, s.presenceManager))
	agentAPI.POST("/actions/claim", handlers.ClaimActionsHandler(s.storage, s.presenceManager, handlers.DefaultLeaseTTL))

	// TODO: Add other node routes (DeleteNode)

	// Reasoner and skill execution endpoints (legacy)
	// When authorization is enabled, these require the same permission middleware
	// as the unified execute endpoints to prevent policy bypass.
	if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
		legacyReasonerGroup := agentAPI.Group("/reasoners")
		legacySkillGroup := agentAPI.Group("/skills")
		permConfigLegacy := middleware.PermissionConfig{
			Enabled:     true,
			DefaultDeny: s.config.Features.DID.Authorization.DefaultDeny,
		}
		legacyMiddleware := middleware.PermissionCheckMiddleware(
			s.accessPolicyService,
			s.tagVCVerifier,
			s.storage,
			s.didWebService,
			permConfigLegacy,
		)
		legacyReasonerGroup.Use(legacyMiddleware)
		legacySkillGroup.Use(legacyMiddleware)
		legacyReasonerGroup.POST("/:reasoner_id", handlers.ExecuteReasonerHandler(s.storage))
		legacySkillGroup.POST("/:skill_id", handlers.ExecuteSkillHandler(s.storage))
		logger.Logger.Info().Msg("🔒 Permission checking enabled on legacy reasoner/skill endpoints")
	} else {
		agentAPI.POST("/reasoners/:reasoner_id", handlers.ExecuteReasonerHandler(s.storage))
		agentAPI.POST("/skills/:skill_id", handlers.ExecuteSkillHandler(s.storage))
	}

	// Unified execution endpoints (path-based)
	// These routes may have permission middleware applied if authorization is enabled.
	executeGroup := agentAPI.Group("/execute")
	{
		if s.rateLimitExecute != nil {
			executeGroup.Use(middleware.RateLimit(s.rateLimitExecute))
		}
		if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
			permConfig := middleware.PermissionConfig{
				Enabled:     true,
				DefaultDeny: s.config.Features.DID.Authorization.DefaultDeny,
			}
			executeGroup.Use(middleware.PermissionCheckMiddleware(
				s.accessPolicyService,
				s.tagVCVerifier,
				s.storage,
				s.didWebService,
				permConfig,
			))
			logger.Logger.Info().Msg("🔒 Permission checking enabled on execute endpoints")
		}

		executeGroup.POST("/:target", handlers.ExecuteHandlerWithARD(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken, func() config.ARDConfig {
			s.configMu.RLock()
			defer s.configMu.RUnlock()
			return s.config.AgentField.ARD
		}))
		executeGroup.POST("/async/:target", handlers.ExecuteAsyncHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken))
	}
	// Static "active" coexists with the :execution_id param route (same
	// pattern as POST /executions/batch-status); gin matches static first.
	agentAPI.GET("/executions/active", handlers.ActiveExecutionsHandler(s.storage))
	agentAPI.GET("/executions/:execution_id", handlers.GetExecutionStatusHandler(s.storage))
	agentAPI.GET("/executions/:execution_id/events", s.streamHandler(handlers.StreamExecutionEventsHandler(s.storage)))
	agentAPI.POST("/executions/batch-status", handlers.BatchExecutionStatusHandler(s.storage))
	agentAPI.POST("/executions/:execution_id/status", handlers.UpdateExecutionStatusHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout))
	agentAPI.POST("/executions/:execution_id/logs", handlers.StructuredExecutionLogsHandler(s.storage, func() config.ExecutionLogsConfig {
		return s.config.AgentField.ExecutionLogs
	}))
	agentAPI.POST("/executions/:execution_id/cancel", handlers.CancelExecutionHandler(s.storage))
	agentAPI.POST("/executions/:execution_id/pause", handlers.PauseExecutionHandler(s.storage))
	agentAPI.POST("/executions/:execution_id/resume", handlers.ResumeExecutionHandler(s.storage))
	agentAPI.POST("/executions/:execution_id/restart", handlers.RestartExecutionHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken))
	agentAPI.POST("/workflows/:workflowId/cancel-tree", handlers.CancelWorkflowTreeHandler(s.storage))

	// Approval workflow endpoints — CP manages execution state only;
	// agents handle external approval service communication directly.
	agentAPI.POST("/executions/:execution_id/request-approval", handlers.RequestApprovalHandler(s.storage))
	agentAPI.GET("/executions/:execution_id/approval-status", handlers.GetApprovalStatusHandler(s.storage))
	// Authenticated approval resolution by execution ID — same resolution
	// logic as the HMAC webhook below, for operators/CLIs that hold an API
	// key rather than the webhook secret.
	agentAPI.POST("/executions/:execution_id/approval-response", handlers.ResolveApprovalHandler(s.storage))

	// Agent-scoped approval routes — enforce that the execution belongs to the requesting agent.
	agentAPI.POST("/agents/:node_id/executions/:execution_id/request-approval", handlers.AgentScopedRequestApprovalHandler(s.storage))
	agentAPI.GET("/agents/:node_id/executions/:execution_id/approval-status", handlers.AgentScopedGetApprovalStatusHandler(s.storage))

	// Multi-hop pause propagation: an agent's app.call wait loop pushes its
	// OWN status to WAITING when its awaited child enters WAITING, so any
	// further ancestor sees WAITING transitively. Without this, pause only
	// propagates one hop up the call tree and 3+-deep chains time out at
	// wallclock on the great-grandparent.
	agentAPI.POST("/agents/:node_id/executions/:execution_id/awaiter-status", handlers.UpdateAwaiterStatusHandler(s.storage))

	// Approval resolution webhook (called by agents or external services when approval resolves)
	agentAPI.POST("/webhooks/approval-response", handlers.ApprovalWebhookHandler(s.storage, s.config.AgentField.Approval.WebhookSecret))

	// Execution notes endpoints for app.note() feature
	agentAPI.POST("/executions/note", handlers.AddExecutionNoteHandler(s.storage, s.noteOwnershipEnforced()))
	agentAPI.GET("/executions/:execution_id/notes", handlers.GetExecutionNotesHandler(s.storage, s.noteOwnershipEnforced()))
	agentAPI.POST("/workflow/executions/events", handlers.WorkflowExecutionEventHandler(s.storage))

	sessionTargetGroup := agentAPI.Group("/session-targets")
	{
		if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
			permConfig := middleware.PermissionConfig{
				Enabled:     true,
				DefaultDeny: s.config.Features.DID.Authorization.DefaultDeny,
			}
			sessionTargetGroup.Use(middleware.PermissionCheckMiddleware(
				s.accessPolicyService,
				s.tagVCVerifier,
				s.storage,
				s.didWebService,
				permConfig,
			))
			logger.Logger.Info().Msg("🔒 Permission checking enabled on session target endpoints")
		}
		sessionTargetGroup.POST("/:target/start", handlers.StartSessionHandler(s.storage))
	}
	sessionInstanceGroup := agentAPI.Group("/session-instances")
	{
		sessionInstanceGroup.POST("/:session_id/realtime-offer", handlers.SessionRealtimeOfferHandler(s.storage))
		sessionInstanceGroup.POST("/:session_id/tools/:tool", handlers.SessionToolHandler(s.storage, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken))
	}

	// Workflow endpoints will be reintroduced once the simplified execution pipeline lands.
}

// unregisterAgentFromMonitoring removes an agent from health monitoring.
func (s *AgentFieldServer) unregisterAgentFromMonitoring(c *gin.Context) {
	nodeID := c.Param("node_id")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
		return
	}

	if s.healthMonitor != nil {
		s.healthMonitor.UnregisterAgent(nodeID)
		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"message": fmt.Sprintf("Agent %s unregistered from health monitoring", nodeID),
		})
	} else {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "health monitor not available"})
	}
}

// healthCheckHandler provides a comprehensive health check for container
// orchestration (Railway, K8s readiness probes, etc).
func (s *AgentFieldServer) healthCheckHandler(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	healthStatus := gin.H{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   buildVersion,
		"checks":    gin.H{},
	}
	if furrowPublicAddr := os.Getenv("FURROW_PUBLIC_ADDR"); furrowPublicAddr != "" {
		healthStatus["furrow_public_addr"] = furrowPublicAddr
	}

	allHealthy := true
	checks := healthStatus["checks"].(gin.H)

	// Storage health check
	if s.storage != nil || s.storageHealthOverride != nil {
		storageHealth := s.checkStorageHealth(ctx)
		checks["storage"] = storageHealth
		if storageHealth["status"] != "healthy" {
			allHealthy = false
		}
	} else {
		checks["storage"] = gin.H{
			"status":  "unhealthy",
			"message": "storage not initialized",
		}
		allHealthy = false
	}

	// Cache health check
	if s.cache != nil || s.cacheHealthOverride != nil {
		cacheHealth := s.checkCacheHealth(ctx)
		checks["cache"] = cacheHealth
		if cacheHealth["status"] != "healthy" {
			allHealthy = false
		}
	} else {
		checks["cache"] = gin.H{
			"status":  "healthy",
			"message": "cache not configured (optional)",
		}
	}

	if !allHealthy {
		healthStatus["status"] = "unhealthy"
		c.JSON(http.StatusServiceUnavailable, healthStatus)
		return
	}

	c.JSON(http.StatusOK, healthStatus)
}

// readinessHandler answers Kubernetes readiness probes. It reports the same
// dependency checks as healthCheckHandler, except that once BeginDrain has
// run it answers 503 unconditionally so kube-proxy removes this pod from the
// Service endpoints while the listener is still accepting and completing
// requests.
func (s *AgentFieldServer) readinessHandler(c *gin.Context) {
	if s.IsDraining() {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":    "draining",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   buildVersion,
		})
		return
	}
	s.healthCheckHandler(c)
}

type hostingInfo struct {
	Platform      string `json:"platform"`
	ProjectID     string `json:"project_id,omitempty"`
	EnvironmentID string `json:"environment_id,omitempty"`
	ServiceID     string `json:"service_id,omitempty"`
	DeploymentID  string `json:"deployment_id,omitempty"`
	Region        string `json:"region,omitempty"`
}

func (s *AgentFieldServer) versionHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"version":    buildVersion,
		"commit":     buildCommit,
		"build_date": buildDate,
		"hosting":    detectHosting(),
		"features":   []string{"package_updates", "boot_restore"},
	})
}

func detectHosting() hostingInfo {
	platform := packages.HostingPlatform()
	if platform == packages.HostingRailway {
		serviceID := os.Getenv("RAILWAY_SERVICE_ID")
		return hostingInfo{
			Platform:      platform,
			ProjectID:     os.Getenv("RAILWAY_PROJECT_ID"),
			EnvironmentID: os.Getenv("RAILWAY_ENVIRONMENT_ID"),
			ServiceID:     serviceID,
			DeploymentID:  os.Getenv("RAILWAY_DEPLOYMENT_ID"),
			Region:        os.Getenv("RAILWAY_REPLICA_REGION"),
		}
	} else if platform == packages.HostingDocker {
		return hostingInfo{Platform: platform}
	}

	return hostingInfo{Platform: packages.HostingLocal}
}

// checkStorageHealth performs a lightweight storage readiness probe.
func (s *AgentFieldServer) checkStorageHealth(ctx context.Context) gin.H {
	if s.storageHealthOverride != nil {
		return s.storageHealthOverride(ctx)
	}

	startTime := time.Now()

	if err := ctx.Err(); err != nil {
		return gin.H{
			"status":  "unhealthy",
			"message": "context timeout during storage check",
		}
	}

	return gin.H{
		"status":        "healthy",
		"message":       "storage is responsive",
		"response_time": time.Since(startTime).Milliseconds(),
	}
}

// checkCacheHealth performs a lightweight cache round-trip readiness probe.
func (s *AgentFieldServer) checkCacheHealth(ctx context.Context) gin.H {
	if s.cacheHealthOverride != nil {
		return s.cacheHealthOverride(ctx)
	}

	startTime := time.Now()

	testKey := "health_check_" + fmt.Sprintf("%d", time.Now().Unix())
	testValue := "ok"

	if err := s.cache.Set(testKey, testValue, time.Minute); err != nil {
		return gin.H{
			"status":        "unhealthy",
			"message":       fmt.Sprintf("cache set operation failed: %v", err),
			"response_time": time.Since(startTime).Milliseconds(),
		}
	}

	var retrieved string
	if err := s.cache.Get(testKey, &retrieved); err != nil {
		return gin.H{
			"status":        "unhealthy",
			"message":       fmt.Sprintf("cache get operation failed: %v", err),
			"response_time": time.Since(startTime).Milliseconds(),
		}
	}

	if err := s.cache.Delete(testKey); err != nil {
		return gin.H{
			"status":        "unhealthy",
			"message":       fmt.Sprintf("cache delete operation failed: %v", err),
			"response_time": time.Since(startTime).Milliseconds(),
		}
	}

	return gin.H{
		"status":        "healthy",
		"message":       "cache is responsive",
		"response_time": time.Since(startTime).Milliseconds(),
	}
}

// withBulkStatusRateLimit wraps a handler with the bulk-status rate limiter.
// Returns the handler directly if rate limiting is not enabled.
func (s *AgentFieldServer) withBulkStatusRateLimit(h gin.HandlerFunc) gin.HandlerFunc {
	if s.rateLimitBulkStatus == nil {
		return h
	}
	return func(c *gin.Context) {
		key := rateLimitKeyFromContext(c)
		if !s.rateLimitBulkStatus.Allow(key) {
			retryAfter := "1"
			c.Header("Retry-After", retryAfter)
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error":          "rate limit exceeded",
				"error_category": "rate_limit",
				"retry_after":    retryAfter,
			})
			return
		}
		h(c)
	}
}

// rateLimitKeyFromContext derives a rate limit key from the gin context.
func rateLimitKeyFromContext(c *gin.Context) string {
	if apiKey, exists := c.Get("api_key_identity"); exists {
		if key, ok := apiKey.(string); ok && key != "" {
			return "key:" + key
		}
	}
	return "ip:" + c.ClientIP()
}
