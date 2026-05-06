package server

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/agentic"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/ui"
	client "github.com/Agent-Field/agentfield/control-plane/web/client"

	"github.com/gin-gonic/gin"
)

// registerUIStatic wires up UI static-asset serving. When the UI is embedded
// in the binary (`UI.Mode == "embedded"` and `client.IsUIEmbedded()`), routes
// are delegated to the embedded file server. Otherwise files are served from
// `UI.DistPath` on disk and a `/` → `/ui/` redirect is installed.
func (s *AgentFieldServer) registerUIStatic() {
	if !s.config.UI.Enabled {
		return
	}

	if s.config.UI.Mode == "embedded" && client.IsUIEmbedded() {
		client.RegisterUIRoutes(s.Router)
		fmt.Println("Using embedded UI files")
		return
	}

	distPath := resolveUIDistPath(s.config.UI.DistPath)
	s.Router.StaticFS("/ui", http.Dir(distPath))
	s.Router.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/ui/")
	})
	fmt.Printf("Using filesystem UI files from: %s\n", distPath)
}

// registerUIAPI installs the internal browser-facing APIs under /api/ui/v1
// and /api/ui/v2. These are registered before /api/v1 routes to prevent route
// conflicts (see the original setupRoutes comment).
func (s *AgentFieldServer) registerUIAPI() {
	if !s.config.UI.Enabled {
		return
	}

	uiAPI := s.Router.Group("/api/ui/v1")
	{
		// Agents management group - All agent-related operations
		agents := uiAPI.Group("/agents")
		{
			// Package API endpoints
			packagesHandler := ui.NewPackageHandler(s.storage)
			agents.GET("/packages", packagesHandler.ListPackagesHandler)
			agents.GET("/packages/:packageId/details", packagesHandler.GetPackageDetailsHandler)

			// Agent lifecycle management endpoints
			lifecycleHandler := ui.NewLifecycleHandler(s.storage, s.agentService)
			agents.GET("/running", lifecycleHandler.ListRunningAgentsHandler)

			// Individual agent operations
			agents.GET("/:agentId/details", func(c *gin.Context) {
				// TODO: Implement agent details
				c.JSON(http.StatusOK, gin.H{"message": "Agent details endpoint"})
			})
			agents.GET("/:agentId/status", lifecycleHandler.GetAgentStatusHandler)
			agents.POST("/:agentId/start", lifecycleHandler.StartAgentHandler)
			agents.POST("/:agentId/stop", lifecycleHandler.StopAgentHandler)
			agents.POST("/:agentId/reconcile", lifecycleHandler.ReconcileAgentHandler)

			// Configuration endpoints
			configHandler := ui.NewConfigHandler(s.storage)
			agents.GET("/:agentId/config/schema", configHandler.GetConfigSchemaHandler)
			agents.GET("/:agentId/config", configHandler.GetConfigHandler)
			agents.POST("/:agentId/config", configHandler.SetConfigHandler)

			// Environment file endpoints
			envHandler := ui.NewEnvHandler(s.storage, s.agentService, s.agentfieldHome)
			agents.GET("/:agentId/env", envHandler.GetEnvHandler)
			agents.PUT("/:agentId/env", envHandler.PutEnvHandler)
			agents.PATCH("/:agentId/env", envHandler.PatchEnvHandler)
			agents.DELETE("/:agentId/env/:key", envHandler.DeleteEnvVarHandler)

			// Agent execution history endpoints
			agentExecutionHandler := ui.NewExecutionHandler(s.storage, s.payloadStore, s.webhookDispatcher)
			agents.GET("/:agentId/executions", agentExecutionHandler.ListExecutionsHandler)
			agents.GET("/:agentId/executions/:executionId", agentExecutionHandler.GetExecutionDetailsHandler)
		}

		// Nodes management group - All node-related operations
		nodes := uiAPI.Group("/nodes")
		{
			uiNodesHandler := ui.NewNodesHandler(s.uiService)
			nodes.GET("/summary", uiNodesHandler.GetNodesSummaryHandler)
			nodes.GET("/events", uiNodesHandler.StreamNodeEventsHandler)

			// Unified status endpoints
			nodes.GET("/:nodeId/status", uiNodesHandler.GetNodeStatusHandler)
			nodes.POST("/:nodeId/status/refresh", uiNodesHandler.RefreshNodeStatusHandler)
			nodes.POST("/status/bulk", uiNodesHandler.BulkNodeStatusHandler)
			nodes.POST("/status/refresh", uiNodesHandler.RefreshAllNodeStatusHandler)

			// Individual node operations
			nodes.GET("/:nodeId/details", uiNodesHandler.GetNodeDetailsHandler)

			nodeLogsHandler := &ui.NodeLogsProxyHandler{
				Storage: s.storage,
				Snapshot: func() (config.NodeLogProxyConfig, string) {
					s.configMu.RLock()
					defer s.configMu.RUnlock()
					return config.EffectiveNodeLogProxy(s.config.AgentField.NodeLogProxy),
						s.config.Features.DID.Authorization.InternalToken
				},
			}
			nodes.GET("/:nodeId/logs", nodeLogsHandler.ProxyNodeLogsHandler)

			nodeLogSettingsHandler := &ui.NodeLogSettingsHandler{
				Storage: s.storage,
				ReadConfig: func(fn func(*config.Config)) {
					s.configMu.RLock()
					defer s.configMu.RUnlock()
					fn(s.config)
				},
				WriteConfig: func(fn func(*config.Config)) {
					s.configMu.Lock()
					defer s.configMu.Unlock()
					fn(s.config)
				},
			}
			settings := uiAPI.Group("/settings")
			{
				settings.GET("/node-log-proxy", nodeLogSettingsHandler.GetNodeLogProxySettingsHandler)
				settings.PUT("/node-log-proxy", nodeLogSettingsHandler.PutNodeLogProxySettingsHandler)
			}

			// DID and VC management endpoints for nodes
			didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService, s.didWebService)
			nodes.GET("/:nodeId/did", didHandler.GetNodeDIDHandler)
			nodes.GET("/:nodeId/vc-status", didHandler.GetNodeVCStatusHandler)
		}

		// Executions management group
		executions := uiAPI.Group("/executions")
		{
			uiExecutionsHandler := ui.NewExecutionHandler(s.storage, s.payloadStore, s.webhookDispatcher)
			executions.GET("/summary", uiExecutionsHandler.GetExecutionsSummaryHandler)
			executions.GET("/stats", uiExecutionsHandler.GetExecutionStatsHandler)
			executions.GET("/enhanced", uiExecutionsHandler.GetEnhancedExecutionsHandler)
			executions.GET("/events", uiExecutionsHandler.StreamExecutionEventsHandler)

			// Timeline endpoint for hourly aggregated data
			timelineHandler := ui.NewExecutionTimelineHandler(s.storage)
			executions.GET("/timeline", timelineHandler.GetExecutionTimelineHandler)

			// Recent activity endpoint
			recentActivityHandler := ui.NewRecentActivityHandler(s.storage)
			executions.GET("/recent", recentActivityHandler.GetRecentActivityHandler)

			// Individual execution operations
			executions.GET("/:execution_id/details", uiExecutionsHandler.GetExecutionDetailsGlobalHandler)
			executions.POST("/:execution_id/webhook/retry", uiExecutionsHandler.RetryExecutionWebhookHandler)
			executions.POST("/:execution_id/cancel", handlers.CancelExecutionHandler(s.storage))
			executions.POST("/:execution_id/pause", handlers.PauseExecutionHandler(s.storage))
			executions.POST("/:execution_id/resume", handlers.ResumeExecutionHandler(s.storage))

			// Execution notes endpoints for UI
			executions.POST("/note", handlers.AddExecutionNoteHandler(s.storage))
			executions.GET("/:execution_id/notes", handlers.GetExecutionNotesHandler(s.storage))

			// Structured execution logs for the execution detail page
			execLogsHandler := ui.NewExecutionLogsHandler(s.storage, s.llmHealthMonitor, func() config.ExecutionLogsConfig {
				return s.config.AgentField.ExecutionLogs
			})
			executions.GET("/:execution_id/logs", execLogsHandler.GetExecutionLogsHandler)
			executions.GET("/:execution_id/logs/stream", execLogsHandler.StreamExecutionLogsHandler)

			// DID and VC management endpoints for executions
			didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService, s.didWebService)
			executions.GET("/:execution_id/vc", didHandler.GetExecutionVCHandler)
			executions.GET("/:execution_id/vc-status", didHandler.GetExecutionVCStatusHandler)
			executions.POST("/:execution_id/verify-vc", didHandler.VerifyExecutionVCComprehensiveHandler)
		}

		// LLM health status endpoint and execution queue status
		llmHandler := ui.NewExecutionLogsHandler(s.storage, s.llmHealthMonitor, func() config.ExecutionLogsConfig {
			return s.config.AgentField.ExecutionLogs
		})
		uiAPI.GET("/llm/health", llmHandler.GetLLMHealthHandler)
		uiAPI.GET("/queue/status", llmHandler.GetExecutionQueueStatusHandler)

		// Workflows management group
		workflows := uiAPI.Group("/workflows")
		{
			workflows.GET("/:workflowId/dag", handlers.GetWorkflowDAGHandler(s.storage))
			workflows.DELETE("/:workflowId/cleanup", handlers.CleanupWorkflowHandler(s.storage))
			didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService, s.didWebService)
			workflows.POST("/vc-status", didHandler.GetWorkflowVCStatusBatchHandler)
			workflows.GET("/:workflowId/vc-chain", didHandler.GetWorkflowVCChainHandler)
			workflows.POST("/:workflowId/verify-vc", didHandler.VerifyWorkflowVCComprehensiveHandler)

			// Workflow notes SSE streaming
			workflowNotesHandler := ui.NewExecutionHandler(s.storage, s.payloadStore, s.webhookDispatcher)
			workflows.GET("/:workflowId/notes/events", workflowNotesHandler.StreamWorkflowNodeNotesHandler)
		}

		// Reasoners management group
		reasoners := uiAPI.Group("/reasoners")
		{
			reasonersHandler := ui.NewReasonersHandler(s.storage)
			reasoners.GET("/all", reasonersHandler.GetAllReasonersHandler)
			reasoners.GET("/events", reasonersHandler.StreamReasonerEventsHandler)
			reasoners.GET("/:reasonerId/details", reasonersHandler.GetReasonerDetailsHandler)
			reasoners.GET("/:reasonerId/metrics", reasonersHandler.GetPerformanceMetricsHandler)
			reasoners.GET("/:reasonerId/executions", reasonersHandler.GetExecutionHistoryHandler)
			reasoners.GET("/:reasonerId/templates", reasonersHandler.GetExecutionTemplatesHandler)
			reasoners.POST("/:reasonerId/templates", reasonersHandler.SaveExecutionTemplateHandler)
		}

		// Dashboard endpoints
		dashboard := uiAPI.Group("/dashboard")
		{
			dashboardHandler := ui.NewDashboardHandler(s.storage, s.agentService)
			dashboard.GET("/summary", dashboardHandler.GetDashboardSummaryHandler)
			dashboard.GET("/enhanced", dashboardHandler.GetEnhancedDashboardSummaryHandler)
		}

		// DID system-wide endpoints
		did := uiAPI.Group("/did")
		{
			didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService, s.didWebService)
			did.GET("/status", didHandler.GetDIDSystemStatusHandler)
			did.GET("/export/vcs", didHandler.ExportVCsHandler)
			did.POST("/verify", didHandler.VerifyVCHandler)
			did.POST("/verify-audit", didHandler.VerifyAuditBundleHandler)
			did.GET("/:did/resolution-bundle", didHandler.GetDIDResolutionBundleHandler)
			did.GET("/:did/resolution-bundle/download", didHandler.DownloadDIDResolutionBundleHandler)
		}

		// VC system-wide endpoints
		vc := uiAPI.Group("/vc")
		{
			didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService, s.didWebService)
			vc.GET("/:vcId/download", didHandler.DownloadVCHandler)
			vc.POST("/verify", didHandler.VerifyVCHandler)
		}

		// Identity & Trust endpoints (DID Explorer and Credentials)
		identityHandler := ui.NewIdentityHandlers(s.storage, s.didWebService)
		identityHandler.RegisterRoutes(uiAPI)

		// Authorization UI endpoints
		authorization := uiAPI.Group("/authorization")
		{
			authorizationHandler := ui.NewAuthorizationHandler(s.storage)
			authorization.GET("/agents", authorizationHandler.GetAgentsWithTagsHandler)
		}
	}

	uiAPIV2 := s.Router.Group("/api/ui/v2")
	{
		workflowRunsHandler := ui.NewWorkflowRunHandler(s.storage)
		uiAPIV2.GET("/workflow-runs", workflowRunsHandler.ListWorkflowRunsHandler)
		uiAPIV2.GET("/workflow-runs/:run_id", workflowRunsHandler.GetWorkflowRunDetailHandler)
	}
}

// register404 installs the Smart 404 handler that suggests similar endpoints
// for missing API paths and falls back to serving the SPA's index.html for
// client-side routes when the UI is served from disk.
func (s *AgentFieldServer) register404() {
	var uiNoRouteHandler gin.HandlerFunc
	if s.config.UI.Enabled && (s.config.UI.Mode != "embedded" || !client.IsUIEmbedded()) {
		uiNoRouteHandler = func(c *gin.Context) {
			path := strings.ToLower(c.Request.URL.Path)
			isStaticAsset := strings.HasSuffix(path, ".js") ||
				strings.HasSuffix(path, ".css") ||
				strings.HasSuffix(path, ".html") ||
				strings.HasSuffix(path, ".ico") ||
				strings.HasSuffix(path, ".png") ||
				strings.HasSuffix(path, ".jpg") ||
				strings.HasSuffix(path, ".jpeg") ||
				strings.HasSuffix(path, ".gif") ||
				strings.HasSuffix(path, ".svg") ||
				strings.HasSuffix(path, ".woff") ||
				strings.HasSuffix(path, ".woff2") ||
				strings.HasSuffix(path, ".ttf") ||
				strings.HasSuffix(path, ".eot") ||
				strings.HasSuffix(path, ".map") ||
				strings.HasSuffix(path, ".json") ||
				strings.HasSuffix(path, ".xml") ||
				strings.HasSuffix(path, ".txt")

			if isStaticAsset {
				c.JSON(http.StatusNotFound, gin.H{"error": "file not found"})
				return
			}

			distPath := resolveUIDistPath(s.config.UI.DistPath)
			c.File(filepath.Join(distPath, "index.html"))
		}
	}
	s.Router.NoRoute(agentic.Smart404Handler(s.apiCatalog, uiNoRouteHandler))
}

// resolveUIDistPath hunts for the UI dist directory relative to the executable.
// The search order preserves the pre-refactor behavior: explicit override,
// executable-relative `web/client/dist`, the `apps/platform/...` layout, and
// finally cwd fallbacks.
func resolveUIDistPath(override string) string {
	if override != "" {
		return override
	}

	execPath, err := os.Executable()
	if err != nil {
		distPath := filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
		if _, statErr := os.Stat(distPath); os.IsNotExist(statErr) {
			return filepath.Join("web", "client", "dist")
		}
		return distPath
	}

	execDir := filepath.Dir(execPath)
	distPath := filepath.Join(execDir, "web", "client", "dist")
	if _, err := os.Stat(distPath); os.IsNotExist(err) {
		distPath = filepath.Join(filepath.Dir(execDir), "apps", "platform", "agentfield", "web", "client", "dist")
	}
	if _, err := os.Stat(distPath); os.IsNotExist(err) {
		altPath := filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
		if _, altErr := os.Stat(altPath); altErr == nil {
			return altPath
		}
		return filepath.Join("web", "client", "dist")
	}
	return distPath
}
