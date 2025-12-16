package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services" // Core services
	"github.com/Agent-Field/agentfield/control-plane/internal/events"                     // Event system
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"                   // Agent handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/ui"                // UI handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/communication"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/process"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services" // Services
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/adminpb"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	client "github.com/Agent-Field/agentfield/control-plane/web/client"

	"github.com/gin-contrib/cors" // CORS middleware
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// AgentFieldServer represents the core AgentField orchestration service.
type AgentFieldServer struct {
	adminpb.UnimplementedAdminReasonerServiceServer
	storage               storage.StorageProvider
	cache                 storage.CacheProvider
	Router                *gin.Engine
	uiService             *services.UIService           // Add UIService
	executionsUIService   *services.ExecutionsUIService // Add ExecutionsUIService
	healthMonitor         *services.HealthMonitor
	presenceManager       *services.PresenceManager
	statusManager         *services.StatusManager // Add StatusManager for unified status management
	agentService          interfaces.AgentService // Add AgentService for lifecycle management
	agentClient           interfaces.AgentClient  // Add AgentClient for MCP communication
	config                *config.Config
	storageHealthOverride func(context.Context) gin.H
	cacheHealthOverride   func(context.Context) gin.H
	// DID Services
	keystoreService *services.KeystoreService
	didService      *services.DIDService
	vcService       *services.VCService
	didRegistry     *services.DIDRegistry
	agentfieldHome  string
	// Cleanup service
	cleanupService        *handlers.ExecutionCleanupService
	payloadStore          services.PayloadStore
	registryWatcherCancel context.CancelFunc
	adminGRPCServer       *grpc.Server
	adminListener         net.Listener
	adminGRPCPort         int
	webhookDispatcher     services.WebhookDispatcher
}

// NewAgentFieldServer creates a new instance of the AgentFieldServer.
func NewAgentFieldServer(cfg *config.Config) (*AgentFieldServer, error) {
	// Define agentfieldHome at the very top
	agentfieldHome := os.Getenv("AGENTFIELD_HOME")
	if agentfieldHome == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		agentfieldHome = filepath.Join(homeDir, ".agentfield")
	}

	dirs, err := utils.EnsureDataDirectories()
	if err != nil {
		return nil, fmt.Errorf("failed to ensure data directories: %w", err)
	}

	factory := &storage.StorageFactory{}
	storageProvider, cacheProvider, err := factory.CreateStorage(cfg.Storage)
	if err != nil {
		return nil, err
	}

	Router := gin.Default()

	// Sync installed.yaml to database for package visibility
	_ = SyncPackagesFromRegistry(agentfieldHome, storageProvider)

	// Initialize agent client for communication with agent nodes
	agentClient := communication.NewHTTPAgentClient(storageProvider, 5*time.Second)

	// Create infrastructure components for AgentService
	fileSystem := infrastorage.NewFileSystemAdapter()
	registryPath := filepath.Join(agentfieldHome, "installed.json")
	registryStorage := infrastorage.NewLocalRegistryStorage(fileSystem, registryPath)
	processManager := process.NewProcessManager()
	portManager := process.NewPortManager()

	// Create AgentService
	agentService := coreservices.NewAgentService(processManager, portManager, registryStorage, agentClient, agentfieldHome)

	// Initialize StatusManager for unified status management
	statusManagerConfig := services.StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
		StatusCacheTTL:    5 * time.Minute,
		MaxTransitionTime: 2 * time.Minute,
	}

	// Create UIService first (without StatusManager)
	uiService := services.NewUIService(storageProvider, agentClient, agentService, nil)

	// Create StatusManager with UIService and AgentClient
	statusManager := services.NewStatusManager(storageProvider, statusManagerConfig, uiService, agentClient)

	// Update UIService with StatusManager reference
	uiService = services.NewUIService(storageProvider, agentClient, agentService, statusManager)

	// Presence manager tracks node leases so stale nodes age out quickly
	presenceConfig := services.PresenceManagerConfig{
		HeartbeatTTL:  5 * time.Minute,
		SweepInterval: 30 * time.Second,
		HardEvictTTL:  30 * time.Minute,
	}
	presenceManager := services.NewPresenceManager(statusManager, presenceConfig)

	executionsUIService := services.NewExecutionsUIService(storageProvider) // Initialize ExecutionsUIService

	// Initialize health monitor with StatusManager integration
	healthMonitorConfig := services.HealthMonitorConfig{}
	healthMonitor := services.NewHealthMonitor(storageProvider, healthMonitorConfig, uiService, agentClient, statusManager, presenceManager)
	presenceManager.SetExpireCallback(healthMonitor.UnregisterAgent)

	// Initialize DID services if enabled
	var keystoreService *services.KeystoreService
	var didService *services.DIDService
	var vcService *services.VCService
	var didRegistry *services.DIDRegistry

	if cfg.Features.DID.Enabled {
		fmt.Println("🔐 Initializing DID and VC services...")

		// Use universal path management for DID directories
		dirs, err := utils.EnsureDataDirectories()
		if err != nil {
			return nil, fmt.Errorf("failed to create DID directories: %w", err)
		}

		// Update keystore path to use universal paths
		if cfg.Features.DID.Keystore.Path == "./data/keys" {
			cfg.Features.DID.Keystore.Path = dirs.KeysDir
		}

		fmt.Printf("🔑 Creating keystore service at: %s\n", cfg.Features.DID.Keystore.Path)
		// Instantiate services in dependency order: Keystore → DID → VC, Registry
		keystoreService, err = services.NewKeystoreService(&cfg.Features.DID.Keystore)
		if err != nil {
			return nil, fmt.Errorf("failed to create keystore service: %w", err)
		}

		fmt.Println("📋 Creating DID registry...")
		didRegistry = services.NewDIDRegistryWithStorage(storageProvider)

		fmt.Println("🆔 Creating DID service...")
		didService = services.NewDIDService(&cfg.Features.DID, keystoreService, didRegistry)

		fmt.Println("📜 Creating VC service...")
		vcService = services.NewVCService(&cfg.Features.DID, didService, storageProvider)

		// Initialize services
		fmt.Println("🔧 Initializing DID registry...")
		if err = didRegistry.Initialize(); err != nil {
			return nil, fmt.Errorf("failed to initialize DID registry: %w", err)
		}

		fmt.Println("🔧 Initializing VC service...")
		if err = vcService.Initialize(); err != nil {
			return nil, fmt.Errorf("failed to initialize VC service: %w", err)
		}

		// Generate af server ID based on agentfield home directory
		agentfieldServerID := generateAgentFieldServerID(agentfieldHome)

		// Initialize af server DID with dynamic ID
		fmt.Printf("🧠 Initializing af server DID (ID: %s)...\n", agentfieldServerID)
		if err := didService.Initialize(agentfieldServerID); err != nil {
			return nil, fmt.Errorf("failed to initialize af server DID: %w", err)
		}

		// Validate that af server DID was successfully created
		registry, err := didService.GetRegistry(agentfieldServerID)
		if err != nil {
			return nil, fmt.Errorf("failed to validate af server DID creation: %w", err)
		}
		if registry == nil || registry.RootDID == "" {
			return nil, fmt.Errorf("af server DID validation failed: registry or root DID is empty")
		}

		fmt.Printf("✅ AgentField server DID created successfully: %s\n", registry.RootDID)

		// Backfill existing nodes with DIDs
		fmt.Println("🔄 Starting DID backfill for existing nodes...")
		ctx := context.Background()
		if err := didService.BackfillExistingNodes(ctx, storageProvider); err != nil {
			fmt.Printf("⚠️ DID backfill failed: %v\n", err)
		}

		fmt.Println("✅ DID and VC services initialized successfully!")
	} else {
		fmt.Println("⚠️ DID and VC services are DISABLED in configuration")
	}

	payloadStore := services.NewFilePayloadStore(dirs.PayloadsDir)

	webhookDispatcher := services.NewWebhookDispatcher(storageProvider, services.WebhookDispatcherConfig{
		Timeout:         cfg.AgentField.ExecutionQueue.WebhookTimeout,
		MaxAttempts:     cfg.AgentField.ExecutionQueue.WebhookMaxAttempts,
		RetryBackoff:    cfg.AgentField.ExecutionQueue.WebhookRetryBackoff,
		MaxRetryBackoff: cfg.AgentField.ExecutionQueue.WebhookMaxRetryBackoff,
	})
	if err := webhookDispatcher.Start(context.Background()); err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to start webhook dispatcher")
	}

	// Initialize execution cleanup service
	cleanupService := handlers.NewExecutionCleanupService(storageProvider, cfg.AgentField.ExecutionCleanup)

	adminPort := cfg.AgentField.Port + 100
	if envPort := os.Getenv("AGENTFIELD_ADMIN_GRPC_PORT"); envPort != "" {
		if parsedPort, parseErr := strconv.Atoi(envPort); parseErr == nil {
			adminPort = parsedPort
		} else {
			logger.Logger.Warn().Err(parseErr).Str("value", envPort).Msg("invalid AGENTFIELD_ADMIN_GRPC_PORT, using default offset")
		}
	}

	return &AgentFieldServer{
		storage:               storageProvider,
		cache:                 cacheProvider,
		Router:                Router,
		uiService:             uiService,
		executionsUIService:   executionsUIService,
		healthMonitor:         healthMonitor,
		presenceManager:       presenceManager,
		statusManager:         statusManager,
		agentService:          agentService,
		agentClient:           agentClient,
		config:                cfg,
		keystoreService:       keystoreService,
		didService:            didService,
		vcService:             vcService,
		didRegistry:           didRegistry,
		agentfieldHome:        agentfieldHome,
		cleanupService:        cleanupService,
		payloadStore:          payloadStore,
		webhookDispatcher:     webhookDispatcher,
		registryWatcherCancel: nil,
		adminGRPCPort:         adminPort,
	}, nil
}

// Start initializes and starts the AgentFieldServer.
func (s *AgentFieldServer) Start() error {
	// Setup routes
	s.setupRoutes()

	// Start status manager service in background
	go s.statusManager.Start()

	if s.presenceManager != nil {
		go s.presenceManager.Start()

		// Recover presence leases from database
		go func() {
			ctx := context.Background()
			if err := s.presenceManager.RecoverFromDatabase(ctx, s.storage); err != nil {
				logger.Logger.Error().Err(err).Msg("Failed to recover presence leases from database")
			}
		}()
	}

	// Start health monitor service in background
	go s.healthMonitor.Start()

	// Recover previously registered nodes and check their health
	go func() {
		ctx := context.Background()
		if err := s.healthMonitor.RecoverFromDatabase(ctx); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to recover nodes from database")
		}
	}()

	// Start execution cleanup service in background
	ctx := context.Background()
	if err := s.cleanupService.Start(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to start execution cleanup service")
		// Don't fail server startup if cleanup service fails to start
	}

	// Start reasoner event heartbeat (30 second intervals)
	events.StartHeartbeat(30 * time.Second)

	// Start node event heartbeat (30 second intervals)
	events.StartNodeHeartbeat(30 * time.Second)

	if s.registryWatcherCancel == nil {
		cancel, err := StartPackageRegistryWatcher(context.Background(), s.agentfieldHome, s.storage)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to start package registry watcher")
		} else {
			s.registryWatcherCancel = cancel
		}
	}

	if err := s.startAdminGRPCServer(); err != nil {
		return fmt.Errorf("failed to start admin gRPC server: %w", err)
	}

	// TODO: Implement WebSocket, gRPC
	// Start HTTP server
	return s.Router.Run(":" + strconv.Itoa(s.config.AgentField.Port))
}

func (s *AgentFieldServer) startAdminGRPCServer() error {
	if s.adminGRPCServer != nil {
		return nil
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.adminGRPCPort))
	if err != nil {
		return err
	}

	s.adminListener = lis
	opts := []grpc.ServerOption{}
	if s.config.API.Auth.APIKey != "" {
		opts = append(opts, grpc.UnaryInterceptor(
			middleware.APIKeyUnaryInterceptor(s.config.API.Auth.APIKey),
		))
	}
	s.adminGRPCServer = grpc.NewServer(opts...)
	adminpb.RegisterAdminReasonerServiceServer(s.adminGRPCServer, s)

	go func() {
		if serveErr := s.adminGRPCServer.Serve(lis); serveErr != nil && !errors.Is(serveErr, grpc.ErrServerStopped) {
			logger.Logger.Error().Err(serveErr).Msg("admin gRPC server stopped unexpectedly")
		}
	}()

	logger.Logger.Info().Int("port", s.adminGRPCPort).Msg("admin gRPC server listening")
	return nil
}

// ListReasoners implements the admin gRPC surface for listing registered reasoners.
func (s *AgentFieldServer) ListReasoners(ctx context.Context, _ *adminpb.ListReasonersRequest) (*adminpb.ListReasonersResponse, error) {
	nodes, err := s.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list agent nodes: %v", err)
	}

	resp := &adminpb.ListReasonersResponse{}
	for _, node := range nodes {
		if node == nil {
			continue
		}
		for _, reasoner := range node.Reasoners {
			resp.Reasoners = append(resp.Reasoners, &adminpb.Reasoner{
				ReasonerId:    fmt.Sprintf("%s.%s", node.ID, reasoner.ID),
				AgentNodeId:   node.ID,
				Name:          reasoner.ID,
				Description:   fmt.Sprintf("Reasoner %s from node %s", reasoner.ID, node.ID),
				Status:        string(node.HealthStatus),
				NodeVersion:   node.Version,
				LastHeartbeat: node.LastHeartbeat.Format(time.RFC3339),
			})
		}
	}

	return resp, nil
}

// Stop gracefully shuts down the AgentFieldServer.
func (s *AgentFieldServer) Stop() error {
	if s.adminGRPCServer != nil {
		s.adminGRPCServer.GracefulStop()
	}
	if s.adminListener != nil {
		_ = s.adminListener.Close()
	}

	// Stop status manager service
	if s.statusManager != nil {
		s.statusManager.Stop()
	}

	if s.presenceManager != nil {
		s.presenceManager.Stop()
	}

	// Stop health monitor service
	s.healthMonitor.Stop()

	// Stop execution cleanup service
	if s.cleanupService != nil {
		if err := s.cleanupService.Stop(); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to stop execution cleanup service")
		}
	}

	if s.registryWatcherCancel != nil {
		s.registryWatcherCancel()
		s.registryWatcherCancel = nil
	}

	// Stop UI service heartbeat
	if s.uiService != nil {
		s.uiService.StopHeartbeat()
	}

	// TODO: Implement graceful shutdown for HTTP, WebSocket, gRPC
	return nil
}

// unregisterAgentFromMonitoring removes an agent from health monitoring
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

// healthCheckHandler provides comprehensive health check for container orchestration
func (s *AgentFieldServer) healthCheckHandler(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	healthStatus := gin.H{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "1.0.0", // TODO: Get from build info
		"checks":    gin.H{},
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

	// Overall status
	if !allHealthy {
		healthStatus["status"] = "unhealthy"
		c.JSON(http.StatusServiceUnavailable, healthStatus)
		return
	}

	c.JSON(http.StatusOK, healthStatus)
}

// checkStorageHealth performs storage-specific health checks
func (s *AgentFieldServer) checkStorageHealth(ctx context.Context) gin.H {
	if s.storageHealthOverride != nil {
		return s.storageHealthOverride(ctx)
	}

	startTime := time.Now()

	// For local storage, try a basic operation
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

// checkCacheHealth performs cache-specific health checks
func (s *AgentFieldServer) checkCacheHealth(ctx context.Context) gin.H {
	if s.cacheHealthOverride != nil {
		return s.cacheHealthOverride(ctx)
	}

	startTime := time.Now()

	// Try a simple cache operation
	testKey := "health_check_" + fmt.Sprintf("%d", time.Now().Unix())
	testValue := "ok"

	// Set a test value
	if err := s.cache.Set(testKey, testValue, time.Minute); err != nil {
		return gin.H{
			"status":        "unhealthy",
			"message":       fmt.Sprintf("cache set operation failed: %v", err),
			"response_time": time.Since(startTime).Milliseconds(),
		}
	}

	// Get the test value
	var retrieved string
	if err := s.cache.Get(testKey, &retrieved); err != nil {
		return gin.H{
			"status":        "unhealthy",
			"message":       fmt.Sprintf("cache get operation failed: %v", err),
			"response_time": time.Since(startTime).Milliseconds(),
		}
	}

	// Clean up
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

func (s *AgentFieldServer) setupRoutes() {
	// Configure CORS from configuration
	corsConfig := cors.Config{
		AllowOrigins:     s.config.API.CORS.AllowedOrigins,
		AllowMethods:     s.config.API.CORS.AllowedMethods,
		AllowHeaders:     s.config.API.CORS.AllowedHeaders,
		ExposeHeaders:    s.config.API.CORS.ExposedHeaders,
		AllowCredentials: s.config.API.CORS.AllowCredentials,
	}

	// Fallback to defaults if not configured
	if len(corsConfig.AllowOrigins) == 0 {
		corsConfig.AllowOrigins = []string{"http://localhost:3000", "http://localhost:5173"}
	}
	if len(corsConfig.AllowMethods) == 0 {
		corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	}
	if len(corsConfig.AllowHeaders) == 0 {
		corsConfig.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization", "X-API-Key"}
	}

	s.Router.Use(cors.New(corsConfig))

	// Add request logging middleware
	s.Router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC1123),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	}))

	// Add timeout middleware for all routes (1 hour for long-running executions)
	s.Router.Use(func(c *gin.Context) {
		// Set a timeout for the request
		ctx := c.Request.Context()
		timeoutCtx, cancel := context.WithTimeout(ctx, 3600*time.Second)
		defer cancel()

		c.Request = c.Request.WithContext(timeoutCtx)
		c.Next()
	})

	// API key authentication middleware (supports headers + api_key query param)
	s.Router.Use(middleware.APIKeyAuth(middleware.AuthConfig{
		APIKey:    s.config.API.Auth.APIKey,
		SkipPaths: s.config.API.Auth.SkipPaths,
	}))
	if s.config.API.Auth.APIKey != "" {
		logger.Logger.Info().Msg("🔐 API key authentication enabled")
	}

	// Expose Prometheus metrics
	s.Router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Serve UI files - embedded or filesystem based on availability
	if s.config.UI.Enabled {
		// Check if UI is embedded in the binary
		if s.config.UI.Mode == "embedded" && client.IsUIEmbedded() {
			// Use embedded UI
			client.RegisterUIRoutes(s.Router)
			fmt.Println("Using embedded UI files")
		} else {
			// Use filesystem UI
			distPath := s.config.UI.DistPath
			if distPath == "" {
				// Get the executable path and find UI dist relative to it
				execPath, err := os.Executable()
				if err != nil {
					distPath = filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
					if _, statErr := os.Stat(distPath); os.IsNotExist(statErr) {
						distPath = filepath.Join("web", "client", "dist")
					}
				} else {
					execDir := filepath.Dir(execPath)
					// Look for web/client/dist relative to the executable directory
					distPath = filepath.Join(execDir, "web", "client", "dist")

					// If that doesn't exist, try going up one level (if binary is in apps/platform/agentfield/)
					if _, err := os.Stat(distPath); os.IsNotExist(err) {
						distPath = filepath.Join(filepath.Dir(execDir), "apps", "platform", "agentfield", "web", "client", "dist")
					}

					// Final fallback to current working directory
					if _, err := os.Stat(distPath); os.IsNotExist(err) {
						altPath := filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
						if _, altErr := os.Stat(altPath); altErr == nil {
							distPath = altPath
						} else {
							distPath = filepath.Join("web", "client", "dist")
						}
					}
				}
			}

			// Serve static files from filesystem
			s.Router.StaticFS("/ui", http.Dir(distPath))

			// Root redirect
			s.Router.GET("/", func(c *gin.Context) {
				c.Redirect(http.StatusMovedPermanently, "/ui/")
			})

			fmt.Printf("Using filesystem UI files from: %s\n", distPath)
		}
	}

	// UI API routes - Moved before API routes to prevent route conflicts
	if s.config.UI.Enabled { // Only add UI API routes if UI is generally enabled
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
				// Nodes UI endpoints
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

				// DID and VC management endpoints for nodes
				didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService)
				nodes.GET("/:nodeId/did", didHandler.GetNodeDIDHandler)
				nodes.GET("/:nodeId/vc-status", didHandler.GetNodeVCStatusHandler)

				// MCP management endpoints for nodes
				mcpHandler := ui.NewMCPHandler(s.uiService, s.agentClient)
				nodes.GET("/:nodeId/mcp/health", mcpHandler.GetMCPHealthHandler)
				nodes.GET("/:nodeId/mcp/events", mcpHandler.GetMCPEventsHandler)
				nodes.GET("/:nodeId/mcp/metrics", mcpHandler.GetMCPMetricsHandler)
				nodes.POST("/:nodeId/mcp/servers/:alias/restart", mcpHandler.RestartMCPServerHandler)
				nodes.GET("/:nodeId/mcp/servers/:alias/tools", mcpHandler.GetMCPToolsHandler)
			}

			// Executions management group
			executions := uiAPI.Group("/executions")
			{
				// Executions UI endpoints
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

				// Execution notes endpoints for UI
				executions.POST("/note", handlers.AddExecutionNoteHandler(s.storage))
				executions.GET("/:execution_id/notes", handlers.GetExecutionNotesHandler(s.storage))

				// DID and VC management endpoints for executions
				didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService)
				executions.GET("/:execution_id/vc", didHandler.GetExecutionVCHandler)
				executions.GET("/:execution_id/vc-status", didHandler.GetExecutionVCStatusHandler)
				executions.POST("/:execution_id/verify-vc", didHandler.VerifyExecutionVCComprehensiveHandler)
			}

			// Workflows management group
			workflows := uiAPI.Group("/workflows")
			{
				workflows.GET("/:workflowId/dag", handlers.GetWorkflowDAGHandler(s.storage))
				didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService)
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

			// MCP system-wide endpoints
			mcp := uiAPI.Group("/mcp")
			{
				mcpHandler := ui.NewMCPHandler(s.uiService, s.agentClient)
				mcp.GET("/status", mcpHandler.GetMCPStatusHandler)
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
				didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService)
				did.GET("/status", didHandler.GetDIDSystemStatusHandler)
				did.GET("/export/vcs", didHandler.ExportVCsHandler)
				did.GET("/:did/resolution-bundle", didHandler.GetDIDResolutionBundleHandler)
				did.GET("/:did/resolution-bundle/download", didHandler.DownloadDIDResolutionBundleHandler)
			}

			// VC system-wide endpoints
			vc := uiAPI.Group("/vc")
			{
				didHandler := ui.NewDIDHandler(s.storage, s.didService, s.vcService)
				vc.GET("/:vcId/download", didHandler.DownloadVCHandler)
				vc.POST("/verify", didHandler.VerifyVCHandler)
			}

			// Identity & Trust endpoints (DID Explorer and Credentials)
			identityHandler := ui.NewIdentityHandlers(s.storage)
			identityHandler.RegisterRoutes(uiAPI)
		}

		uiAPIV2 := s.Router.Group("/api/ui/v2")
		{
			workflowRunsHandler := ui.NewWorkflowRunHandler(s.storage)
			uiAPIV2.GET("/workflow-runs", workflowRunsHandler.ListWorkflowRunsHandler)
			uiAPIV2.GET("/workflow-runs/:run_id", workflowRunsHandler.GetWorkflowRunDetailHandler)
		}
	}

	// Agent API routes
	agentAPI := s.Router.Group("/api/v1")
	{
		// Health check endpoint for container orchestration
		agentAPI.GET("/health", s.healthCheckHandler)

		// Discovery endpoints
		discovery := agentAPI.Group("/discovery")
		{
			discovery.GET("/capabilities", handlers.DiscoveryCapabilitiesHandler(s.storage))
		}

		// Node management endpoints
		agentAPI.POST("/nodes/register", handlers.RegisterNodeHandler(s.storage, s.uiService, s.didService, s.presenceManager))
		agentAPI.POST("/nodes", handlers.RegisterNodeHandler(s.storage, s.uiService, s.didService, s.presenceManager))
		agentAPI.POST("/nodes/register-serverless", handlers.RegisterServerlessAgentHandler(s.storage, s.uiService, s.didService, s.presenceManager))
		agentAPI.GET("/nodes", handlers.ListNodesHandler(s.storage))
		agentAPI.GET("/nodes/:node_id", handlers.GetNodeHandler(s.storage))
		agentAPI.POST("/nodes/:node_id/heartbeat", handlers.HeartbeatHandler(s.storage, s.uiService, s.healthMonitor, s.statusManager, s.presenceManager))
		agentAPI.DELETE("/nodes/:node_id/monitoring", s.unregisterAgentFromMonitoring)

		// New unified status API endpoints
		agentAPI.GET("/nodes/:node_id/status", handlers.GetNodeStatusHandler(s.statusManager))
		agentAPI.POST("/nodes/:node_id/status/refresh", handlers.RefreshNodeStatusHandler(s.statusManager))
		agentAPI.POST("/nodes/status/bulk", handlers.BulkNodeStatusHandler(s.statusManager, s.storage))
		agentAPI.POST("/nodes/status/refresh", handlers.RefreshAllNodeStatusHandler(s.statusManager, s.storage))

		// Enhanced lifecycle management endpoints
		agentAPI.POST("/nodes/:node_id/start", handlers.StartNodeHandler(s.statusManager, s.storage))
		agentAPI.POST("/nodes/:node_id/stop", handlers.StopNodeHandler(s.statusManager, s.storage))
		agentAPI.POST("/nodes/:node_id/lifecycle/status", handlers.UpdateLifecycleStatusHandler(s.storage, s.uiService, s.statusManager))
		agentAPI.PATCH("/nodes/:node_id/status", handlers.NodeStatusLeaseHandler(s.storage, s.statusManager, s.presenceManager, handlers.DefaultLeaseTTL))
		agentAPI.POST("/nodes/:node_id/actions/ack", handlers.NodeActionAckHandler(s.storage, s.presenceManager, handlers.DefaultLeaseTTL))
		agentAPI.POST("/nodes/:node_id/shutdown", handlers.NodeShutdownHandler(s.storage, s.statusManager, s.presenceManager))
		agentAPI.POST("/actions/claim", handlers.ClaimActionsHandler(s.storage, s.presenceManager, handlers.DefaultLeaseTTL))

		// TODO: Add other node routes (DeleteNode)

		// Reasoner execution endpoints (legacy)
		agentAPI.POST("/reasoners/:reasoner_id", handlers.ExecuteReasonerHandler(s.storage))

		// Skill execution endpoints (legacy)
		agentAPI.POST("/skills/:skill_id", handlers.ExecuteSkillHandler(s.storage))

		// Unified execution endpoints (path-based)
		agentAPI.POST("/execute/:target", handlers.ExecuteHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout))
		agentAPI.POST("/execute/async/:target", handlers.ExecuteAsyncHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout))
		agentAPI.GET("/executions/:execution_id", handlers.GetExecutionStatusHandler(s.storage))
		agentAPI.POST("/executions/batch-status", handlers.BatchExecutionStatusHandler(s.storage))
		agentAPI.POST("/executions/:execution_id/status", handlers.UpdateExecutionStatusHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout))

		// Execution notes endpoints for app.note() feature
		agentAPI.POST("/executions/note", handlers.AddExecutionNoteHandler(s.storage))
		agentAPI.GET("/executions/:execution_id/notes", handlers.GetExecutionNotesHandler(s.storage))
		agentAPI.POST("/workflow/executions/events", handlers.WorkflowExecutionEventHandler(s.storage))

		// Workflow endpoints will be reintroduced once the simplified execution pipeline lands.

		// Memory endpoints
		agentAPI.POST("/memory/set", handlers.SetMemoryHandler(s.storage))
		agentAPI.POST("/memory/get", handlers.GetMemoryHandler(s.storage))
		agentAPI.POST("/memory/delete", handlers.DeleteMemoryHandler(s.storage))
		agentAPI.GET("/memory/list", handlers.ListMemoryHandler(s.storage))
		agentAPI.POST("/memory/vector/set", handlers.SetVectorHandler(s.storage))
		agentAPI.POST("/memory/vector/search", handlers.SimilaritySearchHandler(s.storage))
		agentAPI.POST("/memory/vector/delete", handlers.DeleteVectorHandler(s.storage))
		agentAPI.DELETE("/memory/vector/namespace", handlers.DeleteNamespaceVectorsHandler(s.storage))

		// Memory events endpoints
		memoryEventsHandler := handlers.NewMemoryEventsHandler(s.storage)
		agentAPI.GET("/memory/events/ws", memoryEventsHandler.WebSocketHandler)
		agentAPI.GET("/memory/events/sse", memoryEventsHandler.SSEHandler)
		agentAPI.GET("/memory/events/history", handlers.GetEventHistoryHandler(s.storage))

		// DID/VC endpoints - use service-backed handlers if DID is enabled
		logger.Logger.Debug().
			Bool("did_enabled", s.config.Features.DID.Enabled).
			Bool("did_service_available", s.didService != nil).
			Bool("vc_service_available", s.vcService != nil).
			Msg("DID Route Registration Check")

		if s.config.Features.DID.Enabled && s.didService != nil && s.vcService != nil {
			logger.Logger.Debug().Msg("Registering DID routes - all conditions met")
			// Create DID handlers instance with services
			didHandlers := handlers.NewDIDHandlers(s.didService, s.vcService)

			// Register service-backed DID routes
			didHandlers.RegisterRoutes(agentAPI)

			// Add af server DID endpoint
			agentAPI.GET("/did/agentfield-server", func(c *gin.Context) {
				// Get af server ID dynamically
				agentfieldServerID, err := s.didService.GetAgentFieldServerID()
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "Failed to get af server ID",
						"details": fmt.Sprintf("AgentField server ID error: %v", err),
					})
					return
				}

				// Get the actual af server DID from the registry
				registry, err := s.didService.GetRegistry(agentfieldServerID)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "Failed to get af server DID",
						"details": fmt.Sprintf("Registry error: %v", err),
					})
					return
				}

				if registry == nil {
					c.JSON(http.StatusNotFound, gin.H{
						"error":   "AgentField server DID not found",
						"details": "No DID registry exists for af server 'default'. The DID system may not be properly initialized.",
					})
					return
				}

				if registry.RootDID == "" {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "AgentField server DID is empty",
						"details": "Registry exists but root DID is empty. The DID system may be corrupted.",
					})
					return
				}

				c.JSON(http.StatusOK, gin.H{
					"agentfield_server_id":  "default",
					"agentfield_server_did": registry.RootDID,
					"message":               "AgentField server DID retrieved successfully",
				})
			})
		} else {
			logger.Logger.Warn().
				Bool("did_enabled", s.config.Features.DID.Enabled).
				Bool("did_service_available", s.didService != nil).
				Bool("vc_service_available", s.vcService != nil).
				Msg("DID routes NOT registered - conditions not met")
		}
		// Note: Removed unused/unimplemented DID endpoint placeholders for system simplification
	}

	// SPA fallback - serve index.html for all /ui/* routes that don't match static files
	// Only add this if we're NOT using embedded UI (since embedded UI handles its own NoRoute)
	if s.config.UI.Enabled && (s.config.UI.Mode != "embedded" || !client.IsUIEmbedded()) {
		s.Router.NoRoute(func(c *gin.Context) {
			// Only handle /ui/* paths
			if strings.HasPrefix(c.Request.URL.Path, "/ui/") {
				// Check if it's a static asset by looking for common web asset file extensions
				// This prevents reasoner IDs with dots (like "deepresearchagent.meta_research_methodology_reasoner")
				// from being treated as static assets
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
					// Let it 404 for missing static assets
					c.JSON(http.StatusNotFound, gin.H{"error": "file not found"})
					return
				}

				// For SPA routes (including reasoner detail pages), serve index.html from filesystem
				distPath := s.config.UI.DistPath
				if distPath == "" {
					// Get the executable path and find UI dist relative to it
					execPath, err := os.Executable()
					if err != nil {
						distPath = filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
						if _, statErr := os.Stat(distPath); os.IsNotExist(statErr) {
							distPath = filepath.Join("web", "client", "dist")
						}
					} else {
						execDir := filepath.Dir(execPath)
						// Look for web/client/dist relative to the executable directory
						distPath = filepath.Join(execDir, "web", "client", "dist")

						// If that doesn't exist, try going up one level (if binary is in apps/platform/agentfield/)
						if _, err := os.Stat(distPath); os.IsNotExist(err) {
							distPath = filepath.Join(filepath.Dir(execDir), "apps", "platform", "agentfield", "web", "client", "dist")
						}

						// Final fallback to current working directory
						if _, err := os.Stat(distPath); os.IsNotExist(err) {
							altPath := filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
							if _, altErr := os.Stat(altPath); altErr == nil {
								distPath = altPath
							} else {
								distPath = filepath.Join("web", "client", "dist")
							}
						}
					}
				}
				c.File(filepath.Join(distPath, "index.html"))
			} else {
				// For non-UI paths, return 404
				c.JSON(http.StatusNotFound, gin.H{"error": "endpoint not found"})
			}
		})
	}
}

// generateAgentFieldServerID creates a deterministic af server ID based on the agentfield home directory.
// This ensures each agentfield instance has a unique ID while being deterministic for the same installation.
func generateAgentFieldServerID(agentfieldHome string) string {
	// Use the absolute path of agentfield home to generate a deterministic ID
	absPath, err := filepath.Abs(agentfieldHome)
	if err != nil {
		// Fallback to the original path if absolute path fails
		absPath = agentfieldHome
	}

	// Create a hash of the agentfield home path to generate a unique but deterministic ID
	hash := sha256.Sum256([]byte(absPath))

	// Use first 16 characters of the hex hash as the af server ID
	// This provides uniqueness while keeping the ID manageable
	agentfieldServerID := hex.EncodeToString(hash[:])[:16]

	return agentfieldServerID
}
