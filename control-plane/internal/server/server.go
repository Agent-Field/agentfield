package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services" // Core services
	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"                          // Event system
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"                        // Agent handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/admin"                  // Admin handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/agentic"                // Agentic API handlers
	connectorpkg "github.com/Agent-Field/agentfield/control-plane/internal/handlers/connector" // Connector handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/ui"                     // UI handlers
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/communication"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/process"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/observability"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"    // API catalog
	"github.com/Agent-Field/agentfield/control-plane/internal/server/knowledgebase" // Knowledge base
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services" // Services
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
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
	agentClient           interfaces.AgentClient  // Add AgentClient for agent communication
	config                *config.Config
	storageHealthOverride func(context.Context) gin.H
	cacheHealthOverride   func(context.Context) gin.H
	// DID Services
	keystoreService     *services.KeystoreService
	didService          *services.DIDService
	vcService           *services.VCService
	didRegistry         *services.DIDRegistry
	didWebService       *services.DIDWebService
	accessPolicyService *services.AccessPolicyService
	tagApprovalService  *services.TagApprovalService
	tagVCVerifier       *services.TagVCVerifier
	agentfieldHome      string
	// LLM health monitoring
	llmHealthMonitor *services.LLMHealthMonitor
	// Cleanup service
	cleanupService         *handlers.ExecutionCleanupService
	payloadStore           services.PayloadStore
	registryWatcherCancel  context.CancelFunc
	adminGRPCServer        *grpc.Server
	adminListener          net.Listener
	adminGRPCPort          int
	webhookDispatcher      services.WebhookDispatcher
	observabilityForwarder services.ObservabilityForwarder
	executionTracer        *observability.ExecutionTracer
	tracerShutdown         func(context.Context) error
	configMu               sync.RWMutex
	// Agentic API
	apiCatalog *apicatalog.Catalog
	kb         *knowledgebase.KB
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

	// Overlay database-stored config if AGENTFIELD_CONFIG_SOURCE=db
	if src := os.Getenv("AGENTFIELD_CONFIG_SOURCE"); src == "db" {
		if err := overlayDBConfig(cfg, storageProvider); err != nil {
			fmt.Printf("Warning: failed to load config from database: %v\n", err)
		}
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
		ReconcileInterval:       30 * time.Second,
		StatusCacheTTL:          5 * time.Minute,
		MaxTransitionTime:       2 * time.Minute,
		HeartbeatStaleThreshold: cfg.AgentField.NodeHealth.HeartbeatStaleThreshold,
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

	// Initialize health monitor with configurable settings
	healthMonitorConfig := services.HealthMonitorConfig{
		CheckInterval:       cfg.AgentField.NodeHealth.CheckInterval,
		CheckTimeout:        cfg.AgentField.NodeHealth.CheckTimeout,
		ConsecutiveFailures: cfg.AgentField.NodeHealth.ConsecutiveFailures,
		RecoveryDebounce:    cfg.AgentField.NodeHealth.RecoveryDebounce,
	}
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
		if passphrase := cfg.Features.DID.Keystore.EncryptionPassphrase; passphrase != "" {
			didRegistry.SetEncryptionService(encryption.NewEncryptionService(passphrase))
			fmt.Println("🔐 Master seed encryption enabled")
		}

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

	// Initialize DIDWebService if DID is enabled
	var didWebService *services.DIDWebService

	if cfg.Features.DID.Enabled && didService != nil {
		// Determine domain for did:web identifiers
		domain := cfg.Features.DID.Authorization.Domain
		if domain == "" {
			domain = fmt.Sprintf("localhost:%d", cfg.AgentField.Port)
		}

		// Create DIDWebService
		fmt.Printf("🌐 Creating DID Web service with domain: %s\n", domain)
		didWebService = services.NewDIDWebService(domain, didService, storageProvider)

		if cfg.Features.DID.Authorization.Enabled {
			if cfg.Features.DID.Authorization.AdminToken == "" {
				logger.Logger.Error().Msg("⚠️  SECURITY WARNING: Authorization is enabled but no admin_token is configured! Admin routes (tag approval, policy management) are unprotected. Set AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN for production use.")
			}
			if cfg.Features.DID.Authorization.TagApprovalRules.DefaultMode == "" || cfg.Features.DID.Authorization.TagApprovalRules.DefaultMode == "auto" {
				logger.Logger.Warn().Msg("⚠️  Tag approval default_mode is 'auto' — all agent tags will be auto-approved. Set tag_approval_rules.default_mode to 'manual' for production.")
			}
		}
	}

	// Initialize tag approval service (uses config-based rules)
	var tagApprovalService *services.TagApprovalService
	if cfg.Features.DID.Authorization.Enabled {
		tagApprovalService = services.NewTagApprovalService(
			cfg.Features.DID.Authorization.TagApprovalRules,
			storageProvider,
		)
		if tagApprovalService.IsEnabled() {
			logger.Logger.Info().Msg("🏷️  Tag approval service enabled with rules")
		}
	}

	// Initialize access policy service (tag-based authorization)
	var accessPolicyService *services.AccessPolicyService
	if cfg.Features.DID.Authorization.Enabled {
		accessPolicyService = services.NewAccessPolicyService(storageProvider)
		if err := accessPolicyService.Initialize(context.Background()); err != nil {
			logger.Logger.Warn().Err(err).Msg("Failed to initialize access policy service")
		} else {
			logger.Logger.Info().Msg("📋 Access policy service initialized")
		}

		// Seed access policies from config file
		if len(cfg.Features.DID.Authorization.AccessPolicies) > 0 {
			ctx := context.Background()
			seededCount := 0
			for _, policyCfg := range cfg.Features.DID.Authorization.AccessPolicies {
				desc := ""
				if policyCfg.Name != "" {
					desc = "Seeded from config"
				}
				constraints := make(map[string]types.AccessConstraint)
				for k, v := range policyCfg.Constraints {
					constraints[k] = types.AccessConstraint{
						Operator: v.Operator,
						Value:    v.Value,
					}
				}
				_, err := accessPolicyService.AddPolicy(ctx, &types.AccessPolicyRequest{
					Name:           policyCfg.Name,
					CallerTags:     policyCfg.CallerTags,
					TargetTags:     policyCfg.TargetTags,
					AllowFunctions: policyCfg.AllowFunctions,
					DenyFunctions:  policyCfg.DenyFunctions,
					Constraints:    constraints,
					Action:         policyCfg.Action,
					Priority:       policyCfg.Priority,
					Description:    desc,
				})
				if err != nil {
					logger.Logger.Debug().
						Err(err).
						Str("policy_name", policyCfg.Name).
						Msg("Failed to seed access policy from config (may already exist)")
				} else {
					seededCount++
				}
			}
			if seededCount > 0 {
				logger.Logger.Info().
					Int("seeded_count", seededCount).
					Int("total_config_policies", len(cfg.Features.DID.Authorization.AccessPolicies)).
					Msg("Seeded access policies from config")
			}
		}
	}

	// Initialize tag VC verifier for cryptographic tag verification at call time
	var tagVCVerifier *services.TagVCVerifier
	if cfg.Features.DID.Authorization.Enabled && vcService != nil {
		tagVCVerifier = services.NewTagVCVerifier(storageProvider, vcService)
		logger.Logger.Info().Msg("🔐 Tag VC verifier initialized")
	}

	// Wire VC service into tag approval service for VC issuance on approval
	if tagApprovalService != nil && vcService != nil {
		tagApprovalService.SetVCService(vcService)
		logger.Logger.Info().Msg("🏷️  Tag approval service configured for VC issuance")
	}

	// Wire revocation callback to clear status cache and presence lease
	if tagApprovalService != nil {
		tagApprovalService.SetOnRevokeCallback(func(ctx context.Context, agentID string) {
			presenceManager.Forget(agentID)
			_ = statusManager.RefreshAgentStatus(ctx, agentID)
		})
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

	// Initialize observability forwarder for external webhook integration
	observabilityForwarder := services.NewObservabilityForwarder(storageProvider, services.ObservabilityForwarderConfig{
		BatchSize:       10,
		BatchTimeout:    time.Second,
		HTTPTimeout:     10 * time.Second,
		MaxAttempts:     3,
		RetryBackoff:    time.Second,
		MaxRetryBackoff: 30 * time.Second,
		WorkerCount:     2,
		QueueSize:       1000,
	})
	if err := observabilityForwarder.Start(context.Background()); err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to start observability forwarder")
	}

	// Initialize OpenTelemetry distributed tracing
	var executionTracer *observability.ExecutionTracer
	var tracerShutdown func(context.Context) error
	if cfg.Features.Tracing.Enabled {
		tracer, shutdown, err := observability.InitTracer(context.Background(), observability.TracerConfig{
			Enabled:     cfg.Features.Tracing.Enabled,
			Exporter:    cfg.Features.Tracing.Exporter,
			Endpoint:    cfg.Features.Tracing.Endpoint,
			ServiceName: cfg.Features.Tracing.ServiceName,
			Insecure:    cfg.Features.Tracing.Insecure,
		})
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("failed to initialize OTel tracer")
		} else if tracer != nil {
			executionTracer = observability.NewExecutionTracer(tracer)
			tracerShutdown = shutdown
			logger.Logger.Info().
				Str("endpoint", cfg.Features.Tracing.Endpoint).
				Str("service_name", cfg.Features.Tracing.ServiceName).
				Msg("OpenTelemetry tracing enabled")
		}
	}

	// Initialize LLM health monitor
	var llmHealthMonitor *services.LLMHealthMonitor
	if cfg.AgentField.LLMHealth.Enabled && len(cfg.AgentField.LLMHealth.Endpoints) > 0 {
		llmHealthMonitor = services.NewLLMHealthMonitor(cfg.AgentField.LLMHealth, uiService)
		handlers.SetLLMHealthMonitor(llmHealthMonitor)
		logger.Logger.Info().
			Int("endpoints", len(cfg.AgentField.LLMHealth.Endpoints)).
			Msg("LLM health monitor configured")
	}

	// Initialize per-agent concurrency limiter
	handlers.InitConcurrencyLimiter(cfg.AgentField.ExecutionQueue.MaxConcurrentPerAgent)

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
		storage:                storageProvider,
		cache:                  cacheProvider,
		Router:                 Router,
		uiService:              uiService,
		executionsUIService:    executionsUIService,
		healthMonitor:          healthMonitor,
		presenceManager:        presenceManager,
		statusManager:          statusManager,
		agentService:           agentService,
		agentClient:            agentClient,
		config:                 cfg,
		keystoreService:        keystoreService,
		didService:             didService,
		vcService:              vcService,
		didRegistry:            didRegistry,
		didWebService:          didWebService,
		accessPolicyService:    accessPolicyService,
		tagApprovalService:     tagApprovalService,
		tagVCVerifier:          tagVCVerifier,
		agentfieldHome:         agentfieldHome,
		llmHealthMonitor:       llmHealthMonitor,
		cleanupService:         cleanupService,
		payloadStore:           payloadStore,
		webhookDispatcher:      webhookDispatcher,
		observabilityForwarder: observabilityForwarder,
		executionTracer:        executionTracer,
		tracerShutdown:         tracerShutdown,
		registryWatcherCancel:  nil,
		adminGRPCPort:          adminPort,
		apiCatalog:             initAPICatalog(),
		kb:                     initKnowledgeBase(),
	}, nil
}

// configReloadFn returns a function that reloads config from the database,
// or nil if AGENTFIELD_CONFIG_SOURCE is not set to "db".
// The returned function acquires configMu to prevent data races with
// concurrent readers of s.config.
func (s *AgentFieldServer) configReloadFn() handlers.ConfigReloadFunc {
	if src := os.Getenv("AGENTFIELD_CONFIG_SOURCE"); src != "db" {
		return nil
	}
	return func() error {
		s.configMu.Lock()
		defer s.configMu.Unlock()
		return overlayDBConfig(s.config, s.storage)
	}
}

// initAPICatalog creates and populates the API endpoint catalog.
func initAPICatalog() *apicatalog.Catalog {
	catalog := apicatalog.New()
	catalog.RegisterBatch(apicatalog.DefaultEntries())
	return catalog
}

// initKnowledgeBase creates and populates the built-in knowledge base.
func initKnowledgeBase() *knowledgebase.KB {
	kb := knowledgebase.New()
	knowledgebase.LoadDefaultContent(kb)
	return kb
}

// Start initializes and starts the AgentFieldServer.
func (s *AgentFieldServer) Start() error {
	// Setup routes
	s.setupRoutes()

	// Start status manager service in background
	go s.statusManager.Start()

	if s.presenceManager != nil {
		// Recover presence leases BEFORE starting the sweep loop so the first
		// sweep sees all previously-registered agents instead of an empty map.
		ctx := context.Background()
		if err := s.presenceManager.RecoverFromDatabase(ctx, s.storage); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to recover presence leases from database")
		}

		go s.presenceManager.Start()
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

	// Start LLM health monitor in background
	if s.llmHealthMonitor != nil {
		go s.llmHealthMonitor.Start()
	}

	// Start execution cleanup service in background
	ctx := context.Background()
	if err := s.cleanupService.Start(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to start execution cleanup service")
		// Don't fail server startup if cleanup service fails to start
	}

	// Start OpenTelemetry execution tracer in background
	if s.executionTracer != nil {
		s.executionTracer.Start(ctx)
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

	// Stop observability forwarder
	if s.observabilityForwarder != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.observabilityForwarder.Stop(ctx); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to stop observability forwarder")
		}
	}

	// Stop OpenTelemetry execution tracer and flush remaining spans
	if s.executionTracer != nil {
		s.executionTracer.Stop()
	}
	if s.tracerShutdown != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.tracerShutdown(ctx); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to shutdown OTel tracer provider")
		}
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

// handleDIDWebServerDocument serves the server's root DID document per W3C did:web spec.
// GET /.well-known/did.json -> resolves did:web:{domain}
func (s *AgentFieldServer) handleDIDWebServerDocument(c *gin.Context) {
	serverID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "server DID not available"})
		return
	}
	registry, err := s.didService.GetRegistry(serverID)
	if err != nil || registry == nil || registry.RootDID == "" {
		c.JSON(http.StatusNotFound, gin.H{"error": "server DID not found"})
		return
	}
	s.serveDIDDocument(c, registry.RootDID)
}

// handleDIDWebAgentDocument serves an agent's DID document per W3C did:web spec.
// GET /agents/:agentID/did.json -> resolves did:web:{domain}:agents:{agentID}
func (s *AgentFieldServer) handleDIDWebAgentDocument(c *gin.Context) {
	agentID := c.Param("agentID")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "agent ID is required"})
		return
	}
	did := s.didWebService.GenerateDIDWeb(agentID)
	s.serveDIDDocument(c, did)
}

// serveDIDDocument resolves a DID and returns a W3C-compliant DID document.
// It tries did:web resolution (database) first, then falls back to did:key (in-memory).
func (s *AgentFieldServer) serveDIDDocument(c *gin.Context, did string) {
	// Try did:web resolution via DIDWebService (stored in database)
	if s.didWebService != nil && strings.HasPrefix(did, "did:web:") {
		result, err := s.didWebService.ResolveDID(c.Request.Context(), did)
		if err == nil && result.DIDDocument != nil {
			c.JSON(http.StatusOK, result.DIDDocument)
			return
		}
		if err == nil && result.DIDResolutionMetadata.Error == "deactivated" {
			c.JSON(http.StatusGone, gin.H{"error": "DID has been revoked"})
			return
		}
	}

	// Fall back to did:key resolution via DIDService (in-memory registry)
	identity, err := s.didService.ResolveDID(did)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "DID not found"})
		return
	}

	var publicKeyJWK map[string]interface{}
	if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to parse public key"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"@context": []string{
			"https://www.w3.org/ns/did/v1",
			"https://w3id.org/security/suites/ed25519-2020/v1",
		},
		"id": did,
		"verificationMethod": []gin.H{{
			"id":           did + "#key-1",
			"type":         "Ed25519VerificationKey2020",
			"controller":   did,
			"publicKeyJwk": publicKeyJWK,
		}},
		"authentication":  []string{did + "#key-1"},
		"assertionMethod": []string{did + "#key-1"},
	})
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

	// DID authentication middleware (applied globally, but only validates when headers present)
	if s.config.Features.DID.Enabled && s.config.Features.DID.Authorization.DIDAuthEnabled && s.didWebService != nil {
		didAuthConfig := middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: s.config.Features.DID.Authorization.TimestampWindowSeconds,
			SkipPaths: []string{
				"/health",
				"/metrics",
				"/api/v1/health",
			},
		}
		s.Router.Use(middleware.DIDAuthMiddleware(s.didWebService, didAuthConfig))
		logger.Logger.Info().Msg("🆔 DID authentication middleware enabled")
	}

	// Expose Prometheus metrics
	s.Router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Public health check endpoint for load balancers and container orchestration (e.g., Railway, K8s)
	s.Router.GET("/health", s.healthCheckHandler)

	// W3C did:web resolution endpoints (spec: https://w3c-ccg.github.io/did-method-web/)
	// did:web:{domain} resolves to GET /.well-known/did.json
	// did:web:{domain}:agents:{agentID} resolves to GET /agents/{agentID}/did.json
	if s.config.Features.DID.Enabled && s.didWebService != nil {
		s.Router.GET("/.well-known/did.json", s.handleDIDWebServerDocument)
		s.Router.GET("/agents/:agentID/did.json", s.handleDIDWebAgentDocument)
	}

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

		// Reasoner and skill execution endpoints (legacy)
		// When authorization is enabled, these require the same permission middleware
		// as the unified execute endpoints to prevent policy bypass.
		if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
			legacyReasonerGroup := agentAPI.Group("/reasoners")
			legacySkillGroup := agentAPI.Group("/skills")
			permConfigLegacy := middleware.PermissionConfig{
				Enabled: true,
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
		// These routes may have permission middleware applied if authorization is enabled
		executeGroup := agentAPI.Group("/execute")
		{
			// Apply permission middleware if authorization is enabled
			if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
				permConfig := middleware.PermissionConfig{
					Enabled: true,
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

			executeGroup.POST("/:target", handlers.ExecuteHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken))
			executeGroup.POST("/async/:target", handlers.ExecuteAsyncHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout, s.config.Features.DID.Authorization.InternalToken))
		}
		agentAPI.GET("/executions/:execution_id", handlers.GetExecutionStatusHandler(s.storage))
		agentAPI.POST("/executions/batch-status", handlers.BatchExecutionStatusHandler(s.storage))
		agentAPI.POST("/executions/:execution_id/status", handlers.UpdateExecutionStatusHandler(s.storage, s.payloadStore, s.webhookDispatcher, s.config.AgentField.ExecutionQueue.AgentCallTimeout))
		agentAPI.POST("/executions/:execution_id/logs", handlers.StructuredExecutionLogsHandler(s.storage, func() config.ExecutionLogsConfig {
			return s.config.AgentField.ExecutionLogs
		}))
		agentAPI.POST("/executions/:execution_id/cancel", handlers.CancelExecutionHandler(s.storage))
		agentAPI.POST("/executions/:execution_id/pause", handlers.PauseExecutionHandler(s.storage))
		agentAPI.POST("/executions/:execution_id/resume", handlers.ResumeExecutionHandler(s.storage))

		// Approval workflow endpoints — CP manages execution state only;
		// agents handle external approval service communication directly.
		agentAPI.POST("/executions/:execution_id/request-approval", handlers.RequestApprovalHandler(s.storage))
		agentAPI.GET("/executions/:execution_id/approval-status", handlers.GetApprovalStatusHandler(s.storage))

		// Agent-scoped approval routes — enforce that the execution belongs to the requesting agent.
		// Agents should use these instead of the global routes above.
		agentAPI.POST("/agents/:node_id/executions/:execution_id/request-approval", handlers.AgentScopedRequestApprovalHandler(s.storage))
		agentAPI.GET("/agents/:node_id/executions/:execution_id/approval-status", handlers.AgentScopedGetApprovalStatusHandler(s.storage))

		// Approval resolution webhook (called by agents or external services when approval resolves)
		agentAPI.POST("/webhooks/approval-response", handlers.ApprovalWebhookHandler(s.storage, s.config.AgentField.Approval.WebhookSecret))

		// Execution notes endpoints for app.note() feature
		agentAPI.POST("/executions/note", handlers.AddExecutionNoteHandler(s.storage))
		agentAPI.GET("/executions/:execution_id/notes", handlers.GetExecutionNotesHandler(s.storage))
		agentAPI.POST("/workflow/executions/events", handlers.WorkflowExecutionEventHandler(s.storage))

		// Workflow endpoints will be reintroduced once the simplified execution pipeline lands.

		// Memory endpoints - apply permission middleware when authorization is enabled
		memoryGroup := agentAPI.Group("/memory")
		{
			// Apply memory permission middleware if authorization is enabled
			if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
				memPermConfig := middleware.MemoryPermissionConfig{
					Enabled:               true,
					EnforceScopeOwnership: true,
				}
				memoryGroup.Use(middleware.MemoryPermissionMiddleware(
					s.accessPolicyService,
					s.storage,
					s.didWebService,
					memPermConfig,
				))
				logger.Logger.Info().Msg("Memory permission middleware enabled on memory endpoints")
			}

			// Key-value memory endpoints
			memoryGroup.POST("/set", handlers.SetMemoryHandler(s.storage))
			memoryGroup.POST("/get", handlers.GetMemoryHandler(s.storage))
			memoryGroup.POST("/delete", handlers.DeleteMemoryHandler(s.storage))
			memoryGroup.GET("/list", handlers.ListMemoryHandler(s.storage))

			// Vector Memory endpoints (RESTful)
			memoryGroup.POST("/vector", handlers.SetVectorHandler(s.storage))
			memoryGroup.GET("/vector/:key", handlers.GetVectorHandler(s.storage))
			memoryGroup.POST("/vector/search", handlers.SimilaritySearchHandler(s.storage))
			memoryGroup.DELETE("/vector/:key", handlers.DeleteVectorHandler(s.storage))

			// Legacy Vector Memory endpoints (for backward compatibility)
			memoryGroup.POST("/vector/set", handlers.SetVectorHandler(s.storage))
			memoryGroup.POST("/vector/delete", handlers.DeleteVectorHandler(s.storage))
			memoryGroup.DELETE("/vector/namespace", handlers.DeleteNamespaceVectorsHandler(s.storage))

			// Memory events endpoints
			memoryEventsHandler := handlers.NewMemoryEventsHandler(s.storage)
			memoryGroup.GET("/events/ws", memoryEventsHandler.WebSocketHandler)
			memoryGroup.GET("/events/sse", memoryEventsHandler.SSEHandler)
			memoryGroup.GET("/events/history", handlers.GetEventHistoryHandler(s.storage))
		}

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
			if s.didWebService != nil {
				didHandlers.SetDIDWebService(s.didWebService)
			}

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

		// Agent Tag VC endpoint (for agents to download their own verified tag credential)
		if s.tagVCVerifier != nil {
			agentAPI.GET("/agents/:node_id/tag-vc", func(c *gin.Context) {
				agentID := c.Param("node_id")
				if agentID == "" {
					c.JSON(http.StatusBadRequest, gin.H{"error": "agent_id is required"})
					return
				}
				record, err := s.storage.GetAgentTagVC(c.Request.Context(), agentID)
				if err != nil {
					c.JSON(http.StatusNotFound, gin.H{
						"error":   "tag_vc_not_found",
						"message": fmt.Sprintf("No tag VC found for agent %s", agentID),
					})
					return
				}
				if record.RevokedAt != nil {
					c.JSON(http.StatusGone, gin.H{
						"error":   "tag_vc_revoked",
						"message": "Agent tag VC has been revoked",
					})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"agent_id":    record.AgentID,
					"agent_did":   record.AgentDID,
					"vc_id":       record.VCID,
					"vc_document": json.RawMessage(record.VCDocument),
					"issued_at":   record.IssuedAt,
					"expires_at":  record.ExpiresAt,
				})
			})
			logger.Logger.Info().Msg("🔐 Agent tag VC endpoint registered")
		}

		// Decentralized verification endpoints (for SDK local verification)
		// Policy distribution endpoint — agents cache these for local policy evaluation
		if s.accessPolicyService != nil {
			agentAPI.GET("/policies", func(c *gin.Context) {
				policies, err := s.accessPolicyService.ListPolicies(c.Request.Context())
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "failed_to_list_policies",
						"message": "Failed to list policies",
					})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"policies":   policies,
					"total":      len(policies),
					"fetched_at": time.Now().UTC().Format(time.RFC3339),
				})
			})
			logger.Logger.Info().Msg("📋 Policy distribution endpoint registered (GET /api/v1/policies)")
		}

		// Revocation list endpoint — agents cache revoked DIDs for local verification
		if s.didWebService != nil {
			agentAPI.GET("/revocations", func(c *gin.Context) {
				docs, err := s.storage.ListDIDDocuments(c.Request.Context())
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "failed_to_list_revocations",
						"message": "Failed to list revocations",
					})
					return
				}
				revokedDIDs := make([]string, 0)
				for _, doc := range docs {
					if doc.IsRevoked() {
						revokedDIDs = append(revokedDIDs, doc.DID)
					}
				}
				c.JSON(http.StatusOK, gin.H{
					"revoked_dids": revokedDIDs,
					"total":        len(revokedDIDs),
					"fetched_at":   time.Now().UTC().Format(time.RFC3339),
				})
			})
			logger.Logger.Info().Msg("🚫 Revocation list endpoint registered (GET /api/v1/revocations)")
		}

		// Registered DIDs endpoint — agents cache this set for local verification
		// to ensure only known/registered DIDs are accepted on direct calls.
		agentAPI.GET("/registered-dids", func(c *gin.Context) {
			agentDIDs, err := s.storage.ListAgentDIDs(c.Request.Context())
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "failed_to_list_registered_dids",
					"message": "Failed to list registered DIDs",
				})
				return
			}
			registeredDIDs := make([]string, 0, len(agentDIDs))
			for _, info := range agentDIDs {
				if info.Status == types.AgentDIDStatusActive {
					registeredDIDs = append(registeredDIDs, info.DID)
				}
			}
			c.JSON(http.StatusOK, gin.H{
				"registered_dids": registeredDIDs,
				"total":           len(registeredDIDs),
				"fetched_at":      time.Now().UTC().Format(time.RFC3339),
			})
		})
		logger.Logger.Info().Msg("✅ Registered DIDs endpoint registered (GET /api/v1/registered-dids)")

		// Issuer public key endpoint — agents use this for offline VC signature verification.
		// Registered at /did/issuer-public-key (public, semantic path) and
		// /admin/public-key (legacy alias for backward compatibility).
		if s.didService != nil {
			publicKeyHandler := func(c *gin.Context) {
				issuerDID, err := s.didService.GetControlPlaneIssuerDID()
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "issuer_did_unavailable",
						"message": "Issuer DID unavailable",
					})
					return
				}
				identity, err := s.didService.ResolveDID(issuerDID)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "public_key_unavailable",
						"message": "Public key unavailable",
					})
					return
				}
				var publicKeyJWK map[string]interface{}
				if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{
						"error":   "public_key_parse_error",
						"message": "Failed to parse public key JWK",
					})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"issuer_did":     issuerDID,
					"public_key_jwk": publicKeyJWK,
					"fetched_at":     time.Now().UTC().Format(time.RFC3339),
				})
			}
			agentAPI.GET("/did/issuer-public-key", publicKeyHandler)
			agentAPI.GET("/admin/public-key", publicKeyHandler) // legacy alias
			logger.Logger.Info().Msg("🔑 Issuer public key endpoint registered (GET /api/v1/did/issuer-public-key)")
		}

		// Settings API routes (observability webhook configuration)
		settings := agentAPI.Group("/settings")
		{
			obsHandler := ui.NewObservabilityWebhookHandler(s.storage, s.observabilityForwarder)
			settings.GET("/observability-webhook", obsHandler.GetWebhookHandler)
			settings.POST("/observability-webhook", obsHandler.SetWebhookHandler)
			settings.DELETE("/observability-webhook", obsHandler.DeleteWebhookHandler)
			settings.GET("/observability-webhook/status", obsHandler.GetStatusHandler)
			settings.POST("/observability-webhook/redrive", obsHandler.RedriveHandler)
			settings.GET("/observability-webhook/dlq", obsHandler.GetDeadLetterQueueHandler)
			settings.DELETE("/observability-webhook/dlq", obsHandler.ClearDeadLetterQueueHandler)
		}

		// Admin routes for tag approval and access policy management (VC-based authorization)
		if s.config.Features.DID.Authorization.Enabled {
			adminGroup := agentAPI.Group("")
			adminGroup.Use(middleware.AdminTokenAuth(s.config.Features.DID.Authorization.AdminToken))

			// Tag approval admin routes
			if s.tagApprovalService != nil {
				tagApprovalHandlers := admin.NewTagApprovalHandlers(s.tagApprovalService, s.storage)
				tagApprovalHandlers.RegisterRoutes(adminGroup)
			}

			// Access policy admin routes
			if s.accessPolicyService != nil {
				accessPolicyHandlers := admin.NewAccessPolicyHandlers(s.accessPolicyService)
				accessPolicyHandlers.RegisterRoutes(adminGroup)
			}

			logger.Logger.Info().Msg("📋 Authorization admin routes registered")
		}

		// Config storage routes (admin-authenticated)
		{
			configHandlers := handlers.NewConfigStorageHandlers(s.storage, s.configReloadFn())
			configHandlers.RegisterRoutes(agentAPI)
			logger.Logger.Info().Msg("Config storage routes registered")
		}

		// Connector routes (authenticated with separate connector token)
		if s.config.Features.Connector.Enabled && s.config.Features.Connector.Token != "" {
			connectorGroup := agentAPI.Group("/connector")
			connectorGroup.Use(middleware.ConnectorTokenAuth(s.config.Features.Connector.Token))

			connectorHandlers := connectorpkg.NewHandlers(
				s.config.Features.Connector,
				s.storage,
				s.statusManager,
				s.accessPolicyService,
				s.tagApprovalService,
				s.didService,
			)
			connectorHandlers.RegisterRoutes(connectorGroup)

			// Config management routes for connector
			configGroup := connectorGroup.Group("")
			configGroup.Use(middleware.ConnectorCapabilityCheck("config_management", s.config.Features.Connector.Capabilities))
			{
				configHandlers := handlers.NewConfigStorageHandlers(s.storage, s.configReloadFn())
				configHandlers.RegisterRoutes(configGroup)
			}

			logger.Logger.Info().Msg("🔌 Connector routes registered")
		}

		// Agentic API routes — agent-optimized endpoints for discovery, query, and operations
		agenticGroup := agentAPI.Group("/agentic")
		{
			agenticGroup.GET("/discover", agentic.DiscoverHandler(s.apiCatalog))
			agenticGroup.POST("/query", agentic.QueryHandler(s.storage))
			agenticGroup.GET("/run/:run_id", agentic.RunOverviewHandler(s.storage))
			agenticGroup.GET("/agent/:agent_id/summary", agentic.AgentSummaryHandler(s.storage))
			agenticGroup.POST("/batch", agentic.BatchHandler(s.Router))
			agenticGroup.GET("/status", agentic.StatusHandler(s.storage))
		}
		logger.Logger.Info().Msg("🤖 Agentic API routes registered (discover, query, run, agent, batch, status)")
	}

	// Knowledge Base routes — public, no auth required (registered outside agentAPI group)
	kbGroup := s.Router.Group("/api/v1/agentic/kb")
	{
		kbGroup.GET("/topics", agentic.KBTopicsHandler(s.kb))
		kbGroup.GET("/articles", agentic.KBArticlesHandler(s.kb))
		kbGroup.GET("/articles/:article_id/:sub_id", agentic.KBArticleHandler(s.kb))
		kbGroup.GET("/articles/:article_id", agentic.KBArticleHandler(s.kb))
		kbGroup.GET("/guide", agentic.KBGuideHandler(s.kb))
	}
	logger.Logger.Info().Msg("📚 Knowledge Base routes registered (public, no auth)")

	// Smart 404 handler — provides endpoint suggestions for non-UI paths,
	// preserves SPA fallback for /ui/* paths.
	var uiNoRouteHandler gin.HandlerFunc
	if s.config.UI.Enabled && (s.config.UI.Mode != "embedded" || !client.IsUIEmbedded()) {
		uiNoRouteHandler = func(c *gin.Context) {
			// Check if it's a static asset
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

			// SPA fallback — serve index.html for client-side routing
			distPath := s.config.UI.DistPath
			if distPath == "" {
				execPath, err := os.Executable()
				if err != nil {
					distPath = filepath.Join("apps", "platform", "agentfield", "web", "client", "dist")
					if _, statErr := os.Stat(distPath); os.IsNotExist(statErr) {
						distPath = filepath.Join("web", "client", "dist")
					}
				} else {
					execDir := filepath.Dir(execPath)
					distPath = filepath.Join(execDir, "web", "client", "dist")
					if _, err := os.Stat(distPath); os.IsNotExist(err) {
						distPath = filepath.Join(filepath.Dir(execDir), "apps", "platform", "agentfield", "web", "client", "dist")
					}
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
		}
	}
	s.Router.NoRoute(agentic.Smart404Handler(s.apiCatalog, uiNoRouteHandler))
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
