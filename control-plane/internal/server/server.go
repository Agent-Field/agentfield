package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/communication"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/process"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/observability"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/knowledgebase"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all" // register first-party trigger Sources
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/adminpb"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
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
	// Trigger / webhook plugin system
	triggerDispatcher *services.TriggerDispatcher
	sourceManager     *services.SourceManager
	triggerHandlers   *handlers.TriggerHandlers
	// cancelDispatcher forwards execution-cancelled events to remote SDK
	// workers so they can cooperatively short-circuit in-flight reasoner
	// code (raise CancelledError / abort signals / cancel contexts).
	cancelDispatcher *services.CancelDispatcher
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

	// Configure SSRF-safe webhook client allowlist. Hosts/CIDRs listed here
	// bypass the private-IP check (e.g. for internal Docker/K8s service names).
	services.SetWebhookAllowedHosts(cfg.AgentField.Registration.WebhookAllowedHosts)

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

	triggerDispatcher := services.NewTriggerDispatcher(storageProvider, vcService)
	sourceManager := services.NewSourceManager(storageProvider, triggerDispatcher)
	triggerHandlers := handlers.NewTriggerHandlers(storageProvider, triggerDispatcher, sourceManager)
	handlers.SetTriggerSourceManager(sourceManager)

	cancelDispatcher := services.NewCancelDispatcher(storageProvider, services.CancelDispatcherConfig{
		InternalToken: cfg.Features.DID.Authorization.InternalToken,
	})

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
		triggerDispatcher:      triggerDispatcher,
		sourceManager:          sourceManager,
		triggerHandlers:        triggerHandlers,
		cancelDispatcher:       cancelDispatcher,
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

	// Start cancel dispatcher: forwards bus ExecutionCancelledEvent to
	// remote workers via HTTP callback. Best-effort; missing endpoint on
	// older SDKs is treated as a no-op.
	if s.cancelDispatcher != nil {
		s.cancelDispatcher.Start(ctx)
	}

	// Start OpenTelemetry execution tracer in background
	if s.executionTracer != nil {
		s.executionTracer.Start(ctx)
	}

	// Boot loop-kind triggers (cron etc.) so a server restart resumes existing schedules.
	if s.sourceManager != nil {
		go func() {
			if err := s.sourceManager.LoadAll(context.Background()); err != nil {
				logger.Logger.Warn().Err(err).Msg("source manager: LoadAll failed")
			}
		}()
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

	// Stop cancel dispatcher
	if s.cancelDispatcher != nil {
		s.cancelDispatcher.Stop()
	}

	if s.registryWatcherCancel != nil {
		s.registryWatcherCancel()
		s.registryWatcherCancel = nil
	}

	// Stop UI service heartbeat
	if s.uiService != nil {
		s.uiService.StopHeartbeat()
	}

	// Stop loop-source goroutines.
	if s.sourceManager != nil {
		s.sourceManager.StopAll()
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

// setupRoutes composes the full HTTP surface by delegating to focused
// registration methods. Ordering matters: global middleware must be installed
// before any route; UI API routes are registered before /api/v1 to avoid path
// collisions; and NoRoute/Smart404 must be installed last so it only catches
// truly unknown paths.
//
// Each register* method lives in its own routes_*.go file to keep auth posture
// and concern boundaries obvious. This function is pure composition and
// contains no handler logic.
func (s *AgentFieldServer) setupRoutes() {
	s.applyGlobalMiddleware()

	s.registerPublicRoutes()
	s.registerDIDWellKnownRoutes()

	s.registerUIStatic()
	s.registerUIAPI()

	agentAPI := s.Router.Group("/api/v1")
	{
		s.registerCoreRoutes(agentAPI)
		s.registerMemoryRoutes(agentAPI)
		s.registerDIDRoutes(agentAPI)
		s.registerObservabilityRoutes(agentAPI)
		s.registerAdminRoutes(agentAPI)
		s.registerConnectorRoutes(agentAPI)
		s.registerAgenticRoutes(agentAPI)
		s.registerTriggerRoutes(agentAPI)
	}

	s.registerKBRoutes()
	s.register404()
}

var absPathForServerID = filepath.Abs

// generateAgentFieldServerID creates a deterministic af server ID based on the agentfield home directory.
// This ensures each agentfield instance has a unique ID while being deterministic for the same installation.
func generateAgentFieldServerID(agentfieldHome string) string {
	// Use the absolute path of agentfield home to generate a deterministic ID
	absPath, err := absPathForServerID(agentfieldHome)
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
