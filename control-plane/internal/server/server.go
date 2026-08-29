package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
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
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/communication"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/process"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/knowledge"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/observability"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/knowledgebase"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
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
	storage                  storage.StorageProvider
	cache                    storage.CacheProvider
	Router                   *gin.Engine
	uiService                *services.UIService           // Add UIService
	executionsUIService      *services.ExecutionsUIService // Add ExecutionsUIService
	healthMonitor            *services.HealthMonitor
	presenceManager          *services.PresenceManager
	statusManager            *services.StatusManager // Add StatusManager for unified status management
	agentService             interfaces.AgentService // Add AgentService for lifecycle management
	agentClient              interfaces.AgentClient  // Add AgentClient for agent communication
	packageJobs              *packagejobs.Manager    // Async package install/update jobs
	packageMaintenance       *packagemaint.Service   // Package restore/check/update loop
	packageMaintenanceCancel context.CancelFunc
	maintenanceReady         chan struct{}
	maintenanceReadyOnce     sync.Once
	config                   *config.Config
	storageHealthOverride    func(context.Context) gin.H
	cacheHealthOverride      func(context.Context) gin.H
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
	telemetryService       *observability.TelemetryService
	configMu               sync.RWMutex
	// Rate limiting stores for hot endpoints
	rateLimitExecute    *middleware.RateLimiterStore
	rateLimitDiscovery  *middleware.RateLimiterStore
	rateLimitBulkStatus *middleware.RateLimiterStore
	rateLimitGlobal     *middleware.RateLimiterStore
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
	// Native scope-aware RAG knowledge store (embed-on-write/search).
	knowledgeService *knowledge.Service
	// HTTP server for graceful shutdown support
	httpServerMu      sync.RWMutex
	httpServer        *http.Server
	stopping          bool
	streamCtx         context.Context
	cancelStreams     context.CancelFunc
	cancelStreamsOnce sync.Once
}

// newRouter builds the gin engine the control plane serves from, with the
// structured request logger already installed as its outermost middleware.
//
// gin.New() avoids gin.Default()'s plaintext logger. The logging pair is
// attached here rather than alongside the rest of the global middleware so
// that nothing can be registered ahead of it: gin runs middleware in
// registration order, and a middleware that aborts (CORS rejecting a
// disallowed Origin, for instance) never reaches the handlers behind it — so
// any request handled before the logger would be invisible at every level.
// Recovery is installed after GinLogger so a panicking handler still produces
// a structured http_request at error/500. GIN_MODE still governs the
// [GIN-debug] route table.
func newRouter() *gin.Engine {
	router := gin.New()
	useStructuredRequestLogging(router)
	return router
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

	// Configure execution event payload redaction from logging config.
	handlers.SetRedactPayloads(cfg.Logging.ShouldRedactPayloads())
	if grace := cfg.AgentField.NodeHealth.AgentRestartGrace; grace != 0 {
		handlers.SetAgentRestartGrace(grace)
	}
	if grace := cfg.AgentField.NodeHealth.AgentDrainGrace; grace != 0 {
		handlers.SetAgentDrainGrace(grace)
	}
	logger.Logger.Info().
		Dur("agent_restart_grace", handlers.AgentRestartGrace()).
		Dur("agent_drain_grace", handlers.AgentDrainGrace()).
		Msg("configured agent restart and drain grace windows")

	Router := newRouter()

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

	// Async package install/update job manager (HTTP install API)
	packageJobs := packagejobs.NewManager(registryStorage, agentfieldHome, agentService)
	// Sync the DB immediately after API-driven registry mutations so clients
	// read their own writes; the fsnotify watcher covers CLI-driven changes.
	packageJobs.SetOnRegistryChange(func() {
		_ = SyncPackagesFromRegistry(agentfieldHome, storageProvider)
	})
	packageMaintenanceReady := make(chan struct{})
	packageMaintenance := newPackageMaintenance(agentfieldHome, storageProvider, agentService, packageJobs, packageMaintenanceReady)

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

	var payloadStore services.PayloadStore
	if cfg.Storage.Mode == "postgres" {
		payloadStore = services.NopPayloadStore{}
	} else {
		payloadStore = services.NewFilePayloadStore(dirs.PayloadsDir)
	}

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

	var telemetryService *observability.TelemetryService
	if cfg.Telemetry.IsEnabled() {
		service, err := observability.NewTelemetryService(cfg.Telemetry, agentfieldHome, cfg.Storage.Mode, cfg.Telemetry.AgentFieldVersion)
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("failed to initialize anonymous OSS telemetry")
		} else {
			telemetryService = service
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
	cleanupPayloadStore := services.NewFilePayloadStore(dirs.PayloadsDir)
	cleanupService := handlers.NewExecutionCleanupService(storageProvider, cfg.AgentField.ExecutionCleanup, cleanupPayloadStore)
	logger.Logger.Info().Dur("agent_call_timeout", cfg.AgentField.ExecutionQueue.AgentCallTimeout).Msg("effective agent call timeout")

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

	// Native RAG knowledge store: pick the embedding provider from config,
	// falling back to the deterministic FakeEmbedder when no OpenAI key is set.
	embedder, isOpenAI := embedding.NewFromConfig(embedding.ProviderConfig{
		Provider: cfg.Features.Knowledge.Provider,
		APIKey:   cfg.Features.Knowledge.OpenAI.APIKey,
		Model:    cfg.Features.Knowledge.OpenAI.Model,
	})
	logger.Logger.Info().
		Bool("openai", isOpenAI).
		Int("dimensions", embedder.Dimensions()).
		Msg("knowledge store embedding provider initialized")
	knowledgeService := knowledge.NewService(storageProvider, embedder)

	streamCtx, cancelStreams := context.WithCancel(context.Background())
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
		packageJobs:            packageJobs,
		packageMaintenance:     packageMaintenance,
		maintenanceReady:       packageMaintenanceReady,
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
		streamCtx:              streamCtx,
		cancelStreams:          cancelStreams,
		telemetryService:       telemetryService,
		registryWatcherCancel:  nil,
		adminGRPCPort:          adminPort,
		triggerDispatcher:      triggerDispatcher,
		sourceManager:          sourceManager,
		triggerHandlers:        triggerHandlers,
		cancelDispatcher:       cancelDispatcher,
		apiCatalog:             initAPICatalog(),
		kb:                     initKnowledgeBase(),
		knowledgeService:       knowledgeService,
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

	// Start anonymous OSS usage telemetry. Best-effort only.
	if s.telemetryService != nil {
		s.telemetryService.Start(ctx)
	}

	// Boot loop-kind triggers (cron etc.) so a server restart resumes existing schedules.
	if s.sourceManager != nil {
		go func() {
			if err := s.sourceManager.LoadAll(context.Background()); err != nil {
				logger.Logger.Warn().Err(err).Msg("source manager: LoadAll failed")
			}
		}()
	}
	if s.packageMaintenance != nil {
		maintenanceCtx, cancel := context.WithCancel(context.Background())
		s.packageMaintenanceCancel = cancel
		s.packageMaintenance.SetLifecycleContext(maintenanceCtx)
		go s.packageMaintenance.Run(maintenanceCtx)
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

	// Start HTTP server (using net/http.Server for graceful shutdown support)
	addr := ":" + strconv.Itoa(s.config.AgentField.Port)
	httpServer := s.newHTTPServer(addr)
	if !s.setHTTPServer(httpServer) {
		return nil
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to start HTTP server on %s: %w", addr, err)
	}
	s.maintenanceReadyOnce.Do(func() {
		if s.maintenanceReady != nil {
			close(s.maintenanceReady)
		}
	})
	if err := httpServer.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("failed to start HTTP server on %s: %w", addr, err)
	}
	return nil
}

func (s *AgentFieldServer) newHTTPServer(addr string) *http.Server {
	maxExecuteBodyBytes := int64(32 << 20)
	if raw := strings.TrimSpace(os.Getenv("AGENTFIELD_MAX_EXECUTE_BODY_BYTES")); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil && parsed > 0 {
			maxExecuteBodyBytes = parsed
		} else {
			logger.Logger.Warn().Str("value", raw).Msg("invalid AGENTFIELD_MAX_EXECUTE_BODY_BYTES; using 32 MiB")
		}
	}
	maxRegisterBodyBytes := int64(8 << 20)
	if raw := strings.TrimSpace(os.Getenv("AGENTFIELD_MAX_REGISTER_BODY_BYTES")); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil && parsed > 0 {
			maxRegisterBodyBytes = parsed
		} else {
			logger.Logger.Warn().Str("value", raw).Msg("invalid AGENTFIELD_MAX_REGISTER_BODY_BYTES; using 8 MiB")
		}
	}
	handler := maxRequestBodyHandler(s.Router, maxExecuteBodyBytes, maxRegisterBodyBytes)
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
}

func maxExecuteBodyHandler(next http.Handler, limit int64) http.Handler {
	return maxRequestBodyHandler(next, limit, 0)
}

func maxRequestBodyHandler(next http.Handler, executeLimit, registerLimit int64) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		isExecuteRoute := r.URL.Path == "/api/v1/execute" || strings.HasPrefix(r.URL.Path, "/api/v1/execute/")
		if r.Method == http.MethodPost && isExecuteRoute {
			if r.ContentLength > executeLimit {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.WriteHeader(http.StatusRequestEntityTooLarge)
				_, _ = w.Write([]byte(`{"error":"request body too large"}`))
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, executeLimit)
		}
		isRegisterRoute := r.URL.Path == "/api/v1/nodes/register" || r.URL.Path == "/api/v1/nodes" || r.URL.Path == "/api/v1/nodes/register-serverless"
		if r.Method == http.MethodPost && isRegisterRoute && registerLimit > 0 {
			if r.ContentLength > registerLimit {
				writeBodyTooLarge(w)
				return
			}
			body, err := io.ReadAll(io.LimitReader(r.Body, registerLimit+1))
			if err != nil {
				http.Error(w, "failed to read request body", http.StatusBadRequest)
				return
			}
			if int64(len(body)) > registerLimit {
				writeBodyTooLarge(w)
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
		}
		next.ServeHTTP(w, r)
	})
}

func writeBodyTooLarge(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusRequestEntityTooLarge)
	_, _ = w.Write([]byte(`{"error":"request body too large"}`))
}

func (s *AgentFieldServer) setHTTPServer(httpServer *http.Server) bool {
	s.httpServerMu.Lock()
	defer s.httpServerMu.Unlock()
	if s.stopping {
		return false
	}
	s.httpServer = httpServer
	return true
}

func (s *AgentFieldServer) getHTTPServer() *http.Server {
	s.httpServerMu.RLock()
	defer s.httpServerMu.RUnlock()
	return s.httpServer
}

func (s *AgentFieldServer) shutdownHTTPServerWithContext(ctx context.Context) error {
	httpServer := s.getHTTPServer()
	if httpServer == nil {
		return nil
	}
	if err := httpServer.Shutdown(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("HTTP server shutdown timed out, forcing close")
		_ = httpServer.Close()
		return err
	}
	logger.Logger.Info().Msg("HTTP server shut down gracefully")
	return nil
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
	s.httpServerMu.Lock()
	s.stopping = true
	s.httpServerMu.Unlock()
	s.cancelStreamsOnce.Do(func() {
		if s.cancelStreams != nil {
			s.cancelStreams()
		}
	})

	shutdownTimeout := 30 * time.Second
	if s.config != nil && s.config.AgentField.ShutdownTimeout > 0 {
		shutdownTimeout = s.config.AgentField.ShutdownTimeout
	}
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancelShutdown()
	httpShutdownErr := s.shutdownHTTPServerWithContext(shutdownCtx)
	asyncCtx, cancelAsync := asyncDrainContext(shutdownCtx)
	handlers.StopAsyncWorkerPool(asyncCtx)
	cancelAsync()

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
	if s.healthMonitor != nil {
		s.healthMonitor.Stop()
	}

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

	// Stop rate limiter eviction goroutines
	if s.rateLimitExecute != nil {
		s.rateLimitExecute.Stop()
	}
	if s.rateLimitDiscovery != nil {
		s.rateLimitDiscovery.Stop()
	}
	if s.rateLimitBulkStatus != nil {
		s.rateLimitBulkStatus.Stop()
	}
	if s.rateLimitGlobal != nil {
		s.rateLimitGlobal.Stop()
	}

	if s.registryWatcherCancel != nil {
		s.registryWatcherCancel()
		s.registryWatcherCancel = nil
	}

	if s.packageMaintenanceCancel != nil {
		s.packageMaintenanceCancel()
		s.packageMaintenanceCancel = nil
	}
	if s.packageMaintenance != nil {
		waitCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := s.packageMaintenance.Stop(waitCtx); err != nil {
			logger.Logger.Warn().Err(err).Msg("timed out waiting for package maintenance to stop")
		}
		cancel()
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
	if s.telemetryService != nil {
		s.telemetryService.Stop()
	}
	if s.tracerShutdown != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.tracerShutdown(ctx); err != nil {
			logger.Logger.Error().Err(err).Msg("Failed to shutdown OTel tracer provider")
		}
	}

	// TODO: Implement graceful shutdown for WebSocket
	return httpShutdownErr
}

func (s *AgentFieldServer) streamHandler(handler gin.HandlerFunc) gin.HandlerFunc {
	if s.streamCtx == nil {
		s.streamCtx, s.cancelStreams = context.WithCancel(context.Background())
	}
	return handlers.WithStreamContext(s.streamCtx, handler)
}

func asyncDrainContext(shutdownCtx context.Context) (context.Context, context.CancelFunc) {
	budget := 5 * time.Second
	if deadline, ok := shutdownCtx.Deadline(); ok {
		if remaining := time.Until(deadline); remaining > budget {
			budget = remaining
		}
	}
	// The async pool always gets a useful drain window, so total shutdown may
	// exceed AGENTFIELD_SHUTDOWN_TIMEOUT by up to five seconds.
	return context.WithTimeout(context.Background(), budget)
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
	s.initRateLimiters()

	s.registerPublicRoutes()
	s.registerDIDWellKnownRoutes()
	s.registerARDPublicRoutes()

	s.registerUIStatic()
	s.registerUIAPI()

	agentAPI := s.Router.Group("/api/v1")
	{
		s.registerCoreRoutes(agentAPI)
		s.registerMemoryRoutes(agentAPI)
		s.registerKnowledgeRoutes(agentAPI)
		s.registerDIDRoutes(agentAPI)
		s.registerObservabilityRoutes(agentAPI)
		s.registerAdminRoutes(agentAPI)
		s.registerConnectorRoutes(agentAPI)
		s.registerAgenticRoutes(agentAPI)
		s.registerTriggerRoutes(agentAPI)
		s.registerARDRoutes(agentAPI)
	}

	s.registerKBRoutes()
	s.registerMCPRoutes()
	s.registerPprofRoutes()
	s.register404()
}

// initRateLimiters creates per-endpoint rate limiter stores based on config.
// Keys idle for more than 10 minutes are evicted to avoid unbounded growth.
func (s *AgentFieldServer) initRateLimiters() {
	cfg := s.config.AgentField.RateLimit
	if !cfg.Enabled {
		return
	}

	const evictionTTL = 10 * time.Minute

	// Apply defaults for unconfigured values.
	executeRPS := cfg.ExecuteRPS
	if executeRPS <= 0 {
		executeRPS = 50
	}
	executeBurst := cfg.ExecuteBurst
	if executeBurst <= 0 {
		executeBurst = 100
	}

	discoveryRPS := cfg.DiscoveryRPS
	if discoveryRPS <= 0 {
		discoveryRPS = 20
	}
	discoveryBurst := cfg.DiscoveryBurst
	if discoveryBurst <= 0 {
		discoveryBurst = 40
	}

	bulkStatusRPS := cfg.BulkStatusRPS
	if bulkStatusRPS <= 0 {
		bulkStatusRPS = 30
	}
	bulkStatusBurst := cfg.BulkStatusBurst
	if bulkStatusBurst <= 0 {
		bulkStatusBurst = 60
	}

	globalRPS := cfg.GlobalRPS
	if globalRPS <= 0 {
		globalRPS = 200
	}
	globalBurst := cfg.GlobalBurst
	if globalBurst <= 0 {
		globalBurst = 400
	}

	s.rateLimitExecute = middleware.NewRateLimiterStore(executeRPS, executeBurst, evictionTTL)
	s.rateLimitDiscovery = middleware.NewRateLimiterStore(discoveryRPS, discoveryBurst, evictionTTL)
	s.rateLimitBulkStatus = middleware.NewRateLimiterStore(bulkStatusRPS, bulkStatusBurst, evictionTTL)
	s.rateLimitGlobal = middleware.NewRateLimiterStore(globalRPS, globalBurst, evictionTTL)

	logger.Logger.Info().
		Float64("execute_rps", executeRPS).
		Int("execute_burst", executeBurst).
		Float64("discovery_rps", discoveryRPS).
		Float64("bulk_status_rps", bulkStatusRPS).
		Float64("global_rps", globalRPS).
		Msg("⚡ Rate limiting enabled")
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
