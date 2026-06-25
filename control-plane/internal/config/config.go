package config

import (
	"fmt"           // Added for fmt.Errorf
	"os"            // Added for os.Stat, os.ReadFile
	"path/filepath" // Added for filepath.Join
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3" // Added for yaml.Unmarshal

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

// Config holds the entire configuration for the AgentField server.
type Config struct {
	AgentField AgentFieldConfig `yaml:"agentfield" mapstructure:"agentfield"`
	Features   FeatureConfig    `yaml:"features" mapstructure:"features"`
	Storage    StorageConfig    `yaml:"storage" mapstructure:"storage"`
	UI         UIConfig         `yaml:"ui" mapstructure:"ui"`
	API        APIConfig        `yaml:"api" mapstructure:"api"`
	Telemetry  TelemetryConfig  `yaml:"telemetry" mapstructure:"telemetry"`
}

// TelemetryConfig controls anonymous OSS usage telemetry. It is separate from
// Prometheus metrics and OpenTelemetry tracing, which remain local/self-hosted
// observability surfaces.
type TelemetryConfig struct {
	Enabled       *bool         `yaml:"enabled" mapstructure:"enabled"`
	Mode          string        `yaml:"mode" mapstructure:"mode"`
	Endpoint      string        `yaml:"endpoint" mapstructure:"endpoint"`
	InstallIDPath string        `yaml:"install_id_path" mapstructure:"install_id_path"`
	InstallID     string        `yaml:"install_id" mapstructure:"install_id"`
	Timeout       time.Duration `yaml:"timeout" mapstructure:"timeout"`
}

// IsEnabled returns true unless telemetry was explicitly disabled.
func (c TelemetryConfig) IsEnabled() bool {
	return c.Enabled == nil || *c.Enabled
}

// UIConfig holds configuration for the web UI.
type UIConfig struct {
	Enabled    bool   `yaml:"enabled" mapstructure:"enabled"`
	Mode       string `yaml:"mode" mapstructure:"mode"`               // "embedded", "dev", "separate"
	SourcePath string `yaml:"source_path" mapstructure:"source_path"` // Path to UI source for building
	DistPath   string `yaml:"dist_path" mapstructure:"dist_path"`     // Path to built UI assets for serving
	DevPort    int    `yaml:"dev_port" mapstructure:"dev_port"`       // Port for UI dev server
}

// AgentFieldConfig holds the core AgentField server configuration.
type AgentFieldConfig struct {
	Port             int                    `yaml:"port"`
	ARD              ARDConfig              `yaml:"ard" mapstructure:"ard"`
	Registration     RegistrationConfig     `yaml:"registration" mapstructure:"registration"`
	NodeHealth       NodeHealthConfig       `yaml:"node_health" mapstructure:"node_health"`
	LLMHealth        LLMHealthConfig        `yaml:"llm_health" mapstructure:"llm_health"`
	ExecutionCleanup ExecutionCleanupConfig `yaml:"execution_cleanup" mapstructure:"execution_cleanup"`
	ExecutionQueue   ExecutionQueueConfig   `yaml:"execution_queue" mapstructure:"execution_queue"`
	Approval         ApprovalConfig         `yaml:"approval" mapstructure:"approval"`
	NodeLogProxy     NodeLogProxyConfig     `yaml:"node_log_proxy" mapstructure:"node_log_proxy"`
	ExecutionLogs    ExecutionLogsConfig    `yaml:"execution_logs" mapstructure:"execution_logs"`
}

// ARDConfig controls Agentic Resource Discovery exposure. Config/env values are
// deployment guardrails; runtime publish/import state is stored in the DB.
type ARDConfig struct {
	Enabled         bool              `yaml:"enabled" mapstructure:"enabled"`
	PublicBaseURL   string            `yaml:"public_base_url" mapstructure:"public_base_url"`
	PublisherDomain string            `yaml:"publisher_domain" mapstructure:"publisher_domain"`
	Host            ARDHostConfig     `yaml:"host" mapstructure:"host"`
	Publish         ARDPublishConfig  `yaml:"publish" mapstructure:"publish"`
	Registry        ARDRegistryConfig `yaml:"registry" mapstructure:"registry"`
	External        ARDExternalConfig `yaml:"external" mapstructure:"external"`
}

type ARDHostConfig struct {
	DisplayName      string `yaml:"display_name" mapstructure:"display_name"`
	Identifier       string `yaml:"identifier" mapstructure:"identifier"`
	DocumentationURL string `yaml:"documentation_url" mapstructure:"documentation_url"`
	LogoURL          string `yaml:"logo_url" mapstructure:"logo_url"`
}

type ARDPublishConfig struct {
	Enabled               bool     `yaml:"enabled" mapstructure:"enabled"`
	IncludeHealthStatuses []string `yaml:"include_health_statuses" mapstructure:"include_health_statuses"`
	DefaultType           string   `yaml:"default_type" mapstructure:"default_type"`
}

type ARDRegistryConfig struct {
	Enabled bool `yaml:"enabled" mapstructure:"enabled"`
	Public  bool `yaml:"public" mapstructure:"public"`
}

type ARDExternalConfig struct {
	SearchEnabled      bool     `yaml:"search_enabled" mapstructure:"search_enabled"`
	InvocationEnabled  bool     `yaml:"invocation_enabled" mapstructure:"invocation_enabled"`
	AllowedRegistries  []string `yaml:"allowed_registries" mapstructure:"allowed_registries"`
	DefaultSearchLimit int      `yaml:"default_search_limit" mapstructure:"default_search_limit"`
}

// RegistrationConfig governs validation of agent-supplied registration endpoints.
type RegistrationConfig struct {
	ServerlessDiscoveryAllowedHosts []string `yaml:"serverless_discovery_allowed_hosts" mapstructure:"serverless_discovery_allowed_hosts"`
	// WebhookAllowedHosts lists hostnames, wildcards, or CIDRs that bypass
	// SSRF filtering for execution and observability webhooks. Useful for
	// deployments where webhooks legitimately target internal RFC-1918 hosts
	// (e.g. Docker service names, Kubernetes cluster DNS).
	WebhookAllowedHosts []string `yaml:"webhook_allowed_hosts" mapstructure:"webhook_allowed_hosts"`
}

// NodeLogProxyConfig limits the control plane proxy to agent process logs (NDJSON).
type NodeLogProxyConfig struct {
	ConnectTimeout    time.Duration `yaml:"connect_timeout" mapstructure:"connect_timeout"`
	StreamIdleTimeout time.Duration `yaml:"stream_idle_timeout" mapstructure:"stream_idle_timeout"`
	MaxStreamDuration time.Duration `yaml:"max_stream_duration" mapstructure:"max_stream_duration"`
	MaxTailLines      int           `yaml:"max_tail_lines" mapstructure:"max_tail_lines"`
}

// EffectiveNodeLogProxy returns proxy settings with defaults for zero values.
func EffectiveNodeLogProxy(c NodeLogProxyConfig) NodeLogProxyConfig {
	out := c
	if out.ConnectTimeout <= 0 {
		out.ConnectTimeout = 5 * time.Second
	}
	if out.StreamIdleTimeout <= 0 {
		out.StreamIdleTimeout = 60 * time.Second
	}
	if out.MaxStreamDuration <= 0 {
		out.MaxStreamDuration = 15 * time.Minute
	}
	if out.MaxTailLines <= 0 {
		out.MaxTailLines = 10000
	}
	return out
}

// ExecutionLogsConfig governs structured execution-correlated logs stored by the control plane.
type ExecutionLogsConfig struct {
	RetentionPeriod        time.Duration `yaml:"retention_period" mapstructure:"retention_period"`
	MaxEntriesPerExecution int           `yaml:"max_entries_per_execution" mapstructure:"max_entries_per_execution"`
	MaxTailEntries         int           `yaml:"max_tail_entries" mapstructure:"max_tail_entries"`
	StreamIdleTimeout      time.Duration `yaml:"stream_idle_timeout" mapstructure:"stream_idle_timeout"`
	MaxStreamDuration      time.Duration `yaml:"max_stream_duration" mapstructure:"max_stream_duration"`
}

// EffectiveExecutionLogs returns execution-log settings with defaults for zero values.
func EffectiveExecutionLogs(c ExecutionLogsConfig) ExecutionLogsConfig {
	out := c
	if out.RetentionPeriod <= 0 {
		out.RetentionPeriod = 24 * time.Hour
	}
	if out.MaxEntriesPerExecution <= 0 {
		out.MaxEntriesPerExecution = 5000
	}
	if out.MaxTailEntries <= 0 {
		out.MaxTailEntries = 1000
	}
	if out.StreamIdleTimeout <= 0 {
		out.StreamIdleTimeout = 60 * time.Second
	}
	if out.MaxStreamDuration <= 0 {
		out.MaxStreamDuration = 15 * time.Minute
	}
	return out
}

// ApprovalConfig holds configuration for the execution approval workflow.
// The control plane manages execution state only — agents are responsible for
// communicating with external approval services (e.g. hax-sdk).
type ApprovalConfig struct {
	WebhookSecret      string `yaml:"webhook_secret" mapstructure:"webhook_secret"`             // Required for HMAC auth on /api/v1/webhooks/approval-response (empty disables the endpoint)
	DefaultExpiryHours int    `yaml:"default_expiry_hours" mapstructure:"default_expiry_hours"` // Default approval expiry (hours); 0 = 72h
}

// NodeHealthConfig holds configuration for agent node health monitoring.
// Zero values are treated as "use default" — set explicitly to override.
type NodeHealthConfig struct {
	CheckInterval           time.Duration `yaml:"check_interval" mapstructure:"check_interval"`                       // How often to HTTP health check nodes (0 = default 10s)
	CheckTimeout            time.Duration `yaml:"check_timeout" mapstructure:"check_timeout"`                         // Timeout per HTTP health check (0 = default 5s)
	ConsecutiveFailures     int           `yaml:"consecutive_failures" mapstructure:"consecutive_failures"`           // Failures before marking inactive (0 = default 3; set 1 for instant)
	RecoveryDebounce        time.Duration `yaml:"recovery_debounce" mapstructure:"recovery_debounce"`                 // Wait before allowing inactive->active (0 = default 5s)
	HeartbeatStaleThreshold time.Duration `yaml:"heartbeat_stale_threshold" mapstructure:"heartbeat_stale_threshold"` // Heartbeat age before marking stale (0 = default 60s)
}

// ExecutionCleanupConfig holds configuration for execution cleanup and garbage collection
type ExecutionCleanupConfig struct {
	Enabled                bool          `yaml:"enabled" mapstructure:"enabled" default:"true"`
	RetentionPeriod        time.Duration `yaml:"retention_period" mapstructure:"retention_period" default:"24h"`
	CleanupInterval        time.Duration `yaml:"cleanup_interval" mapstructure:"cleanup_interval" default:"1h"`
	BatchSize              int           `yaml:"batch_size" mapstructure:"batch_size" default:"100"`
	PreserveRecentDuration time.Duration `yaml:"preserve_recent_duration" mapstructure:"preserve_recent_duration" default:"1h"`
	StaleExecutionTimeout  time.Duration `yaml:"stale_execution_timeout" mapstructure:"stale_execution_timeout" default:"30m"`
	MaxRetries             int           `yaml:"max_retries" mapstructure:"max_retries" default:"0"`
	RetryBackoff           time.Duration `yaml:"retry_backoff" mapstructure:"retry_backoff" default:"30s"`
}

// ExecutionQueueConfig configures execution and webhook settings.
type ExecutionQueueConfig struct {
	AgentCallTimeout       time.Duration `yaml:"agent_call_timeout" mapstructure:"agent_call_timeout"`
	MaxConcurrentPerAgent  int           `yaml:"max_concurrent_per_agent" mapstructure:"max_concurrent_per_agent"` // 0 = unlimited
	WebhookTimeout         time.Duration `yaml:"webhook_timeout" mapstructure:"webhook_timeout"`
	WebhookMaxAttempts     int           `yaml:"webhook_max_attempts" mapstructure:"webhook_max_attempts"`
	WebhookRetryBackoff    time.Duration `yaml:"webhook_retry_backoff" mapstructure:"webhook_retry_backoff"`
	WebhookMaxRetryBackoff time.Duration `yaml:"webhook_max_retry_backoff" mapstructure:"webhook_max_retry_backoff"`
}

// LLMHealthConfig configures LLM backend health monitoring with circuit breaker.
type LLMHealthConfig struct {
	Enabled           bool          `yaml:"enabled" mapstructure:"enabled"`
	Endpoints         []LLMEndpoint `yaml:"endpoints" mapstructure:"endpoints"`
	CheckInterval     time.Duration `yaml:"check_interval" mapstructure:"check_interval"`             // How often to probe (default 15s)
	CheckTimeout      time.Duration `yaml:"check_timeout" mapstructure:"check_timeout"`               // Timeout per probe (default 5s)
	FailureThreshold  int           `yaml:"failure_threshold" mapstructure:"failure_threshold"`       // Failures before opening circuit (default 3)
	RecoveryTimeout   time.Duration `yaml:"recovery_timeout" mapstructure:"recovery_timeout"`         // How long circuit stays open before half-open (default 30s)
	HalfOpenMaxProbes int           `yaml:"half_open_max_probes" mapstructure:"half_open_max_probes"` // Probes in half-open before closing (default 2)
}

// LLMEndpoint defines a single LLM backend to monitor.
type LLMEndpoint struct {
	Name   string `yaml:"name" mapstructure:"name"`     // Display name (e.g. "litellm")
	URL    string `yaml:"url" mapstructure:"url"`       // Health check URL (e.g. "http://localhost:4000/health")
	Method string `yaml:"method" mapstructure:"method"` // HTTP method (default GET)
	Header string `yaml:"header" mapstructure:"header"` // Optional auth header value
}

// FeatureConfig holds configuration for enabling/disabling features.
type FeatureConfig struct {
	DID       DIDConfig       `yaml:"did" mapstructure:"did"`
	Connector ConnectorConfig `yaml:"connector" mapstructure:"connector"`
	Tracing   TracingConfig   `yaml:"tracing" mapstructure:"tracing"`
	Knowledge KnowledgeConfig `yaml:"knowledge" mapstructure:"knowledge"`
}

// KnowledgeConfig configures the native, scope-aware RAG knowledge store and its
// embedding provider. The embedding dimension is pinned in code (see
// internal/embedding) and is intentionally NOT configurable per-caller — the
// shared vector index is fixed-dimension.
type KnowledgeConfig struct {
	// Enabled turns the /api/v1/knowledge endpoints on. Default: true.
	Enabled *bool `yaml:"enabled" mapstructure:"enabled"`
	// Provider selects the embedder: "openai", "fake", or "" (auto: openai when
	// an API key is present, otherwise fake).
	Provider string `yaml:"provider" mapstructure:"provider"`
	// OpenAI holds OpenAI embedding-provider settings.
	OpenAI OpenAIEmbeddingConfig `yaml:"openai" mapstructure:"openai"`
}

// IsEnabled reports whether the knowledge store is enabled (default true).
func (c KnowledgeConfig) IsEnabled() bool {
	return c.Enabled == nil || *c.Enabled
}

// OpenAIEmbeddingConfig holds OpenAI embedding-provider settings. When APIKey is
// empty the knowledge store falls back to the deterministic FakeEmbedder so it
// works locally with zero external dependencies.
type OpenAIEmbeddingConfig struct {
	APIKey string `yaml:"api_key" mapstructure:"api_key"`
	Model  string `yaml:"model" mapstructure:"model"`
}

// TracingConfig holds configuration for OpenTelemetry distributed tracing.
type TracingConfig struct {
	Enabled     bool   `yaml:"enabled" mapstructure:"enabled"`           // Enable OTel trace export (default: false)
	Exporter    string `yaml:"exporter" mapstructure:"exporter"`         // "otlp-http" (default) or "otlp-grpc"
	Endpoint    string `yaml:"endpoint" mapstructure:"endpoint"`         // OTLP endpoint (default: "localhost:4318")
	ServiceName string `yaml:"service_name" mapstructure:"service_name"` // Service name for traces (default: "agentfield")
	Insecure    bool   `yaml:"insecure" mapstructure:"insecure"`         // Skip TLS verification
}

// ConnectorConfig holds configuration for the connector service integration.
type ConnectorConfig struct {
	Enabled      bool                           `yaml:"enabled" mapstructure:"enabled"`
	Token        string                         `yaml:"token" mapstructure:"token"`
	Capabilities map[string]ConnectorCapability `yaml:"capabilities" mapstructure:"capabilities"`
}

// ConnectorCapability defines whether a capability domain is enabled and its access mode.
type ConnectorCapability struct {
	Enabled  bool `yaml:"enabled" mapstructure:"enabled"`
	ReadOnly bool `yaml:"read_only" mapstructure:"read_only"`
}

// DIDConfig holds configuration for DID identity system.
type DIDConfig struct {
	Enabled          bool                `yaml:"enabled" mapstructure:"enabled" default:"true"`
	Method           string              `yaml:"method" mapstructure:"method" default:"did:key"`
	KeyAlgorithm     string              `yaml:"key_algorithm" mapstructure:"key_algorithm" default:"Ed25519"`
	DerivationMethod string              `yaml:"derivation_method" mapstructure:"derivation_method" default:"BIP32"`
	KeyRotationDays  int                 `yaml:"key_rotation_days" mapstructure:"key_rotation_days" default:"90"`
	VCRequirements   VCRequirements      `yaml:"vc_requirements" mapstructure:"vc_requirements"`
	Keystore         KeystoreConfig      `yaml:"keystore" mapstructure:"keystore"`
	Authorization    AuthorizationConfig `yaml:"authorization" mapstructure:"authorization"`
}

// AuthorizationConfig holds configuration for VC-based authorization.
type AuthorizationConfig struct {
	// Enabled determines if the authorization system is active
	Enabled bool `yaml:"enabled" mapstructure:"enabled" default:"false"`
	// DIDAuthEnabled enables DID-based authentication on API routes
	DIDAuthEnabled bool `yaml:"did_auth_enabled" mapstructure:"did_auth_enabled" default:"false"`
	// Domain is the domain used for did:web identifiers (e.g., "localhost:8080")
	Domain string `yaml:"domain" mapstructure:"domain" default:"localhost:8080"`
	// TimestampWindowSeconds is the allowed time drift for DID signature timestamps
	TimestampWindowSeconds int64 `yaml:"timestamp_window_seconds" mapstructure:"timestamp_window_seconds" default:"300"`
	// DefaultApprovalDurationHours is the default duration for permission approvals
	DefaultApprovalDurationHours int `yaml:"default_approval_duration_hours" mapstructure:"default_approval_duration_hours" default:"720"`
	// AdminToken is a separate token required for admin operations (tag approval,
	// policy management). When empty and InsecureDisableAdminAuth is false, admin
	// routes will reject requests at startup or return 401.
	AdminToken string `yaml:"admin_token" mapstructure:"admin_token"`
	// InsecureDisableAdminAuth explicitly permits running admin routes without
	// an admin token. This should only be enabled for trusted local development.
	InsecureDisableAdminAuth bool `yaml:"insecure_disable_admin_auth" mapstructure:"insecure_disable_admin_auth"`
	// InternalToken is sent as Authorization: Bearer header when the control plane
	// forwards execution requests to agents. Agents with RequireOriginAuth enabled
	// validate this token, preventing direct access to their HTTP ports.
	InternalToken string `yaml:"internal_token" mapstructure:"internal_token"`
	// TagApprovalRules configures how proposed tags are handled at registration time.
	// Default mode is "auto" (all tags auto-approved) for backward compatibility.
	TagApprovalRules TagApprovalRulesConfig `yaml:"tag_approval_rules" mapstructure:"tag_approval_rules"`
	// AccessPolicies defines tag-based authorization policies for cross-agent calls.
	AccessPolicies []AccessPolicyConfig `yaml:"access_policies" mapstructure:"access_policies"`
	// DefaultDeny, when true, causes the permission middleware to return 403 if
	// no access policy matches a request. Default false preserves the existing
	// behavior of allowing unmatched requests (backward compat for untagged agents).
	DefaultDeny bool `yaml:"default_deny" mapstructure:"default_deny" default:"false"`
}

// TagApprovalRulesConfig configures tag approval behavior at registration.
type TagApprovalRulesConfig struct {
	// DefaultMode is the approval mode for tags not matching any rule: "auto", "manual", or "forbidden".
	// Default: "auto" (backward compat — all tags auto-approved when no rules configured).
	DefaultMode string            `yaml:"default_mode" mapstructure:"default_mode"`
	Rules       []TagApprovalRule `yaml:"rules" mapstructure:"rules"`
}

// TagApprovalRule defines the approval mode for a set of tags.
type TagApprovalRule struct {
	Tags     []string `yaml:"tags" mapstructure:"tags"`
	Approval string   `yaml:"approval" mapstructure:"approval"` // "auto", "manual", "forbidden"
	Reason   string   `yaml:"reason" mapstructure:"reason"`
}

// AccessPolicyConfig defines a tag-based authorization policy for cross-agent calls.
type AccessPolicyConfig struct {
	Name           string                      `yaml:"name" mapstructure:"name"`
	CallerTags     []string                    `yaml:"caller_tags" mapstructure:"caller_tags"`
	TargetTags     []string                    `yaml:"target_tags" mapstructure:"target_tags"`
	AllowFunctions []string                    `yaml:"allow_functions" mapstructure:"allow_functions"`
	DenyFunctions  []string                    `yaml:"deny_functions" mapstructure:"deny_functions"`
	Constraints    map[string]ConstraintConfig `yaml:"constraints" mapstructure:"constraints"`
	Action         string                      `yaml:"action" mapstructure:"action"`     // "allow" or "deny"
	Priority       int                         `yaml:"priority" mapstructure:"priority"` // higher = evaluated first
}

// ConstraintConfig defines a parameter constraint for a policy.
type ConstraintConfig struct {
	Operator string `yaml:"operator" mapstructure:"operator"` // "<=", ">=", "==", "!=", "<", ">"
	Value    any    `yaml:"value" mapstructure:"value"`
}

// VCRequirements holds VC generation requirements.
type VCRequirements struct {
	RequireVCForRegistration bool   `yaml:"require_vc_registration" mapstructure:"require_vc_registration" default:"true"`
	RequireVCForExecution    bool   `yaml:"require_vc_execution" mapstructure:"require_vc_execution" default:"true"`
	RequireVCForCrossAgent   bool   `yaml:"require_vc_cross_agent" mapstructure:"require_vc_cross_agent" default:"true"`
	StoreInputOutput         bool   `yaml:"store_input_output" mapstructure:"store_input_output" default:"false"`
	HashSensitiveData        bool   `yaml:"hash_sensitive_data" mapstructure:"hash_sensitive_data" default:"true"`
	PersistExecutionVC       bool   `yaml:"persist_execution_vc" mapstructure:"persist_execution_vc" default:"true"`
	StorageMode              string `yaml:"storage_mode" mapstructure:"storage_mode" default:"inline"`
}

// KeystoreConfig holds keystore configuration.
type KeystoreConfig struct {
	Type                 string `yaml:"type" mapstructure:"type" default:"local"`
	Path                 string `yaml:"path" mapstructure:"path" default:"./data/keys"`
	Encryption           string `yaml:"encryption" mapstructure:"encryption" default:"AES-256-GCM"`
	EncryptionPassphrase string `yaml:"encryption_passphrase" mapstructure:"encryption_passphrase"`
	BackupEnabled        bool   `yaml:"backup_enabled" mapstructure:"backup_enabled" default:"true"`
	BackupInterval       string `yaml:"backup_interval" mapstructure:"backup_interval" default:"24h"`
}

// APIConfig holds configuration for API settings
type APIConfig struct {
	CORS CORSConfig `yaml:"cors" mapstructure:"cors"`
	Auth AuthConfig `yaml:"auth" mapstructure:"auth"`
}

// CORSConfig holds CORS configuration
type CORSConfig struct {
	AllowedOrigins   []string `yaml:"allowed_origins" mapstructure:"allowed_origins"`
	AllowedMethods   []string `yaml:"allowed_methods" mapstructure:"allowed_methods"`
	AllowedHeaders   []string `yaml:"allowed_headers" mapstructure:"allowed_headers"`
	ExposedHeaders   []string `yaml:"exposed_headers" mapstructure:"exposed_headers"`
	AllowCredentials bool     `yaml:"allow_credentials" mapstructure:"allow_credentials"`
}

// AuthConfig holds API authentication configuration.
type AuthConfig struct {
	// APIKey is checked against headers or query params.
	APIKey string `yaml:"api_key" mapstructure:"api_key"`
	// InsecureDisableAuth explicitly permits running without API-key authentication.
	// This should only be enabled for trusted local development environments.
	InsecureDisableAuth bool `yaml:"insecure_disable_auth" mapstructure:"insecure_disable_auth"`
	// SkipPaths allows bypassing auth for specific endpoints (e.g., health).
	SkipPaths []string `yaml:"skip_paths" mapstructure:"skip_paths"`
}

// StorageConfig is an alias of the storage layer's configuration so callers can
// work with a single definition while keeping the canonical struct colocated
// with the implementation in the storage package.
type StorageConfig = storage.StorageConfig

// DefaultConfigPath is the default path for the af configuration file.
const DefaultConfigPath = "agentfield.yaml" // Or "./agentfield.yaml", "config/agentfield.yaml" depending on convention

// LoadConfig reads the configuration from the given path or default paths.
func LoadConfig(configPath string) (*Config, error) {
	if configPath == "" {
		configPath = DefaultConfigPath
	}

	// Check if the specific path exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// Fallback: try to find it in common locations relative to executable or CWD
		// This part might need more sophisticated logic depending on project structure
		// For now, let's assume configPath is either absolute or relative to CWD.
		// If not found, try a common "config/" subdirectory
		altPath := filepath.Join("config", "agentfield.yaml")
		if _, err2 := os.Stat(altPath); err2 == nil {
			configPath = altPath
		} else {
			// If still not found, return the original error for the specified/default path
			return nil, fmt.Errorf("configuration file not found at %s or default locations: %w", configPath, err)
		}
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read configuration file %s: %w", configPath, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse configuration file %s: %w", configPath, err)
	}

	ApplyDefaults(&cfg)

	// Apply environment variable overrides
	ApplyEnvOverrides(&cfg)

	return &cfg, nil
}

// ApplyDefaults fills values that should be stable across config loaders.
func ApplyDefaults(cfg *Config) {
	if cfg.AgentField.ARD.Publish.DefaultType == "" {
		cfg.AgentField.ARD.Publish.DefaultType = "application/openapi+json"
	}
	if len(cfg.AgentField.ARD.Publish.IncludeHealthStatuses) == 0 {
		cfg.AgentField.ARD.Publish.IncludeHealthStatuses = []string{"active", "unknown"}
	}
	if cfg.AgentField.ARD.External.DefaultSearchLimit <= 0 {
		cfg.AgentField.ARD.External.DefaultSearchLimit = 10
	}
	if cfg.Telemetry.Mode == "" {
		cfg.Telemetry.Mode = "anonymous"
	}
	if cfg.Telemetry.Endpoint == "" {
		cfg.Telemetry.Endpoint = "https://agentfield.ai/api/oss/telemetry"
	}
	if cfg.Telemetry.Timeout <= 0 {
		cfg.Telemetry.Timeout = 800 * time.Millisecond
	}
}

// ApplyEnvOverrides applies environment variable overrides to the config.
// Environment variables take precedence over YAML config values.
// Exported so the main server startup (which uses Viper for file loading)
// can call it after Viper unmarshal to apply the shorter env var names.
func ApplyEnvOverrides(cfg *Config) {
	// Knowledge / embedding overrides. OPENAI_API_KEY is the standard OpenAI env
	// var; AGENTFIELD_KNOWLEDGE_* allow explicit control.
	if val := os.Getenv("OPENAI_API_KEY"); val != "" && cfg.Features.Knowledge.OpenAI.APIKey == "" {
		cfg.Features.Knowledge.OpenAI.APIKey = strings.TrimSpace(val)
	}
	if val := os.Getenv("AGENTFIELD_KNOWLEDGE_OPENAI_API_KEY"); val != "" {
		cfg.Features.Knowledge.OpenAI.APIKey = strings.TrimSpace(val)
	}
	if val := os.Getenv("AGENTFIELD_KNOWLEDGE_PROVIDER"); val != "" {
		cfg.Features.Knowledge.Provider = strings.TrimSpace(val)
	}
	if val := os.Getenv("AGENTFIELD_KNOWLEDGE_OPENAI_MODEL"); val != "" {
		cfg.Features.Knowledge.OpenAI.Model = strings.TrimSpace(val)
	}
	if val := os.Getenv("AGENTFIELD_KNOWLEDGE_ENABLED"); val != "" {
		enabled := parseEnvBool(val)
		cfg.Features.Knowledge.Enabled = &enabled
	}

	if val := os.Getenv("AGENTFIELD_ARD_ENABLED"); val != "" {
		cfg.AgentField.ARD.Enabled = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_PUBLIC_BASE_URL"); val != "" {
		cfg.AgentField.ARD.PublicBaseURL = strings.TrimRight(strings.TrimSpace(val), "/")
	}
	if val := os.Getenv("AGENTFIELD_ARD_PUBLISHER_DOMAIN"); val != "" {
		cfg.AgentField.ARD.PublisherDomain = strings.TrimSpace(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_PUBLISH_ENABLED"); val != "" {
		cfg.AgentField.ARD.Publish.Enabled = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_REGISTRY_ENABLED"); val != "" {
		cfg.AgentField.ARD.Registry.Enabled = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_REGISTRY_PUBLIC"); val != "" {
		cfg.AgentField.ARD.Registry.Public = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_EXTERNAL_SEARCH_ENABLED"); val != "" {
		cfg.AgentField.ARD.External.SearchEnabled = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_EXTERNAL_INVOCATION_ENABLED"); val != "" {
		cfg.AgentField.ARD.External.InvocationEnabled = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_ARD_EXTERNAL_ALLOWED_REGISTRIES"); val != "" {
		cfg.AgentField.ARD.External.AllowedRegistries = splitEnvCSV(val)
	}

	// API Authentication
	if apiKey := os.Getenv("AGENTFIELD_API_KEY"); apiKey != "" {
		cfg.API.Auth.APIKey = apiKey
	}
	// Also support the nested path format for consistency
	if apiKey := os.Getenv("AGENTFIELD_API_AUTH_API_KEY"); apiKey != "" {
		cfg.API.Auth.APIKey = apiKey
	}
	if val, ok := os.LookupEnv("AGENTFIELD_INSECURE_DISABLE_AUTH"); ok {
		cfg.API.Auth.InsecureDisableAuth = parseEnvBool(val)
	}
	// Also support the nested path format for consistency.
	if val, ok := os.LookupEnv("AGENTFIELD_API_AUTH_INSECURE_DISABLE_AUTH"); ok {
		cfg.API.Auth.InsecureDisableAuth = parseEnvBool(val)
	}

	if val := os.Getenv("AGENTFIELD_REGISTRATION_SERVERLESS_DISCOVERY_ALLOWED_HOSTS"); val != "" {
		parts := strings.Split(val, ",")
		cfg.AgentField.Registration.ServerlessDiscoveryAllowedHosts = cfg.AgentField.Registration.ServerlessDiscoveryAllowedHosts[:0]
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed != "" {
				cfg.AgentField.Registration.ServerlessDiscoveryAllowedHosts = append(cfg.AgentField.Registration.ServerlessDiscoveryAllowedHosts, trimmed)
			}
		}
	}

	if val := os.Getenv("AGENTFIELD_WEBHOOK_ALLOWED_HOSTS"); val != "" {
		parts := strings.Split(val, ",")
		cfg.AgentField.Registration.WebhookAllowedHosts = cfg.AgentField.Registration.WebhookAllowedHosts[:0]
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed != "" {
				cfg.AgentField.Registration.WebhookAllowedHosts = append(cfg.AgentField.Registration.WebhookAllowedHosts, trimmed)
			}
		}
	}

	// Node health monitoring overrides
	if val := os.Getenv("AGENTFIELD_HEALTH_CHECK_INTERVAL"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeHealth.CheckInterval = d
		}
	}
	if val := os.Getenv("AGENTFIELD_HEALTH_CHECK_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeHealth.CheckTimeout = d
		}
	}
	if val := os.Getenv("AGENTFIELD_HEALTH_CONSECUTIVE_FAILURES"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.NodeHealth.ConsecutiveFailures = i
		}
	}
	if val := os.Getenv("AGENTFIELD_HEALTH_RECOVERY_DEBOUNCE"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeHealth.RecoveryDebounce = d
		}
	}
	if val := os.Getenv("AGENTFIELD_HEARTBEAT_STALE_THRESHOLD"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeHealth.HeartbeatStaleThreshold = d
		}
	}

	// LLM health monitoring overrides
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_ENABLED"); val != "" {
		cfg.AgentField.LLMHealth.Enabled = val == "true" || val == "1"
	}
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_CHECK_INTERVAL"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.LLMHealth.CheckInterval = d
		}
	}
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_CHECK_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.LLMHealth.CheckTimeout = d
		}
	}
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_FAILURE_THRESHOLD"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.LLMHealth.FailureThreshold = i
		}
	}
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_RECOVERY_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.LLMHealth.RecoveryTimeout = d
		}
	}
	// Single LLM endpoint via env var (convenience for simple setups)
	if val := os.Getenv("AGENTFIELD_LLM_HEALTH_ENDPOINT"); val != "" {
		name := os.Getenv("AGENTFIELD_LLM_HEALTH_ENDPOINT_NAME")
		if name == "" {
			name = "default"
		}
		cfg.AgentField.LLMHealth.Endpoints = append(cfg.AgentField.LLMHealth.Endpoints, LLMEndpoint{
			Name: name,
			URL:  val,
		})
	}

	// Execution queue overrides
	if val := os.Getenv("AGENTFIELD_MAX_CONCURRENT_PER_AGENT"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.ExecutionQueue.MaxConcurrentPerAgent = i
		}
	}

	// Execution retry overrides
	if val := os.Getenv("AGENTFIELD_EXECUTION_MAX_RETRIES"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.ExecutionCleanup.MaxRetries = i
		}
	}
	if val := os.Getenv("AGENTFIELD_EXECUTION_RETRY_BACKOFF"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.ExecutionCleanup.RetryBackoff = d
		}
	}

	// Authorization overrides
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_ENABLED"); val != "" {
		cfg.Features.DID.Authorization.Enabled = val == "true" || val == "1"
	}
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_DID_AUTH_ENABLED"); val != "" {
		cfg.Features.DID.Authorization.DIDAuthEnabled = val == "true" || val == "1"
	}
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_DOMAIN"); val != "" {
		cfg.Features.DID.Authorization.Domain = val
	}
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN"); val != "" {
		cfg.Features.DID.Authorization.AdminToken = val
	}
	if val, ok := os.LookupEnv("AGENTFIELD_INSECURE_ADMIN_NO_TOKEN"); ok {
		cfg.Features.DID.Authorization.InsecureDisableAdminAuth = parseEnvBool(val)
	}
	// Also support the nested path format for consistency.
	if val, ok := os.LookupEnv("AGENTFIELD_AUTHORIZATION_INSECURE_DISABLE_ADMIN_AUTH"); ok {
		cfg.Features.DID.Authorization.InsecureDisableAdminAuth = parseEnvBool(val)
	}
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN"); val != "" {
		cfg.Features.DID.Authorization.InternalToken = val
	}
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_DEFAULT_DENY"); val != "" {
		cfg.Features.DID.Authorization.DefaultDeny = val == "true" || val == "1"
	}

	// Node log proxy (UI → agent NDJSON)
	if val := os.Getenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeLogProxy.ConnectTimeout = d
		}
	}
	if val := os.Getenv("AGENTFIELD_NODE_LOG_PROXY_STREAM_IDLE_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeLogProxy.StreamIdleTimeout = d
		}
	}
	if val := os.Getenv("AGENTFIELD_NODE_LOG_PROXY_MAX_DURATION"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.NodeLogProxy.MaxStreamDuration = d
		}
	}
	if val := os.Getenv("AGENTFIELD_NODE_LOG_MAX_TAIL_LINES"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.NodeLogProxy.MaxTailLines = i
		}
	}

	// Structured execution log storage and streaming
	if val := os.Getenv("AGENTFIELD_EXECUTION_LOG_RETENTION_PERIOD"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.ExecutionLogs.RetentionPeriod = d
		}
	}
	if val := os.Getenv("AGENTFIELD_EXECUTION_LOG_MAX_ENTRIES_PER_EXECUTION"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.ExecutionLogs.MaxEntriesPerExecution = i
		}
	}
	if val := os.Getenv("AGENTFIELD_EXECUTION_LOG_MAX_TAIL_ENTRIES"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.ExecutionLogs.MaxTailEntries = i
		}
	}
	if val := os.Getenv("AGENTFIELD_EXECUTION_LOG_STREAM_IDLE_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.ExecutionLogs.StreamIdleTimeout = d
		}
	}
	if val := os.Getenv("AGENTFIELD_EXECUTION_LOG_MAX_DURATION"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.AgentField.ExecutionLogs.MaxStreamDuration = d
		}
	}

	// Approval workflow overrides
	if val := os.Getenv("AGENTFIELD_APPROVAL_WEBHOOK_SECRET"); val != "" {
		cfg.AgentField.Approval.WebhookSecret = val
	}
	if val := os.Getenv("AGENTFIELD_APPROVAL_DEFAULT_EXPIRY_HOURS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			cfg.AgentField.Approval.DefaultExpiryHours = i
		}
	}

	// OpenTelemetry tracing overrides (also supports standard OTEL_* env vars)
	if val := os.Getenv("AGENTFIELD_TRACING_ENABLED"); val != "" {
		cfg.Features.Tracing.Enabled = val == "true" || val == "1"
	}
	if val := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); val != "" {
		cfg.Features.Tracing.Endpoint = val
		cfg.Features.Tracing.Enabled = true
	}
	if val := os.Getenv("OTEL_SERVICE_NAME"); val != "" {
		cfg.Features.Tracing.ServiceName = val
	}
	if val := os.Getenv("AGENTFIELD_TRACING_INSECURE"); val != "" {
		cfg.Features.Tracing.Insecure = val == "true" || val == "1"
	}

	// Anonymous OSS usage telemetry overrides.
	if val := os.Getenv("AGENTFIELD_TELEMETRY_ENABLED"); val != "" {
		enabled := val == "true" || val == "1"
		cfg.Telemetry.Enabled = &enabled
	}
	if val := os.Getenv("AGENTFIELD_TELEMETRY_ENDPOINT"); val != "" {
		cfg.Telemetry.Endpoint = val
	}
	if val := os.Getenv("AGENTFIELD_TELEMETRY_INSTALL_ID"); val != "" {
		cfg.Telemetry.InstallID = val
	}
	if val := os.Getenv("AGENTFIELD_TELEMETRY_INSTALL_ID_PATH"); val != "" {
		cfg.Telemetry.InstallIDPath = val
	}
	if val := os.Getenv("AGENTFIELD_TELEMETRY_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			cfg.Telemetry.Timeout = d
		}
	}

	// Connector overrides
	if val := os.Getenv("AGENTFIELD_CONNECTOR_ENABLED"); val != "" {
		cfg.Features.Connector.Enabled = val == "true" || val == "1"
	}
	if val := os.Getenv("AGENTFIELD_CONNECTOR_TOKEN"); val != "" {
		cfg.Features.Connector.Token = val
	}
	// Connector capability overrides (true / false / readonly)
	connectorCapEnvMap := map[string]string{
		"AGENTFIELD_CONNECTOR_CAP_POLICY_MANAGEMENT":    "policy_management",
		"AGENTFIELD_CONNECTOR_CAP_TAG_MANAGEMENT":       "tag_management",
		"AGENTFIELD_CONNECTOR_CAP_DID_MANAGEMENT":       "did_management",
		"AGENTFIELD_CONNECTOR_CAP_REASONER_MANAGEMENT":  "reasoner_management",
		"AGENTFIELD_CONNECTOR_CAP_STATUS_READ":          "status_read",
		"AGENTFIELD_CONNECTOR_CAP_OBSERVABILITY_CONFIG": "observability_config",
		"AGENTFIELD_CONNECTOR_CAP_CONFIG_MANAGEMENT":    "config_management",
	}
	for envKey, capName := range connectorCapEnvMap {
		if val := os.Getenv(envKey); val != "" {
			if cfg.Features.Connector.Capabilities == nil {
				cfg.Features.Connector.Capabilities = make(map[string]ConnectorCapability)
			}
			switch strings.ToLower(val) {
			case "true":
				cfg.Features.Connector.Capabilities[capName] = ConnectorCapability{Enabled: true, ReadOnly: false}
			case "readonly":
				cfg.Features.Connector.Capabilities[capName] = ConnectorCapability{Enabled: true, ReadOnly: true}
			default:
				cfg.Features.Connector.Capabilities[capName] = ConnectorCapability{Enabled: false}
			}
		}
	}
}

func parseEnvBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "y", "on", "enabled":
		return true
	default:
		return false
	}
}

func splitEnvCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
