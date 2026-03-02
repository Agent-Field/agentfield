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
	NodeHealth       NodeHealthConfig       `yaml:"node_health" mapstructure:"node_health"`
	ExecutionCleanup ExecutionCleanupConfig `yaml:"execution_cleanup" mapstructure:"execution_cleanup"`
	ExecutionQueue   ExecutionQueueConfig   `yaml:"execution_queue" mapstructure:"execution_queue"`
	Approval         ApprovalConfig         `yaml:"approval" mapstructure:"approval"`
}

// ApprovalConfig holds configuration for the execution approval workflow.
// The control plane manages execution state only — agents are responsible for
// communicating with external approval services (e.g. hax-sdk).
type ApprovalConfig struct {
	WebhookSecret      string `yaml:"webhook_secret" mapstructure:"webhook_secret"`             // Optional HMAC-SHA256 secret for verifying webhook callbacks
	DefaultExpiryHours int    `yaml:"default_expiry_hours" mapstructure:"default_expiry_hours"` // Default approval expiry (hours); 0 = 72h
}

// NodeHealthConfig holds configuration for agent node health monitoring.
// Zero values are treated as "use default" — set explicitly to override.
type NodeHealthConfig struct {
	CheckInterval           time.Duration `yaml:"check_interval" mapstructure:"check_interval"`                       // How often to HTTP health check nodes (0 = default 10s)
	CheckTimeout            time.Duration `yaml:"check_timeout" mapstructure:"check_timeout"`                         // Timeout per HTTP health check (0 = default 5s)
	ConsecutiveFailures     int           `yaml:"consecutive_failures" mapstructure:"consecutive_failures"`            // Failures before marking inactive (0 = default 3; set 1 for instant)
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
}

// ExecutionQueueConfig configures execution and webhook settings.
type ExecutionQueueConfig struct {
	AgentCallTimeout       time.Duration `yaml:"agent_call_timeout" mapstructure:"agent_call_timeout"`
	WebhookTimeout         time.Duration `yaml:"webhook_timeout" mapstructure:"webhook_timeout"`
	WebhookMaxAttempts     int           `yaml:"webhook_max_attempts" mapstructure:"webhook_max_attempts"`
	WebhookRetryBackoff    time.Duration `yaml:"webhook_retry_backoff" mapstructure:"webhook_retry_backoff"`
	WebhookMaxRetryBackoff time.Duration `yaml:"webhook_max_retry_backoff" mapstructure:"webhook_max_retry_backoff"`
}

// FeatureConfig holds configuration for enabling/disabling features.
type FeatureConfig struct {
	DID       DIDConfig       `yaml:"did" mapstructure:"did"`
	Connector ConnectorConfig `yaml:"connector" mapstructure:"connector"`
}

// ConnectorConfig holds configuration for the connector service integration.
type ConnectorConfig struct {
	Enabled      bool                              `yaml:"enabled" mapstructure:"enabled"`
	Token        string                            `yaml:"token" mapstructure:"token"`
	Capabilities map[string]ConnectorCapability     `yaml:"capabilities" mapstructure:"capabilities"`
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
	// policy management). If empty, admin routes fall back to the standard API key.
	AdminToken string `yaml:"admin_token" mapstructure:"admin_token"`
	// InternalToken is sent as Authorization: Bearer header when the control plane
	// forwards execution requests to agents. Agents with RequireOriginAuth enabled
	// validate this token, preventing direct access to their HTTP ports.
	InternalToken string `yaml:"internal_token" mapstructure:"internal_token"`
	// TagApprovalRules configures how proposed tags are handled at registration time.
	// Default mode is "auto" (all tags auto-approved) for backward compatibility.
	TagApprovalRules TagApprovalRulesConfig `yaml:"tag_approval_rules" mapstructure:"tag_approval_rules"`
	// AccessPolicies defines tag-based authorization policies for cross-agent calls.
	AccessPolicies []AccessPolicyConfig `yaml:"access_policies" mapstructure:"access_policies"`
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
	Name           string                        `yaml:"name" mapstructure:"name"`
	CallerTags     []string                      `yaml:"caller_tags" mapstructure:"caller_tags"`
	TargetTags     []string                      `yaml:"target_tags" mapstructure:"target_tags"`
	AllowFunctions []string                      `yaml:"allow_functions" mapstructure:"allow_functions"`
	DenyFunctions  []string                      `yaml:"deny_functions" mapstructure:"deny_functions"`
	Constraints    map[string]ConstraintConfig    `yaml:"constraints" mapstructure:"constraints"`
	Action         string                        `yaml:"action" mapstructure:"action"`     // "allow" or "deny"
	Priority       int                           `yaml:"priority" mapstructure:"priority"` // higher = evaluated first
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
	// APIKey is checked against headers or query params. Empty disables auth.
	APIKey string `yaml:"api_key" mapstructure:"api_key"`
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

	// Apply environment variable overrides
	applyEnvOverrides(&cfg)

	return &cfg, nil
}

// applyEnvOverrides applies environment variable overrides to the config.
// Environment variables take precedence over YAML config values.
func applyEnvOverrides(cfg *Config) {
	// API Authentication
	if apiKey := os.Getenv("AGENTFIELD_API_KEY"); apiKey != "" {
		cfg.API.Auth.APIKey = apiKey
	}
	// Also support the nested path format for consistency
	if apiKey := os.Getenv("AGENTFIELD_API_AUTH_API_KEY"); apiKey != "" {
		cfg.API.Auth.APIKey = apiKey
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
	if val := os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN"); val != "" {
		cfg.Features.DID.Authorization.InternalToken = val
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

	// Connector overrides
	if val := os.Getenv("AGENTFIELD_CONNECTOR_ENABLED"); val != "" {
		cfg.Features.Connector.Enabled = val == "true" || val == "1"
	}
	if val := os.Getenv("AGENTFIELD_CONNECTOR_TOKEN"); val != "" {
		cfg.Features.Connector.Token = val
	}
	// Connector capability overrides (true / false / readonly)
	connectorCapEnvMap := map[string]string{
		"AGENTFIELD_CONNECTOR_CAP_POLICY_MANAGEMENT":   "policy_management",
		"AGENTFIELD_CONNECTOR_CAP_TAG_MANAGEMENT":      "tag_management",
		"AGENTFIELD_CONNECTOR_CAP_DID_MANAGEMENT":      "did_management",
		"AGENTFIELD_CONNECTOR_CAP_REASONER_MANAGEMENT": "reasoner_management",
		"AGENTFIELD_CONNECTOR_CAP_STATUS_READ":          "status_read",
		"AGENTFIELD_CONNECTOR_CAP_OBSERVABILITY_CONFIG": "observability_config",
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
