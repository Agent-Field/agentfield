package config

import (
	"fmt"              // Added for fmt.Errorf
	"gopkg.in/yaml.v3" // Added for yaml.Unmarshal
	"os"               // Added for os.Stat, os.ReadFile
	"path/filepath"    // Added for filepath.Join
	"time"

	"github.com/your-org/agentfield/control-plane/internal/storage"
)

// Config holds the entire configuration for the AgentField server.
type Config struct {
	AgentField      AgentFieldConfig      `yaml:"agentfield" mapstructure:"agentfield"`
	Agents          AgentsConfig          `yaml:"agents" mapstructure:"agents"`
	Features        FeatureConfig         `yaml:"features" mapstructure:"features"`
	Storage         StorageConfig         `yaml:"storage" mapstructure:"storage"`                   // Added storage config
	UI              UIConfig              `yaml:"ui" mapstructure:"ui"`                             // Added UI config
	API             APIConfig             `yaml:"api" mapstructure:"api"`                           // Added API config
	DataDirectories DataDirectoriesConfig `yaml:"data_directories" mapstructure:"data_directories"` // Added data directories config
}

// UIConfig holds configuration for the web UI.
type UIConfig struct {
	Enabled    bool   `yaml:"enabled" mapstructure:"enabled"`
	Mode       string `yaml:"mode" mapstructure:"mode"`               // "embedded", "dev", "separate"
	SourcePath string `yaml:"source_path" mapstructure:"source_path"` // Path to UI source for building
	DistPath   string `yaml:"dist_path" mapstructure:"dist_path"`     // Path to built UI assets for serving
	DevPort    int    `yaml:"dev_port" mapstructure:"dev_port"`       // Port for UI dev server
	BackendURL string `yaml:"backend_url" mapstructure:"backend_url"` // URL of the backend if UI is separate
}

// AgentFieldConfig holds the core AgentField server configuration.
type AgentFieldConfig struct {
	Port                    int                    `yaml:"port"`
	DatabaseURL             string                 `yaml:"database_url"`
	MaxConcurrentRequests   int                    `yaml:"max_concurrent_requests"`
	RequestTimeout          time.Duration          `yaml:"request_timeout"`
	CircuitBreakerThreshold int                    `yaml:"circuit_breaker_threshold"`
	Mode                    string                 `yaml:"mode"`
	ExecutionCleanup        ExecutionCleanupConfig `yaml:"execution_cleanup" mapstructure:"execution_cleanup"`
	ExecutionQueue          ExecutionQueueConfig   `yaml:"execution_queue" mapstructure:"execution_queue"`
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

// ExecutionQueueConfig configures the durable execution worker pool.
type ExecutionQueueConfig struct {
	WorkerCount            int           `yaml:"worker_count" mapstructure:"worker_count"`
	RequestTimeout         time.Duration `yaml:"request_timeout" mapstructure:"request_timeout"`
	AgentCallTimeout       time.Duration `yaml:"agent_call_timeout" mapstructure:"agent_call_timeout"`
	LeaseDuration          time.Duration `yaml:"lease_duration" mapstructure:"lease_duration"`
	MaxAttempts            int           `yaml:"max_attempts" mapstructure:"max_attempts"`
	FailureBackoff         time.Duration `yaml:"failure_backoff" mapstructure:"failure_backoff"`
	MaxFailureBackoff      time.Duration `yaml:"max_failure_backoff" mapstructure:"max_failure_backoff"`
	PollInterval           time.Duration `yaml:"poll_interval" mapstructure:"poll_interval"`
	ResultPreviewBytes     int           `yaml:"result_preview_bytes" mapstructure:"result_preview_bytes"`
	QueueSoftLimit         int           `yaml:"queue_soft_limit" mapstructure:"queue_soft_limit"`
	WaiterMapLimit         int           `yaml:"waiter_map_limit" mapstructure:"waiter_map_limit"`
	WebhookTimeout         time.Duration `yaml:"webhook_timeout" mapstructure:"webhook_timeout"`
	WebhookMaxAttempts     int           `yaml:"webhook_max_attempts" mapstructure:"webhook_max_attempts"`
	WebhookRetryBackoff    time.Duration `yaml:"webhook_retry_backoff" mapstructure:"webhook_retry_backoff"`
	WebhookMaxRetryBackoff time.Duration `yaml:"webhook_max_retry_backoff" mapstructure:"webhook_max_retry_backoff"`
}

// AgentsConfig holds configuration related to agent management.
type AgentsConfig struct {
	Discovery DiscoveryConfig `yaml:"discovery"`
	Scaling   ScalingConfig   `yaml:"scaling"`
}

// DiscoveryConfig holds configuration for agent discovery.
type DiscoveryConfig struct {
	ScanInterval        time.Duration `yaml:"scan_interval"`
	HealthCheckInterval time.Duration `yaml:"health_check_interval"`
}

// ScalingConfig holds configuration for agent scaling.
type ScalingConfig struct {
	AutoScale   bool `yaml:"auto_scale"`
	MinReplicas int  `yaml:"min_replicas"`
	MaxReplicas int  `yaml:"max_replicas"`
}

// FeatureConfig holds configuration for enabling/disabling features.
type FeatureConfig struct {
	Core       CoreFeatures       `yaml:"core"`
	Enterprise EnterpriseFeatures `yaml:"enterprise"`
	DID        DIDConfig          `yaml:"did"`
}

// CoreFeatures holds configuration for core features.
type CoreFeatures struct {
	MemoryManagement bool `yaml:"memory_management" default:"true"`
	ExecutionLogging bool `yaml:"execution_logging" default:"true"`
	BasicMetrics     bool `yaml:"basic_metrics" default:"true"`
	WebSocketSupport bool `yaml:"websocket_support" default:"true"`
}

// EnterpriseFeatures holds configuration for enterprise features.
type EnterpriseFeatures struct {
	Enabled         bool `yaml:"enabled" default:"false"`
	ComplianceMode  bool `yaml:"compliance_mode" default:"false"`
	AuditLogging    bool `yaml:"audit_logging" default:"false"`
	RoleBasedAccess bool `yaml:"role_based_access" default:"false"`
	DataEncryption  bool `yaml:"data_encryption" default:"false"`
}

// DIDConfig holds configuration for DID identity system.
type DIDConfig struct {
	Enabled          bool           `yaml:"enabled" default:"true"`
	Method           string         `yaml:"method" default:"did:key"`
	KeyAlgorithm     string         `yaml:"key_algorithm" default:"Ed25519"`
	DerivationMethod string         `yaml:"derivation_method" default:"BIP32"`
	KeyRotationDays  int            `yaml:"key_rotation_days" default:"90"`
	VCRequirements   VCRequirements `yaml:"vc_requirements"`
	Keystore         KeystoreConfig `yaml:"keystore"`
}

// VCRequirements holds VC generation requirements.
type VCRequirements struct {
	RequireVCForRegistration bool   `yaml:"require_vc_registration" default:"true"`
	RequireVCForExecution    bool   `yaml:"require_vc_execution" default:"true"`
	RequireVCForCrossAgent   bool   `yaml:"require_vc_cross_agent" default:"true"`
	StoreInputOutput         bool   `yaml:"store_input_output" default:"false"`
	HashSensitiveData        bool   `yaml:"hash_sensitive_data" default:"true"`
	PersistExecutionVC       bool   `yaml:"persist_execution_vc" default:"true"`
	StorageMode              string `yaml:"storage_mode" default:"inline"`
}

// KeystoreConfig holds keystore configuration.
type KeystoreConfig struct {
	Type           string `yaml:"type" default:"local"`
	Path           string `yaml:"path" default:"./data/keys"`
	Encryption     string `yaml:"encryption" default:"AES-256-GCM"`
	BackupEnabled  bool   `yaml:"backup_enabled" default:"true"`
	BackupInterval string `yaml:"backup_interval" default:"24h"`
}

// APIConfig holds configuration for API settings
type APIConfig struct {
	CORS CORSConfig `yaml:"cors" mapstructure:"cors"`
}

// CORSConfig holds CORS configuration
type CORSConfig struct {
	AllowedOrigins   []string `yaml:"allowed_origins" mapstructure:"allowed_origins"`
	AllowedMethods   []string `yaml:"allowed_methods" mapstructure:"allowed_methods"`
	AllowedHeaders   []string `yaml:"allowed_headers" mapstructure:"allowed_headers"`
	ExposedHeaders   []string `yaml:"exposed_headers" mapstructure:"exposed_headers"`
	AllowCredentials bool     `yaml:"allow_credentials" mapstructure:"allow_credentials"`
}

// StorageConfig is an alias of the storage layer's configuration so callers can
// work with a single definition while keeping the canonical struct colocated
// with the implementation in the storage package.
type StorageConfig = storage.StorageConfig

// DataDirectoriesConfig holds configuration for AgentField data directory paths
type DataDirectoriesConfig struct {
	AgentFieldHome   string `yaml:"agentfield_home" mapstructure:"agentfield_home"`       // Can be overridden by AGENTFIELD_HOME env var
	DatabaseDir      string `yaml:"database_dir" mapstructure:"database_dir"`             // Relative to agentfield_home
	KeysDir          string `yaml:"keys_dir" mapstructure:"keys_dir"`                     // Relative to agentfield_home
	DIDRegistriesDir string `yaml:"did_registries_dir" mapstructure:"did_registries_dir"` // Relative to agentfield_home
	VCsDir           string `yaml:"vcs_dir" mapstructure:"vcs_dir"`                       // Relative to agentfield_home
	AgentsDir        string `yaml:"agents_dir" mapstructure:"agents_dir"`                 // Relative to agentfield_home
	LogsDir          string `yaml:"logs_dir" mapstructure:"logs_dir"`                     // Relative to agentfield_home
	ConfigDir        string `yaml:"config_dir" mapstructure:"config_dir"`                 // Relative to agentfield_home
	TempDir          string `yaml:"temp_dir" mapstructure:"temp_dir"`                     // Relative to agentfield_home
}

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

	// Apply defaults or perform validation if needed here
	// e.g., cfg.UI.Mode = "embedded" if cfg.UI.Mode == ""

	return &cfg, nil
}
