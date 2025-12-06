package config

import (
	"fmt"           // Added for fmt.Errorf
	"os"            // Added for os.Stat, os.ReadFile
	"path/filepath" // Added for filepath.Join
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
	ExecutionCleanup ExecutionCleanupConfig `yaml:"execution_cleanup" mapstructure:"execution_cleanup"`
	ExecutionQueue   ExecutionQueueConfig   `yaml:"execution_queue" mapstructure:"execution_queue"`
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
	DID DIDConfig `yaml:"did"`
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
}
