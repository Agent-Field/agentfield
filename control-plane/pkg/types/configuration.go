package types

import (
	"encoding/json"
	"time"
)

// AgentConfiguration represents a stored configuration for an agent package
type AgentConfiguration struct {
	ID              int64                  `json:"id" db:"id"`
	AgentID         string                 `json:"agent_id" db:"agent_id"`
	PackageID       string                 `json:"package_id" db:"package_id"`
	Configuration   map[string]interface{} `json:"configuration" db:"configuration"`
	EncryptedFields []string               `json:"encrypted_fields" db:"encrypted_fields"`
	Status          ConfigurationStatus    `json:"status" db:"status"`
	Version         int                    `json:"version" db:"version"`
	CreatedAt       time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at" db:"updated_at"`
	CreatedBy       *string                `json:"created_by,omitempty" db:"created_by"`
	UpdatedBy       *string                `json:"updated_by,omitempty" db:"updated_by"`
}

// ConfigurationStatus represents the status of an agent configuration
type ConfigurationStatus string

const (
	ConfigurationStatusDraft    ConfigurationStatus = "draft"
	ConfigurationStatusActive   ConfigurationStatus = "active"
	ConfigurationStatusInactive ConfigurationStatus = "inactive"
	ConfigurationStatusError    ConfigurationStatus = "error"
)

// AgentPackage represents an installed agent package
type AgentPackage struct {
	ID                  string              `json:"id" db:"id"`
	Name                string              `json:"name" db:"name"`
	Version             string              `json:"version" db:"version"`
	Description         *string             `json:"description,omitempty" db:"description"`
	Author              *string             `json:"author,omitempty" db:"author"`
	Repository          *string             `json:"repository,omitempty" db:"repository"`
	InstallPath         string              `json:"install_path" db:"install_path"`
	ConfigurationSchema json.RawMessage     `json:"configuration_schema" db:"configuration_schema"`
	Status              PackageStatus       `json:"status" db:"status"`
	ConfigurationStatus ConfigurationStatus `json:"configuration_status" db:"configuration_status"`
	InstalledAt         time.Time           `json:"installed_at" db:"installed_at"`
	UpdatedAt           time.Time           `json:"updated_at" db:"updated_at"`
	Metadata            PackageMetadata     `json:"metadata" db:"metadata"`
}

// PackageStatus represents the status of an agent package
type PackageStatus string

const (
	PackageStatusInstalled   PackageStatus = "installed"
	PackageStatusRunning     PackageStatus = "running"
	PackageStatusStopped     PackageStatus = "stopped"
	PackageStatusError       PackageStatus = "error"
	PackageStatusUninstalled PackageStatus = "uninstalled"
)

// PackageMetadata holds extensible metadata for an agent package
type PackageMetadata struct {
	Dependencies  []string               `json:"dependencies,omitempty"`
	Runtime       *RuntimeMetadata       `json:"runtime,omitempty"`
	Configuration *ConfigurationMetadata `json:"configuration,omitempty"`
	Custom        map[string]interface{} `json:"custom,omitempty"`
}

// RuntimeMetadata holds runtime-related metadata for a package
type RuntimeMetadata struct {
	Language       string            `json:"language"`
	Version        string            `json:"version"`
	Environment    map[string]string `json:"environment,omitempty"`
	ProcessID      *int              `json:"process_id,omitempty"`
	StartedAt      *time.Time        `json:"started_at,omitempty"`
	HealthCheckURL *string           `json:"health_check_url,omitempty"`
}

// ConfigurationMetadata holds configuration-related metadata
type ConfigurationMetadata struct {
	RequiredFields   []string   `json:"required_fields,omitempty"`
	OptionalFields   []string   `json:"optional_fields,omitempty"`
	SecretFields     []string   `json:"secret_fields,omitempty"`
	LastValidated    *time.Time `json:"last_validated,omitempty"`
	ValidationErrors []string   `json:"validation_errors,omitempty"`
}

// ConfigurationFilters holds filters for querying agent configurations
type ConfigurationFilters struct {
	AgentID   *string              `json:"agent_id,omitempty"`
	PackageID *string              `json:"package_id,omitempty"`
	Status    *ConfigurationStatus `json:"status,omitempty"`
	CreatedBy *string              `json:"created_by,omitempty"`
	StartTime *time.Time           `json:"start_time,omitempty"`
	EndTime   *time.Time           `json:"end_time,omitempty"`
	Limit     int                  `json:"limit,omitempty"`
	Offset    int                  `json:"offset,omitempty"`
}

// PackageFilters holds filters for querying agent packages
type PackageFilters struct {
	Status              *PackageStatus       `json:"status,omitempty"`
	ConfigurationStatus *ConfigurationStatus `json:"configuration_status,omitempty"`
	Name                *string              `json:"name,omitempty"`
	Author              *string              `json:"author,omitempty"`
	Limit               int                  `json:"limit,omitempty"`
	Offset              int                  `json:"offset,omitempty"`
}

// ConfigurationValidationResult represents the result of configuration validation
type ConfigurationValidationResult struct {
	Valid  bool     `json:"valid"`
	Errors []string `json:"errors,omitempty"`
}

// EncryptedValue represents an encrypted configuration value
type EncryptedValue struct {
	Value     string `json:"value"`
	KeyID     string `json:"key_id"`
	Algorithm string `json:"algorithm"`
}
