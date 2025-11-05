package config

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// ConfigurationField represents a single configuration field in haxen-package.yaml
type ConfigurationField struct {
	Name        string `yaml:"name" json:"name"`
	Description string `yaml:"description" json:"description"`
	Type        string `yaml:"type" json:"type"` // "string", "secret", "integer", "float", "boolean", "select"
	Default     string `yaml:"default" json:"default,omitempty"`
	Validation  string `yaml:"validation" json:"validation,omitempty"` // regex pattern
	Options     []string `yaml:"options" json:"options,omitempty"`     // for select type
	Min         *int   `yaml:"min" json:"min,omitempty"`               // for integer/float
	Max         *int   `yaml:"max" json:"max,omitempty"`               // for integer/float
}

// ConfigurationSchema represents the configuration schema from haxen-package.yaml
type ConfigurationSchema struct {
	Required []ConfigurationField `yaml:"required" json:"required"`
	Optional []ConfigurationField `yaml:"optional" json:"optional"`
}

// HaxenPackageConfig represents the structure of haxen-package.yaml
type HaxenPackageConfig struct {
	Name            string              `yaml:"name"`
	Version         string              `yaml:"version"`
	Description     string              `yaml:"description"`
	Author          string              `yaml:"author"`
	Type            string              `yaml:"type"`
	Main            string              `yaml:"main"`
	AgentNode       AgentNodeConfig     `yaml:"agent_node"`
	Dependencies    DependenciesConfig  `yaml:"dependencies"`
	Capabilities    CapabilitiesConfig  `yaml:"capabilities"`
	Runtime         RuntimeConfig       `yaml:"runtime"`
	Environment     map[string]string   `yaml:"environment"`
	UserEnvironment ConfigurationSchema `yaml:"user_environment"`
	Metadata        MetadataConfig      `yaml:"metadata"`
}

type AgentNodeConfig struct {
	NodeID      string `yaml:"node_id"`
	DefaultPort int    `yaml:"default_port"`
}

type DependenciesConfig struct {
	Python []string `yaml:"python"`
}

type CapabilitiesConfig struct {
	Reasoners []CapabilityItem `yaml:"reasoners"`
	Skills    []CapabilityItem `yaml:"skills"`
}

type CapabilityItem struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

type RuntimeConfig struct {
	AutoPort          bool `yaml:"auto_port"`
	HeartbeatInterval int  `yaml:"heartbeat_interval"`
	DevMode           bool `yaml:"dev_mode"`
}

type MetadataConfig struct {
	CreatedAt   string `yaml:"created_at"`
	SDKVersion  string `yaml:"sdk_version"`
	Language    string `yaml:"language"`
	Platform    string `yaml:"platform"`
}

// LoadHaxenPackageConfig loads and parses a haxen-package.yaml file
func LoadHaxenPackageConfig(packagePath string) (*HaxenPackageConfig, error) {
	configPath := filepath.Join(packagePath, "haxen-package.yaml")
	
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("haxen-package.yaml not found at %s", configPath)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read haxen-package.yaml: %w", err)
	}

	var config HaxenPackageConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse haxen-package.yaml: %w", err)
	}

	// Validate the configuration schema
	if err := validateConfigurationSchema(&config.UserEnvironment); err != nil {
		return nil, fmt.Errorf("invalid configuration schema: %w", err)
	}

	return &config, nil
}

// validateConfigurationSchema validates the configuration schema
func validateConfigurationSchema(schema *ConfigurationSchema) error {
	// Validate required fields
	for i, field := range schema.Required {
		if err := validateConfigurationField(&field); err != nil {
			return fmt.Errorf("required field %d (%s): %w", i, field.Name, err)
		}
	}

	// Validate optional fields
	for i, field := range schema.Optional {
		if err := validateConfigurationField(&field); err != nil {
			return fmt.Errorf("optional field %d (%s): %w", i, field.Name, err)
		}
	}

	return nil
}

// validateConfigurationField validates a single configuration field
func validateConfigurationField(field *ConfigurationField) error {
	if field.Name == "" {
		return fmt.Errorf("field name is required")
	}

	if field.Description == "" {
		return fmt.Errorf("field description is required")
	}

	// Validate field type
	validTypes := []string{"string", "secret", "integer", "float", "boolean", "select"}
	if !contains(validTypes, field.Type) {
		return fmt.Errorf("invalid field type '%s', must be one of: %s", field.Type, strings.Join(validTypes, ", "))
	}

	// Validate select type has options
	if field.Type == "select" && len(field.Options) == 0 {
		return fmt.Errorf("select type field must have options")
	}

	// Validate regex pattern if provided
	if field.Validation != "" {
		if _, err := regexp.Compile(field.Validation); err != nil {
			return fmt.Errorf("invalid validation regex: %w", err)
		}
	}

	// Validate default value against type and validation
	if field.Default != "" {
		if err := validateFieldValue(field, field.Default); err != nil {
			return fmt.Errorf("invalid default value: %w", err)
		}
	}

	return nil
}

// validateFieldValue validates a value against a configuration field
func validateFieldValue(field *ConfigurationField, value string) error {
	// Type validation
	switch field.Type {
	case "integer":
		val, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("value must be an integer")
		}
		if field.Min != nil && val < *field.Min {
			return fmt.Errorf("value must be at least %d", *field.Min)
		}
		if field.Max != nil && val > *field.Max {
			return fmt.Errorf("value must be at most %d", *field.Max)
		}
	case "float":
		val, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("value must be a float")
		}
		if field.Min != nil && val < float64(*field.Min) {
			return fmt.Errorf("value must be at least %d", *field.Min)
		}
		if field.Max != nil && val > float64(*field.Max) {
			return fmt.Errorf("value must be at most %d", *field.Max)
		}
	case "boolean":
		if value != "true" && value != "false" {
			return fmt.Errorf("value must be 'true' or 'false'")
		}
	case "select":
		if !contains(field.Options, value) {
			return fmt.Errorf("value must be one of: %s", strings.Join(field.Options, ", "))
		}
	}

	// Regex validation
	if field.Validation != "" {
		matched, err := regexp.MatchString(field.Validation, value)
		if err != nil {
			return fmt.Errorf("validation regex error: %w", err)
		}
		if !matched {
			return fmt.Errorf("value does not match validation pattern")
		}
	}

	return nil
}

// ValidateConfiguration validates a complete configuration against a schema
func ValidateConfiguration(schema *ConfigurationSchema, config map[string]string) error {
	// Check all required fields are present
	for _, field := range schema.Required {
		value, exists := config[field.Name]
		if !exists {
			return fmt.Errorf("required field '%s' is missing", field.Name)
		}
		if err := validateFieldValue(&field, value); err != nil {
			return fmt.Errorf("field '%s': %w", field.Name, err)
		}
	}

	// Validate optional fields if present
	for _, field := range schema.Optional {
		if value, exists := config[field.Name]; exists {
			if err := validateFieldValue(&field, value); err != nil {
				return fmt.Errorf("field '%s': %w", field.Name, err)
			}
		}
	}

	// Check for unknown fields
	allFieldNames := make(map[string]bool)
	for _, field := range schema.Required {
		allFieldNames[field.Name] = true
	}
	for _, field := range schema.Optional {
		allFieldNames[field.Name] = true
	}

	for name := range config {
		if !allFieldNames[name] {
			return fmt.Errorf("unknown field '%s'", name)
		}
	}

	return nil
}

// GetConfigurationWithDefaults returns configuration with default values applied
func GetConfigurationWithDefaults(schema *ConfigurationSchema, config map[string]string) map[string]string {
	result := make(map[string]string)
	
	// Copy provided values
	for k, v := range config {
		result[k] = v
	}
	
	// Apply defaults for missing optional fields
	for _, field := range schema.Optional {
		if _, exists := result[field.Name]; !exists && field.Default != "" {
			result[field.Name] = field.Default
		}
	}
	
	return result
}

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
