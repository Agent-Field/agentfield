package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAgentFieldPackageConfig(t *testing.T) {
	// Create a temporary test directory
	tempDir := t.TempDir()

	// Create a test agentfield-package.yaml
	testConfig := `
name: "test-agent"
version: "1.0.0"
description: "Test agent for configuration parsing"
author: "Test Author"
type: "agent-node"
main: "main.py"

agent_node:
  node_id: "testagent"
  default_port: 8001

dependencies:
  python: ["agentfield>=1.0.0"]

capabilities:
  reasoners:
    - name: "test_reasoner"
      description: "Test reasoner"
  skills:
    - name: "test_skill"
      description: "Test skill"

runtime:
  auto_port: true
  heartbeat_interval: 30
  dev_mode: true

environment:
  LOG_LEVEL: "info"

user_environment:
  required:
    - name: "API_KEY"
      description: "API key for external service"
      type: "secret"
      validation: "^sk-[a-zA-Z0-9]+$"
    - name: "MODEL_NAME"
      description: "AI model to use"
      type: "select"
      options: ["gpt-4", "gpt-3.5-turbo", "claude-3"]
  optional:
    - name: "MAX_TOKENS"
      description: "Maximum tokens per request"
      type: "integer"
      default: "2000"
      min: 100
      max: 4000
    - name: "TEMPERATURE"
      description: "Model temperature"
      type: "float"
      default: "0.7"
      validation: "^[0-1](\\.[0-9]+)?$"
    - name: "ENABLE_LOGGING"
      description: "Enable detailed logging"
      type: "boolean"
      default: "true"

metadata:
  created_at: "2025-06-20"
  sdk_version: "1.0.0"
  language: "python"
  platform: "agentfield-agent-node"
`

	configPath := filepath.Join(tempDir, "agentfield-package.yaml")
	err := os.WriteFile(configPath, []byte(testConfig), 0644)
	if err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	// Test loading the configuration
	config, err := LoadAgentFieldPackageConfig(tempDir)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Verify basic fields
	if config.Name != "test-agent" {
		t.Errorf("Expected name 'test-agent', got '%s'", config.Name)
	}

	if config.Version != "1.0.0" {
		t.Errorf("Expected version '1.0.0', got '%s'", config.Version)
	}

	// Verify required fields
	if len(config.UserEnvironment.Required) != 2 {
		t.Errorf("Expected 2 required fields, got %d", len(config.UserEnvironment.Required))
	}

	// Check specific required field
	apiKeyField := config.UserEnvironment.Required[0]
	if apiKeyField.Name != "API_KEY" {
		t.Errorf("Expected first required field to be 'API_KEY', got '%s'", apiKeyField.Name)
	}
	if apiKeyField.Type != "secret" {
		t.Errorf("Expected API_KEY type to be 'secret', got '%s'", apiKeyField.Type)
	}

	// Verify optional fields
	if len(config.UserEnvironment.Optional) != 3 {
		t.Errorf("Expected 3 optional fields, got %d", len(config.UserEnvironment.Optional))
	}

	// Check specific optional field
	maxTokensField := config.UserEnvironment.Optional[0]
	if maxTokensField.Name != "MAX_TOKENS" {
		t.Errorf("Expected first optional field to be 'MAX_TOKENS', got '%s'", maxTokensField.Name)
	}
	if maxTokensField.Type != "integer" {
		t.Errorf("Expected MAX_TOKENS type to be 'integer', got '%s'", maxTokensField.Type)
	}
	if maxTokensField.Default != "2000" {
		t.Errorf("Expected MAX_TOKENS default to be '2000', got '%s'", maxTokensField.Default)
	}
}

func TestValidateConfiguration(t *testing.T) {
	schema := &ConfigurationSchema{
		Required: []ConfigurationField{
			{
				Name:        "API_KEY",
				Description: "API key",
				Type:        "secret",
				Validation:  "^sk-[a-zA-Z0-9]+$",
			},
			{
				Name:        "MODEL",
				Description: "Model selection",
				Type:        "select",
				Options:     []string{"gpt-4", "gpt-3.5-turbo"},
			},
		},
		Optional: []ConfigurationField{
			{
				Name:        "MAX_TOKENS",
				Description: "Max tokens",
				Type:        "integer",
				Default:     "2000",
				Min:         intPtr(100),
				Max:         intPtr(4000),
			},
		},
	}

	// Test valid configuration
	validConfig := map[string]string{
		"API_KEY":    "sk-abc123",
		"MODEL":      "gpt-4",
		"MAX_TOKENS": "2000",
	}

	err := ValidateConfiguration(schema, validConfig)
	if err != nil {
		t.Errorf("Valid configuration should not have errors: %v", err)
	}

	// Test missing required field
	invalidConfig1 := map[string]string{
		"MODEL": "gpt-4",
	}

	err = ValidateConfiguration(schema, invalidConfig1)
	if err == nil {
		t.Error("Should have error for missing required field")
	}

	// Test invalid field value
	invalidConfig2 := map[string]string{
		"API_KEY": "invalid-key",
		"MODEL":   "gpt-4",
	}

	err = ValidateConfiguration(schema, invalidConfig2)
	if err == nil {
		t.Error("Should have error for invalid API key format")
	}

	// Test invalid select option
	invalidConfig3 := map[string]string{
		"API_KEY": "sk-abc123",
		"MODEL":   "invalid-model",
	}

	err = ValidateConfiguration(schema, invalidConfig3)
	if err == nil {
		t.Error("Should have error for invalid select option")
	}

	// Test integer out of range
	invalidConfig4 := map[string]string{
		"API_KEY":    "sk-abc123",
		"MODEL":      "gpt-4",
		"MAX_TOKENS": "5000",
	}

	err = ValidateConfiguration(schema, invalidConfig4)
	if err == nil {
		t.Error("Should have error for integer out of range")
	}
}

func TestGetConfigurationWithDefaults(t *testing.T) {
	schema := &ConfigurationSchema{
		Required: []ConfigurationField{
			{
				Name:        "API_KEY",
				Description: "API key",
				Type:        "secret",
			},
		},
		Optional: []ConfigurationField{
			{
				Name:        "MAX_TOKENS",
				Description: "Max tokens",
				Type:        "integer",
				Default:     "2000",
			},
			{
				Name:        "TEMPERATURE",
				Description: "Temperature",
				Type:        "float",
				Default:     "0.7",
			},
		},
	}

	config := map[string]string{
		"API_KEY":    "sk-abc123",
		"MAX_TOKENS": "3000", // Override default
	}

	result := GetConfigurationWithDefaults(schema, config)

	// Should keep provided values
	if result["API_KEY"] != "sk-abc123" {
		t.Errorf("Expected API_KEY to be 'sk-abc123', got '%s'", result["API_KEY"])
	}
	if result["MAX_TOKENS"] != "3000" {
		t.Errorf("Expected MAX_TOKENS to be '3000', got '%s'", result["MAX_TOKENS"])
	}

	// Should apply default for missing optional field
	if result["TEMPERATURE"] != "0.7" {
		t.Errorf("Expected TEMPERATURE to be '0.7', got '%s'", result["TEMPERATURE"])
	}
}

func TestValidateFieldValue(t *testing.T) {
	// Test integer validation
	intField := &ConfigurationField{
		Name: "TEST_INT",
		Type: "integer",
		Min:  intPtr(10),
		Max:  intPtr(100),
	}

	if err := validateFieldValue(intField, "50"); err != nil {
		t.Errorf("Valid integer should not have error: %v", err)
	}

	if err := validateFieldValue(intField, "5"); err == nil {
		t.Error("Integer below min should have error")
	}

	if err := validateFieldValue(intField, "150"); err == nil {
		t.Error("Integer above max should have error")
	}

	if err := validateFieldValue(intField, "not-a-number"); err == nil {
		t.Error("Non-integer should have error")
	}

	// Test boolean validation
	boolField := &ConfigurationField{
		Name: "TEST_BOOL",
		Type: "boolean",
	}

	if err := validateFieldValue(boolField, "true"); err != nil {
		t.Errorf("Valid boolean should not have error: %v", err)
	}

	if err := validateFieldValue(boolField, "false"); err != nil {
		t.Errorf("Valid boolean should not have error: %v", err)
	}

	if err := validateFieldValue(boolField, "yes"); err == nil {
		t.Error("Invalid boolean should have error")
	}

	// Test regex validation
	regexField := &ConfigurationField{
		Name:       "TEST_REGEX",
		Type:       "string",
		Validation: "^sk-[a-zA-Z0-9]+$",
	}

	if err := validateFieldValue(regexField, "sk-abc123"); err != nil {
		t.Errorf("Valid regex match should not have error: %v", err)
	}

	if err := validateFieldValue(regexField, "invalid-format"); err == nil {
		t.Error("Invalid regex match should have error")
	}
}

// Helper function to create int pointer
func intPtr(i int) *int {
	return &i
}
