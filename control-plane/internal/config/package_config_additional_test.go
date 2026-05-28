package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadAgentFieldPackageConfigErrors(t *testing.T) {
	t.Run("missing package config", func(t *testing.T) {
		_, err := LoadAgentFieldPackageConfig(t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "agentfield-package.yaml not found") {
			t.Fatalf("expected missing file error, got %v", err)
		}
	})

	t.Run("invalid yaml", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "agentfield-package.yaml")
		if err := os.WriteFile(path, []byte("name: [\n"), 0o644); err != nil {
			t.Fatalf("write invalid yaml: %v", err)
		}

		_, err := LoadAgentFieldPackageConfig(dir)
		if err == nil || !strings.Contains(err.Error(), "failed to parse agentfield-package.yaml") {
			t.Fatalf("expected yaml parse error, got %v", err)
		}
	})

	t.Run("invalid schema", func(t *testing.T) {
		dir := t.TempDir()
		cfg := `
name: "test-agent"
user_environment:
  required:
    - name: "BROKEN"
      description: ""
      type: "string"
`
		path := filepath.Join(dir, "agentfield-package.yaml")
		if err := os.WriteFile(path, []byte(cfg), 0o644); err != nil {
			t.Fatalf("write invalid schema config: %v", err)
		}

		_, err := LoadAgentFieldPackageConfig(dir)
		if err == nil || !strings.Contains(err.Error(), "invalid configuration schema") {
			t.Fatalf("expected schema validation error, got %v", err)
		}
	})
}

func TestValidateConfigurationSchema(t *testing.T) {
	validSchema := &ConfigurationSchema{
		Required: []ConfigurationField{{
			Name:        "API_KEY",
			Description: "API key",
			Type:        "secret",
		}},
		Optional: []ConfigurationField{{
			Name:        "MODEL",
			Description: "Model",
			Type:        "select",
			Options:     []string{"a", "b"},
		}},
	}
	if err := validateConfigurationSchema(validSchema); err != nil {
		t.Fatalf("expected valid schema, got %v", err)
	}

	tests := []struct {
		name   string
		schema *ConfigurationSchema
		want   string
	}{
		{
			name: "required field failure is wrapped",
			schema: &ConfigurationSchema{
				Required: []ConfigurationField{{
					Description: "missing name",
					Type:        "string",
				}},
			},
			want: "required field 0",
		},
		{
			name: "optional field failure is wrapped",
			schema: &ConfigurationSchema{
				Optional: []ConfigurationField{{
					Name:        "BROKEN",
					Description: "broken regex",
					Type:        "string",
					Validation:  "[",
				}},
			},
			want: "optional field 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConfigurationSchema(tt.schema)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}
}

func TestValidateConfigurationField(t *testing.T) {
	min := 1
	max := 3

	tests := []struct {
		name  string
		field ConfigurationField
		want  string
	}{
		{
			name: "missing name",
			field: ConfigurationField{
				Description: "desc",
				Type:        "string",
			},
			want: "field name is required",
		},
		{
			name: "missing description",
			field: ConfigurationField{
				Name: "NAME",
				Type: "string",
			},
			want: "field description is required",
		},
		{
			name: "invalid type",
			field: ConfigurationField{
				Name:        "NAME",
				Description: "desc",
				Type:        "map",
			},
			want: "invalid field type",
		},
		{
			name: "select missing options",
			field: ConfigurationField{
				Name:        "CHOICE",
				Description: "desc",
				Type:        "select",
			},
			want: "select type field must have options",
		},
		{
			name: "invalid regex",
			field: ConfigurationField{
				Name:        "NAME",
				Description: "desc",
				Type:        "string",
				Validation:  "[",
			},
			want: "invalid validation regex",
		},
		{
			name: "invalid default value",
			field: ConfigurationField{
				Name:        "COUNT",
				Description: "desc",
				Type:        "integer",
				Default:     "0",
				Min:         &min,
				Max:         &max,
			},
			want: "invalid default value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConfigurationField(&tt.field)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}

	valid := ConfigurationField{
		Name:        "COUNT",
		Description: "desc",
		Type:        "integer",
		Default:     "2",
		Min:         &min,
		Max:         &max,
	}
	if err := validateConfigurationField(&valid); err != nil {
		t.Fatalf("expected valid field, got %v", err)
	}
}

func TestValidateFieldValueAdditionalCases(t *testing.T) {
	min := 2
	max := 4

	tests := []struct {
		name  string
		field ConfigurationField
		value string
		want  string
	}{
		{
			name: "integer parse failure",
			field: ConfigurationField{
				Type: "integer",
			},
			value: "abc",
			want:  "value must be an integer",
		},
		{
			name: "integer below min",
			field: ConfigurationField{
				Type: "integer",
				Min:  &min,
			},
			value: "1",
			want:  "value must be at least 2",
		},
		{
			name: "integer above max",
			field: ConfigurationField{
				Type: "integer",
				Max:  &max,
			},
			value: "5",
			want:  "value must be at most 4",
		},
		{
			name: "float parse failure",
			field: ConfigurationField{
				Type: "float",
			},
			value: "abc",
			want:  "value must be a float",
		},
		{
			name: "float below min",
			field: ConfigurationField{
				Type: "float",
				Min:  &min,
			},
			value: "1.5",
			want:  "value must be at least 2",
		},
		{
			name: "float above max",
			field: ConfigurationField{
				Type: "float",
				Max:  &max,
			},
			value: "4.5",
			want:  "value must be at most 4",
		},
		{
			name: "boolean invalid",
			field: ConfigurationField{
				Type: "boolean",
			},
			value: "yes",
			want:  "value must be 'true' or 'false'",
		},
		{
			name: "select invalid option",
			field: ConfigurationField{
				Type:    "select",
				Options: []string{"a", "b"},
			},
			value: "c",
			want:  "value must be one of: a, b",
		},
		{
			name: "regex mismatch",
			field: ConfigurationField{
				Type:       "string",
				Validation: "^ok$",
			},
			value: "bad",
			want:  "value does not match validation pattern",
		},
		{
			name: "regex evaluation error",
			field: ConfigurationField{
				Type:       "string",
				Validation: "[",
			},
			value: "bad",
			want:  "validation regex error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateFieldValue(&tt.field, tt.value)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}

	validCases := []struct {
		name  string
		field ConfigurationField
		value string
	}{
		{
			name:  "string without validation",
			field: ConfigurationField{Type: "string"},
			value: "anything",
		},
		{
			name:  "integer valid",
			field: ConfigurationField{Type: "integer", Min: &min, Max: &max},
			value: "3",
		},
		{
			name:  "float valid",
			field: ConfigurationField{Type: "float", Min: &min, Max: &max},
			value: "2.5",
		},
		{
			name:  "boolean valid",
			field: ConfigurationField{Type: "boolean"},
			value: "true",
		},
		{
			name:  "select valid",
			field: ConfigurationField{Type: "select", Options: []string{"a", "b"}},
			value: "a",
		},
		{
			name:  "regex valid",
			field: ConfigurationField{Type: "string", Validation: "^ok$"},
			value: "ok",
		},
	}

	for _, tt := range validCases {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateFieldValue(&tt.field, tt.value); err != nil {
				t.Fatalf("expected valid value, got %v", err)
			}
		})
	}
}

func TestValidateConfigurationRejectsUnknownOptionalFieldValues(t *testing.T) {
	schema := &ConfigurationSchema{
		Required: []ConfigurationField{{
			Name:        "REQ",
			Description: "req",
			Type:        "string",
		}},
		Optional: []ConfigurationField{{
			Name:        "OPT",
			Description: "opt",
			Type:        "boolean",
		}},
	}

	tests := []struct {
		name   string
		config map[string]string
		want   string
	}{
		{
			name: "optional field validation fails",
			config: map[string]string{
				"REQ": "ok",
				"OPT": "maybe",
			},
			want: "field 'OPT'",
		},
		{
			name: "unknown field rejected",
			config: map[string]string{
				"REQ":   "ok",
				"EXTRA": "value",
			},
			want: "unknown field 'EXTRA'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfiguration(schema, tt.config)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}
}
