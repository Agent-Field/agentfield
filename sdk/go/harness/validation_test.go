package harness

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sequenceWriterProvider writes a different output-file body on each Execute
// call (simulating a coding agent that fixes its output across retries).
type sequenceWriterProvider struct {
	outputPath string
	contents   [][]byte
	results    []*RawResult
	calls      int
}

func (m *sequenceWriterProvider) Execute(_ context.Context, _ string, _ Options) (*RawResult, error) {
	idx := m.calls
	m.calls++
	if idx < len(m.contents) && m.contents[idx] != nil {
		_ = os.WriteFile(m.outputPath, m.contents[idx], 0o644)
	}
	if idx < len(m.results) {
		return m.results[idx], nil
	}
	return &RawResult{Result: "done", Metrics: Metrics{NumTurns: 1}}, nil
}

// TestValidation_MissingRequiredFieldTriggersRetry maps to the contract:
// "output missing a required field -> validateAgainstSchema error -> retry
// fires -> second attempt valid -> success". Without real validation the
// initial (missing-field) output would unmarshal cleanly and no retry would
// fire, so mock.calls==0 would be the (buggy) old behavior.
func TestValidation_MissingRequiredFieldTriggersRetry(t *testing.T) {
	dir := t.TempDir()
	outputPath := OutputPath(dir)

	// Initial attempt: valid JSON but missing the required "value" field.
	require.NoError(t, os.WriteFile(outputPath, []byte(`{}`), 0o644))

	schema := map[string]any{
		"type":     "object",
		"required": []any{"value"},
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
	}

	provider := &sequenceWriterProvider{
		outputPath: outputPath,
		// Retry writes a valid object.
		contents: [][]byte{[]byte(`{"value":"fixed"}`)},
		results:  []*RawResult{{Result: "retry", Metrics: Metrics{NumTurns: 1}}},
	}

	var dest struct {
		Value string `json:"value"`
	}
	runner := NewRunner(Options{Provider: "opencode"})
	result := runner.handleSchemaWithRetry(
		context.Background(),
		&RawResult{Result: "initial", Metrics: Metrics{NumTurns: 1}},
		schema, &dest, dir, time.Now(), provider,
		Options{Provider: "opencode", SchemaMaxRetries: 2}, "prompt", false,
	)

	assert.False(t, result.IsError, "expected success after retry, got: %s", result.ErrorMessage)
	assert.Equal(t, "fixed", dest.Value)
	assert.Equal(t, 1, provider.calls, "exactly one retry should have fired")
}

// TestValidation_InvalidEnumTriggersRetry maps to the contract's invalid-enum
// case.
func TestValidation_InvalidEnumTriggersRetry(t *testing.T) {
	dir := t.TempDir()
	outputPath := OutputPath(dir)

	// Initial attempt: "severity" present but not one of the allowed values.
	require.NoError(t, os.WriteFile(outputPath, []byte(`{"severity":"bogus"}`), 0o644))

	schema := map[string]any{
		"type":     "object",
		"required": []any{"severity"},
		"properties": map[string]any{
			"severity": map[string]any{
				"type": "string",
				"enum": []any{"low", "high"},
			},
		},
	}

	provider := &sequenceWriterProvider{
		outputPath: outputPath,
		contents:   [][]byte{[]byte(`{"severity":"high"}`)},
		results:    []*RawResult{{Result: "retry", Metrics: Metrics{NumTurns: 1}}},
	}

	var dest struct {
		Severity string `json:"severity"`
	}
	runner := NewRunner(Options{Provider: "opencode"})
	result := runner.handleSchemaWithRetry(
		context.Background(),
		&RawResult{Result: "initial", Metrics: Metrics{NumTurns: 1}},
		schema, &dest, dir, time.Now(), provider,
		Options{Provider: "opencode", SchemaMaxRetries: 2}, "prompt", false,
	)

	assert.False(t, result.IsError, "expected success after enum retry, got: %s", result.ErrorMessage)
	assert.Equal(t, "high", dest.Severity)
	assert.Equal(t, 1, provider.calls)
}

// TestValidation_ExtraFieldsAllowedWhenAdditionalPropertiesUnset maps to the
// contract: "valid-but-extra-fields output passes when schema allows
// additionalProperties". No retry should fire.
func TestValidation_ExtraFieldsAllowedWhenAdditionalPropertiesUnset(t *testing.T) {
	dir := t.TempDir()
	outputPath := OutputPath(dir)

	require.NoError(t, os.WriteFile(outputPath, []byte(`{"a":"x","extra":"y"}`), 0o644))

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{"type": "string"},
		},
	}

	provider := &sequenceWriterProvider{outputPath: outputPath}

	var dest struct {
		A string `json:"a"`
	}
	runner := NewRunner(Options{Provider: "opencode"})
	result := runner.handleSchemaWithRetry(
		context.Background(),
		&RawResult{Result: "initial", Metrics: Metrics{NumTurns: 1}},
		schema, &dest, dir, time.Now(), provider,
		Options{Provider: "opencode", SchemaMaxRetries: 2}, "prompt", false,
	)

	assert.False(t, result.IsError)
	assert.Equal(t, "x", dest.A)
	assert.Equal(t, 0, provider.calls, "valid output must not trigger a retry")
}

// TestValidation_RetriesExhaustedSurfacesError maps to the contract: "retries
// exhausted -> error surfaces as before".
func TestValidation_RetriesExhaustedSurfacesError(t *testing.T) {
	dir := t.TempDir()
	outputPath := OutputPath(dir)

	require.NoError(t, os.WriteFile(outputPath, []byte(`{}`), 0o644))

	schema := map[string]any{
		"type":     "object",
		"required": []any{"value"},
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
	}

	// Every retry keeps writing an invalid (missing-field) object.
	provider := &sequenceWriterProvider{
		outputPath: outputPath,
		contents:   [][]byte{[]byte(`{}`), []byte(`{}`)},
		results: []*RawResult{
			{Result: "r1", Metrics: Metrics{NumTurns: 1}},
			{Result: "r2", Metrics: Metrics{NumTurns: 1}},
		},
	}

	var dest struct {
		Value string `json:"value"`
	}
	runner := NewRunner(Options{Provider: "opencode"})
	result := runner.handleSchemaWithRetry(
		context.Background(),
		&RawResult{Result: "initial", Metrics: Metrics{NumTurns: 1}},
		schema, &dest, dir, time.Now(), provider,
		Options{Provider: "opencode", SchemaMaxRetries: 2}, "prompt", false,
	)

	assert.True(t, result.IsError)
	assert.Equal(t, FailureSchema, result.FailureType)
	assert.Contains(t, result.ErrorMessage, "Schema validation failed")
	assert.Equal(t, 2, provider.calls, "both retries should have been attempted")
}

// TestValidateAgainstSchema unit-tests the validator directly.
func TestValidateAgainstSchema(t *testing.T) {
	schema := map[string]any{
		"type":     "object",
		"required": []any{"name", "level"},
		"properties": map[string]any{
			"name":  map[string]any{"type": "string"},
			"level": map[string]any{"type": "string", "enum": []any{"low", "high"}},
		},
		"additionalProperties": false,
	}

	t.Run("valid", func(t *testing.T) {
		assert.NoError(t, validateAgainstSchema(map[string]any{"name": "x", "level": "high"}, schema))
	})
	t.Run("missing required", func(t *testing.T) {
		err := validateAgainstSchema(map[string]any{"name": "x"}, schema)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "schema validation failed")
	})
	t.Run("invalid enum", func(t *testing.T) {
		err := validateAgainstSchema(map[string]any{"name": "x", "level": "nope"}, schema)
		require.Error(t, err)
	})
	t.Run("extra field rejected when additionalProperties false", func(t *testing.T) {
		err := validateAgainstSchema(map[string]any{"name": "x", "level": "low", "junk": 1}, schema)
		require.Error(t, err)
	})
	t.Run("uncompilable schema skips validation", func(t *testing.T) {
		// A schema whose "type" is not a valid keyword value fails to compile;
		// validation is skipped (nil) rather than blocking — no regression.
		bad := map[string]any{"type": 123}
		assert.NoError(t, validateAgainstSchema(map[string]any{"any": "thing"}, bad))
	})
}

// TestRunSchemaValidation_Gating verifies validation only applies when both a
// destination and a schema are present.
func TestRunSchemaValidation_Gating(t *testing.T) {
	schema := map[string]any{
		"type":     "object",
		"required": []any{"value"},
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
	}
	invalid := map[string]any{} // missing required "value"

	var dest struct {
		Value string `json:"value"`
	}
	// Both present -> validation runs and fails.
	assert.Error(t, runSchemaValidation(invalid, schema, &dest))
	// No dest -> skipped.
	assert.NoError(t, runSchemaValidation(invalid, schema, nil))
	// No schema -> skipped.
	assert.NoError(t, runSchemaValidation(invalid, nil, &dest))
	// No data -> skipped.
	assert.NoError(t, runSchemaValidation(nil, schema, &dest))
}
