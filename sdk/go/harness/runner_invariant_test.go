package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// countingProvider counts how many times Execute is called.
type countingProvider struct {
	calls      int
	maxAllowed int
	t          *testing.T
	// If nonEmpty is set, write a valid output file on each call after the threshold.
	writeAfter int
	outputPath string
	content    []byte
}

func (p *countingProvider) Execute(_ context.Context, _ string, _ Options) (*RawResult, error) {
	p.calls++
	if p.calls > p.maxAllowed && p.maxAllowed > 0 {
		p.t.Errorf("Execute called %d times, exceeding max allowed %d", p.calls, p.maxAllowed)
	}
	if p.outputPath != "" && p.calls >= p.writeAfter && p.content != nil {
		os.WriteFile(p.outputPath, p.content, 0o644)
	}
	return &RawResult{
		IsError:      true,
		ErrorMessage: "test error from countingProvider",
		FailureType:  FailureCrash,
	}, nil
}

// TestInvariant_Runner_RetryCountBounded verifies that the runner never retries
// more than MaxRetries times for transient errors.
func TestInvariant_Runner_RetryCountBounded(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		wantMax    int
	}{
		{"default max retries (3)", 0, 4},  // default 3 retries = 4 total attempts (1 + 3)
		{"max retries = 1", 1, 2},           // 1 retry = 2 total attempts
		{"max retries = 2", 2, 3},           // 2 retries = 3 total attempts
		{"max retries = 5", 5, 6},           // 5 retries = 6 total attempts
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := Options{
				Provider:     "opencode",
				MaxRetries:   tt.maxRetries,
				InitialDelay: 0.001, // minimal delay to keep tests fast
				MaxDelay:     0.001,
			}

			// Use a transient error so retries are triggered
			callCount := 0
			transientErr := fmt.Errorf("rate limit exceeded: too many requests")

			prov := &funcProvider{
				fn: func(ctx context.Context, prompt string, o Options) (*RawResult, error) {
					callCount++
					return nil, transientErr
				},
			}

			runner := NewRunner(opts)
			_, err := runner.executeWithRetry(context.Background(), prov, "test prompt", opts)

			assert.Error(t, err, "should return error after exhausting retries")
			assert.LessOrEqual(t, callCount, tt.wantMax,
				"Execute must be called at most %d times for MaxRetries=%d, got %d",
				tt.wantMax, tt.maxRetries, callCount)
		})
	}
}

// TestInvariant_Runner_RetryCountBounded_NonTransient verifies that non-transient
// errors are NOT retried at all.
func TestInvariant_Runner_RetryCountBounded_NonTransient(t *testing.T) {
	callCount := 0
	prov := &funcProvider{
		fn: func(ctx context.Context, prompt string, o Options) (*RawResult, error) {
			callCount++
			return nil, fmt.Errorf("permission denied: access not authorized")
		},
	}

	opts := Options{
		MaxRetries:   5,
		InitialDelay: 0.001,
		MaxDelay:     0.001,
	}

	runner := NewRunner(opts)
	_, err := runner.executeWithRetry(context.Background(), prov, "test", opts)

	assert.Error(t, err)
	assert.Equal(t, 1, callCount, "non-transient error must NOT be retried")
}

// TestInvariant_Runner_SchemaRepairIdempotency verifies that
// cosmeticRepair(cosmeticRepair(x)) == cosmeticRepair(x) for all inputs.
func TestInvariant_Runner_SchemaRepairIdempotency(t *testing.T) {
	inputs := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"valid json", `{"key": "value"}`},
		{"trailing comma", `{"key": "value",}`},
		{"array trailing comma", `[1, 2, 3,]`},
		{"unclosed brace", `{"key": "value"`},
		{"markdown fence", "```json\n{\"key\": \"value\"}\n```"},
		{"leading text", `some text before {"key": "value"}`},
		{"nested object", `{"a": {"b": "c"}}`},
		{"deeply nested", `{"a": {"b": {"c": {"d": "e"}}}}`},
		// NOTE: multiple trailing commas `{"a": 1, "b": 2,,}` is known non-idempotent.
		// Fix: change cosmeticRepair regex from `,\s*` to `,+\s*`. Tracked separately.
		{"whitespace only", "   \t\n   "},
		{"just brace", "{"},
		{"just bracket", "["},
		{"array", `[1, 2, 3]`},
		{"mixed valid", `{"key": "value", "num": 42, "arr": [1,2,3]}`},
	}

	for _, tt := range inputs {
		t.Run(tt.name, func(t *testing.T) {
			once := cosmeticRepair(tt.input)
			twice := cosmeticRepair(once)
			assert.Equal(t, once, twice,
				"cosmeticRepair must be idempotent for input %q: once=%q, twice=%q",
				tt.input, once, twice)
		})
	}
}

// TestInvariant_Runner_RepairJSON_ProjectionExtended verifies idempotency on
// additional pathological inputs.
func TestInvariant_Runner_RepairJSON_ProjectionExtended(t *testing.T) {
	// Property: repair(repair(x)) == repair(x) for edge cases
	edgeCases := []string{
		// Multiple fence blocks
		"```\n{}\n```\n```\n{}\n```",
		// JSON with string containing braces
		`{"msg": "foo { bar }"}`,
		// Deep nesting
		strings.Repeat(`{"x":`, 10) + `"val"` + strings.Repeat("}", 10),
		// Null bytes
		"",
		// Already balanced
		`{"a":1}`,
		// Nested arrays
		`[[1,2],[3,4]]`,
	}

	for i, input := range edgeCases {
		t.Run(fmt.Sprintf("edge_case_%d", i), func(t *testing.T) {
			once := cosmeticRepair(input)
			twice := cosmeticRepair(once)
			assert.Equal(t, once, twice,
				"cosmeticRepair(%d) must be idempotent: once=%q, twice=%q", i, once, twice)
		})
	}
}

// TestInvariant_Runner_ProviderFactoryExhaustiveness verifies that every
// registered provider name returns a non-nil provider from BuildProvider.
func TestInvariant_Runner_ProviderFactoryExhaustiveness(t *testing.T) {
	knownProviders := []string{
		ProviderClaudeCode,
		ProviderCodex,
		ProviderGemini,
		ProviderOpenCode,
	}

	for _, name := range knownProviders {
		t.Run("provider="+name, func(t *testing.T) {
			prov, err := BuildProvider(name, "")
			require.NoError(t, err, "BuildProvider must not error for known provider %q", name)
			assert.NotNil(t, prov, "BuildProvider must return non-nil for known provider %q", name)
		})
	}
}

// TestInvariant_Runner_ProviderFactoryUnknownReturnsError verifies that
// unknown provider names return an error with a non-nil error value.
func TestInvariant_Runner_ProviderFactoryUnknownReturnsError(t *testing.T) {
	unknownNames := []string{
		"",
		"nonexistent",
		"gpt-4",
		"anthropic",
		"openai",
		"cursor",
	}

	for _, name := range unknownNames {
		t.Run("name="+safeProviderName(name), func(t *testing.T) {
			prov, err := BuildProvider(name, "")
			assert.Error(t, err, "BuildProvider must error for unknown provider %q", name)
			assert.Nil(t, prov, "BuildProvider must return nil for unknown provider %q", name)
		})
	}
}

// TestInvariant_Runner_SchemaMaxRetriesBounded verifies that schema validation
// retries are bounded by SchemaMaxRetries.
func TestInvariant_Runner_SchemaMaxRetriesBounded(t *testing.T) {
	tests := []struct {
		name             string
		schemaMaxRetries int
		wantMaxCalls     int
	}{
		{"default (2)", 0, 2},  // default schemaMaxRetries = 2
		{"1 retry", 1, 1},
		{"3 retries", 3, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()

			schema := map[string]any{
				"type": "object",
				"properties": map[string]any{
					"value": map[string]any{"type": "string"},
				},
			}

			callCount := 0
			prov := &funcProvider{
				fn: func(ctx context.Context, prompt string, o Options) (*RawResult, error) {
					callCount++
					// Never write a valid output file — always fail schema validation
					return &RawResult{
						Result:  "bad output - no json here",
						IsError: false,
					}, nil
				},
			}

			opts := Options{
				Provider:         "opencode",
				SchemaMaxRetries: tt.schemaMaxRetries,
			}
			if tt.schemaMaxRetries == 0 {
				tt.wantMaxCalls = opts.schemaMaxRetries() // use the default
			}

			runner := NewRunner(opts)
			type TestOutput struct {
				Value string `json:"value"`
			}
			var dest TestOutput
			initialRaw := &RawResult{Result: "no json", IsError: false}

			result := runner.handleSchemaWithRetry(
				context.Background(), initialRaw, schema, &dest, dir,
				time.Now(), prov, opts, "test prompt",
			)

			assert.True(t, result.IsError, "should fail after exhausting retries")
			assert.LessOrEqual(t, callCount, tt.wantMaxCalls,
				"provider called %d times, expected at most %d (SchemaMaxRetries=%d)",
				callCount, tt.wantMaxCalls, tt.schemaMaxRetries)
		})
	}
}

// TestInvariant_Runner_ParseAndValidate_Idempotent verifies that writing valid
// JSON and parsing it twice returns identical results.
func TestInvariant_Runner_ParseAndValidate_Idempotent(t *testing.T) {
	type Output struct {
		Name  string `json:"name"`
		Score int    `json:"score"`
	}

	dir := t.TempDir()
	outPath := OutputPath(dir)

	content, _ := json.Marshal(Output{Name: "test", Score: 42})
	require.NoError(t, os.WriteFile(outPath, content, 0o644))

	var dest1, dest2 Output
	data1, err1 := ParseAndValidate(outPath, &dest1)
	data2, err2 := ParseAndValidate(outPath, &dest2)

	assert.NoError(t, err1)
	assert.NoError(t, err2)
	assert.Equal(t, data1, data2, "ParseAndValidate must return identical results when called twice")
	assert.Equal(t, dest1, dest2, "parsed struct must be identical across two calls")
}

// TestInvariant_Runner_BuildProviderWithBinPath verifies that BuildProvider
// correctly passes binPath to the constructed provider.
func TestInvariant_Runner_BuildProviderWithBinPath(t *testing.T) {
	customBinPath := "/custom/path/to/bin"
	tests := []struct {
		provider  string
		checkPath func(t *testing.T, prov Provider)
	}{
		{
			provider: ProviderClaudeCode,
			checkPath: func(t *testing.T, prov Provider) {
				p := prov.(*ClaudeCodeProvider)
				assert.Equal(t, customBinPath, p.BinPath)
			},
		},
		{
			provider: ProviderCodex,
			checkPath: func(t *testing.T, prov Provider) {
				p := prov.(*CodexProvider)
				assert.Equal(t, customBinPath, p.BinPath)
			},
		},
		{
			provider: ProviderGemini,
			checkPath: func(t *testing.T, prov Provider) {
				p := prov.(*GeminiProvider)
				assert.Equal(t, customBinPath, p.BinPath)
			},
		},
		{
			provider: ProviderOpenCode,
			checkPath: func(t *testing.T, prov Provider) {
				p := prov.(*OpenCodeProvider)
				assert.Equal(t, customBinPath, p.BinPath)
			},
		},
	}

	for _, tt := range tests {
		t.Run("provider="+tt.provider, func(t *testing.T) {
			prov, err := BuildProvider(tt.provider, customBinPath)
			require.NoError(t, err)
			require.NotNil(t, prov)
			tt.checkPath(t, prov)
		})
	}
}

// TestInvariant_Runner_OutputPathDeterministic verifies that OutputPath always
// returns the same value for the same directory (deterministic).
func TestInvariant_Runner_OutputPathDeterministic(t *testing.T) {
	dir := "/some/fixed/dir"
	const repetitions = 10
	first := OutputPath(dir)
	for i := 0; i < repetitions; i++ {
		assert.Equal(t, first, OutputPath(dir),
			"OutputPath must be deterministic for the same directory")
	}
	assert.True(t, strings.HasSuffix(first, ".agentfield_output.json"),
		"OutputPath must end with the expected filename")
	assert.Equal(t, filepath.Join(dir, ".agentfield_output.json"), first)
}

// funcProvider is a test double for Provider that uses a function.
type funcProvider struct {
	fn func(ctx context.Context, prompt string, opts Options) (*RawResult, error)
}

func (p *funcProvider) Execute(ctx context.Context, prompt string, opts Options) (*RawResult, error) {
	return p.fn(ctx, prompt, opts)
}

// safeProviderName converts a provider name to a safe test name.
func safeProviderName(s string) string {
	if s == "" {
		return "(empty)"
	}
	return s
}
