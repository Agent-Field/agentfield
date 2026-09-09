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

// mockProvider is a test double for Provider.
type mockProvider struct {
	results []*RawResult
	errors  []error
	calls   int
}

func (m *mockProvider) Execute(_ context.Context, prompt string, opts Options) (*RawResult, error) {
	idx := m.calls
	m.calls++
	if idx < len(m.errors) && m.errors[idx] != nil {
		return nil, m.errors[idx]
	}
	if idx < len(m.results) {
		return m.results[idx], nil
	}
	return &RawResult{IsError: true, ErrorMessage: "no more results"}, nil
}

func TestRunner_Run_NoSchema(t *testing.T) {
	runner := NewRunner(Options{Provider: "opencode"})

	// We can't easily test with real providers, so test the merge/validation logic
	t.Run("empty provider defaults to aforge", func(t *testing.T) {
		t.Setenv(ProviderEnvVar, "")
		r := NewRunner(Options{BinPath: filepath.Join(t.TempDir(), "missing-aforge")})
		result, err := r.Run(context.Background(), "test", nil, nil, Options{})
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.True(t, result.IsError)
		assert.Contains(t, result.ErrorMessage, "missing-aforge")
		assert.NotContains(t, result.ErrorMessage, "no harness provider specified")
	})

	t.Run("unknown provider", func(t *testing.T) {
		_, err := runner.Run(context.Background(), "test", nil, nil, Options{Provider: "unknown"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unknown harness provider")
	})
}

func TestRunner_MergeOptions(t *testing.T) {
	runner := NewRunner(Options{
		Provider:       "opencode",
		Model:          "default-model",
		MaxTurns:       10,
		PermissionMode: "plan",
		Env:            map[string]string{"A": "1"},
	})

	merged := runner.mergeOptions(Options{
		Model: "override-model",
		Env:   map[string]string{"B": "2"},
	})

	assert.Equal(t, "opencode", merged.Provider)
	assert.Equal(t, "override-model", merged.Model)
	assert.Equal(t, 10, merged.MaxTurns)
	assert.Equal(t, "plan", merged.PermissionMode)
	assert.Equal(t, "1", merged.Env["A"])
	assert.Equal(t, "2", merged.Env["B"])
}

func TestRunner_HandleSchemaWithRetry_FirstAttemptSuccess(t *testing.T) {
	dir := t.TempDir()

	type TestOutput struct {
		Finding  string `json:"finding"`
		Severity string `json:"severity"`
	}

	// Write a valid output file
	outputPath := OutputPath(dir)
	content, _ := json.Marshal(TestOutput{Finding: "bug", Severity: "high"})
	err := os.WriteFile(outputPath, content, 0o644)
	require.NoError(t, err)

	runner := NewRunner(Options{Provider: "opencode"})
	mock := &mockProvider{}

	var dest TestOutput
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"finding":  map[string]any{"type": "string"},
			"severity": map[string]any{"type": "string"},
		},
	}

	raw := &RawResult{Result: "done", Metrics: Metrics{NumTurns: 1}}
	result := runner.handleSchemaWithRetry(
		context.Background(), raw, schema, &dest, dir,
		time.Now(), mock, Options{Provider: "opencode"}, "test prompt", false,
	)

	assert.False(t, result.IsError)
	assert.Equal(t, "bug", dest.Finding)
	assert.Equal(t, "high", dest.Severity)
	assert.Equal(t, 0, mock.calls) // No retries needed
}

func TestRunner_HandleSchemaWithRetry_StdoutFallback(t *testing.T) {
	dir := t.TempDir()

	type TestOutput struct {
		Status string `json:"status"`
	}

	// No output file — should fall back to parsing stdout
	runner := NewRunner(Options{Provider: "opencode"})
	mock := &mockProvider{}

	var dest TestOutput
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string"},
		},
	}

	raw := &RawResult{
		Result:  `Here is the result: {"status": "ok"} done.`,
		Metrics: Metrics{NumTurns: 1},
	}

	result := runner.handleSchemaWithRetry(
		context.Background(), raw, schema, &dest, dir,
		time.Now(), mock, Options{Provider: "opencode"}, "test prompt", false,
	)

	assert.False(t, result.IsError)
	assert.Equal(t, "ok", dest.Status)
}

func TestRunner_HandleSchemaWithRetry_RetrySuccess(t *testing.T) {
	dir := t.TempDir()

	type TestOutput struct {
		Value string `json:"value"`
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
	}

	// First attempt: no output file
	runner := NewRunner(Options{Provider: "opencode"})

	// Mock will be called for retry — write file on retry
	outputPath := OutputPath(dir)
	mock := &mockProvider{
		results: []*RawResult{
			{
				Result:  "retry success",
				Metrics: Metrics{NumTurns: 1, SessionID: "sess-1"},
			},
		},
	}

	// Write the output file that the retry "creates"
	content, _ := json.Marshal(TestOutput{Value: "retried"})
	os.WriteFile(outputPath, content, 0o644)

	var dest TestOutput
	initialRaw := &RawResult{
		Result:  "no json here",
		Metrics: Metrics{NumTurns: 1},
	}

	// Remove output file to simulate first failure, then recreate for retry
	os.Remove(outputPath)

	// Actually for this test, let's have the mock "create" the file
	mock2 := &writerMockProvider{
		inner:      mock.Execute,
		outputPath: outputPath,
		content:    content,
		results: []*RawResult{
			{Result: "done", Metrics: Metrics{NumTurns: 1}},
		},
	}

	result := runner.handleSchemaWithRetry(
		context.Background(), initialRaw, schema, &dest, dir,
		time.Now(), mock2, Options{Provider: "opencode", SchemaMaxRetries: 2}, "test prompt", false,
	)

	assert.False(t, result.IsError)
	assert.Equal(t, "retried", dest.Value)
}

// writerMockProvider writes an output file on execute (simulating the coding agent).
type writerMockProvider struct {
	inner      func(context.Context, string, Options) (*RawResult, error)
	outputPath string
	content    []byte
	results    []*RawResult
	calls      int
}

func (m *writerMockProvider) Execute(_ context.Context, _ string, _ Options) (*RawResult, error) {
	idx := m.calls
	m.calls++
	// Write the output file
	os.WriteFile(m.outputPath, m.content, 0o644)
	if idx < len(m.results) {
		return m.results[idx], nil
	}
	return &RawResult{Result: "done"}, nil
}

func TestRunner_HandleSchemaWithRetry_AllRetriesFail(t *testing.T) {
	dir := t.TempDir()

	type TestOutput struct {
		Value string `json:"value"`
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
	}

	runner := NewRunner(Options{Provider: "opencode"})
	mock := &mockProvider{
		results: []*RawResult{
			{Result: "bad output", Metrics: Metrics{NumTurns: 1}},
			{Result: "still bad", Metrics: Metrics{NumTurns: 1}},
		},
	}

	var dest TestOutput
	initialRaw := &RawResult{Result: "no json", Metrics: Metrics{NumTurns: 1}}

	result := runner.handleSchemaWithRetry(
		context.Background(), initialRaw, schema, &dest, dir,
		time.Now(), mock, Options{Provider: "opencode", SchemaMaxRetries: 2}, "test prompt", false,
	)

	assert.True(t, result.IsError)
	assert.Equal(t, FailureSchema, result.FailureType)
	assert.Contains(t, result.ErrorMessage, "Schema validation failed")
}

func TestIsTransient(t *testing.T) {
	assert.True(t, isTransient("rate limit exceeded"))
	assert.True(t, isTransient("Error 503: service unavailable"))
	assert.True(t, isTransient("connection refused"))
	assert.False(t, isTransient("invalid input"))
	assert.False(t, isTransient("permission denied"))
}

func TestOptionsDefaults(t *testing.T) {
	opts := Options{}
	assert.Equal(t, 3, opts.maxRetries())
	assert.Equal(t, 1.0, opts.initialDelay())
	assert.Equal(t, 30.0, opts.maxDelay())
	assert.Equal(t, 2.0, opts.backoffFactor())
	assert.Equal(t, 2, opts.schemaMaxRetries())
	assert.Equal(t, 600, opts.timeout())

	opts2 := Options{
		MaxRetries:       5,
		InitialDelay:     2.0,
		MaxDelay:         60.0,
		BackoffFactor:    3.0,
		SchemaMaxRetries: 4,
		Timeout:          300,
	}
	assert.Equal(t, 5, opts2.maxRetries())
	assert.Equal(t, 2.0, opts2.initialDelay())
	assert.Equal(t, 60.0, opts2.maxDelay())
	assert.Equal(t, 3.0, opts2.backoffFactor())
	assert.Equal(t, 4, opts2.schemaMaxRetries())
	assert.Equal(t, 300, opts2.timeout())
}

func TestNewOpenCodeProvider(t *testing.T) {
	p := NewOpenCodeProvider("", "")
	assert.Equal(t, "opencode", p.BinPath)

	p2 := NewOpenCodeProvider("/usr/bin/opencode", "http://localhost:3000")
	assert.Equal(t, "/usr/bin/opencode", p2.BinPath)
	assert.Equal(t, "http://localhost:3000", p2.ServerURL)
}

func TestNewClaudeCodeProvider(t *testing.T) {
	p := NewClaudeCodeProvider("")
	assert.Equal(t, "claude", p.BinPath)

	p2 := NewClaudeCodeProvider("/usr/bin/claude")
	assert.Equal(t, "/usr/bin/claude", p2.BinPath)
}

func TestOpenCodeProvider_BinaryNotFound(t *testing.T) {
	p := NewOpenCodeProvider("/nonexistent/binary/opencode-fake", "")
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err) // Error is in RawResult, not returned
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "not found")
}

func TestClaudeCodeProvider_BinaryNotFound(t *testing.T) {
	p := NewClaudeCodeProvider("/nonexistent/binary/claude-fake")
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "not found")
}

// Helper to create a simple test script that outputs JSON
func writeTestScript(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	err := os.WriteFile(path, []byte(content), 0o755)
	require.NoError(t, err)
	return path
}

func TestOpenCodeProvider_SuccessfulExecution(t *testing.T) {
	dir := t.TempDir()
	// Create a fake opencode binary that outputs text
	script := writeTestScript(t, dir, "opencode", "#!/bin/sh\necho 'Hello from opencode'\n")

	p := NewOpenCodeProvider(script, "")
	raw, err := p.Execute(context.Background(), "test prompt", Options{})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Contains(t, raw.Result, "Hello from opencode")
}

func TestClaudeCodeProvider_SuccessfulExecution(t *testing.T) {
	dir := t.TempDir()
	// Create a fake claude binary that outputs JSON
	script := writeTestScript(t, dir, "claude",
		`#!/bin/sh
echo '{"type":"result","result":"test output","session_id":"s123","num_turns":3}'
`)

	p := NewClaudeCodeProvider(script)
	raw, err := p.Execute(context.Background(), "test prompt", Options{})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, "test output", raw.Result)
	assert.Equal(t, "s123", raw.Metrics.SessionID)
	assert.Equal(t, 3, raw.Metrics.NumTurns)
}

func TestOpenCodeProvider_NonZeroExit(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "opencode", "#!/bin/sh\necho 'error output' >&2\nexit 1\n")

	p := NewOpenCodeProvider(script, "")
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
}

func TestOpenCodeProvider_WithOptions(t *testing.T) {
	dir := t.TempDir()
	// Script that echoes its arguments
	script := writeTestScript(t, dir, "opencode", "#!/bin/sh\necho \"args: $@\"\n")

	p := NewOpenCodeProvider(script, "")
	raw, err := p.Execute(context.Background(), "test prompt", Options{
		Model:        "my-model",
		ProjectDir:   "/some/project",
		SystemPrompt: "Be helpful",
	})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)
	// The script should have received --model and --dir flags
	assert.Contains(t, raw.Result, "args:")
}

func TestOpenCodeProvider_OpenRouterConfigOverlay(t *testing.T) {
	p := NewOpenCodeProvider("opencode", "")
	var capturedEnv map[string]string
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, _ []byte) (*CLIResult, error) {
		capturedEnv = env
		return &CLIResult{Stdout: "ok\n", ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "test prompt", Options{
		Model: "openrouter/openai/gpt-4o",
	})
	require.NoError(t, err)
	require.False(t, raw.IsError)

	var overlay map[string]any
	require.NoError(t, json.Unmarshal([]byte(capturedEnv["OPENCODE_CONFIG_CONTENT"]), &overlay))
	provider := overlay["provider"].(map[string]any)
	openrouter := provider["openrouter"].(map[string]any)
	models := openrouter["models"].(map[string]any)
	model := models["openai/gpt-4o"].(map[string]any)
	headers := model["headers"].(map[string]any)
	assert.Equal(t, "https://agentfield.ai", headers["HTTP-Referer"])
	assert.Equal(t, "AgentField AI", headers["X-OpenRouter-Title"])
	assert.Equal(t, "AgentField AI", headers["X-Title"])
	assert.Equal(t, defaultOpenRouterCategories, headers["X-OpenRouter-Categories"])
}

// --- Codex provider tests ---

func TestNewCodexProvider(t *testing.T) {
	p := NewCodexProvider("")
	assert.Equal(t, "codex", p.BinPath)

	p2 := NewCodexProvider("/usr/bin/codex")
	assert.Equal(t, "/usr/bin/codex", p2.BinPath)
}

func TestCodexProvider_BinaryNotFound(t *testing.T) {
	p := NewCodexProvider("/nonexistent/binary/codex-fake")
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "not found")
}

func TestCodexProvider_SuccessfulExecution(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "codex",
		`#!/bin/sh
echo '{"type":"thread.started","thread_id":"019d-abc"}'
echo '{"type":"turn.started"}'
echo '{"type":"item.completed","item":{"type":"agent_message","text":"codex result"}}'
echo '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":10}}'
`)

	p := NewCodexProvider(script)
	raw, err := p.Execute(context.Background(), "test prompt", Options{})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, "codex result", raw.Result)
	assert.Equal(t, "019d-abc", raw.Metrics.SessionID)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}

func TestCodexProvider_NonZeroExit(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "codex", "#!/bin/sh\necho 'error' >&2\nexit 1\n")

	p := NewCodexProvider(script)
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
}

// TestCodexProvider_AutoPermissionUsesBypassNotFullAuto verifies the deprecated
// --full-auto flag is gone and the "auto" permission mode maps to the bypass
// flag (codex_harness_patch.py:165-166). Argv is captured via an injected
// runCLI rather than echoed to stdout, because the provider no longer surfaces
// raw stdout as the result.
func TestCodexProvider_AutoPermissionUsesBypassNotFullAuto(t *testing.T) {
	var gotCmd []string
	var gotStdin []byte
	p := NewCodexProvider("codex")
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, stdin []byte) (*CLIResult, error) {
		gotCmd = cmd
		gotStdin = stdin
		return &CLIResult{Stdout: `{"type":"result","result":"done"}`, ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "test prompt", Options{
		Model:          "gpt-5.5",
		PermissionMode: "auto",
	})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)

	joined := strings.Join(gotCmd, " ")
	assert.Contains(t, joined, "-m gpt-5.5")
	assert.Contains(t, gotCmd, "--dangerously-bypass-approvals-and-sandbox")
	assert.Contains(t, gotCmd, "--skip-git-repo-check")
	assert.NotContains(t, gotCmd, "--full-auto")
	// The prompt rides on stdin, never as a positional argv entry.
	assert.Equal(t, "test prompt", string(gotStdin))
	assert.NotContains(t, gotCmd, "test prompt")
}

// --- Gemini provider tests ---

func TestNewGeminiProvider(t *testing.T) {
	p := NewGeminiProvider("")
	assert.Equal(t, "gemini", p.BinPath)

	p2 := NewGeminiProvider("/usr/bin/gemini")
	assert.Equal(t, "/usr/bin/gemini", p2.BinPath)
}

func TestGeminiProvider_BinaryNotFound(t *testing.T) {
	p := NewGeminiProvider("/nonexistent/binary/gemini-fake")
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "not found")
}

func TestGeminiProvider_SuccessfulExecution(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\necho 'Hello from gemini'\n")

	p := NewGeminiProvider(script)
	raw, err := p.Execute(context.Background(), "test prompt", Options{})
	assert.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Contains(t, raw.Result, "Hello from gemini")
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}

func TestGeminiProvider_NonZeroExit(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\necho 'error' >&2\nexit 1\n")

	p := NewGeminiProvider(script)
	raw, err := p.Execute(context.Background(), "test", Options{})
	assert.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
}

func TestGeminiProvider_WithModel(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\nprintf '%s\\n' \"$@\"\n")

	tests := []struct {
		name    string
		options Options
		want    []string
	}{
		{
			name:    "auto uses yolo",
			options: Options{Cwd: dir, PermissionMode: "auto"},
			want:    []string{"--yolo", "-p", "test prompt"},
		},
		{
			name:    "plan precedes stripped model and prompt",
			options: Options{PermissionMode: "plan", Model: "gemini-2.5-pro#high"},
			want:    []string{"--approval-mode", "plan", "-m", "gemini-2.5-pro", "-p", "test prompt"},
		},
		{
			name: "unset permission mode adds no flag",
			want: []string{"-p", "test prompt"},
		},
		{
			name:    "unknown permission mode adds no flag",
			options: Options{PermissionMode: "unknown", Model: "gemini-2.5-flash"},
			want:    []string{"-m", "gemini-2.5-flash", "-p", "test prompt"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := NewGeminiProvider(script).Execute(context.Background(), "test prompt", tc.options)
			require.NoError(t, err)
			assert.False(t, raw.IsError)
			assert.Equal(t, tc.want, strings.Split(raw.Result, "\n"))
			assert.NotContains(t, raw.Result, "-C")
			assert.NotContains(t, raw.Result, "--sandbox")
		})
	}
}

// --- BuildProvider factory tests ---

func TestBuildProvider(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		wantType string
		wantErr  bool
	}{
		{"aforge", "aforge", "*harness.AforgeProvider", false},
		{"claude-code", "claude-code", "*harness.ClaudeCodeProvider", false},
		{"codex", "codex", "*harness.CodexProvider", false},
		{"gemini", "gemini", "*harness.GeminiProvider", false},
		{"opencode", "opencode", "*harness.OpenCodeProvider", false},
		{"pi", "pi", "*harness.PiProvider", false},
		{"omp", "omp", "*harness.OMPProvider", false},
		{"unknown", "unknown-agent", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := BuildProvider(tt.provider, "")
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unknown harness provider")
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantType, fmt.Sprintf("%T", p))
			}
		})
	}
}

func TestRunner_BuildProvider_UsesFactory(t *testing.T) {
	// Verify the runner can build every registered provider.
	for _, name := range []string{"aforge", "claude-code", "codex", "gemini", "opencode", "pi", "omp"} {
		t.Run(name, func(t *testing.T) {
			runner := NewRunner(Options{Provider: name})
			provider, err := runner.buildProvider(runner.DefaultOptions)
			require.NoError(t, err)
			assert.NotNil(t, provider)
		})
	}
}

// TestOpenCodeProvider_OpenRouterOverlayUsesBaseModel ensures the OpenRouter
// attribution overlay keys off the BASE model: a "#variant" reasoning-effort
// suffix neither defeats the openrouter/ prefix match nor leaks into the
// per-model config overlay key.
func TestOpenCodeProvider_OpenRouterOverlayUsesBaseModel(t *testing.T) {
	p := NewOpenCodeProvider("opencode", "")
	var capturedEnv map[string]string
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, _ []byte) (*CLIResult, error) {
		capturedEnv = env
		return &CLIResult{Stdout: "ok\n", ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "test prompt", Options{
		Model: "openrouter/openai/gpt-4o#high",
	})
	require.NoError(t, err)
	require.False(t, raw.IsError)

	var overlay map[string]any
	require.NoError(t, json.Unmarshal([]byte(capturedEnv["OPENCODE_CONFIG_CONTENT"]), &overlay))
	models := overlay["provider"].(map[string]any)["openrouter"].(map[string]any)["models"].(map[string]any)
	_, hasBase := models["openai/gpt-4o"]
	assert.True(t, hasBase, "overlay must key the BASE model slug")
	_, hasSuffixed := models["openai/gpt-4o#high"]
	assert.False(t, hasSuffixed, "the #variant suffix must not leak into the overlay key")
}
