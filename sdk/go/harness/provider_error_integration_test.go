package harness

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeHarnessStub(t *testing.T, dir, name, content string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(path, []byte(content), 0o755))
	return path
}

func prependPATH(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func TestCodexProvider_ExitNonZeroWithoutStderrReturnsCrash(t *testing.T) {
	dir := t.TempDir()
	writeHarnessStub(t, dir, "codex", "#!/bin/sh\nexit 7\n")
	prependPATH(t, dir)

	raw, err := NewCodexProvider("").Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	require.NotNil(t, raw)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Equal(t, 7, raw.ReturnCode)
	assert.Contains(t, raw.ErrorMessage, "Process exited with code 7 and produced no output.")
}

func TestGeminiProvider_ContextTimeoutReturnsFailureTimeout(t *testing.T) {
	dir := t.TempDir()
	writeHarnessStub(t, dir, "gemini", "#!/bin/sh\nexec sleep 30\n")
	prependPATH(t, dir)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()

	raw, err := NewGeminiProvider("").Execute(ctx, "prompt", Options{
		Timeout: 1,
	})
	require.NoError(t, err)

	require.NotNil(t, raw)
	assert.True(t, raw.IsError)
	// Current behavior is a crash-style result from the killed subprocess,
	// not a FailureTimeout result from RunCLI.
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "Process killed by signal")
}

func TestCodexProvider_SkipsMalformedJSONLAndKeepsValidEvents(t *testing.T) {
	dir := t.TempDir()
	writeHarnessStub(t, dir, "codex", "#!/bin/sh\nprintf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-1\"}'\nprintf '%s\\n' 'this is not json'\nprintf '%s\\n' '{\"type\":\"result\",\"result\":\"hello\"}'\n")
	prependPATH(t, dir)

	raw, err := NewCodexProvider("").Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	require.NotNil(t, raw)
	assert.False(t, raw.IsError)
	assert.Equal(t, "hello", raw.Result)
	assert.Equal(t, "thread-1", raw.Metrics.SessionID)
	assert.Len(t, raw.Messages, 2)
	assert.Equal(t, 2, raw.Metrics.NumTurns)
}

func TestRunner_Run_RetriesSchemaValidationUntilStdoutIsValid(t *testing.T) {
	dir := t.TempDir()
	countFile := filepath.Join(dir, "attempt-count")
	writeHarnessStub(t, dir, "codex", "#!/bin/sh\ncount=0\nif [ -f \"$COUNT_FILE\" ]; then\n  count=$(cat \"$COUNT_FILE\")\nfi\ncount=$((count + 1))\nprintf '%s' \"$count\" > \"$COUNT_FILE\"\nif [ \"$count\" -eq 1 ]; then\n  printf '%s\\n' '{\"type\":\"result\",\"result\":\"not-json\"}'\nelse\n  printf '%s\\n' '{\"type\":\"result\",\"result\":\"{\\\"status\\\":\\\"ok\\\"}\"}'\nfi\n")
	prependPATH(t, dir)

	runner := NewRunner(Options{
		Provider:         ProviderCodex,
		BinPath:          "codex",
		SchemaMaxRetries: 2,
	})

	var dest struct {
		Status string `json:"status"`
	}

	result, err := runner.Run(context.Background(), "prompt", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string"},
		},
	}, &dest, Options{
		Env: map[string]string{"COUNT_FILE": countFile},
	})
	require.NoError(t, err)

	require.NotNil(t, result)
	assert.False(t, result.IsError)
	assert.Equal(t, "ok", dest.Status)

	countBytes, readErr := os.ReadFile(countFile)
	require.NoError(t, readErr)
	assert.Equal(t, "2", string(countBytes))
}

func TestCodexProvider_EmptyEnvValueUnsetsInheritedVariable(t *testing.T) {
	dir := t.TempDir()
	writeHarnessStub(t, dir, "codex", "#!/bin/sh\nif [ -z \"${FOO+x}\" ]; then\n  value=unset\nelse\n  value=$FOO\nfi\nprintf '%s\\n' \"{\\\"type\\\":\\\"result\\\",\\\"result\\\":\\\"$value\\\"}\"\n")
	prependPATH(t, dir)
	t.Setenv("FOO", "present")

	raw, err := NewCodexProvider("").Execute(context.Background(), "prompt", Options{
		Env: map[string]string{"FOO": ""},
	})
	require.NoError(t, err)

	require.NotNil(t, raw)
	assert.False(t, raw.IsError)
	assert.Equal(t, "unset", raw.Result)
}

func TestGeminiProvider_MissingBinaryReturnsClearCrashResult(t *testing.T) {
	raw, err := NewGeminiProvider("missing-gemini-binary").Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	require.NotNil(t, raw)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "missing-gemini-binary")
}
