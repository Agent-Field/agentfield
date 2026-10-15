package harness

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// previewTestDir returns a temp directory whose path contains no digits. The
// runner's transient classifier matches substrings such as "500" anywhere in
// the error text, and a path with random digits can match by accident, so the
// fixture keeps the executable path free of them.
func previewTestDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "agentfieldpreview")
	require.NoError(t, err)
	safe := filepath.Join(filepath.Dir(dir), strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return 'x'
		}
		return r
	}, filepath.Base(dir)))
	require.NoError(t, os.Rename(dir, safe))
	t.Cleanup(func() { os.RemoveAll(safe) })
	require.False(t, isTransient(safe), "temp fixture path looks transient: %s", safe)
	return safe
}

func TestFormatCommandPreviewNamesOnlyTheExecutable(t *testing.T) {
	assert.Equal(t, "", formatCommandPreview(nil))
	assert.Equal(t, "/usr/local/bin/opencode", formatCommandPreview([]string{"/usr/local/bin/opencode"}))
	assert.Equal(t, "opencode [6 args omitted]", formatCommandPreview([]string{
		"opencode", "run", "--format", "json", "--agent", "agentfield-harness", "return 500 error",
	}))
}

func TestNoProgressErrorOmitsCommandArguments(t *testing.T) {
	dir := previewTestDir(t)
	script := writeTestScript(t, dir, "opencode-hang", "#!/bin/sh\nsleep 30\n")

	_, err := runCLIWithStdinIdle(context.Background(), []string{script, "run", "return 500 error"}, nil, dir, 0, 1, nil)

	require.Error(t, err)
	message := err.Error()
	assert.Contains(t, message, "CLI command made no progress for 1s: ")
	assert.Contains(t, message, script+" [2 args omitted]")
	assert.NotContains(t, message, "return 500 error")
	assert.False(t, isTransient(message), "prompt text must not make a no-progress error retryable")
}

func TestNoProgressWithStatusLikePromptRunsOnce(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("opencode sends the prompt over stdin on Windows")
	}
	dir := previewTestDir(t)
	counter := filepath.Join(dir, "runs")
	script := writeTestScript(t, dir, "opencode-hang", "#!/bin/sh\necho x >> "+counter+"\nsleep 30\n")
	t.Setenv("AGENTFIELD_HARNESS_IDLE_SECONDS", "1")

	provider := NewOpenCodeProvider(script, "")
	runner := NewRunner(Options{Provider: "opencode"})
	opts := Options{
		Provider:     "opencode",
		MaxRetries:   3,
		InitialDelay: 0.001,
		MaxDelay:     0.002,
	}

	_, err := runner.executeWithRetry(context.Background(), provider, "return 500 error", opts)
	require.Error(t, err)

	data, readErr := os.ReadFile(counter)
	require.NoError(t, readErr)
	assert.Equal(t, 1, len(strings.Fields(string(data))), "a timed out run must not be retried because of prompt text")
}

func TestTransientStatusErrorStillRetries(t *testing.T) {
	calls := 0
	provider := &funcProvider{fn: func(_ context.Context, _ string, _ Options) (*RawResult, error) {
		calls++
		if calls == 1 {
			return &RawResult{IsError: true, ErrorMessage: "Error 503", FailureType: FailureAPIError}, nil
		}
		return &RawResult{Result: "ok", Metrics: Metrics{NumTurns: 1}}, nil
	}}
	runner := NewRunner(Options{Provider: "opencode"})
	opts := Options{Provider: "opencode", MaxRetries: 3, InitialDelay: 0.001, MaxDelay: 0.002}

	raw, err := runner.executeWithRetry(context.Background(), provider, "hello", opts)

	require.NoError(t, err)
	require.NotNil(t, raw)
	assert.False(t, raw.IsError)
	assert.Equal(t, 2, calls)
}
