package harness

import (
	"context"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiProviderBuildsCommandAndMapsSuccess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture requires POSIX")
	}
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\nprintf '%s|%s\\n' \"$PWD\" \"$*\"\n")
	raw, err := NewGeminiProvider(script).Execute(context.Background(), "hello", Options{PermissionMode: "auto", Model: "gemini-2.5-pro#high", ProjectDir: dir})
	require.NoError(t, err)
	assert.Contains(t, raw.Result, dir+"|--yolo -m gemini-2.5-pro -p hello")
	assert.False(t, raw.IsError)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}

func TestGeminiProviderUsesPlanModeAndCwdPrecedence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture requires POSIX")
	}
	dir := t.TempDir()
	script := writeTestScript(t, dir, "gemini", "#!/bin/sh\nprintf '%s|%s' \"$PWD\" \"$*\"\n")
	raw, err := NewGeminiProvider(script).Execute(context.Background(), "prompt", Options{PermissionMode: "plan", Model: "model#low", Cwd: dir, ProjectDir: "/project"})
	require.NoError(t, err)
	assert.Equal(t, dir+"|--approval-mode plan -m model -p prompt", raw.Result)
}

func TestGeminiProviderMapsExecutionErrors(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture requires POSIX")
	}
	tests := []struct{ name, script, wantText string }{{"missing binary", "", "not found"}, {"empty output", "#!/bin/sh\nexit 1\n", "code 1"}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bin := "gemini-does-not-exist"
			if test.script != "" {
				bin = writeTestScript(t, t.TempDir(), "gemini", test.script)
			}
			raw, err := NewGeminiProvider(bin).Execute(context.Background(), "prompt", Options{})
			require.NoError(t, err)
			assert.True(t, raw.IsError)
			assert.Equal(t, FailureCrash, raw.FailureType)
			assert.Contains(t, raw.ErrorMessage, test.wantText)
			assert.Zero(t, raw.Metrics.NumTurns)
		})
	}
}
