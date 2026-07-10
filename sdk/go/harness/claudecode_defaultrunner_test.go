package harness

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClaudeCodeProvider_NilRunCLIUsesDefaultRunner covers the runCLI == nil
// fallback in Execute. NewClaudeCodeProvider always wires runCLI to
// RunCLIWithStdin, so a provider built via the constructor never exercises the
// fallback. A provider constructed as a struct literal (e.g. zero-valued
// runCLI) must still run: Execute defaults runCLI to RunCLIWithStdin and drives
// the real subprocess.
//
// Contract: with runCLI unset, Execute delivers the prompt to the default
// runner, runs the binary, and parses its stream-json result.
func TestClaudeCodeProvider_NilRunCLIUsesDefaultRunner(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "claude",
		`#!/bin/sh
echo '{"type":"result","result":"defaulted","session_id":"s-default","num_turns":1}'
`)

	// Struct literal (not NewClaudeCodeProvider) leaves runCLI nil, forcing the
	// default-runner path in Execute.
	p := &ClaudeCodeProvider{BinPath: script}
	require.Nil(t, p.runCLI, "precondition: runCLI must be nil to exercise the fallback")

	raw, err := p.Execute(context.Background(), "hello", Options{})
	require.NoError(t, err)
	require.False(t, raw.IsError, "unexpected error: %s", raw.ErrorMessage)
	assert.Equal(t, "defaulted", raw.Result)
	assert.Equal(t, "s-default", raw.Metrics.SessionID)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}
