package harness

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClaudeCodeProvider_PromptDeliveredViaStdin pins the fix for the variadic
// --allowedTools bug in the claude CLI (verified against 2.1.191).
//
// Contract:
//   - The prompt must be handed to the subprocess on stdin.
//   - The prompt must NOT appear as an argument, in particular not as a trailing
//     positional after --allowedTools — claude's --allowedTools is variadic and
//     greedily absorbs a following positional, leaving `--print` with no prompt
//     and a non-zero exit ("Input must be provided ... when using --print").
func TestClaudeCodeProvider_PromptDeliveredViaStdin(t *testing.T) {
	p := NewClaudeCodeProvider("claude")

	var gotCmd []string
	var gotStdin []byte
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, stdin []byte) (*CLIResult, error) {
		gotCmd = append([]string(nil), cmd...)
		gotStdin = append([]byte(nil), stdin...)
		return &CLIResult{
			Stdout:     `{"type":"result","result":"OK","session_id":"s1","num_turns":1}`,
			ReturnCode: 0,
		}, nil
	}

	const prompt = "please reply with exactly OK"
	raw, err := p.Execute(context.Background(), prompt, Options{
		Model: "haiku",
		Tools: []string{"Read", "Write"},
	})
	require.NoError(t, err)
	require.False(t, raw.IsError, "unexpected error: %s", raw.ErrorMessage)

	// 1. Prompt delivered via stdin.
	assert.Equal(t, prompt, string(gotStdin), "prompt must be piped to the CLI on stdin")

	// 2. Prompt must not appear anywhere in the arg vector.
	assert.NotContains(t, gotCmd, prompt, "prompt must not be passed as a CLI argument")

	// 3. The arg vector must end at the last --allowedTools value, never at a
	//    positional prompt — this is the exact regression being guarded.
	require.NotEmpty(t, gotCmd)
	assert.Equal(t, "Write", gotCmd[len(gotCmd)-1],
		"arg vector must end at the last --allowedTools value, not a trailing positional prompt")

	// Sanity: the flags we expect are still present.
	assert.Contains(t, gotCmd, "--allowedTools")
	assert.Contains(t, gotCmd, "--print")

	// 4. Watchdog contract: the CLI must stream per-message events so RunCLI's
	//    idle watchdog sees progress — plain `--output-format json` is silent
	//    until completion and any turn quieter than the idle window was
	//    SIGKILLed mid-run. In --print mode the claude CLI hard-requires
	//    --verbose alongside stream-json.
	assert.Contains(t, gotCmd, "--verbose")
	for i, arg := range gotCmd {
		if arg == "--output-format" {
			require.Greater(t, len(gotCmd), i+1)
			assert.Equal(t, "stream-json", gotCmd[i+1],
				"output format must be stream-json to keep the idle watchdog fed")
		}
	}
}

// TestClaudeCodeProvider_EmptyToolsStillStdin ensures the prompt is delivered on
// stdin even when no tools are set (no trailing positional in any case).
func TestClaudeCodeProvider_EmptyToolsStillStdin(t *testing.T) {
	p := NewClaudeCodeProvider("claude")

	var gotCmd []string
	var gotStdin []byte
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, stdin []byte) (*CLIResult, error) {
		gotCmd = append([]string(nil), cmd...)
		gotStdin = append([]byte(nil), stdin...)
		return &CLIResult{Stdout: `{"type":"result","result":"OK"}`, ReturnCode: 0}, nil
	}

	const prompt = "hello there"
	_, err := p.Execute(context.Background(), prompt, Options{})
	require.NoError(t, err)

	assert.Equal(t, prompt, string(gotStdin))
	assert.NotContains(t, gotCmd, prompt)
	require.NotEmpty(t, gotCmd)
	assert.Equal(t, "--verbose", gotCmd[len(gotCmd)-1],
		"vector ends at the base flags (--output-format stream-json --verbose), no positional prompt")
}

// TestClaudeCodeProvider_ParsesStreamJSONFixture asserts the parser against a
// REAL captured stream: testdata/claude_stream_json_2.1.191.jsonl was produced
// by `claude --print --output-format stream-json --verbose --model haiku
// --allowedTools Read --allowedTools Write` (claude CLI 2.1.191) with the
// prompt "reply with just OK" delivered on stdin. Only environment-identifying
// inventories in the system/init event were sanitized; the event sequence and
// the terminal result event are byte-real.
//
// Contract (unchanged from the plain-json output format):
//   - Result is the final result event's "result" text — not the raw stream.
//   - SessionID / CostUSD / NumTurns come from the final result event.
//   - Every stream event is retained in Messages (Python-provider parity: the
//     Python claude provider appends every SDK stream message).
func TestClaudeCodeProvider_ParsesStreamJSONFixture(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("testdata", "claude_stream_json_2.1.191.jsonl"))
	require.NoError(t, err)

	t.Run("parser consumes the raw capture", func(t *testing.T) {
		raw := &RawResult{}
		NewClaudeCodeProvider("").parseJSONOutput(string(fixture), raw)

		assert.Equal(t, "OK", raw.Result, "Result must be the final result event's text")
		assert.Equal(t, "48a64026-56f9-4203-8220-1c099dd8378a", raw.Metrics.SessionID)
		require.NotNil(t, raw.Metrics.CostUSD)
		assert.InDelta(t, 0.0326441, *raw.Metrics.CostUSD, 1e-9)
		assert.Equal(t, 1, raw.Metrics.NumTurns)
		assert.Len(t, raw.Messages, 11,
			"all stream events retained (init, rate_limit, thinking deltas, assistant, result)")
	})

	t.Run("Execute end-to-end over a subprocess emitting the capture", func(t *testing.T) {
		dir := t.TempDir()
		streamPath := filepath.Join(dir, "stream.jsonl")
		require.NoError(t, os.WriteFile(streamPath, fixture, 0o644))
		script := writeTestScript(t, dir, "claude-stream",
			"#!/bin/sh\ncat "+streamPath+"\n")

		raw, err := NewClaudeCodeProvider(script).Execute(context.Background(), "reply with just OK", Options{
			Tools: []string{"Read", "Write"},
		})
		require.NoError(t, err)
		require.False(t, raw.IsError, "unexpected error: %s", raw.ErrorMessage)
		assert.Equal(t, "OK", raw.Result)
		assert.Equal(t, "48a64026-56f9-4203-8220-1c099dd8378a", raw.Metrics.SessionID)
		require.NotNil(t, raw.Metrics.CostUSD)
		assert.InDelta(t, 0.0326441, *raw.Metrics.CostUSD, 1e-9)
		assert.Equal(t, 1, raw.Metrics.NumTurns)
	})
}
