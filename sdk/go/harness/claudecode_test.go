package harness

import (
	"context"
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
	assert.Equal(t, "json", gotCmd[len(gotCmd)-1], "vector ends at --output-format json, no positional prompt")
}
