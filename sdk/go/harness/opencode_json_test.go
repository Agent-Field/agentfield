package harness

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeOpenCode returns an OpenCodeProvider whose subprocess is replaced by a
// runCLI that yields the given stdout/stderr/returncode and captures the argv.
func fakeOpenCode(stdout, stderr string, code int, gotCmd *[]string) *OpenCodeProvider {
	p := NewOpenCodeProvider("opencode", "")
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		if gotCmd != nil {
			*gotCmd = cmd
		}
		return &CLIResult{Stdout: stdout, Stderr: stderr, ReturnCode: code}, nil
	}
	return p
}

// TestOpenCode_JSONEventsParsed maps to the contract: a fake CLI emitting JSON
// events yields a parsed result plus event-derived cost and turn count, and the
// argv requests --format json.
func TestOpenCode_JSONEventsParsed(t *testing.T) {
	stream := strings.Join([]string{
		`{"type":"step_start"}`,
		`{"type":"text","part":{"text":"first "}}`,
		`{"type":"step_finish","part":{"cost":0.01}}`,
		`{"type":"step_start"}`,
		`{"type":"text","part":{"text":"final answer"}}`,
		`{"type":"step_finish","part":{"cost":0.02}}`,
	}, "\n")

	var gotCmd []string
	p := fakeOpenCode(stream, "", 0, &gotCmd)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.False(t, raw.IsError)
	assert.Equal(t, "final answer", raw.Result, "step_start resets accumulated text")
	assert.Equal(t, 2, raw.Metrics.NumTurns, "one turn per step_start")
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.InDelta(t, 0.03, *raw.Metrics.CostUSD, 1e-9)
	assert.Len(t, raw.Messages, 6)

	joined := strings.Join(gotCmd, " ")
	assert.Contains(t, joined, "run --format json")
}

// TestOpenCode_ToolUseTurnFallback verifies the turn count falls back to
// tool_use events when there are no step markers, and that cost is nil when no
// step_finish carried a cost.
func TestOpenCode_ToolUseTurnFallback(t *testing.T) {
	stream := strings.Join([]string{
		`{"type":"tool_use"}`,
		`{"type":"tool_use"}`,
		`{"type":"result","result":"done"}`,
	}, "\n")

	p := fakeOpenCode(stream, "", 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.False(t, raw.IsError)
	assert.Equal(t, "done", raw.Result)
	assert.Equal(t, 2, raw.Metrics.NumTurns)
	assert.Nil(t, raw.Metrics.CostUSD, "no step_finish cost -> unknown, not zero")
}

// TestOpenCode_Exit0AuthErrorSurfaced maps to the contract: exit 0 + stderr
// "AuthenticationError..." + empty result must surface IsError carrying the
// matched message rather than silently returning empty output.
func TestOpenCode_Exit0AuthErrorSurfaced(t *testing.T) {
	stderr := "Performing one time database migration...\nError: AuthenticationError: invalid api key\nmore context\n"

	p := fakeOpenCode("", stderr, 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.True(t, raw.IsError, "auth error on exit-0 must be surfaced")
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "AuthenticationError")
	// The migration prelude should not be all that surfaces.
	assert.NotEqual(t, "Performing one time database migration...", raw.ErrorMessage)
}

// TestOpenCode_Exit0ModelNotFoundSurfaced covers the "Model not found" pattern.
func TestOpenCode_Exit0ModelNotFoundSurfaced(t *testing.T) {
	p := fakeOpenCode("", "Model not found: gpt-nonexistent", 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.True(t, raw.IsError)
	assert.Contains(t, raw.ErrorMessage, "Model not found")
}

// TestOpenCode_InBandErrorEvent verifies an in-band JSON error event is
// surfaced when there is no final result text.
func TestOpenCode_InBandErrorEvent(t *testing.T) {
	stream := `{"type":"error","message":"provider APIError: 500"}`
	p := fakeOpenCode(stream, "", 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "APIError: 500")
}

// TestOpenCode_HealthyRunUnaffected maps to the contract: a healthy run is
// unaffected. Benign stderr noise (no error markers) with a valid result does
// not flip IsError.
func TestOpenCode_HealthyRunUnaffected(t *testing.T) {
	stream := strings.Join([]string{
		`{"type":"step_start"}`,
		`{"type":"text","part":{"text":"all good"}}`,
	}, "\n")

	p := fakeOpenCode(stream, "Performing one time database migration...", 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.False(t, raw.IsError, "benign stderr + valid result must stay healthy")
	assert.Equal(t, "all good", raw.Result)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}

// TestExtractOpenCodeFinalText covers each final-text event shape.
func TestExtractOpenCodeFinalText(t *testing.T) {
	cases := []struct {
		name   string
		events []map[string]any
		want   string
	}{
		{
			name: "item.completed agent_message",
			events: []map[string]any{
				{"type": "item.completed", "item": map[string]any{"type": "agent_message", "text": "hi there"}},
			},
			want: "hi there",
		},
		{
			name:   "result field",
			events: []map[string]any{{"type": "result", "result": "R"}},
			want:   "R",
		},
		{
			name:   "result via text field",
			events: []map[string]any{{"type": "result", "text": "RT"}},
			want:   "RT",
		},
		{
			name:   "turn.completed text",
			events: []map[string]any{{"type": "turn.completed", "text": "turn done"}},
			want:   "turn done",
		},
		{
			name:   "assistant content",
			events: []map[string]any{{"type": "assistant", "content": "assistant says"}},
			want:   "assistant says",
		},
		{
			name:   "message via text field",
			events: []map[string]any{{"type": "message", "text": "msg text"}},
			want:   "msg text",
		},
		{
			name:   "text direct field accumulates",
			events: []map[string]any{{"type": "text", "text": "a"}, {"type": "text", "text": "b"}},
			want:   "ab",
		},
		{
			name:   "text content field",
			events: []map[string]any{{"type": "text", "content": "cc"}},
			want:   "cc",
		},
		{
			name:   "no final text",
			events: []map[string]any{{"type": "step_start"}, {"type": "tool_use"}},
			want:   "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, extractOpenCodeFinalText(tc.events))
		})
	}
}

// TestExtractOpenCodeEventError covers the error-event extraction shapes.
func TestExtractOpenCodeEventError(t *testing.T) {
	t.Run("top-level error key", func(t *testing.T) {
		got := extractOpenCodeEventError([]map[string]any{{"type": "error", "error": "boom"}})
		assert.Equal(t, "boom", got)
	})
	t.Run("part-nested message", func(t *testing.T) {
		got := extractOpenCodeEventError([]map[string]any{
			{"type": "error", "part": map[string]any{"message": "nested failure"}},
		})
		assert.Equal(t, "nested failure", got)
	})
	t.Run("marshal fallback when no known key", func(t *testing.T) {
		got := extractOpenCodeEventError([]map[string]any{{"type": "error", "code": 500.0}})
		assert.Contains(t, got, "500")
	})
	t.Run("no error event", func(t *testing.T) {
		assert.Equal(t, "", extractOpenCodeEventError([]map[string]any{{"type": "text", "text": "x"}}))
	})
}

// TestOpenCode_PlainTextFallback verifies non-JSON stdout still surfaces as the
// result (older opencode versions / degraded output).
func TestOpenCode_PlainTextFallback(t *testing.T) {
	p := fakeOpenCode("just plain text\n", "", 0, nil)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)

	assert.False(t, raw.IsError)
	assert.Equal(t, "just plain text", raw.Result)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}
