package harness

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeCursorCLI returns a runCLI stub that records the command it was invoked
// with and returns the supplied result.
func fakeCursorCLI(captured *[]string, result *CLIResult, err error) func(context.Context, []string, map[string]string, string, int) (*CLIResult, error) {
	return func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int) (*CLIResult, error) {
		if captured != nil {
			*captured = cmd
		}
		return result, err
	}
}

func TestCursorProvider_DefaultBinAndInterface(t *testing.T) {
	p := NewCursorProvider("")
	assert.Equal(t, "agent", p.BinPath)
	// Compile-time guarantee it satisfies the Provider interface.
	var _ Provider = p

	custom := NewCursorProvider("/opt/cursor/agent")
	assert.Equal(t, "/opt/cursor/agent", custom.BinPath)
}

func TestBuildProvider_Cursor(t *testing.T) {
	prov, err := BuildProvider(ProviderCursor, "")
	require.NoError(t, err)
	cursor, ok := prov.(*CursorProvider)
	require.True(t, ok, "expected *CursorProvider")
	assert.Equal(t, "agent", cursor.BinPath)

	withBin, err := BuildProvider("cursor", "myagent")
	require.NoError(t, err)
	assert.Equal(t, "myagent", withBin.(*CursorProvider).BinPath)
}

func TestCursorProvider_SuccessfulExecution(t *testing.T) {
	var captured []string
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(&captured, &CLIResult{
		Stdout:     `{"type":"result","subtype":"success","result":"all done","session_id":"sess-123","duration_ms":4200}`,
		ReturnCode: 0,
	}, nil)

	raw, err := p.Execute(context.Background(), "do the thing", Options{})
	require.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, "all done", raw.Result)
	assert.Equal(t, "sess-123", raw.Metrics.SessionID)
	assert.Equal(t, 4200, raw.Metrics.DurationAPIMS)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
	require.Len(t, raw.Messages, 1)

	// Command construction: headless flags, JSON output, positional prompt last.
	assert.Equal(t, "agent", captured[0])
	assert.Contains(t, captured, "-p")
	assert.Contains(t, captured, "--force")
	assert.Contains(t, captured, "--trust")
	assertFlagValue(t, captured, "--output-format", "json")
	assert.Equal(t, "do the thing", captured[len(captured)-1])
}

func TestCursorProvider_FlagWiring(t *testing.T) {
	var captured []string
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(&captured, &CLIResult{
		Stdout:     `{"type":"result","result":"ok"}`,
		ReturnCode: 0,
	}, nil)

	_, err := p.Execute(context.Background(), "prompt", Options{
		ProjectDir:     "/work/project",
		Model:          "gpt-5#high", // variant suffix must be stripped (cursor has no effort flag)
		PermissionMode: "plan",
	})
	require.NoError(t, err)

	assertFlagValue(t, captured, "--workspace", "/work/project")
	assertFlagValue(t, captured, "--model", "gpt-5")
	assertFlagValue(t, captured, "--mode", "plan")
	// The "#high" variant must not leak into the model id.
	assert.NotContains(t, strings.Join(captured, " "), "#high")
}

func TestCursorProvider_CwdFallbackForWorkspace(t *testing.T) {
	var captured []string
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(&captured, &CLIResult{Stdout: `{"result":"ok"}`, ReturnCode: 0}, nil)

	_, err := p.Execute(context.Background(), "prompt", Options{Cwd: "/tmp/here"})
	require.NoError(t, err)
	assertFlagValue(t, captured, "--workspace", "/tmp/here")
}

func TestCursorProvider_SessionResume(t *testing.T) {
	var captured []string
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(&captured, &CLIResult{
		Stdout:     `{"type":"result","result":"resumed","session_id":"sess-999"}`,
		ReturnCode: 0,
	}, nil)

	raw, err := p.Execute(context.Background(), "continue", Options{ResumeSessionID: "sess-999"})
	require.NoError(t, err)
	assertFlagValue(t, captured, "--resume", "sess-999")
	assert.Equal(t, "sess-999", raw.Metrics.SessionID)
}

func TestCursorProvider_NonJSONStdoutFallsBackToRawText(t *testing.T) {
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, &CLIResult{
		Stdout:     "plain text answer, not json",
		ReturnCode: 0,
	}, nil)

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, "plain text answer, not json", raw.Result)
	assert.Equal(t, 1, raw.Metrics.NumTurns)
}

func TestCursorProvider_MissingBinaryReportsFailureCrash(t *testing.T) {
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, nil, fmt.Errorf("exec: %q: executable file not found in $PATH", "agent"))

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "Cursor binary not found")
}

func TestCursorProvider_TimeoutReportsFailureTimeout(t *testing.T) {
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, nil, fmt.Errorf("CLI command timed out after 1s: agent -p"))

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureTimeout, raw.FailureType)
}

func TestCursorProvider_NonZeroExitWithNoOutputIsCrash(t *testing.T) {
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, &CLIResult{
		Stdout:     "",
		Stderr:     "boom: something failed",
		ReturnCode: 2,
	}, nil)

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "boom: something failed")
}

func TestCursorProvider_NonZeroExitWithNoStderrReportsExitCode(t *testing.T) {
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, &CLIResult{Stdout: "", Stderr: "", ReturnCode: 3}, nil)

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "exited with code 3")
}

func TestCursorProvider_NonZeroExitButHasResultIsErrorNotCrash(t *testing.T) {
	// A non-zero exit that still produced a parseable result is flagged as an
	// error but not classified as a crash (mirrors the codex/opencode contract).
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, &CLIResult{
		Stdout:     `{"type":"result","result":"partial answer"}`,
		ReturnCode: 1,
	}, nil)

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.Equal(t, "partial answer", raw.Result)
	assert.True(t, raw.IsError)
	assert.NotEqual(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "exited with code 1")
}

func TestCursorProvider_KilledBySignalIsCrash(t *testing.T) {
	// RunCLI encodes a signal kill as a negative return code.
	p := NewCursorProvider("agent")
	p.runCLI = fakeCursorCLI(nil, &CLIResult{
		Stdout:     "",
		Stderr:     "terminated",
		ReturnCode: -9,
	}, nil)

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "killed by signal 9")
}

func TestCursorProvider_NilRunCLIFallsBackToDefault(t *testing.T) {
	// A CursorProvider built without the constructor (nil runCLI) must not
	// panic; Execute falls back to the real RunCLI, which reports the missing
	// binary as a crash rather than dereferencing a nil func.
	p := &CursorProvider{BinPath: "definitely-not-a-real-binary-xyz"}
	raw, err := p.Execute(context.Background(), "prompt", Options{Timeout: 5})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Contains(t, raw.ErrorMessage, "Cursor binary not found")
}
