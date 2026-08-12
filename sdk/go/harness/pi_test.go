package harness

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const piEventStream = `{"type":"session","id":"session-123"}
{"type":"turn_start"}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"thinking","thinking":"internal"},{"type":"text","text":"done"}],"model":"google/gemini-2.5-flash","usage":{"input":120,"output":30,"cacheRead":10,"cacheWrite":4,"cost":{"total":0.0025}},"stopReason":"stop"}}
{"type":"turn_end"}
{"type":"agent_end"}`

func TestPiFamilyCommandAndMetrics(t *testing.T) {
	tests := []struct {
		name           string
		newProvider    func() (Provider, *piFamilyProvider)
		permissionFlag string
		globTool       string
		wantPrefix     []string
	}{
		{
			name: "pi",
			newProvider: func() (Provider, *piFamilyProvider) {
				provider := NewPiProvider("/opt/pi")
				return provider, provider.piFamilyProvider
			},
			permissionFlag: "--approve",
			globTool:       "find",
			wantPrefix:     []string{"/opt/pi", "--print", "--mode", "json"},
		},
		{
			name: "omp",
			newProvider: func() (Provider, *piFamilyProvider) {
				provider := NewOMPProvider("/opt/omp")
				return provider, provider.piFamilyProvider
			},
			permissionFlag: "--auto-approve",
			globTool:       "glob",
			wantPrefix:     []string{"/opt/omp", "--print", "--mode", "json", "--cwd", "/tmp/project"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			provider, base := tc.newProvider()

			var gotCmd []string
			var gotEnv map[string]string
			var gotCwd string
			var gotPrompt string
			base.runCLI = func(_ context.Context, cmd []string, env map[string]string, cwd string, _ int, stdin []byte) (*CLIResult, error) {
				gotCmd = append([]string(nil), cmd...)
				gotEnv = env
				gotCwd = cwd
				gotPrompt = string(stdin)
				return &CLIResult{Stdout: piEventStream}, nil
			}

			raw, err := provider.Execute(context.Background(), "implement this", Options{
				ProjectDir:     "/tmp/project",
				Model:          "openrouter/google/gemini-2.5-flash#high",
				PermissionMode: "auto",
				SystemPrompt:   "Be precise.",
				Tools:          []string{"Read", "Write", "Edit", "Bash", "Glob", "Grep"},
				Env:            map[string]string{"EXTRA": "1"},
			})
			require.NoError(t, err)
			require.GreaterOrEqual(t, len(gotCmd), len(tc.wantPrefix))
			assert.Equal(t, tc.wantPrefix, gotCmd[:len(tc.wantPrefix)])
			assert.Contains(t, gotCmd, tc.permissionFlag)
			assertFlagValue(t, gotCmd, "--model", "openrouter/google/gemini-2.5-flash")
			assertFlagValue(t, gotCmd, "--thinking", "high")
			assertFlagValue(t, gotCmd, "--tools", "read,write,edit,bash,"+tc.globTool+",grep")
			assert.Equal(t, map[string]string{"EXTRA": "1"}, gotEnv)
			assert.Equal(t, "/tmp/project", gotCwd)
			assert.Equal(t, "implement this", gotPrompt)

			assert.False(t, raw.IsError)
			assert.Equal(t, "done", raw.Result)
			assert.Equal(t, "session-123", raw.Metrics.SessionID)
			assert.Equal(t, 1, raw.Metrics.NumTurns)
			assert.Equal(t, 120, raw.Metrics.InputTokens)
			assert.Equal(t, 30, raw.Metrics.OutputTokens)
			assert.Equal(t, 10, raw.Metrics.CacheReadTokens)
			assert.Equal(t, 4, raw.Metrics.CacheCreationTokens)
			require.NotNil(t, raw.Metrics.CostUSD)
			assert.InDelta(t, 0.0025, *raw.Metrics.CostUSD, 0.000001)
		})
	}
}

func TestPiFamilyPlanModeIsReadOnlyAndResumes(t *testing.T) {
	tests := []struct {
		name        string
		newProvider func() (Provider, *piFamilyProvider)
		resumeFlag  string
		tools       string
	}{
		{"pi", func() (Provider, *piFamilyProvider) {
			provider := NewPiProvider("pi")
			return provider, provider.piFamilyProvider
		}, "--session", "read,grep,find"},
		{"omp", func() (Provider, *piFamilyProvider) {
			provider := NewOMPProvider("omp")
			return provider, provider.piFamilyProvider
		}, "--resume", "read,grep,glob"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			provider, base := tc.newProvider()
			var gotCmd []string
			base.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
				gotCmd = append([]string(nil), cmd...)
				return &CLIResult{Stdout: piEventStream}, nil
			}
			_, err := provider.Execute(context.Background(), "plan", Options{
				PermissionMode:  "plan",
				Tools:           []string{"Read", "Write", "Bash", "Grep", "Glob"},
				ResumeSessionID: "abc123",
			})
			require.NoError(t, err)
			assertFlagValue(t, gotCmd, "--tools", tc.tools)
			assertFlagValue(t, gotCmd, tc.resumeFlag, "abc123")
		})
	}
}

func TestPiFamilyNonzeroExitIsError(t *testing.T) {
	p := NewPiProvider("pi")
	p.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stderr: "authentication failed", ReturnCode: 2}, nil
	}
	raw, err := p.Execute(context.Background(), "hello", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureCrash, raw.FailureType)
	assert.Equal(t, "authentication failed", raw.ErrorMessage)
}

func TestBuildProviderPiFamily(t *testing.T) {
	pi, err := BuildProvider(ProviderPi, "/opt/pi")
	require.NoError(t, err)
	omp, err := BuildProvider(ProviderOMP, "/opt/omp")
	require.NoError(t, err)
	assert.Equal(t, "*harness.PiProvider", fmt.Sprintf("%T", pi))
	assert.Equal(t, "*harness.OMPProvider", fmt.Sprintf("%T", omp))
}

func assertFlagValue(t *testing.T, cmd []string, flag, value string) {
	t.Helper()
	for i := range cmd {
		if cmd[i] == flag && i+1 < len(cmd) {
			assert.Equal(t, value, cmd[i+1])
			return
		}
	}
	t.Fatalf("flag %q not found in %s", flag, strings.Join(cmd, " "))
}
