package harness

import (
	"context"
	"errors"
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
			permissionFlag: "",
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
			if tc.permissionFlag != "" {
				assert.Contains(t, gotCmd, tc.permissionFlag)
			}
			assertNoApprovalFlags(t, gotCmd, tc.permissionFlag)
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

func TestPiConfiguredModelOverridesReportedModel(t *testing.T) {
	provider := NewPiProvider("pi")
	provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stdout: piEventStream}, nil
	}

	raw, err := provider.Execute(context.Background(), "inspect", Options{Model: "openrouter/x/y"})
	require.NoError(t, err)
	assert.Equal(t, "openrouter/x/y", raw.Metrics.Model)
}

func TestPiUsesReportedModelWithoutConfiguredModel(t *testing.T) {
	provider := NewPiProvider("pi")
	provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stdout: piEventStream}, nil
	}

	raw, err := provider.Execute(context.Background(), "inspect", Options{})
	require.NoError(t, err)
	assert.Equal(t, "google/gemini-2.5-flash", raw.Metrics.Model)
}

func TestPiModelIsEmptyWhenNotConfiguredOrReported(t *testing.T) {
	provider := NewPiProvider("pi")
	provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stdout: `{"type":"message_end","message":{"role":"assistant","content":"done"}}`}, nil
	}

	raw, err := provider.Execute(context.Background(), "inspect", Options{})
	require.NoError(t, err)
	assert.Empty(t, raw.Metrics.Model)
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
			assertNoApprovalFlags(t, gotCmd, "")
		})
	}
}

func assertNoApprovalFlags(t *testing.T, cmd []string, allowed string) {
	t.Helper()
	for _, flag := range []string{"--approve", "--auto-approve", "--yolo", "-y", "--approval-mode", "--permission-mode"} {
		if flag != allowed {
			assert.NotContains(t, cmd, flag)
		}
	}
}

func TestPiFamilyToolEdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		provider    *piFamilyProvider
		options     Options
		wantFlag    string
		wantNoTools bool
	}{
		{
			name:     "pi plan restores default read-only tools",
			provider: NewPiProvider("pi").piFamilyProvider,
			options:  Options{PermissionMode: "plan", Tools: []string{"Write", "Bash"}},
			wantFlag: "read,grep,find",
		},
		{
			name:     "omp plan restores default read-only tools",
			provider: NewOMPProvider("omp").piFamilyProvider,
			options:  Options{PermissionMode: "plan", Tools: []string{"Write", "Bash"}},
			wantFlag: "read,grep,glob",
		},
		{
			name:        "explicit empty tools disables tools",
			provider:    NewPiProvider("pi").piFamilyProvider,
			options:     Options{Tools: []string{}},
			wantNoTools: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotCmd []string
			tc.provider.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
				gotCmd = append([]string(nil), cmd...)
				return &CLIResult{Stdout: piEventStream}, nil
			}

			_, err := tc.provider.execute(context.Background(), "inspect", tc.options)
			require.NoError(t, err)
			if tc.wantNoTools {
				assert.Contains(t, gotCmd, "--no-tools")
				assert.NotContains(t, gotCmd, "--tools")
				return
			}
			assertFlagValue(t, gotCmd, "--tools", tc.wantFlag)
		})
	}
}

func TestPiFamilyExecutionFailures(t *testing.T) {
	tests := []struct {
		name        string
		result      *CLIResult
		runErr      error
		wantErr     bool
		failureType FailureType
		message     string
	}{
		{
			name:        "timeout",
			runErr:      errors.New("command timed out"),
			failureType: FailureTimeout,
			message:     "command timed out",
		},
		{
			name:    "unexpected runner error",
			runErr:  errors.New("pipe failed"),
			wantErr: true,
		},
		{
			name:        "nonzero exit without stderr",
			result:      &CLIResult{ReturnCode: 7},
			failureType: FailureCrash,
			message:     "Process exited with code 7.",
		},
		{
			name:        "signal kill reports the signal, not an exit code",
			result:      &CLIResult{ReturnCode: -9},
			failureType: FailureCrash,
			message:     "Process killed by signal 9.",
		},
		{
			name:        "api error event",
			result:      &CLIResult{Stdout: `{"type":"message_end","message":{"role":"assistant","stopReason":"error","errorMessage":"quota exceeded"}}`},
			failureType: FailureAPIError,
			message:     "quota exceeded",
		},
		{
			name:        "successful process without assistant output",
			result:      &CLIResult{Stderr: "empty response"},
			failureType: FailureNoOutput,
			message:     "empty response",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewPiProvider("pi").piFamilyProvider
			provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
				return tc.result, tc.runErr
			}

			raw, err := provider.execute(context.Background(), "inspect", Options{})
			if tc.wantErr {
				require.EqualError(t, err, tc.runErr.Error())
				assert.Nil(t, raw)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, raw)
			assert.True(t, raw.IsError)
			assert.Equal(t, tc.failureType, raw.FailureType)
			assert.Equal(t, tc.message, raw.ErrorMessage)
		})
	}
}

func TestPiFamilyRecoveredTurnIsNotAnError(t *testing.T) {
	tests := []struct {
		name            string
		stdout          string
		wantResult      string
		wantIsError     bool
		wantFailureType FailureType
		wantError       string
	}{
		{
			name: "error then recovery",
			stdout: `{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"partial"}],"stopReason":"error","errorMessage":"upstream 503"}}
{"type":"turn_end"}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"FINAL ANSWER"}],"stopReason":"stop"}}
{"type":"turn_end"}`,
			wantResult:      "FINAL ANSWER",
			wantFailureType: FailureNone,
		},
		{
			name: "last message is an error",
			stdout: `{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"partial"}],"stopReason":"stop"}}
{"type":"turn_end"}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"FINAL ANSWER"}],"stopReason":"error","errorMessage":"upstream 503"}}
{"type":"turn_end"}`,
			wantResult:      "FINAL ANSWER",
			wantIsError:     true,
			wantFailureType: FailureAPIError,
			wantError:       "upstream 503",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewPiProvider("pi").piFamilyProvider
			provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
				return &CLIResult{Stdout: tc.stdout, ReturnCode: 0}, nil
			}

			raw, err := provider.execute(context.Background(), "inspect", Options{})
			require.NoError(t, err)
			require.NotNil(t, raw)
			assert.Equal(t, tc.wantResult, raw.Result)
			assert.Equal(t, tc.wantIsError, raw.IsError)
			assert.Equal(t, tc.wantFailureType, raw.FailureType)
			assert.Equal(t, tc.wantError, raw.ErrorMessage)
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

	// Pi and OMP are additional providers: an empty name still resolves to the
	// SDK default, aforge.
	t.Setenv(ProviderEnvVar, "")
	defaultProvider, err := BuildProvider("", "")
	require.NoError(t, err)
	assert.Equal(t, "*harness.AforgeProvider", fmt.Sprintf("%T", defaultProvider))
}

func TestPiFamilyMissingBinaryIncludesInstallGuidance(t *testing.T) {
	tests := []struct {
		name        string
		provider    *piFamilyProvider
		installHint string
	}{
		{"pi", NewPiProvider("pi").piFamilyProvider, "npm install -g --ignore-scripts @earendil-works/pi-coding-agent"},
		{"omp", NewOMPProvider("omp").piFamilyProvider, "curl -fsSL https://omp.sh/install | sh"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
				return nil, fmt.Errorf("exec: executable file not found in $PATH")
			}
			raw, err := tc.provider.execute(context.Background(), "hello", Options{})
			require.NoError(t, err)
			require.True(t, raw.IsError)
			assert.Contains(t, raw.ErrorMessage, tc.installHint)
			assert.Equal(t, FailureCrash, raw.FailureType)
		})
	}
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
