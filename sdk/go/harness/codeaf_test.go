package harness

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func isolateCodeAF(t *testing.T) {
	t.Helper()
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OPENROUTER_API_KEY", "test-codeaf-key")
	t.Setenv("TMPDIR", t.TempDir())
	for _, key := range []string{"CODEAF_BIN", "AFORGE_BIN", "AGENTFIELD_AFORGE_COMMAND", "AGENTFIELD_CODEAF_COMMAND", "AGENTFIELD_SKIP_AFORGE", "CODEAF_HOME", "CODEAF_NO_UPDATE_CHECK", "CODEAF_TELEMETRY", "CI", "DO_NOT_TRACK", "AGENTFIELD_TELEMETRY_ENABLED"} {
		t.Setenv(key, "")
		require.NoError(t, os.Unsetenv(key))
	}
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "1")
	resetCodeAFResolution()
	codeAFAliasOnce, codeAFLegacyOnce, codeAFCommandOnce = sync.Once{}, sync.Once{}, sync.Once{}
	t.Cleanup(resetCodeAFResolution)
}

func resetCodeAFResolution() {
	codeAFResolver.Lock()
	defer codeAFResolver.Unlock()
	codeAFResolver.cache = make(map[string]codeAFBinary)
	codeAFResolver.lastDownloadAttempt = time.Time{}
	codeAFResolver.lastDownloadError = nil
}

func captureCodeAFLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(old) })
	return &buf
}

func stubCodeAF() *CodeAFProvider {
	p := NewCodeAFProvider("/opt/codeaf")
	p.resolveBinary = func(context.Context, string) (codeAFBinary, error) {
		return codeAFBinary{path: p.BinPath}, nil
	}
	return p
}

// V7: exact argv, stdin, model/variant handling, landing window, and disabled idle watchdog.
func TestCodeAFInvocation(t *testing.T) {
	for _, command := range []string{"exec", "do"} {
		t.Run(command, func(t *testing.T) {
			isolateCodeAF(t)
			t.Setenv("AGENTFIELD_CODEAF_COMMAND", command)
			p := stubCodeAF()
			p.runCLI = func(_ context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, input []byte) (*CLIResult, error) {
				want := []string{"/opt/codeaf", command, "--json", "--dir", "/project"}
				if command == "exec" {
					want = append(want, "--model", "vendor/model", "--timeout", "85s", "--max-turns", "7", "--system", "be precise")
					assert.Equal(t, "private task", string(input))
				} else {
					want = append(want, "--yes-spend", "--model", "vendor/model", "--plan-model", "vendor/model", "--check-model", "vendor/model", "--timeout", "85s", "-")
					assert.Equal(t, "be precise\n\nTask:\nprivate task", string(input))
				}
				assert.Equal(t, want, normalizeCodeAFTimeout(t, cmd, want))
				assert.Equal(t, "high", env["CODEAF_EXEC_REASONING"])
				assert.NotContains(t, env, "AFORGE_EXEC_REASONING")
				assert.NotContains(t, env, "AFORGE_MODEL")
				assert.Empty(t, cwd)
				assert.Equal(t, 90, timeout)
				assert.Zero(t, idle)
				return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"answer"}`}, nil
			}
			raw, err := p.Execute(context.Background(), "private task", Options{ProjectDir: "/project", Cwd: "/nested", Model: "openrouter/vendor/model#high", SystemPrompt: "  be precise  ", MaxTurns: 7, Timeout: 90})
			require.NoError(t, err)
			assert.Equal(t, "answer", raw.Result)
			assert.Equal(t, "vendor/model", raw.Metrics.Model)
		})
	}
}

func TestCodeAFDefaultModelTimeoutAndCommand(t *testing.T) {
	for _, tc := range []struct {
		name, model, envTimeout, command string
		timeout                          int
		wantModel, wantInner             string
		wantOuter                        int
	}{
		{"defaults", "", "", "", 0, DefaultCodeAFModel, "1795s", 1800},
		{"strip only one prefix", "openrouter/openrouter/model", "", "exec", 0, "openrouter/model", "1795s", 1800},
		{"env timeout", "model", "40", "exec", 0, "model", "35s", 40},
		{"bad env timeout", "model", "bad", "exec", 0, "model", "1795s", 1800},
		{"minimum inner", "model", "", "exec", 3, "model", "1s", 3},
		{"invalid command", "model", "", "run", 0, "model", "1795s", 1800},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolateCodeAF(t)
			logs := captureCodeAFLogs(t)
			t.Setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", tc.envTimeout)
			t.Setenv("AGENTFIELD_CODEAF_COMMAND", tc.command)
			p := stubCodeAF()
			p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, timeout, _ int, input []byte) (*CLIResult, error) {
				want := []string{"/opt/codeaf", "exec", "--json", "--dir", ".", "--model", tc.wantModel, "--timeout", tc.wantInner}
				assert.Equal(t, want, normalizeCodeAFTimeout(t, cmd, want))
				assert.Equal(t, tc.wantOuter, timeout)
				assert.Equal(t, "task", string(input))
				return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"ok"}`}, nil
			}
			for i := 0; i < 2; i++ {
				raw, err := p.Execute(context.Background(), "task", Options{Model: tc.model, Timeout: tc.timeout})
				require.NoError(t, err)
				assert.Equal(t, tc.wantModel, raw.Metrics.Model)
			}
			if tc.command == "run" {
				assert.Equal(t, 1, strings.Count(logs.String(), "using exec"))
			}
		})
	}
}

// V8: defaults apply only to absent keys, panel variables are removed, overlay wins.
func TestCodeAFEnvironment(t *testing.T) {
	isolateCodeAF(t)
	env, err := codeAFEnvironment(Options{}, "HIGH")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(os.Getenv("AGENTFIELD_HOME"), "codeaf"), env["CODEAF_HOME"])
	assert.DirExists(t, env["CODEAF_HOME"])
	assert.Equal(t, "1", env["CODEAF_NO_UPDATE_CHECK"])
	assert.Equal(t, "high", env["CODEAF_EXEC_REASONING"])
	assert.NotContains(t, env, "CODEAF_MODELS")
	assert.NotContains(t, env, "AFORGE_MODELS")
	assert.NotContains(t, env, "CODEAF_TELEMETRY")

	for _, tc := range []struct{ key, value string }{
		{"CI", "YeS"}, {"DO_NOT_TRACK", "true"},
		{"AGENTFIELD_TELEMETRY_ENABLED", "0"}, {"AGENTFIELD_TELEMETRY_ENABLED", "FALSE"},
		{"AGENTFIELD_TELEMETRY_ENABLED", "no"}, {"AGENTFIELD_TELEMETRY_ENABLED", "off"},
	} {
		t.Run(tc.key+tc.value, func(t *testing.T) {
			t.Setenv(tc.key, tc.value)
			env, err := codeAFEnvironment(Options{}, "nonsense")
			require.NoError(t, err)
			assert.Equal(t, "off", env["CODEAF_TELEMETRY"])
			assert.NotContains(t, env, "CODEAF_EXEC_REASONING")
		})
	}

	t.Setenv("CODEAF_HOME", filepath.Join(t.TempDir(), "inherited"))
	t.Setenv("CODEAF_NO_UPDATE_CHECK", "0")
	t.Setenv("CODEAF_TELEMETRY", "on")
	t.Setenv("CI", "1")
	env, err = codeAFEnvironment(Options{}, "")
	require.NoError(t, err)
	for _, key := range []string{"CODEAF_HOME", "CODEAF_NO_UPDATE_CHECK", "CODEAF_TELEMETRY"} {
		assert.NotContains(t, env, key)
	}
	overlay := map[string]string{"CODEAF_HOME": filepath.Join(t.TempDir(), "overlay"), "CODEAF_NO_UPDATE_CHECK": "0", "CODEAF_TELEMETRY": "on", "CODEAF_MODELS": "panel", "AFORGE_MODELS": "legacy-panel", "CODEAF_EXEC_REASONING": "low"}
	env, err = codeAFEnvironment(Options{Env: overlay}, "high")
	require.NoError(t, err)
	for key, value := range overlay {
		assert.Equal(t, value, env[key])
	}
	env, err = codeAFEnvironment(Options{Env: map[string]string{"CODEAF_HOME": "", "CODEAF_TELEMETRY": "", "CODEAF_NO_UPDATE_CHECK": ""}}, "")
	require.NoError(t, err)
	assert.Empty(t, env["CODEAF_HOME"])
	assert.Empty(t, env["CODEAF_TELEMETRY"])
}

// V9: canonical envelopes only; last eligible JSON line wins.
func TestParseCodeAFEnvelope(t *testing.T) {
	for _, tc := range []struct{ name, stdout, answer string }{
		{"whole pretty object", "{\n\"ok\": true,\n\"stop\": \"done\",\n\"answer\": \"pretty\"\n}", "pretty"},
		{"last envelope", "diagnostic\n{\"ok\":false,\"stop\":\"error\",\"answer\":\"old\"}\n{\"ok\":true,\"stop\":\"done\",\"answer\":\"last\"}\n{\"other\":true}\nnoise", "last"},
		{"legacy exec", `{"text":"legacy","stop":"done","usage":{}}`, ""},
		{"legacy do", `{"deliverable":"legacy","settled":true}`, ""},
		{"no stop", `{"ok":true,"answer":"invalid"}`, ""},
		{"invalid", "\nnot json\n[]\nnull\n", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			envelope := parseCodeAFEnvelope(tc.stdout)
			if tc.answer == "" {
				assert.Nil(t, envelope)
			} else {
				require.NotNil(t, envelope)
				assert.Equal(t, tc.answer, envelope["answer"])
			}
		})
	}
}

// V10: each outcome is derived from the shared exit/stop table.
func TestCodeAFOutcomes(t *testing.T) {
	for _, tc := range []struct {
		name, stop, answer string
		code               int
		ok                 bool
		failure            FailureType
	}{
		{"done", "done", "answer", 0, true, FailureNone},
		{"done false", "done", "answer", 0, false, FailureCrash},
		{"unchecked", "unchecked", "answer", 2, false, FailureNone},
		{"unchecked empty", "unchecked", "", 2, false, FailureCrash},
		{"budget partial", "budget", "partial", 3, false, FailureNone},
		{"turn cap partial", "turn-cap", "partial", 3, false, FailureNone},
		{"budget empty", "budget", "", 3, false, FailureNoOutput},
		{"turn cap empty", "turn-cap", "", 3, false, FailureNoOutput},
		{"exec deadline", "deadline", "partial", 3, false, FailureTimeout},
		{"do deadline", "deadline", "partial", 124, false, FailureTimeout},
		{"question", "question", "question", 4, false, FailureCrash},
		{"incomplete", "incomplete", "partial", 2, false, FailureCrash},
		{"error", "error", "partial", 1, false, FailureCrash},
		{"price", "price", "partial", 3, false, FailureCrash},
		{"unknown success", "future", "answer", 0, false, FailureNone},
		{"unknown failure", "future", "answer", 3, true, FailureCrash},
		{"unknown empty", "future", "", 0, true, FailureCrash},
		{"signal", "done", "answer", -9, true, FailureCrash},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stdout := fmt.Sprintf(`{"ok":%t,"stop":%q,"answer":%q,"blocked_on":"need approval"}`, tc.ok, tc.stop, tc.answer)
			raw := codeAFResult(&CLIResult{Stdout: stdout, ReturnCode: tc.code}, "requested", 42)
			assert.Equal(t, tc.failure, raw.FailureType)
			assert.Equal(t, tc.failure != FailureNone, raw.IsError)
			assert.Equal(t, ProviderCodeAF, raw.Provider)
			if raw.IsError {
				assert.Empty(t, raw.Result)
				assert.Contains(t, raw.ErrorMessage, fmt.Sprintf("codeaf exit code %d (stop=%s):", tc.code, tc.stop))
				if tc.stop == "question" {
					assert.Contains(t, raw.ErrorMessage, "need approval")
				}
			} else {
				assert.Equal(t, tc.answer, raw.Result)
			}
		})
	}
	for _, tc := range []struct {
		stdout  string
		code    int
		failure FailureType
	}{
		{" \n", 0, FailureNoOutput}, {"", 1, FailureCrash},
		{"plain text", 0, FailureCrash}, {`{"text":"legacy"}`, 0, FailureCrash}, {"", -9, FailureCrash},
	} {
		raw := codeAFResult(&CLIResult{Stdout: tc.stdout, ReturnCode: tc.code}, "model", 1)
		assert.Equal(t, tc.failure, raw.FailureType)
	}
}

func TestCodeAFMetricsAndErrorDetails(t *testing.T) {
	raw := codeAFResult(&CLIResult{Stdout: `{"ok":true,"stop":"done","answer":" result ","steps":4,"calls":99,"tokens":{"in":100,"out":50},"spend_usd":0.0,"model":"returned","seconds":999,"usage":{"cost":99},"turns":99}`}, "requested", 12)
	assert.Equal(t, "result", raw.Result)
	assert.Equal(t, 4, raw.Metrics.NumTurns)
	assert.Equal(t, 100, raw.Metrics.InputTokens)
	assert.Equal(t, 50, raw.Metrics.OutputTokens)
	assert.Equal(t, "returned", raw.Metrics.Model)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.Zero(t, *raw.Metrics.CostUSD)
	assert.Equal(t, 12, raw.Metrics.DurationAPIMS)
	assert.Equal(t, 12, raw.Metrics.DurationMS)
	assert.Zero(t, raw.Metrics.CacheReadTokens)
	require.Len(t, raw.Messages, 1)

	raw = codeAFResult(&CLIResult{Stdout: `{"ok":false,"stop":"error","error":"specific","answer":"answer"}`, Stderr: "stderr", ReturnCode: 1}, "requested", 0)
	assert.Contains(t, raw.ErrorMessage, "specific")
	assert.NotContains(t, raw.ErrorMessage, "stderr")
	assert.Nil(t, raw.Metrics.CostUSD)
	assert.Equal(t, "requested", raw.Metrics.Model)
	message := codeAFErrorMessage(1, "error", "", "", "\x1b[31m"+strings.Repeat("x", 1200)+"tail\x1b[0m")
	assert.True(t, strings.HasSuffix(message, "tail"))
	assert.Less(t, len(message), 1100)
	assert.NotContains(t, message, "\x1b")
	assert.Contains(t, codeAFErrorMessage(4, "question", strings.Repeat("x", 1500), "approval", ""), "approval")
}

func TestCodeAFExecutionErrors(t *testing.T) {
	isolateCodeAF(t)
	p := stubCodeAF()
	for _, tc := range []struct {
		err     error
		failure FailureType
	}{
		{context.DeadlineExceeded, FailureTimeout}, {errors.New("CLI timed out"), FailureTimeout},
		{errors.New("transport failed"), FailureNone}, {os.ErrNotExist, FailureNone},
	} {
		p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
			return nil, tc.err
		}
		raw, err := p.Execute(context.Background(), "task", Options{})
		if tc.failure == FailureTimeout {
			require.NoError(t, err)
			assert.Equal(t, FailureTimeout, raw.FailureType)
		} else {
			require.Error(t, err)
			assert.Nil(t, raw)
		}
	}
	p.resolveBinary = func(context.Context, string) (codeAFBinary, error) {
		return codeAFBinary{}, errors.New("resolution failed")
	}
	_, err := p.Execute(context.Background(), "task", Options{})
	require.ErrorContains(t, err, "resolution failed")
}

func TestCodeAFOuterTimeoutAndConcurrency(t *testing.T) {
	isolateCodeAF(t)
	script := filepath.Join(t.TempDir(), "quiet-codeaf")
	require.NoError(t, os.WriteFile(script, []byte("#!/bin/sh\nsleep 5\n"), 0o755))
	raw, err := NewCodeAFProvider(script).Execute(context.Background(), "task", Options{Timeout: 1})
	require.NoError(t, err)
	assert.Equal(t, FailureTimeout, raw.FailureType)
	assert.Empty(t, raw.Result)

	oldSem := getCodeAFSemaphore()
	codeAFSemaphore = make(chan struct{}, 2)
	t.Cleanup(func() { codeAFSemaphore = oldSem })
	var current, peak atomic.Int32
	p := stubCodeAF()
	p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _, _ int, _ []byte) (*CLIResult, error) {
		active := current.Add(1)
		for old := peak.Load(); active > old && !peak.CompareAndSwap(old, active); old = peak.Load() {
		}
		time.Sleep(10 * time.Millisecond)
		current.Add(-1)
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"ok"}`}, nil
	}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := p.Execute(context.Background(), "task", Options{})
			assert.NoError(t, err)
		}()
	}
	wg.Wait()
	assert.Equal(t, int32(2), peak.Load())
	codeAFSemaphore <- struct{}{}
	codeAFSemaphore <- struct{}{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = p.Execute(ctx, "task", Options{})
	assert.ErrorIs(t, err, context.Canceled)
	<-codeAFSemaphore
	<-codeAFSemaphore
}

// V12: invalid environment never prevents construction or execution.
func TestCodeAFConcurrencyParsing(t *testing.T) {
	for _, tc := range []struct {
		codeaf, aforge string
		set            bool
		want           int
	}{
		{"", "", false, 8}, {"", "3", false, 3}, {"2", "3", true, 2},
		{"", "3", true, 8}, {"bad", "3", true, 8}, {"0", "3", true, 8},
		{"-1", "3", true, 8}, {"999999999999999999999999999999", "", true, 8}, {" 4 ", "", true, 4},
	} {
		t.Run(fmt.Sprintf("%s/%s/%t", tc.codeaf, tc.aforge, tc.set), func(t *testing.T) {
			t.Setenv("AFORGE_MAX_CONCURRENT", tc.aforge)
			t.Setenv("CODEAF_MAX_CONCURRENT", tc.codeaf)
			if !tc.set {
				require.NoError(t, os.Unsetenv("CODEAF_MAX_CONCURRENT"))
			}
			assert.Equal(t, tc.want, codeAFMaxConcurrent())
		})
	}
}

// V8: validate the actual child environment after RunCLI merges and unsets keys.
func TestCodeAFChildEnvironment(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv("CI", "true")
	t.Setenv("CODEAF_MODELS", "inherited")
	t.Setenv("AFORGE_MODELS", "inherited")
	path := filepath.Join(t.TempDir(), "codeaf")
	script := `#!/bin/sh
printf '{"ok":true,"stop":"done","answer":"%s|%s|%s|%s|%s|%s"}\n' "$CODEAF_HOME" "$CODEAF_NO_UPDATE_CHECK" "$CODEAF_TELEMETRY" "${CODEAF_MODELS-unset}" "${AFORGE_MODELS-unset}" "$CODEAF_EXEC_REASONING"
`
	require.NoError(t, os.WriteFile(path, []byte(script), 0o755))
	p := NewCodeAFProvider(path)
	raw, err := p.Execute(context.Background(), "task", Options{Variant: "high"})
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(os.Getenv("AGENTFIELD_HOME"), "codeaf")+"|1|off|unset|unset|high", raw.Result)
	raw, err = p.Execute(context.Background(), "task", Options{Variant: "high", Env: map[string]string{"CODEAF_TELEMETRY": "on", "CODEAF_NO_UPDATE_CHECK": "0", "CODEAF_MODELS": "overlay", "AFORGE_MODELS": "overlay", "CODEAF_EXEC_REASONING": "low"}})
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(os.Getenv("AGENTFIELD_HOME"), "codeaf")+"|0|on|overlay|overlay|low", raw.Result)
}

func TestCodeAFSchemaUsesExistingRunnerPath(t *testing.T) {
	isolateCodeAF(t)
	path := writeCodeAFCLI(t, filepath.Join(t.TempDir(), "codeaf"), `{"ok":true,"stop":"done","answer":"{\"value\":\"ok\"}","spend_usd":0}`, 0)
	var dest struct {
		Value string `json:"value"`
	}
	schema := map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}, "required": []string{"value"}}
	result, err := NewRunner(Options{Provider: ProviderCodeAF, BinPath: path, ProjectDir: t.TempDir()}).Run(context.Background(), "task", schema, &dest, Options{})
	require.NoError(t, err)
	assert.False(t, result.IsError, result.ErrorMessage)
	assert.Equal(t, "ok", dest.Value)
	assert.Equal(t, ProviderCodeAF, result.Provider)
	require.NotNil(t, result.CostUSD)
	assert.Zero(t, *result.CostUSD)
}

func TestCodeAFEnvironmentErrorsAndOverlayHome(t *testing.T) {
	isolateCodeAF(t)
	childHome := t.TempDir()
	env, err := codeAFEnvironment(Options{Env: map[string]string{"AGENTFIELD_HOME": childHome}}, "")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(childHome, "codeaf"), env["CODEAF_HOME"])
	blocked := filepath.Join(t.TempDir(), "file")
	require.NoError(t, os.WriteFile(blocked, []byte("file"), 0o644))
	env, err = codeAFEnvironment(Options{Env: map[string]string{"CODEAF_HOME": filepath.Join(blocked, "home")}}, "")
	require.NoError(t, err)
}
