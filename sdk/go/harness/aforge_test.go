package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func aforgeEnvelope(deliverable string, settled bool, blockedOn, usage string) string {
	if usage == "" {
		usage = `{"calls":1,"prompt_tokens":0,"completion_tokens":0,"cached_tokens":0,"cost":0}`
	}
	return fmt.Sprintf(`{"settled":%t,"deliverable":%q,"blocked_on":%q,"spend_usd":0.0123,"elapsed_ms":12,"usage":%s}`,
		settled, deliverable, blockedOn, usage)
}

func aforgeExecEnvelope(text, stop, usage string, turns int) string {
	if usage == "" {
		usage = `{"calls":1,"prompt_tokens":0,"completion_tokens":0,"cached_tokens":0,"cost":0}`
	}
	return fmt.Sprintf(`{"text":%q,"stop":%q,"usage":%s,"artifacts":[],"turns":%d,"elapsed_ms":12}`,
		text, stop, usage, turns)
}

func useAforgeDo(t *testing.T) {
	t.Helper()
	t.Setenv("AGENTFIELD_AFORGE_COMMAND", "do")
}

func TestAforgeProviderMapsDoCommandEnvelopeAndMetrics(t *testing.T) {
	useAforgeDo(t)
	var capturedCmd []string
	var capturedEnv map[string]string
	var capturedCwd string
	var capturedTimeout, capturedIdle int
	var capturedStdin []byte

	p := NewAforgeProvider("/opt/aforge")
	p.runCLI = func(_ context.Context, cmd []string, env map[string]string, cwd string, timeout, idleSeconds int, stdin []byte) (*CLIResult, error) {
		capturedCmd = append([]string(nil), cmd...)
		capturedEnv = env
		capturedCwd = cwd
		capturedTimeout = timeout
		capturedIdle = idleSeconds
		capturedStdin = append([]byte(nil), stdin...)
		return &CLIResult{
			Stdout: aforgeEnvelope(" final answer ", true, "",
				`{"calls":3,"prompt_tokens":100,"completion_tokens":50,"cached_tokens":20,"cost":0.0123}`),
			ReturnCode: 0,
		}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt that stays off argv", Options{
		ProjectDir:   "/project",
		Cwd:          "/project/nested",
		SystemPrompt: "  be precise  ",
		Model:        "openrouter/z-ai/glm-5.2#high",
	})
	require.NoError(t, err)
	require.NotNil(t, raw)
	assert.Equal(t, []string{
		"/opt/aforge", "do", "--json", "--yes-spend", "-w", "/project",
		"--timeout", "1795",
	}, capturedCmd)
	assert.Equal(t, "z-ai/glm-5.2", capturedEnv["AFORGE_MODEL"])
	assert.Equal(t, "high", capturedEnv["AFORGE_EXEC_REASONING"])
	assert.Empty(t, capturedCwd)
	assert.Equal(t, defaultAforgeTimeout, capturedTimeout)
	assert.Zero(t, capturedIdle)
	assert.Equal(t, "be precise\n\nTask:\nprompt that stays off argv", string(capturedStdin))
	for _, arg := range capturedCmd {
		assert.NotContains(t, arg, "prompt that stays off argv")
	}

	assert.Equal(t, "final answer", raw.Result)
	assert.False(t, raw.IsError)
	assert.Equal(t, FailureNone, raw.FailureType)
	assert.Equal(t, 0, raw.ReturnCode)
	assert.Equal(t, 3, raw.Metrics.NumTurns)
	assert.Equal(t, 100, raw.Metrics.InputTokens)
	assert.Equal(t, 50, raw.Metrics.OutputTokens)
	assert.Equal(t, 20, raw.Metrics.CacheReadTokens)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.InDelta(t, 0.0123, *raw.Metrics.CostUSD, 1e-9)
	require.Len(t, raw.Messages, 1)
	assert.Equal(t, " final answer ", raw.Messages[0]["deliverable"])
}

func TestAforgeProviderMapsExecCommandEnvelopeAndMetrics(t *testing.T) {
	t.Setenv("AGENTFIELD_AFORGE_COMMAND", "")
	var capturedCmd []string
	var capturedEnv map[string]string
	var capturedStdin []byte
	p := NewAforgeProvider("/opt/aforge")
	p.runCLI = func(_ context.Context, cmd []string, env map[string]string, _ string, _, _ int, stdin []byte) (*CLIResult, error) {
		capturedCmd = append([]string(nil), cmd...)
		capturedEnv = env
		capturedStdin = append([]byte(nil), stdin...)
		return &CLIResult{Stdout: aforgeExecEnvelope(" linear answer ", "done",
			`{"calls":3,"prompt_tokens":100,"completion_tokens":50,"cached_tokens":20,"cost":0.0123}`, 4)}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt that stays off argv", Options{
		ProjectDir:   "/project",
		SystemPrompt: "  be precise  ",
		Model:        "openrouter/deepseek/deepseek-v4-flash-0731",
	})
	require.NoError(t, err)
	assert.Equal(t, []string{
		"/opt/aforge", "exec", "--json", "-w", "/project",
		"--timeout", "1795", "--context-fill", "60", "--completion-reserve", "65536",
		"--system", "be precise",
		"--model", "deepseek/deepseek-v4-flash-0731",
		"--plan-model", "deepseek/deepseek-v4-flash-0731",
	}, capturedCmd)
	assert.Equal(t, "", capturedEnv["AFORGE_MODELS"])
	assert.Equal(t, "deepseek/deepseek-v4-flash-0731", capturedEnv["AFORGE_MODEL"])
	assert.Equal(t, "prompt that stays off argv", string(capturedStdin))
	assert.Equal(t, "linear answer", raw.Result)
	assert.False(t, raw.IsError)
	assert.Equal(t, 4, raw.Metrics.NumTurns)
	assert.Equal(t, 100, raw.Metrics.InputTokens)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.InDelta(t, 0.0123, *raw.Metrics.CostUSD, 1e-9)
}

func TestAforgeProviderExecBudgetPartialIsUsable(t *testing.T) {
	t.Setenv("AGENTFIELD_AFORGE_COMMAND", "exec")
	p := NewAforgeProvider("aforge")
	p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stdout: aforgeExecEnvelope("usable", "budget", "", 2), ReturnCode: 2}, nil
	}

	raw, err := p.Execute(context.Background(), "hello", Options{})
	require.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, FailureNone, raw.FailureType)
	assert.Equal(t, "usable", raw.Result)
}

func TestAforgeProviderBinaryEnvironmentOverride(t *testing.T) {
	t.Setenv("AFORGE_BIN", "/opt/aforge-env")
	assert.Equal(t, "/opt/aforge-env", NewAforgeProvider("").BinPath)
	assert.Equal(t, "/explicit/aforge", NewAforgeProvider("/explicit/aforge").BinPath)
}

func TestAforgeProviderModelVariantAndEnvironmentPrecedence(t *testing.T) {
	useAforgeDo(t)
	var captured []map[string]string
	p := NewAforgeProvider("aforge")
	p.runCLI = func(_ context.Context, _ []string, env map[string]string, _ string, _, _ int, _ []byte) (*CLIResult, error) {
		copyEnv := make(map[string]string, len(env))
		for key, value := range env {
			copyEnv[key] = value
		}
		captured = append(captured, copyEnv)
		return &CLIResult{Stdout: aforgeEnvelope("done", true, "", ""), ReturnCode: 0}, nil
	}

	_, err := p.Execute(context.Background(), "hello", Options{Model: "openrouter/x/y#turbo"})
	require.NoError(t, err)
	_, err = p.Execute(context.Background(), "hello", Options{
		Model:   "openrouter/x/y#low",
		Variant: " HIGH ",
		Env: map[string]string{
			"AFORGE_MODEL":          "override/model",
			"AFORGE_EXEC_REASONING": "off",
			"EXTRA":                 "1",
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "x/y", captured[0]["AFORGE_MODEL"])
	assert.NotContains(t, captured[0], "AFORGE_EXEC_REASONING")
	assert.Equal(t, "override/model", captured[1]["AFORGE_MODEL"])
	assert.Equal(t, "off", captured[1]["AFORGE_EXEC_REASONING"])
	assert.Equal(t, "1", captured[1]["EXTRA"])
}

func TestAforgeProviderRootAndTimeoutResolution(t *testing.T) {
	useAforgeDo(t)
	t.Setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "2400")
	var commands [][]string
	var timeouts []int
	p := NewAforgeProvider("aforge")
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, timeout, idleSeconds int, _ []byte) (*CLIResult, error) {
		commands = append(commands, append([]string(nil), cmd...))
		timeouts = append(timeouts, timeout)
		assert.Zero(t, idleSeconds)
		return &CLIResult{Stdout: aforgeEnvelope("done", true, "", ""), ReturnCode: 0}, nil
	}

	_, err := p.Execute(context.Background(), "hello", Options{Cwd: "/cwd-only"})
	require.NoError(t, err)
	_, err = p.Execute(context.Background(), "hello", Options{Timeout: 7})
	require.NoError(t, err)

	assert.Equal(t, []string{"aforge", "do", "--json", "--yes-spend", "-w", "/cwd-only", "--timeout", "2395"}, commands[0])
	assert.Equal(t, []string{"aforge", "do", "--json", "--yes-spend", "-w", ".", "--timeout", "2"}, commands[1])
	assert.Equal(t, []int{2400, 7}, timeouts)
}

func TestAforgeProviderExitSemantics(t *testing.T) {
	useAforgeDo(t)
	tests := []struct {
		name        string
		code        int
		deliverable string
		blockedOn   string
		stderr      string
		wantFailure FailureType
		wantMessage string
	}{
		{name: "success", code: 0, deliverable: "done", wantFailure: FailureNone},
		{name: "timeout with partial", code: 2, deliverable: "usable", wantFailure: FailureTimeout, wantMessage: "partial: usable"},
		{name: "blocked", code: 1, blockedOn: "Which repository?", wantFailure: FailureCrash, wantMessage: "blocked_on: Which repository?"},
		{name: "error", code: 1, stderr: "\x1b[31mauthentication exploded\x1b[0m", wantFailure: FailureCrash, wantMessage: "authentication exploded"},
		{name: "zero without deliverable", code: 0, wantFailure: FailureCrash, wantMessage: "aforge exit code 0"},
		{name: "signal", code: -9, wantFailure: FailureCrash, wantMessage: "Process killed by signal 9"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := NewAforgeProvider("aforge")
			p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
				return &CLIResult{
					Stdout:     aforgeEnvelope(test.deliverable, test.code == 0, test.blockedOn, ""),
					Stderr:     test.stderr,
					ReturnCode: test.code,
				}, nil
			}

			raw, err := p.Execute(context.Background(), "hello", Options{})
			require.NoError(t, err)
			assert.Equal(t, test.wantFailure != FailureNone, raw.IsError)
			assert.Equal(t, test.wantFailure, raw.FailureType)
			if test.wantMessage != "" {
				assert.Contains(t, raw.ErrorMessage, test.wantMessage)
			}
		})
	}
}

func TestAforgeProviderParsesLastEnvelopeAndLeavesZeroCostUnknown(t *testing.T) {
	useAforgeDo(t)
	p := NewAforgeProvider("aforge")
	p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
		return &CLIResult{Stdout: "stray diagnostic\n{\"type\":\"event\"}\n" +
			aforgeEnvelope("real result", true, "", `{"calls":1,"cost":0}`), ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "hello", Options{})
	require.NoError(t, err)
	assert.Equal(t, "real result", raw.Result)
	assert.Nil(t, raw.Metrics.CostUSD, "zero provider cost remains unknown")
}

func TestAforgeProviderParsesPrettyPrintedEnvelope(t *testing.T) {
	useAforgeDo(t)
	p := NewAforgeProvider("aforge")
	p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
		var value map[string]any
		require.NoError(t, json.Unmarshal([]byte(aforgeEnvelope("pretty result", true, "", "")), &value))
		pretty, err := json.MarshalIndent(value, "", "  ")
		require.NoError(t, err)
		return &CLIResult{Stdout: string(pretty), ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "hello", Options{})
	require.NoError(t, err)
	assert.Equal(t, "pretty result", raw.Result)
	assert.False(t, raw.IsError)
}

func TestAforgeProviderMissingBinaryAndTimeout(t *testing.T) {
	useAforgeDo(t)
	t.Run("missing binary", func(t *testing.T) {
		p := NewAforgeProvider("aforge-missing")
		p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
			return nil, fmt.Errorf("exec: executable file not found in $PATH")
		}
		raw, err := p.Execute(context.Background(), "hello", Options{})
		require.NoError(t, err)
		assert.True(t, raw.IsError)
		assert.Equal(t, FailureCrash, raw.FailureType)
		assert.Contains(t, raw.ErrorMessage, "aforge-missing")
	})

	t.Run("timeout", func(t *testing.T) {
		p := NewAforgeProvider("aforge")
		p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
			return nil, fmt.Errorf("CLI command timed out after 1s: aforge do")
		}
		raw, err := p.Execute(context.Background(), "hello", Options{})
		require.NoError(t, err)
		assert.True(t, raw.IsError)
		assert.Equal(t, FailureTimeout, raw.FailureType)
	})
}

func TestAforgeProviderConcurrencyLimit(t *testing.T) {
	useAforgeDo(t)
	t.Setenv("AFORGE_MAX_CONCURRENT", "2")
	aforgeSemaphore = nil
	aforgeSemOnce = sync.Once{}
	t.Cleanup(func() {
		aforgeSemaphore = nil
		aforgeSemOnce = sync.Once{}
	})

	var current int64
	var maxSeen int64
	p := NewAforgeProvider("aforge")
	p.runCLI = func(context.Context, []string, map[string]string, string, int, int, []byte) (*CLIResult, error) {
		active := atomic.AddInt64(&current, 1)
		for {
			previous := atomic.LoadInt64(&maxSeen)
			if active <= previous || atomic.CompareAndSwapInt64(&maxSeen, previous, active) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		atomic.AddInt64(&current, -1)
		return &CLIResult{Stdout: aforgeEnvelope("done", true, "", ""), ReturnCode: 0}, nil
	}

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = p.Execute(context.Background(), "hello", Options{})
		}()
	}
	wg.Wait()
	assert.LessOrEqual(t, maxSeen, int64(2))
}

func TestAforgeRunnerConcurrentSameCwdUsesIsolatedSchemaFiles(t *testing.T) {
	useAforgeDo(t)
	cwd := t.TempDir()
	script := writeTestScript(t, cwd, "aforge-test", `#!/bin/sh
prompt=$(cat)
output_path=$(printf '%s' "$prompt" | tr '\n' ' ' | sed -n 's/.*create this file: \([^ ]*\.agentfield_output\.json\).*/\1/p')
case "$prompt" in
  *first*) payload='{"name":"first","count":1}' ;;
  *) payload='{"name":"second","count":2}' ;;
esac
mkdir -p "$(dirname "$output_path")"
printf '%s' "$payload" > "$output_path"
printf '%s\n' '{"settled":true,"deliverable":"done","blocked_on":"","spend_usd":0,"elapsed_ms":1,"usage":{"calls":1,"prompt_tokens":0,"completion_tokens":0,"cached_tokens":0,"cost":0}}'
`)

	type output struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":  map[string]any{"type": "string"},
			"count": map[string]any{"type": "integer"},
		},
		"required": []any{"name", "count"},
	}
	runner := NewRunner(Options{Provider: ProviderAforge, BinPath: script})

	type runResult struct {
		result *Result
		dest   output
		err    error
	}
	results := make(chan runResult, 2)
	for _, prompt := range []string{"first", "second"} {
		prompt := prompt
		go func() {
			var dest output
			result, err := runner.Run(context.Background(), prompt, schema, &dest, Options{Cwd: cwd})
			results <- runResult{result: result, dest: dest, err: err}
		}()
	}

	seen := map[string]int{}
	for i := 0; i < 2; i++ {
		got := <-results
		require.NoError(t, got.err)
		require.NotNil(t, got.result)
		assert.False(t, got.result.IsError, got.result.ErrorMessage)
		seen[got.dest.Name] = got.dest.Count
	}
	assert.Equal(t, map[string]int{"first": 1, "second": 2}, seen)
	matches, err := filepath.Glob(filepath.Join(cwd, ".agentfield-out-*"))
	require.NoError(t, err)
	assert.Empty(t, matches)
	_, err = os.Stat(filepath.Join(cwd, outputFilename))
	assert.True(t, os.IsNotExist(err))
}
