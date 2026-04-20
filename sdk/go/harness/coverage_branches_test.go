package harness

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResultText(t *testing.T) {
	assert.Equal(t, "hello", (&Result{Result: "hello"}).Text())
}

func TestRunnerMergeOptions_AllOverrideBranches(t *testing.T) {
	defaults := Options{
		Provider:        ProviderOpenCode,
		Model:           "base-model",
		MaxTurns:        1,
		PermissionMode:  "plan",
		SystemPrompt:    "base system",
		Env:             map[string]string{"BASE": "1"},
		Cwd:             "/base/cwd",
		ProjectDir:      "/base/project",
		Tools:           []string{"base-tool"},
		MaxBudgetUSD:    1.25,
		ResumeSessionID: "base-session",
		BinPath:         "/base/bin",
		Timeout:         10,
		MaxRetries:      1,
		InitialDelay:    1.5,
		MaxDelay:        2.5,
		BackoffFactor:   3.5,
		SchemaMaxRetries: 1,
	}

	merged := NewRunner(defaults).mergeOptions(Options{
		Provider:         ProviderCodex,
		Model:            "override-model",
		MaxTurns:         9,
		PermissionMode:   "auto",
		SystemPrompt:     "override system",
		Env:              map[string]string{"EXTRA": "2"},
		Cwd:              "/override/cwd",
		ProjectDir:       "/override/project",
		Tools:            []string{"tool-a", "tool-b"},
		MaxBudgetUSD:     9.99,
		ResumeSessionID:  "override-session",
		BinPath:          "/override/bin",
		Timeout:          99,
		MaxRetries:       4,
		InitialDelay:     0.25,
		MaxDelay:         4.25,
		BackoffFactor:    1.75,
		SchemaMaxRetries: 5,
	})

	assert.Equal(t, ProviderCodex, merged.Provider)
	assert.Equal(t, "override-model", merged.Model)
	assert.Equal(t, 9, merged.MaxTurns)
	assert.Equal(t, "auto", merged.PermissionMode)
	assert.Equal(t, "override system", merged.SystemPrompt)
	assert.Equal(t, "/override/cwd", merged.Cwd)
	assert.Equal(t, "/override/project", merged.ProjectDir)
	assert.Equal(t, []string{"tool-a", "tool-b"}, merged.Tools)
	assert.Equal(t, 9.99, merged.MaxBudgetUSD)
	assert.Equal(t, "override-session", merged.ResumeSessionID)
	assert.Equal(t, "/override/bin", merged.BinPath)
	assert.Equal(t, 99, merged.Timeout)
	assert.Equal(t, 4, merged.MaxRetries)
	assert.Equal(t, 0.25, merged.InitialDelay)
	assert.Equal(t, 4.25, merged.MaxDelay)
	assert.Equal(t, 1.75, merged.BackoffFactor)
	assert.Equal(t, 5, merged.SchemaMaxRetries)
	assert.Equal(t, "1", merged.Env["BASE"])
	assert.Equal(t, "2", merged.Env["EXTRA"])
}

func TestRunnerRun_SuccessBranches(t *testing.T) {
	t.Run("no schema returns provider result", func(t *testing.T) {
		runner := NewRunner(Options{
			Provider: ProviderOpenCode,
			BinPath:  "opencode",
		})

		result, err := runner.Run(context.Background(), "prompt", nil, nil, Options{})
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.False(t, result.IsError)
		assert.Equal(t, "stub opencode result", strings.TrimSpace(result.Result))
		assert.GreaterOrEqual(t, result.DurationMS, 0)
	})

	t.Run("schema + project dir uses temp output dir and parses file", func(t *testing.T) {
		projectDir := t.TempDir()
		script := writeTestScript(t, projectDir, "opencode-write-json", `#!/bin/sh
for last; do :; done
output_path=$(printf '%s' "$last" | tr '\n' ' ' | sed -n 's/.*create this file: \([^ ]*\.agentfield_output\.json\).*/\1/p')
dirname=$(dirname "$output_path")
mkdir -p "$dirname"
printf '%s' '{"status":"ok"}' > "$output_path"
printf '%s\n' 'wrote structured output'
`)

		runner := NewRunner(Options{
			Provider: ProviderOpenCode,
			BinPath:  script,
		})

		var dest struct {
			Status string `json:"status"`
		}

		result, err := runner.Run(context.Background(), "prompt", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"status": map[string]any{"type": "string"},
			},
		}, &dest, Options{
			ProjectDir: projectDir,
		})
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.False(t, result.IsError)
		assert.Equal(t, "ok", dest.Status)

		matches, globErr := filepath.Glob(filepath.Join(projectDir, ".agentfield-out-*"))
		require.NoError(t, globErr)
		assert.Empty(t, matches)
	})
}

func TestProviderExecuteAdditionalBranches(t *testing.T) {
	t.Run("ClaudeCode Execute passes optional flags and env unsets CLAUDECODE", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "claude-echo", `#!/bin/sh
printf '%s\n' '{"type":"result","result":"ok","session_id":"claude-session","num_turns":2}'
printf '%s\n' "ARGS:$*" >&2
if [ -z "${CLAUDECODE+x}" ]; then
  printf '%s\n' 'CLAUDECODE=unset' >&2
else
  printf '%s\n' "CLAUDECODE=$CLAUDECODE" >&2
fi
`)

		raw, err := NewClaudeCodeProvider(script).Execute(context.Background(), "prompt", Options{
			Model:           "sonnet",
			MaxTurns:        3,
			PermissionMode:  "auto",
			SystemPrompt:    "system",
			ResumeSessionID: "resume-me",
			MaxBudgetUSD:    1.5,
			Tools:           []string{"Read", "Write"},
			Env:             map[string]string{"EXTRA": "1"},
			Cwd:             dir,
		})
		require.NoError(t, err)
		require.NotNil(t, raw)
		assert.False(t, raw.IsError)
		assert.Equal(t, "ok", raw.Result)
	})

	t.Run("ClaudeCode Execute marks non-zero exit with stdout as non-fatal provider error", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "claude-exit-with-output", "#!/bin/sh\necho '{\"type\":\"result\",\"result\":\"partial\"}'\nexit 3\n")

		raw, err := NewClaudeCodeProvider(script).Execute(context.Background(), "prompt", Options{})
		require.NoError(t, err)
		assert.True(t, raw.IsError)
		assert.Equal(t, "partial", raw.Result)
		assert.Equal(t, "Process exited with code 3", raw.ErrorMessage)
	})

	t.Run("Gemini Execute with cwd/project dir and stdout on non-zero keeps result", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "gemini-exit-with-output", "#!/bin/sh\necho \"$PWD|$*\"\nexit 2\n")

		raw, err := NewGeminiProvider(script).Execute(context.Background(), "prompt", Options{
			Cwd:            dir,
			ProjectDir:     "/ignored/project",
			Model:          "gemini-model",
			PermissionMode: "auto",
		})
		require.NoError(t, err)
		assert.False(t, raw.IsError)
		assert.Contains(t, raw.Result, dir)
		assert.Contains(t, raw.Result, "--sandbox")
		assert.Contains(t, raw.Result, "-m gemini-model")
	})

	t.Run("OpenCode Execute prefers cwd over project dir and keeps stdout on non-zero", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "opencode-exit-with-output", "#!/bin/sh\necho \"$PWD|$*\"\nexit 2\n")

		raw, err := NewOpenCodeProvider(script, "").Execute(context.Background(), "prompt", Options{
			Cwd:        dir,
			ProjectDir: "/ignored/project",
		})
		require.NoError(t, err)
		assert.False(t, raw.IsError)
		assert.Contains(t, raw.Result, dir)
		assert.Contains(t, raw.Result, "-q")
		assert.Contains(t, raw.Result, "-p prompt")
	})
}

func TestSchemaAdditionalBranches(t *testing.T) {
	t.Run("writeSchemaFile returns an error when parent path is a file", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "not-a-dir")
		require.NoError(t, os.WriteFile(root, []byte("x"), 0o644))

		err := writeSchemaFile(`{"type":"object"}`, filepath.Join(root, "child"))
		require.Error(t, err)
	})

	t.Run("ReadRepairAndParse rejects empty and unrecoverable files", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.json")
		require.NoError(t, os.WriteFile(path, []byte(""), 0o644))
		_, err := ReadRepairAndParse(path)
		require.Error(t, err)

		require.NoError(t, os.WriteFile(path, []byte("still not json"), 0o644))
		_, err = ReadRepairAndParse(path)
		require.Error(t, err)
	})

	t.Run("unmarshalInto returns an error for incompatible destination", func(t *testing.T) {
		err := unmarshalInto(map[string]any{"value": "x"}, &struct {
			Value int `json:"value"`
		}{})
		require.Error(t, err)
	})
}

func TestRunnerRetryAdditionalBranches(t *testing.T) {
	t.Run("executeWithRetry retries transient raw errors until success", func(t *testing.T) {
		attempts := 0
		prov := &funcProvider{
			fn: func(context.Context, string, Options) (*RawResult, error) {
				attempts++
				if attempts < 3 {
					return &RawResult{IsError: true, ErrorMessage: "503 service unavailable"}, nil
				}
				return &RawResult{Result: "ok"}, nil
			},
		}

		raw, err := NewRunner(Options{}).executeWithRetry(context.Background(), prov, "prompt", Options{
			MaxRetries:   3,
			InitialDelay: 0.001,
			MaxDelay:     0.001,
		})
		require.NoError(t, err)
		assert.Equal(t, "ok", raw.Result)
		assert.Equal(t, 3, attempts)
	})

	t.Run("handleSchemaWithRetry returns timeout when context is cancelled during retry delay", func(t *testing.T) {
		dir := t.TempDir()
		ctx, cancel := context.WithCancel(context.Background())

		prov := &funcProvider{
			fn: func(context.Context, string, Options) (*RawResult, error) {
				cancel()
				return &RawResult{Result: "still bad"}, nil
			},
		}

		result := NewRunner(Options{}).handleSchemaWithRetry(
			ctx,
			&RawResult{Result: "bad first result"},
			map[string]any{"properties": map[string]any{"value": map[string]any{"type": "string"}}},
			&struct {
				Value string `json:"value"`
			}{},
			dir,
			time.Now(),
			prov,
			Options{SchemaMaxRetries: 2},
			"prompt",
		)

		assert.True(t, result.IsError)
		assert.Equal(t, FailureTimeout, result.FailureType)
		assert.Contains(t, result.ErrorMessage, "context cancelled during schema retry")
	})
}
