package harness

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCLIHelpers(t *testing.T) {
	t.Run("StripANSI removes escape sequences", func(t *testing.T) {
		assert.Equal(t, "plain text", StripANSI("\x1b[31mplain\x1b[0m text"))
	})

	t.Run("isExecNotFound handles nil and common messages", func(t *testing.T) {
		assert.False(t, isExecNotFound(nil))
		assert.True(t, isExecNotFound(errors.New("exec: executable file not found in $PATH")))
		assert.True(t, isExecNotFound(errors.New("fork/exec /missing/tool: no such file or directory")))
		assert.False(t, isExecNotFound(errors.New("permission denied")))
	})

	t.Run("truncate preserves short strings and trims long ones", func(t *testing.T) {
		assert.Equal(t, "abc", truncate("abc", 10))
		assert.Equal(t, "abc", truncate("abcdef", 3))
	})
}

func TestRunCLI(t *testing.T) {
	t.Run("empty command returns error", func(t *testing.T) {
		result, err := RunCLI(context.Background(), nil, nil, "", 0)
		assert.Nil(t, result)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty command")
	})

	t.Run("merges env, unsets vars, and honors cwd", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "print-env", `#!/bin/sh
printf '%s\n' "PWD=$PWD"
if [ -z "${REMOVE_ME+x}" ]; then
  printf '%s\n' 'REMOVE_ME=unset'
else
  printf '%s\n' "REMOVE_ME=$REMOVE_ME"
fi
printf '%s\n' "KEEP_ME=$KEEP_ME"
`)

		t.Setenv("REMOVE_ME", "present")

		result, err := RunCLI(context.Background(), []string{script}, map[string]string{
			"REMOVE_ME": "",
			"KEEP_ME":   "set",
		}, dir, 0)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, 0, result.ReturnCode)
		assert.Contains(t, result.Stdout, "PWD="+dir)
		assert.Contains(t, result.Stdout, "REMOVE_ME=unset")
		assert.Contains(t, result.Stdout, "KEEP_ME=set")
	})

	t.Run("context cancellation returns a killed-process result with partial stdout", func(t *testing.T) {
		dir := t.TempDir()
		script := writeTestScript(t, dir, "sleepy", "#!/bin/sh\nprintf 'before-timeout'\nsleep 5\n")

		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()

		result, err := RunCLI(ctx, []string{script}, nil, "", 0)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEqual(t, 0, result.ReturnCode)
		assert.Equal(t, "before-timeout", result.Stdout)
	})
}

func TestProviderParsingHelpers(t *testing.T) {
	t.Run("extractAssistantText prefers direct content then nested blocks", func(t *testing.T) {
		assert.Equal(t, "direct", extractAssistantText(map[string]any{
			"content": "direct",
		}))
		assert.Equal(t, "nested", extractAssistantText(map[string]any{
			"message": map[string]any{"content": "nested"},
		}))
		assert.Equal(t, "block", extractAssistantText(map[string]any{
			"message": map[string]any{
				"content": []any{
					map[string]any{"type": "text", "text": "block"},
				},
			},
		}))
		assert.Equal(t, "", extractAssistantText(map[string]any{
			"content": []any{map[string]any{"type": "image"}},
		}))
	})

	t.Run("claude parser falls back to assistant text and counts messages", func(t *testing.T) {
		raw := &RawResult{}
		NewClaudeCodeProvider("").parseJSONOutput(strings.Join([]string{
			`{"type":"assistant","message":{"content":[{"type":"text","text":"from-assistant"}]}}`,
			`{"type":"ignored","value":1}`,
		}, "\n"), raw)

		assert.Equal(t, "from-assistant", raw.Result)
		assert.Len(t, raw.Messages, 2)
		assert.Equal(t, 2, raw.Metrics.NumTurns)
	})

	t.Run("codex parser handles content field and explicit num_turns", func(t *testing.T) {
		raw := &RawResult{}
		NewCodexProvider("").parseJSONLOutput(strings.Join([]string{
			`{"type":"thread.started","session_id":"sess-2"}`,
			`{"type":"item.completed","item":{"type":"agent_message","content":"from-content"}}`,
			`{"type":"result","session_id":"sess-3","num_turns":4}`,
		}, "\n"), raw)

		assert.Equal(t, "from-content", raw.Result)
		assert.Equal(t, "sess-3", raw.Metrics.SessionID)
		assert.Equal(t, 4, raw.Metrics.NumTurns)
		assert.Len(t, raw.Messages, 3)
	})
}

func TestSchemaHelperBranches(t *testing.T) {
	t.Run("estimateTokens is a simple four-char approximation", func(t *testing.T) {
		assert.Equal(t, 3, estimateTokens("abcdefghijkl"))
	})

	t.Run("BuildPromptSuffix falls back when schema cannot marshal", func(t *testing.T) {
		suffix := BuildPromptSuffix(map[string]any{
			"bad": make(chan int),
		}, "/tmp/out")
		assert.Contains(t, suffix, "CRITICAL OUTPUT REQUIREMENTS")
		assert.Contains(t, suffix, OutputPath("/tmp/out"))
		assert.NotContains(t, suffix, "Required JSON Schema:")
	})

	t.Run("BuildFollowupPrompt handles nil schema and existing schema file", func(t *testing.T) {
		dir := t.TempDir()
		require.NoError(t, os.WriteFile(SchemaPath(dir), []byte(`{"type":"object"}`), 0o600))

		prompt := BuildFollowupPrompt("bad output", dir, nil)
		assert.Contains(t, prompt, SchemaPath(dir))
		assert.Contains(t, prompt, OutputPath(dir))
	})

	t.Run("BuildFollowupPrompt reports marshal failures", func(t *testing.T) {
		prompt := BuildFollowupPrompt("bad output", t.TempDir(), map[string]any{
			"bad": make(chan int),
		})
		assert.Contains(t, prompt, "could not be serialized")
	})

	t.Run("DiagnoseOutputFailure truncates large invalid files", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "bad.json")
		require.NoError(t, os.WriteFile(path, []byte("{"+strings.Repeat("x", 700)), 0o644))

		diagnosis := DiagnoseOutputFailure(path, map[string]any{"properties": map[string]any{}})
		assert.Contains(t, diagnosis, "invalid JSON")
		assert.Contains(t, diagnosis, "first 500 chars")
		assert.LessOrEqual(t, len(diagnosis), 800)
	})

	t.Run("CleanupTempFiles is safe for empty and dot dirs", func(t *testing.T) {
		CleanupTempFiles("")
		CleanupTempFiles(".")
	})
}

func TestRunnerHelperBranches(t *testing.T) {
	t.Run("accumulateMetrics merges turns, session ids, and messages", func(t *testing.T) {
		turns, sid, msgs := accumulateMetrics([]*RawResult{
			{Metrics: Metrics{NumTurns: 1, SessionID: "old"}, Messages: []map[string]any{{"a": 1}}},
			{Metrics: Metrics{NumTurns: 2, SessionID: "new"}, Messages: []map[string]any{{"b": 2}}},
		})
		assert.Equal(t, 3, turns)
		assert.Equal(t, "new", sid)
		assert.Len(t, msgs, 2)
	})

	t.Run("fileExists reflects file presence", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "file.txt")
		assert.False(t, fileExists(path))
		require.NoError(t, os.WriteFile(path, []byte("x"), 0o644))
		assert.True(t, fileExists(path))
	})

	t.Run("StructToJSONSchema rejects nil and non-struct", func(t *testing.T) {
		_, err := StructToJSONSchema(nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "nil value")

		_, err = StructToJSONSchema([]string{"bad"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "expects a struct")
	})

	t.Run("StructToJSONSchema handles pointers, omitempty, ignored, and unexported fields", func(t *testing.T) {
		type nested struct {
			Flag bool `json:"flag,omitempty"`
		}
		type sample struct {
			Name   string   `json:"name"`
			Count  *int     `json:"count,omitempty"`
			Score  float64  `json:"score"`
			Tags   []string `json:"tags"`
			Nested nested   `json:"nested"`
			Skip   string   `json:"-"`
			hidden string
		}

		schema, err := StructToJSONSchema(&sample{})
		require.NoError(t, err)
		props := schema["properties"].(map[string]any)
		required := schema["required"].([]string)

		assert.Equal(t, "string", props["name"].(map[string]any)["type"])
		assert.Equal(t, "integer", props["count"].(map[string]any)["type"])
		assert.Equal(t, "number", props["score"].(map[string]any)["type"])
		assert.Equal(t, "array", props["tags"].(map[string]any)["type"])
		assert.Equal(t, "object", props["nested"].(map[string]any)["type"])
		assert.NotContains(t, props, "Skip")
		assert.NotContains(t, props, "hidden")
		assert.Contains(t, required, "name")
		assert.Contains(t, required, "score")
		assert.Contains(t, required, "tags")
		assert.Contains(t, required, "nested")
		assert.NotContains(t, required, "count")
	})

	t.Run("executeWithRetry returns provider raw on non-transient raw error", func(t *testing.T) {
		runner := NewRunner(Options{})
		prov := &funcProvider{
			fn: func(context.Context, string, Options) (*RawResult, error) {
				return &RawResult{IsError: true, ErrorMessage: "bad request", FailureType: FailureAPIError}, nil
			},
		}

		raw, err := runner.executeWithRetry(context.Background(), prov, "prompt", Options{MaxRetries: 2})
		require.NoError(t, err)
		assert.True(t, raw.IsError)
		assert.Equal(t, FailureAPIError, raw.FailureType)
	})

	t.Run("handleSchemaWithRetry returns provider failure when output file is absent and failure is non-retryable", func(t *testing.T) {
		runner := NewRunner(Options{})
		result := runner.handleSchemaWithRetry(
			context.Background(),
			&RawResult{
				IsError:      true,
				ErrorMessage: "api unavailable",
				FailureType:  FailureAPIError,
				Metrics:      Metrics{NumTurns: 2, SessionID: "sess-9"},
			},
			map[string]any{"properties": map[string]any{"value": map[string]any{"type": "string"}}},
			&struct {
				Value string `json:"value"`
			}{},
			t.TempDir(),
			time.Now(),
			&funcProvider{fn: func(context.Context, string, Options) (*RawResult, error) {
				return nil, fmt.Errorf("should not retry")
			}},
			Options{SchemaMaxRetries: 2},
			"prompt",
		)

		assert.True(t, result.IsError)
		assert.Equal(t, FailureAPIError, result.FailureType)
		assert.Contains(t, result.ErrorMessage, "Output file was not created")
		assert.Equal(t, "sess-9", result.SessionID)
	})
}
