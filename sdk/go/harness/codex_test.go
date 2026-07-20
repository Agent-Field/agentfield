package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCodexStrictJSONSchema exercises the strict rewrite against a nested
// fixture: objects inside array items, an anyOf branch, and $defs. On every
// object node defaults must be stripped, required must list all property keys,
// and additionalProperties must be false.
func TestCodexStrictJSONSchema(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string", "default": "anon"},
			// A boolean-schema property value (non-map) is passed through as-is.
			"flag": true,
			"items": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":    map[string]any{"type": "integer", "default": 0},
						"label": map[string]any{"type": "string"},
					},
				},
			},
			"choice": map[string]any{
				"anyOf": []any{
					map[string]any{
						"type": "object",
						"properties": map[string]any{
							"a": map[string]any{"type": "string", "default": "z"},
						},
					},
					// A non-map branch entry is passed through untouched.
					true,
				},
			},
		},
		"$defs": map[string]any{
			"Sub": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"v": map[string]any{"type": "string", "default": "d"},
				},
			},
		},
		// definitions with a non-map value exercises the pass-through branch.
		"definitions": map[string]any{
			"Legacy": false,
		},
	}

	strict := codexStrictJSONSchema(schema)

	// Top-level object: additionalProperties false, all keys required, no default.
	assert.Equal(t, false, strict["additionalProperties"])
	assert.ElementsMatch(t, []string{"name", "flag", "items", "choice"}, strict["required"].([]string))
	// Non-map property values are passed through untouched.
	assert.Equal(t, true, strict["properties"].(map[string]any)["flag"])

	props := strict["properties"].(map[string]any)
	name := props["name"].(map[string]any)
	_, nameHasDefault := name["default"]
	assert.False(t, nameHasDefault, "top-level default must be stripped")

	// Array items object.
	items := props["items"].(map[string]any)
	itemObj := items["items"].(map[string]any)
	assert.Equal(t, false, itemObj["additionalProperties"])
	assert.ElementsMatch(t, []string{"id", "label"}, itemObj["required"].([]string))
	idProp := itemObj["properties"].(map[string]any)["id"].(map[string]any)
	_, idHasDefault := idProp["default"]
	assert.False(t, idHasDefault, "array-item default must be stripped")

	// anyOf branch object.
	anyOf := props["choice"].(map[string]any)["anyOf"].([]any)
	branch0 := anyOf[0].(map[string]any)
	assert.Equal(t, false, branch0["additionalProperties"])
	assert.ElementsMatch(t, []string{"a"}, branch0["required"].([]string))
	aProp := branch0["properties"].(map[string]any)["a"].(map[string]any)
	_, aHasDefault := aProp["default"]
	assert.False(t, aHasDefault, "anyOf-branch default must be stripped")
	// The non-map branch entry is passed through untouched.
	assert.Equal(t, true, anyOf[1])

	// $defs recursion.
	sub := strict["$defs"].(map[string]any)["Sub"].(map[string]any)
	assert.Equal(t, false, sub["additionalProperties"])
	assert.ElementsMatch(t, []string{"v"}, sub["required"].([]string))
	// definitions with a non-map value is passed through untouched.
	assert.Equal(t, false, strict["definitions"].(map[string]any)["Legacy"])

	// The source schema must be left unmutated (no additionalProperties added).
	_, srcMutated := schema["additionalProperties"]
	assert.False(t, srcMutated, "input schema must not be mutated")
}

// TestCodexProvider_SandboxMapping asserts the permission_mode → sandbox flag
// mapping from codex_harness_patch.py:165-170.
func TestCodexProvider_SandboxMapping(t *testing.T) {
	cases := []struct {
		mode string
		want []string
	}{
		{"auto", []string{"--dangerously-bypass-approvals-and-sandbox"}},
		{"read-only", []string{"--sandbox", "read-only"}},
		{"workspace-write", []string{"--sandbox", "workspace-write"}},
		{"danger-full-access", []string{"--sandbox", "danger-full-access"}},
		{"", []string{"--sandbox", "workspace-write"}},
		{"something-else", []string{"--sandbox", "workspace-write"}},
	}

	for _, tc := range cases {
		t.Run("mode="+tc.mode, func(t *testing.T) {
			var gotCmd []string
			p := NewCodexProvider("codex")
			p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
				gotCmd = cmd
				return &CLIResult{Stdout: `{"type":"result","result":"ok"}`, ReturnCode: 0}, nil
			}

			_, err := p.Execute(context.Background(), "prompt", Options{PermissionMode: tc.mode})
			require.NoError(t, err)

			joined := strings.Join(gotCmd, " ")
			assert.Contains(t, joined, strings.Join(tc.want, " "))
			assert.NotContains(t, gotCmd, "--full-auto")
		})
	}
}

// TestCodexProvider_NativeSchemaArgv verifies that when a schema file is set,
// the argv points --output-schema at it and --output-last-message at the
// output file, and that the schema file on disk is the strict rewrite.
func TestCodexProvider_NativeSchemaArgv(t *testing.T) {
	dir := t.TempDir()
	schemaPath := SchemaPath(dir)
	outputPath := OutputPath(dir)

	// Emulate what the runner writes: the strict rewrite of the source schema.
	source := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string", "default": "unknown"},
		},
	}
	strictJSON, err := json.MarshalIndent(codexStrictJSONSchema(source), "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(schemaPath, strictJSON, 0o600))

	var gotCmd []string
	p := NewCodexProvider("codex")
	p.SetSchema(schemaPath, outputPath)
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		gotCmd = cmd
		return &CLIResult{Stdout: `{"type":"result","result":"{\"status\":\"ok\"}"}`, ReturnCode: 0}, nil
	}

	_, err = p.Execute(context.Background(), "prompt", Options{Model: "gpt-5.5"})
	require.NoError(t, err)

	joined := strings.Join(gotCmd, " ")
	assert.Contains(t, joined, "--output-schema "+schemaPath)
	assert.Contains(t, joined, "--output-last-message "+outputPath)

	// The schema file on disk must be the strict rewrite.
	var written map[string]any
	raw, err := os.ReadFile(schemaPath)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(raw, &written))
	assert.Equal(t, false, written["additionalProperties"])
	assert.ElementsMatch(t, []string{"status"}, toStringSlice(written["required"]))
	statusProp := written["properties"].(map[string]any)["status"].(map[string]any)
	_, hasDefault := statusProp["default"]
	assert.False(t, hasDefault)
}

// TestCodexProvider_LastMessageFallback verifies that when stdout carries no
// parseable final text but codex persisted its answer to the output file, the
// result is read from the file (codex_harness_patch.py:236-243).
func TestCodexProvider_LastMessageFallback(t *testing.T) {
	dir := t.TempDir()
	schemaPath := SchemaPath(dir)
	outputPath := OutputPath(dir)
	require.NoError(t, os.WriteFile(schemaPath, []byte(`{"type":"object"}`), 0o600))

	p := NewCodexProvider("codex")
	p.SetSchema(schemaPath, outputPath)
	p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		// codex writes its final message to the last-message file; stdout has
		// only non-final events, so parsing yields no result text.
		require.NoError(t, os.WriteFile(outputPath, []byte(`{"status":"ok"}`), 0o600))
		return &CLIResult{Stdout: `{"type":"turn.started"}`, ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.False(t, raw.IsError)
	assert.Equal(t, `{"status":"ok"}`, raw.Result)
}

// TestCodexProvider_TimeoutSurfacesFailureTimeout verifies a timed-out run is
// classified as FailureTimeout, not a generic crash.
func TestCodexProvider_TimeoutSurfacesFailureTimeout(t *testing.T) {
	p := NewCodexProvider("codex")
	p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		return nil, fmt.Errorf("CLI command timed out after 1s: codex exec")
	}

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.True(t, raw.IsError)
	assert.Equal(t, FailureTimeout, raw.FailureType)
}

// TestCodexProvider_StdoutPreferredOverFile verifies the fallback does NOT fire
// when stdout already yields a final message.
func TestCodexProvider_StdoutPreferredOverFile(t *testing.T) {
	dir := t.TempDir()
	schemaPath := SchemaPath(dir)
	outputPath := OutputPath(dir)
	require.NoError(t, os.WriteFile(schemaPath, []byte(`{"type":"object"}`), 0o600))

	p := NewCodexProvider("codex")
	p.SetSchema(schemaPath, outputPath)
	p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		require.NoError(t, os.WriteFile(outputPath, []byte(`{"from":"file"}`), 0o600))
		return &CLIResult{Stdout: `{"type":"result","result":"from-stdout"}`, ReturnCode: 0}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.Equal(t, "from-stdout", raw.Result)
}

// TestRunner_Run_CodexNativeSchemaEndToEnd drives the runner with a fake codex
// binary and asserts the native path end-to-end: the runner writes the strict
// schema and hands codex a codex-native suffix (not the Write-tool one), codex
// persists its answer to --output-last-message, and the runner reads it back.
func TestRunner_Run_CodexNativeSchemaEndToEnd(t *testing.T) {
	dir := t.TempDir()
	promptFile := filepath.Join(dir, "captured-prompt.txt")

	// The fake codex reads the prompt from stdin (capturing it), locates the
	// --output-last-message path in its args, writes valid JSON there, and emits
	// an empty result event so the runner must use the file, not stdout.
	script := writeTestScript(t, dir, "codex",
		"#!/bin/sh\n"+
			"cat > \""+promptFile+"\"\n"+
			"out=\"\"\n"+
			"prev=\"\"\n"+
			"for a in \"$@\"; do\n"+
			"  if [ \"$prev\" = \"--output-last-message\" ]; then out=\"$a\"; fi\n"+
			"  prev=\"$a\"\n"+
			"done\n"+
			"if [ -n \"$out\" ]; then printf '%s' '{\"status\":\"ok\"}' > \"$out\"; fi\n"+
			"printf '%s\\n' '{\"type\":\"result\",\"result\":\"\"}'\n")

	runner := NewRunner(Options{Provider: ProviderCodex, BinPath: script})

	var dest struct {
		Status string `json:"status"`
	}
	result, err := runner.Run(context.Background(), "do the work", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string"},
		},
	}, &dest, Options{ProjectDir: dir})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.IsError, "expected success, got: %s", result.ErrorMessage)
	assert.Equal(t, "ok", dest.Status)

	captured, err := os.ReadFile(promptFile)
	require.NoError(t, err)
	prompt := string(captured)
	assert.Contains(t, prompt, "CRITICAL CODEX STRUCTURED OUTPUT REQUIREMENTS")
	assert.NotContains(t, prompt, "use your Write tool")
}

func toStringSlice(v any) []string {
	switch vv := v.(type) {
	case []string:
		return vv
	case []any:
		out := make([]string, 0, len(vv))
		for _, item := range vv {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

// TestCodexProvider_ModelVariantReasoningEffort pins the model#variant parity
// matrix from the Python provider tests (test_harness_provider_codex.py):
// -m always carries the BASE model, a "#variant" suffix (or an explicit
// Options.Variant, which wins) becomes -c model_reasoning_effort=<v>, and a
// bare model adds no -c at all.
func TestCodexProvider_ModelVariantReasoningEffort(t *testing.T) {
	run := func(t *testing.T, options Options) []string {
		t.Helper()
		var gotCmd []string
		p := NewCodexProvider("codex")
		p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
			gotCmd = append([]string(nil), cmd...)
			return &CLIResult{Stdout: `{"type":"turn.completed","text":"ok"}`, ReturnCode: 0}, nil
		}
		raw, err := p.Execute(context.Background(), "hello", options)
		require.NoError(t, err)
		require.False(t, raw.IsError, "unexpected error: %s", raw.ErrorMessage)
		return gotCmd
	}

	index := func(cmd []string, flag string) int {
		for i, arg := range cmd {
			if arg == flag {
				return i
			}
		}
		return -1
	}

	t.Run("suffix maps to model_reasoning_effort", func(t *testing.T) {
		cmd := run(t, Options{Model: "gpt-5.3-codex#high"})
		mIdx := index(cmd, "-m")
		require.GreaterOrEqual(t, mIdx, 0)
		assert.Equal(t, "gpt-5.3-codex", cmd[mIdx+1], "-m must carry the base model, never the suffixed string")
		cIdx := index(cmd, "-c")
		require.GreaterOrEqual(t, cIdx, 0)
		assert.Equal(t, "model_reasoning_effort=high", cmd[cIdx+1])
	})

	t.Run("explicit variant wins over suffix", func(t *testing.T) {
		cmd := run(t, Options{Model: "gpt-5.5#low", Variant: "max"})
		mIdx := index(cmd, "-m")
		require.GreaterOrEqual(t, mIdx, 0)
		assert.Equal(t, "gpt-5.5", cmd[mIdx+1])
		cIdx := index(cmd, "-c")
		require.GreaterOrEqual(t, cIdx, 0)
		assert.Equal(t, "model_reasoning_effort=max", cmd[cIdx+1])
	})

	t.Run("bare model has no effort config", func(t *testing.T) {
		cmd := run(t, Options{Model: "gpt-5.5"})
		mIdx := index(cmd, "-m")
		require.GreaterOrEqual(t, mIdx, 0)
		assert.Equal(t, "gpt-5.5", cmd[mIdx+1])
		assert.NotContains(t, cmd, "-c")
	})
}
