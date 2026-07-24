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

// codexSchemaRejectionEvent is the real error event codex exec --json emits
// when the server's strict validator refuses --output-schema (captured live on
// codex-cli 0.144.1, issue: invalid_json_schema for codex_output_schema).
const codexSchemaRejectionEvent = `{"type":"error","message":"{\n  \"type\": \"error\",\n  \"error\": {\n    \"type\": \"invalid_request_error\",\n    \"code\": \"invalid_json_schema\",\n    \"message\": \"Invalid schema for response_format 'codex_output_schema'. Please ensure it is a valid JSON Schema.\",\n    \"param\": \"text.format.schema\"\n  },\n  \"status\": 400\n}"}`

// TestCodexSchemaStrictExpressible pins the live-probed strict-validator rules:
// which schema shapes may be sent through --output-schema and which must fall
// back to last-message + local validation.
func TestCodexSchemaStrictExpressible(t *testing.T) {
	strictObject := func(props map[string]any) map[string]any {
		return codexStrictJSONSchema(map[string]any{
			"type":       "object",
			"properties": props,
		})
	}

	cases := []struct {
		name   string
		schema map[string]any
		want   bool
	}{
		{
			name:   "flat strict object",
			schema: strictObject(map[string]any{"status": map[string]any{"type": "string"}}),
			want:   true,
		},
		{
			name:   "empty properties object",
			schema: strictObject(map[string]any{}),
			want:   true,
		},
		{
			name: "free-form map property",
			schema: strictObject(map[string]any{
				"meta": map[string]any{"type": "object"},
			}),
			want: false,
		},
		{
			name: "typed map property",
			schema: strictObject(map[string]any{
				"meta": map[string]any{
					"type":                 "object",
					"additionalProperties": map[string]any{"type": "string"},
				},
			}),
			want: false,
		},
		{
			name: "bare Any node",
			schema: strictObject(map[string]any{
				"value": map[string]any{},
			}),
			want: false,
		},
		{
			name: "anyOf with null",
			schema: strictObject(map[string]any{
				"note": map[string]any{"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "null"},
				}},
			}),
			want: true,
		},
		{
			name: "array of strict objects via $defs ref",
			schema: codexStrictJSONSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"items": map[string]any{
						"type":  "array",
						"items": map[string]any{"$ref": "#/$defs/Sub"},
					},
				},
				"$defs": map[string]any{
					"Sub": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"x": map[string]any{"type": "string"},
						},
					},
				},
			}),
			want: true,
		},
		{
			name: "inexpressible $defs entry poisons the schema",
			schema: codexStrictJSONSchema(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"sub": map[string]any{"$ref": "#/$defs/Sub"},
				},
				"$defs": map[string]any{
					"Sub": map[string]any{"type": "object"},
				},
			}),
			want: false,
		},
		{
			name: "array without items",
			schema: strictObject(map[string]any{
				"xs": map[string]any{"type": "array"},
			}),
			want: false,
		},
		{
			name: "type list of primitives",
			schema: strictObject(map[string]any{
				"a": map[string]any{"type": []any{"string", "null"}},
			}),
			want: true,
		},
		{
			name: "type list containing object",
			schema: strictObject(map[string]any{
				"a": map[string]any{"type": []any{"object", "null"}},
			}),
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, codexSchemaStrictExpressible(tc.schema))
		})
	}
}

// TestCodexProvider_LastMessageOnlyArgv verifies the not-strict-expressible
// contract: an empty schemaPath with a non-empty outputPath yields
// --output-last-message but NO --output-schema.
func TestCodexProvider_LastMessageOnlyArgv(t *testing.T) {
	dir := t.TempDir()
	outputPath := OutputPath(dir)

	var gotCmd []string
	p := NewCodexProvider("codex")
	p.SetSchema("", outputPath)
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		gotCmd = cmd
		return &CLIResult{Stdout: `{"type":"result","result":"{}"}`, ReturnCode: 0}, nil
	}

	_, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.NotContains(t, gotCmd, "--output-schema")
	assert.Contains(t, strings.Join(gotCmd, " "), "--output-last-message "+outputPath)
}

// TestCodexProvider_OutputSchemaRejectionFallback verifies the reactive
// fallback: a run whose --output-schema is refused server-side
// (invalid_json_schema) is rerun exactly once without the flag, keeping
// --output-last-message, and the retry's result is returned.
func TestCodexProvider_OutputSchemaRejectionFallback(t *testing.T) {
	dir := t.TempDir()
	schemaPath := SchemaPath(dir)
	outputPath := OutputPath(dir)
	require.NoError(t, os.WriteFile(schemaPath, []byte(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`), 0o600))

	var calls [][]string
	p := NewCodexProvider("codex")
	p.SetSchema(schemaPath, outputPath)
	p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		calls = append(calls, cmd)
		if len(calls) == 1 {
			return &CLIResult{Stdout: codexSchemaRejectionEvent, ReturnCode: 1}, nil
		}
		return &CLIResult{
			Stdout:     `{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"{\"ok\":true}"}}`,
			ReturnCode: 0,
		}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	require.Len(t, calls, 2)

	first := strings.Join(calls[0], " ")
	assert.Contains(t, first, "--output-schema "+schemaPath)
	second := strings.Join(calls[1], " ")
	assert.NotContains(t, calls[1], "--output-schema")
	assert.Contains(t, second, "--output-last-message "+outputPath)

	assert.False(t, raw.IsError)
	assert.Equal(t, `{"ok":true}`, raw.Result)
}

// TestCodexProvider_NonSchemaFailureNoFallbackRetry verifies an ordinary
// failure (no invalid_json_schema marker) is NOT rerun.
func TestCodexProvider_NonSchemaFailureNoFallbackRetry(t *testing.T) {
	dir := t.TempDir()
	schemaPath := SchemaPath(dir)
	outputPath := OutputPath(dir)
	require.NoError(t, os.WriteFile(schemaPath, []byte(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`), 0o600))

	callCount := 0
	p := NewCodexProvider("codex")
	p.SetSchema(schemaPath, outputPath)
	p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _ int, _ []byte) (*CLIResult, error) {
		callCount++
		return &CLIResult{Stderr: "stream error: something unrelated", ReturnCode: 1}, nil
	}

	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	assert.Equal(t, 1, callCount)
	assert.True(t, raw.IsError)
}

// TestRunner_Run_CodexInexpressibleSchemaSkipsFlag drives the runner end-to-end
// with a schema containing a free-form map field: the codex argv must carry
// --output-last-message but not --output-schema, and the answer written to the
// last-message file must still parse and validate locally.
func TestRunner_Run_CodexInexpressibleSchemaSkipsFlag(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "captured-args.txt")

	script := writeTestScript(t, dir, "codex",
		"#!/bin/sh\n"+
			"cat > /dev/null\n"+
			"printf '%s\\n' \"$@\" > \""+argsFile+"\"\n"+
			"out=\"\"\n"+
			"prev=\"\"\n"+
			"for a in \"$@\"; do\n"+
			"  if [ \"$prev\" = \"--output-last-message\" ]; then out=\"$a\"; fi\n"+
			"  prev=\"$a\"\n"+
			"done\n"+
			"if [ -n \"$out\" ]; then printf '%s' '{\"status\":\"ok\",\"meta\":{\"k\":\"v\"}}' > \"$out\"; fi\n"+
			"printf '%s\\n' '{\"type\":\"result\",\"result\":\"\"}'\n")

	runner := NewRunner(Options{Provider: ProviderCodex, BinPath: script})

	var dest struct {
		Status string         `json:"status"`
		Meta   map[string]any `json:"meta"`
	}
	result, err := runner.Run(context.Background(), "do the work", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string"},
			"meta":   map[string]any{"type": "object"},
		},
	}, &dest, Options{ProjectDir: dir})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.IsError, "expected success, got: %s", result.ErrorMessage)
	assert.Equal(t, "ok", dest.Status)
	assert.Equal(t, map[string]any{"k": "v"}, dest.Meta)

	captured, err := os.ReadFile(argsFile)
	require.NoError(t, err)
	args := string(captured)
	assert.NotContains(t, args, "--output-schema")
	assert.Contains(t, args, "--output-last-message")
}
