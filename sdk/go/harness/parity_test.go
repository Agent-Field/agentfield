package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Per-call cost reporting ---------------------------------------------

// TestExtractCost pins the Python `cost_usd or total_cost_usd` semantics from
// agentfield/harness/providers/claude.py.
func TestExtractCost(t *testing.T) {
	t.Run("total_cost_usd only", func(t *testing.T) {
		c := extractCost(map[string]any{"total_cost_usd": 0.42})
		require.NotNil(t, c)
		assert.InDelta(t, 0.42, *c, 1e-9)
	})

	t.Run("cost_usd takes precedence when non-zero", func(t *testing.T) {
		c := extractCost(map[string]any{"cost_usd": 1.5, "total_cost_usd": 9.9})
		require.NotNil(t, c)
		assert.InDelta(t, 1.5, *c, 1e-9)
	})

	t.Run("zero cost_usd falls through to total_cost_usd", func(t *testing.T) {
		c := extractCost(map[string]any{"cost_usd": 0.0, "total_cost_usd": 2.0})
		require.NotNil(t, c)
		assert.InDelta(t, 2.0, *c, 1e-9)
	})

	t.Run("neither present yields nil", func(t *testing.T) {
		assert.Nil(t, extractCost(map[string]any{"type": "result"}))
	})

	t.Run("zero cost_usd with no total yields nil (0.0 or None == None)", func(t *testing.T) {
		assert.Nil(t, extractCost(map[string]any{"cost_usd": 0.0}))
	})
}

// TestClaudeCodeProvider_ExtractsCost verifies cost is pulled out of a Claude
// Code JSON result fixture (contract: cost extracted from a claude JSON output).
func TestClaudeCodeProvider_ExtractsCost(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "claude",
		`#!/bin/sh
echo '{"type":"result","result":"ok","session_id":"s1","num_turns":2,"total_cost_usd":0.0731}'
`)

	p := NewClaudeCodeProvider(script)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	require.False(t, raw.IsError)
	require.NotNil(t, raw.Metrics.CostUSD)
	assert.InDelta(t, 0.0731, *raw.Metrics.CostUSD, 1e-9)
}

// TestClaudeCodeProvider_NoCostIsNil confirms nil (unknown), not 0.0, when the
// provider does not report cost.
func TestClaudeCodeProvider_NoCostIsNil(t *testing.T) {
	dir := t.TempDir()
	script := writeTestScript(t, dir, "claude",
		`#!/bin/sh
echo '{"type":"result","result":"ok","session_id":"s1","num_turns":1}'
`)

	p := NewClaudeCodeProvider(script)
	raw, err := p.Execute(context.Background(), "prompt", Options{})
	require.NoError(t, err)
	require.False(t, raw.IsError)
	assert.Nil(t, raw.Metrics.CostUSD)
}

// TestAccumulateMetrics_Cost verifies cost sums over attempts and stays nil
// when nothing reported a cost (Python _accumulate_metrics semantics).
func TestAccumulateMetrics_Cost(t *testing.T) {
	t.Run("nil when no attempt reports cost", func(t *testing.T) {
		cost, _, _, _, _ := accumulateMetrics([]*RawResult{
			{Metrics: Metrics{NumTurns: 1}},
			{Metrics: Metrics{NumTurns: 1}},
		})
		assert.Nil(t, cost)
	})

	t.Run("sums present costs", func(t *testing.T) {
		c1, c2 := 0.10, 0.25
		cost, _, _, _, _ := accumulateMetrics([]*RawResult{
			{Metrics: Metrics{CostUSD: &c1}},
			{Metrics: Metrics{NumTurns: 1}}, // no cost — skipped
			{Metrics: Metrics{CostUSD: &c2}},
		})
		require.NotNil(t, cost)
		assert.InDelta(t, 0.35, *cost, 1e-9)
	})
}

// TestHandleSchemaWithRetry_AccumulatesCostIncludingFailedAttempts proves the
// end-to-end result cost sums across every attempt, INCLUDING failed retry
// attempts (Python appends retry_raw to all_raws before the is_error check).
func TestHandleSchemaWithRetry_AccumulatesCostIncludingFailedAttempts(t *testing.T) {
	dir := t.TempDir()
	type Out struct {
		Value string `json:"value"`
	}
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"value": map[string]any{"type": "string"}},
	}

	c0, c1, c2 := 0.10, 0.20, 0.30
	initialRaw := &RawResult{Result: "no json", Metrics: Metrics{NumTurns: 1, CostUSD: &c0}}
	mock := &mockProvider{results: []*RawResult{
		{Result: "still bad", IsError: true, ErrorMessage: "nope", Metrics: Metrics{NumTurns: 1, CostUSD: &c1}},
		{Result: "still bad", IsError: true, ErrorMessage: "nope", Metrics: Metrics{NumTurns: 1, CostUSD: &c2}},
	}}

	var dest Out
	result := NewRunner(Options{Provider: "opencode"}).handleSchemaWithRetry(
		context.Background(), initialRaw, schema, &dest, dir,
		time.Now(), mock, Options{Provider: "opencode", SchemaMaxRetries: 2}, "test prompt", false,
	)

	require.True(t, result.IsError)
	require.NotNil(t, result.CostUSD)
	// 0.10 (initial) + 0.20 + 0.30 (both failed retries) = 0.60
	assert.InDelta(t, 0.60, *result.CostUSD, 1e-9)
}

// --- Incremental schema mode ---------------------------------------------

// TestResolveIncremental covers the mode-selection rules from the Python
// HarnessRunner._resolve_incremental.
func TestResolveIncremental(t *testing.T) {
	small := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
	}

	assert.False(t, resolveIncremental(nil, Options{SchemaMode: "incremental"}), "nil schema never incremental")
	assert.True(t, resolveIncremental(small, Options{SchemaMode: "incremental"}))
	assert.True(t, resolveIncremental(small, Options{SchemaMode: "INCREMENTAL"}), "mode is case-insensitive")
	assert.False(t, resolveIncremental(small, Options{}), "default (empty) is single-shot")
	assert.False(t, resolveIncremental(small, Options{SchemaMode: "single"}))
	assert.False(t, resolveIncremental(small, Options{SchemaMode: "auto"}), "small schema stays single-shot under auto")
}

// TestResolveIncremental_AutoEngagesOnLargeSchema constructs the auto trigger:
// a schema whose compact JSON exceeds the large-schema token threshold engages
// incremental, matching Python (which measures the compact json.dumps output).
func TestResolveIncremental_AutoEngagesOnLargeSchema(t *testing.T) {
	large := largeParitySchema()

	compact, err := json.Marshal(large)
	require.NoError(t, err)
	require.Greater(t, estimateTokens(string(compact)), largeSchemaTokenThreshold,
		"fixture must exceed the threshold on its COMPACT encoding")

	assert.True(t, resolveIncremental(large, Options{SchemaMode: "auto"}))
}

// TestBuildIncrementalPromptSuffix checks the byte-verbatim incremental build
// instructions and the per-field required/optional listing.
func TestBuildIncrementalPromptSuffix(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{"type": "string"},
			"body":  map[string]any{"type": "string"},
		},
		"required": []any{"title"},
	}
	dir := t.TempDir()
	s := BuildIncrementalPromptSuffix(schema, dir)

	assert.Contains(t, s, "CRITICAL OUTPUT REQUIREMENTS (incremental build):")
	assert.Contains(t, s, "Build it ONE FIELD AT A TIME so nothing gets truncated:")
	assert.Contains(t, s, "  1. First create the file with an empty object: {}")
	assert.Contains(t, s, "     each edit re-read the file to confirm it is still valid JSON.")
	assert.Contains(t, s, "  4. Do not finish until every required field is present.")
	assert.Contains(t, s, "Top-level fields to add:")
	assert.Contains(t, s, "  - title (required)")
	assert.Contains(t, s, "  - body (optional)")
	assert.Contains(t, s, "Full JSON Schema:")
	assert.Contains(t, s, OutputPath(dir))
	assert.Contains(t, s, "no markdown fences, no commentary, no extra text.")
}

// TestBuildIncrementalFollowup checks the byte-verbatim patch-only follow-up.
func TestBuildIncrementalFollowup(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"body": map[string]any{"type": "string"}},
	}
	dir := t.TempDir()
	s := BuildIncrementalFollowup(map[string]string{"body": "missing required field"}, dir, schema)

	assert.Contains(t, s, "PARTIAL OUTPUT NEEDS FIXES. The JSON at ")
	assert.Contains(t, s, " is incomplete or invalid.")
	assert.Contains(t, s, "Patch ONLY these fields, one at a time, using Edit, keeping the file valid JSON after each change:")
	assert.Contains(t, s, "  - body: missing required field")
	assert.Contains(t, s, "Full schema:")
	assert.Contains(t, s, "Leave every already-correct field unchanged. Do NOT rewrite the whole file.")
}

// TestDiagnoseFieldFailures covers the field-level diagnosis reasons used to
// drive incremental recovery.
func TestDiagnoseFieldFailures(t *testing.T) {
	dir := t.TempDir()
	type Out struct {
		Title string `json:"title"`
		Body  string `json:"body"`
	}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{"type": "string"},
			"body":  map[string]any{"type": "string"},
		},
		"required": []any{"title", "body"},
	}
	path := OutputPath(dir)

	t.Run("missing required field", func(t *testing.T) {
		require.NoError(t, os.WriteFile(path, []byte(`{"title":"x"}`), 0o644))
		var dest Out
		f := DiagnoseFieldFailures(path, schema, &dest)
		assert.Equal(t, "missing required field", f["body"])
		_, hasTitle := f["title"]
		assert.False(t, hasTitle, "present required field is not flagged")
	})

	t.Run("file not a JSON object reports required fields", func(t *testing.T) {
		require.NoError(t, os.WriteFile(path, []byte(`not json at all`), 0o644))
		var dest Out
		f := DiagnoseFieldFailures(path, schema, &dest)
		assert.Equal(t, "output file missing or not a JSON object", f["title"])
		assert.Equal(t, "output file missing or not a JSON object", f["body"])
	})

	t.Run("clean object has no failures", func(t *testing.T) {
		require.NoError(t, os.WriteFile(path, []byte(`{"title":"x","body":"y"}`), 0o644))
		var dest Out
		f := DiagnoseFieldFailures(path, schema, &dest)
		assert.Empty(t, f)
	})

	t.Run("type mismatch reports _root", func(t *testing.T) {
		require.NoError(t, os.WriteFile(path, []byte(`{"title":"x","body":123}`), 0o644))
		var dest Out
		f := DiagnoseFieldFailures(path, schema, &dest)
		_, has := f["_root"]
		assert.True(t, has)
		// The caller's dest must NOT be mutated by diagnosis.
		assert.Equal(t, "", dest.Title)
	})
}

// TestHandleSchemaWithRetry_IncrementalRecovery drives the incremental retry
// path: an initial partial/invalid output file triggers a patch-only follow-up
// (with the original goal preserved), and the corrected file assembles a valid
// dest.
func TestHandleSchemaWithRetry_IncrementalRecovery(t *testing.T) {
	dir := t.TempDir()
	type Out struct {
		Title string `json:"title"`
		Body  string `json:"body"`
	}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{"type": "string"},
			"body":  map[string]any{"type": "string"},
		},
		"required": []any{"title", "body"},
	}
	outputPath := OutputPath(dir)
	// Initial file has a type-mismatched required field, so validation fails
	// and the incremental recovery path runs (Go struct unmarshal, unlike
	// pydantic, does not fail on a merely-missing field).
	require.NoError(t, os.WriteFile(outputPath, []byte(`{"title":"hello","body":123}`), 0o644))

	var capturedPrompt string
	prov := &funcProvider{fn: func(_ context.Context, prompt string, _ Options) (*RawResult, error) {
		capturedPrompt = prompt
		// Agent patches the file to a valid object.
		_ = os.WriteFile(outputPath, []byte(`{"title":"hello","body":"world"}`), 0o644)
		return &RawResult{Result: "patched", Metrics: Metrics{NumTurns: 1}}, nil
	}}

	var dest Out
	result := NewRunner(Options{}).handleSchemaWithRetry(
		context.Background(),
		&RawResult{Result: "", Metrics: Metrics{NumTurns: 1}},
		schema, &dest, dir, time.Now(), prov,
		Options{SchemaMaxRetries: 2}, "ORIGINAL GOAL", true, // useIncremental
	)

	require.False(t, result.IsError)
	assert.Equal(t, "hello", dest.Title)
	assert.Equal(t, "world", dest.Body)

	assert.Contains(t, capturedPrompt, "PARTIAL OUTPUT NEEDS FIXES")
	assert.Contains(t, capturedPrompt, "Patch ONLY these fields")
	assert.Contains(t, capturedPrompt, "ORIGINAL GOAL", "original goal is prepended on incremental retry")
}

// TestSchemaMode_SingleShotUnchanged asserts the single-shot production path is
// selected and unchanged when the mode is off.
func TestSchemaMode_SingleShotUnchanged(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
	}

	assert.False(t, resolveIncremental(schema, Options{}))
	assert.False(t, resolveIncremental(schema, Options{SchemaMode: "single"}))

	single := BuildPromptSuffix(schema, "/tmp/parity")
	assert.Contains(t, single, "CRITICAL OUTPUT REQUIREMENTS:")
	assert.NotContains(t, single, "incremental build")

	inc := BuildIncrementalPromptSuffix(schema, "/tmp/parity")
	assert.Contains(t, inc, "CRITICAL OUTPUT REQUIREMENTS (incremental build):")
}

// TestMergeOptions_SchemaMode verifies SchemaMode flows through the default/
// override merge.
func TestMergeOptions_SchemaMode(t *testing.T) {
	r := NewRunner(Options{Provider: "opencode", SchemaMode: "auto"})
	assert.Equal(t, "auto", r.mergeOptions(Options{}).SchemaMode, "default carried through")
	assert.Equal(t, "incremental", r.mergeOptions(Options{SchemaMode: "incremental"}).SchemaMode, "override wins")
}

// largeParitySchema builds a JSON schema whose COMPACT encoding exceeds the
// large-schema token threshold (~16000 chars).
func largeParitySchema() map[string]any {
	props := make(map[string]any)
	for i := 0; i < 600; i++ {
		key := fmt.Sprintf("field_%04d", i)
		props[key] = map[string]any{
			"type":        "string",
			"description": "A padded description that helps push the schema past the large-schema token threshold.",
		}
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
	}
}
