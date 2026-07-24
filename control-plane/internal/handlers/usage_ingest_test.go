package handlers

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// fakeUsageStore embeds the ExecutionStore interface (nil) and implements the
// usageWriter capability so ingestUsage can persist through it. Only
// CreateExecutionUsage is exercised.
type fakeUsageStore struct {
	ExecutionStore
	created [][]*types.ExecutionUsage
	err     error
}

func (f *fakeUsageStore) CreateExecutionUsage(ctx context.Context, rows []*types.ExecutionUsage) error {
	if f.err != nil {
		return f.err
	}
	f.created = append(f.created, rows)
	return nil
}

func sampleExec() *types.Execution {
	return &types.Execution{
		ExecutionID: "exec_1",
		RunID:       "run_1",
		AgentNodeID: "agent-a",
		ReasonerID:  "default_reasoner",
	}
}

func contractUsage() map[string]interface{} {
	var m map[string]interface{}
	_ = json.Unmarshal([]byte(`{
		"total_cost_usd": 0.0123,
		"total_input_tokens": 123,
		"total_output_tokens": 456,
		"total_tokens": 579,
		"entries": [
			{
				"source": "llm",
				"provider": "anthropic",
				"model": "claude-opus-4-8",
				"harness": null,
				"reasoner": "my_reasoner",
				"input_tokens": 100,
				"output_tokens": 50,
				"cache_read_tokens": 0,
				"cache_creation_tokens": 0,
				"total_tokens": 150,
				"cost_usd": 0.01,
				"cost_source": "litellm"
			},
			{
				"source": "harness",
				"provider": null,
				"model": "claude-opus-4-8",
				"harness": "claude_code",
				"input_tokens": 10,
				"output_tokens": 5,
				"total_tokens": 15,
				"cost_usd": null,
				"cost_source": null
			}
		]
	}`), &m)
	return m
}

func TestParseUsageEntriesContract(t *testing.T) {
	rows := parseUsageEntries(sampleExec(), contractUsage())
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}

	r0 := rows[0]
	if r0.ExecutionID != "exec_1" || r0.WorkflowID != "run_1" || r0.AgentNodeID != "agent-a" {
		t.Errorf("row0 IDs taken from payload instead of execution: %+v", r0)
	}
	if r0.Source != "llm" || r0.Provider != "anthropic" || r0.Model != "claude-opus-4-8" {
		t.Errorf("row0 fields = %+v", r0)
	}
	if r0.Reasoner != "my_reasoner" {
		t.Errorf("row0 reasoner = %q, want my_reasoner", r0.Reasoner)
	}
	if r0.InputTokens != 100 || r0.OutputTokens != 50 || r0.TotalTokens != 150 {
		t.Errorf("row0 tokens = %+v", r0)
	}
	if r0.CostUSD == nil || *r0.CostUSD != 0.01 || r0.CostSource != "litellm" {
		t.Errorf("row0 cost = %v / %q", r0.CostUSD, r0.CostSource)
	}

	r1 := rows[1]
	if r1.Harness != "claude_code" {
		t.Errorf("row1 harness = %q", r1.Harness)
	}
	if r1.CostUSD != nil {
		t.Errorf("row1 cost = %v, want nil", r1.CostUSD)
	}
	// null reasoner falls back to the execution's reasoner id.
	if r1.Reasoner != "default_reasoner" {
		t.Errorf("row1 reasoner = %q, want default_reasoner fallback", r1.Reasoner)
	}
}

func TestParseUsageEntriesJunkTolerance(t *testing.T) {
	var usage map[string]interface{}
	_ = json.Unmarshal([]byte(`{
		"entries": [
			{"model": "m1", "input_tokens": -5, "output_tokens": "not-a-number", "cost_usd": -1.0},
			"i-am-not-an-object",
			42,
			{"model": "m2", "input_tokens": 7.0, "total_tokens": null}
		]
	}`), &usage)

	rows := parseUsageEntries(sampleExec(), usage)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (junk entries skipped)", len(rows))
	}
	// Negative clamped to 0, non-numeric coerced to 0.
	if rows[0].InputTokens != 0 || rows[0].OutputTokens != 0 {
		t.Errorf("row0 tokens not clamped/coerced: %+v", rows[0])
	}
	// Negative cost clamped to 0 (non-nil).
	if rows[0].CostUSD == nil || *rows[0].CostUSD != 0 {
		t.Errorf("row0 cost = %v, want 0", rows[0].CostUSD)
	}
	// total_tokens missing/null -> input+output fallback (7 + 0).
	if rows[1].InputTokens != 7 || rows[1].TotalTokens != 7 {
		t.Errorf("row1 = %+v, want input 7 / total 7", rows[1])
	}
}

func TestParseUsageEntriesCap(t *testing.T) {
	entries := make([]interface{}, 600)
	for i := range entries {
		entries[i] = map[string]interface{}{"model": "m", "input_tokens": float64(1)}
	}
	usage := map[string]interface{}{"entries": entries}
	rows := parseUsageEntries(sampleExec(), usage)
	if len(rows) != maxUsageEntriesPerExecution {
		t.Errorf("got %d rows, want cap %d", len(rows), maxUsageEntriesPerExecution)
	}
}

func TestParseUsageEntriesAbsent(t *testing.T) {
	if rows := parseUsageEntries(sampleExec(), map[string]interface{}{}); rows != nil {
		t.Errorf("empty usage produced %d rows, want nil", len(rows))
	}
	if rows := parseUsageEntries(sampleExec(), map[string]interface{}{"entries": []interface{}{}}); len(rows) != 0 {
		t.Errorf("empty entries produced %d rows, want 0", len(rows))
	}
}

func TestExtractUsageFromResultStrips(t *testing.T) {
	body := []byte(`{"message":"hello","__agentfield_usage__":{"entries":[{"model":"m"}]}}`)
	usage, stripped := extractUsageFromResult(body)
	if usage == nil {
		t.Fatal("usage not extracted")
	}
	var out map[string]interface{}
	if err := json.Unmarshal(stripped, &out); err != nil {
		t.Fatalf("stripped body not valid JSON: %v", err)
	}
	if _, present := out[usageEnvelopeKey]; present {
		t.Errorf("usage envelope key leaked into stripped body: %s", stripped)
	}
	if out["message"] != "hello" {
		t.Errorf("stripped body lost result content: %s", stripped)
	}
}

func TestExtractUsageFromResultNested(t *testing.T) {
	body := []byte(`{"result":{"answer":42,"__agentfield_usage__":{"entries":[{"model":"m"}]}}}`)
	usage, stripped := extractUsageFromResult(body)
	if usage == nil {
		t.Fatal("nested usage not extracted")
	}
	if len(stripped) == 0 || string(stripped) == string(body) {
		t.Errorf("nested usage not stripped: %s", stripped)
	}
	var out map[string]interface{}
	_ = json.Unmarshal(stripped, &out)
	inner, _ := out["result"].(map[string]interface{})
	if _, present := inner[usageEnvelopeKey]; present {
		t.Errorf("nested usage leaked: %s", stripped)
	}
}

// TestExtractUsageFromResultPreservesUserUsageKey pins the compatibility
// guarantee that motivated the namespaced envelope key: an agent whose result
// legitimately contains a top-level (or nested) "usage" key is user payload
// and must pass through byte-for-byte untouched.
func TestExtractUsageFromResultPreservesUserUsageKey(t *testing.T) {
	bodies := [][]byte{
		[]byte(`{"usage":{"entries":[{"model":"user-data"}]},"answer":42}`),
		[]byte(`{"result":{"usage":{"anything":true},"answer":42}}`),
	}
	for _, body := range bodies {
		usage, stripped := extractUsageFromResult(body)
		if usage != nil {
			t.Errorf("user usage key misread as envelope: %s", body)
		}
		if string(stripped) != string(body) {
			t.Errorf("user payload altered: %s -> %s", body, stripped)
		}
	}
}

func TestExtractUsageFromResultNoUsage(t *testing.T) {
	body := []byte(`{"message":"hello"}`)
	usage, stripped := extractUsageFromResult(body)
	if usage != nil {
		t.Errorf("unexpected usage: %v", usage)
	}
	if string(stripped) != string(body) {
		t.Errorf("body altered when no usage present: %s", stripped)
	}

	// Non-object bodies are returned unchanged.
	arr := []byte(`[1,2,3]`)
	usage, stripped = extractUsageFromResult(arr)
	if usage != nil || string(stripped) != string(arr) {
		t.Errorf("array body altered: usage=%v stripped=%s", usage, stripped)
	}
}

func TestIngestUsagePersists(t *testing.T) {
	store := &fakeUsageStore{}
	c := &executionController{store: store}

	c.ingestUsage(context.Background(), sampleExec(), contractUsage())

	if len(store.created) != 1 {
		t.Fatalf("CreateExecutionUsage called %d times, want 1", len(store.created))
	}
	if len(store.created[0]) != 2 {
		t.Errorf("persisted %d rows, want 2", len(store.created[0]))
	}
}

func TestIngestUsageAbsentIsNoOp(t *testing.T) {
	store := &fakeUsageStore{}
	c := &executionController{store: store}

	c.ingestUsage(context.Background(), sampleExec(), nil)
	c.ingestUsage(context.Background(), sampleExec(), map[string]interface{}{})

	if len(store.created) != 0 {
		t.Errorf("CreateExecutionUsage called %d times for absent usage, want 0", len(store.created))
	}
}
