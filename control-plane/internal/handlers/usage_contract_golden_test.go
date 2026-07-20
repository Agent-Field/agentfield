package handlers

import (
	"encoding/json"
	"testing"
)

// goldenPythonUsage is the verbatim output of the Python SDK's
// CostTracker.serialize() (sdk/python/agentfield/cost_tracker.py) for three
// recorded entries: a priced LLM call, an unpriced OpenRouter call, and a
// Claude Code harness run. It pins the cross-language wire contract; if the
// SDK's serialization changes shape, regenerate it with:
//
//	cd sdk/python && python3 -c "from agentfield.cost_tracker import CostTracker; \
//	  t = CostTracker(); ...record entries...; import json; print(json.dumps(t.serialize()))"
const goldenPythonUsage = `{
  "total_cost_usd": 0.5123,
  "total_input_tokens": 600,
  "total_output_tokens": 250,
  "total_tokens": 13195,
  "entries": [
    {
      "source": "llm",
      "provider": "anthropic",
      "model": "claude-opus-4-8",
      "harness": null,
      "reasoner": "summarize",
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_tokens": 2048,
      "cache_creation_tokens": 64,
      "total_tokens": 150,
      "cost_usd": 0.0123,
      "cost_source": "litellm"
    },
    {
      "source": "llm",
      "provider": "openrouter",
      "model": "openrouter/qwen/qwen3-coder",
      "harness": null,
      "reasoner": "code",
      "input_tokens": 500,
      "output_tokens": 200,
      "cache_read_tokens": 0,
      "cache_creation_tokens": 0,
      "total_tokens": 700,
      "cost_usd": null,
      "cost_source": null
    },
    {
      "source": "harness",
      "provider": "anthropic",
      "model": "claude-opus-4-8",
      "harness": "claude_code",
      "reasoner": null,
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_read_tokens": 0,
      "cache_creation_tokens": 0,
      "total_tokens": 12345,
      "cost_usd": 0.5,
      "cost_source": "provider"
    }
  ]
}`

// assertGoldenUsageRows pins the parsed form of the three canonical entries
// every SDK's golden payload must contain. wantRow0CostSource differs by SDK:
// Python prices via litellm ("litellm"); Go/TypeScript only carry
// provider-native costs ("provider").
func assertGoldenUsageRows(t *testing.T, golden string, wantRow0CostSource string) {
	t.Helper()

	var usage map[string]interface{}
	if err := json.Unmarshal([]byte(golden), &usage); err != nil {
		t.Fatalf("golden payload does not decode: %v", err)
	}

	rows := parseUsageEntries(sampleExec(), usage)
	if len(rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(rows))
	}

	llm := rows[0]
	if llm.Source != "llm" || llm.Provider != "anthropic" || llm.Model != "claude-opus-4-8" {
		t.Errorf("row 0 identity mismatch: %+v", llm)
	}
	if llm.Harness != "" || llm.Reasoner != "summarize" {
		t.Errorf("row 0 harness/reasoner mismatch: %q %q", llm.Harness, llm.Reasoner)
	}
	if llm.InputTokens != 100 || llm.OutputTokens != 50 || llm.CacheReadTokens != 2048 ||
		llm.CacheCreationTokens != 64 || llm.TotalTokens != 150 {
		t.Errorf("row 0 token mismatch: %+v", llm)
	}
	if llm.CostUSD == nil || *llm.CostUSD != 0.0123 || llm.CostSource != wantRow0CostSource {
		t.Errorf("row 0 cost mismatch: %v %q", llm.CostUSD, llm.CostSource)
	}

	unpriced := rows[1]
	if unpriced.CostUSD != nil {
		t.Errorf("row 1: null cost_usd must map to nil, got %v", *unpriced.CostUSD)
	}
	if unpriced.CostSource != "" || unpriced.Provider != "openrouter" ||
		unpriced.Model != "openrouter/qwen/qwen3-coder" || unpriced.TotalTokens != 700 {
		t.Errorf("row 1 mismatch: %+v", unpriced)
	}

	harness := rows[2]
	// A null reasoner in the payload falls back to the execution's reasoner ID.
	if harness.Source != "harness" || harness.Harness != "claude_code" || harness.Reasoner != "default_reasoner" {
		t.Errorf("row 2 harness identity mismatch: %+v", harness)
	}
	if harness.TotalTokens != 12345 || harness.CostUSD == nil || *harness.CostUSD != 0.5 ||
		harness.CostSource != "provider" {
		t.Errorf("row 2 mismatch: %+v", harness)
	}

	for i, r := range rows {
		if r.ExecutionID != "exec_1" || r.WorkflowID != "run_1" || r.AgentNodeID != "agent-a" {
			t.Errorf("row %d execution linkage mismatch: %+v", i, r)
		}
	}
}

func TestUsageGoldenFromPythonSDK(t *testing.T) {
	assertGoldenUsageRows(t, goldenPythonUsage, "litellm")
}

// goldenGoUsage is the verbatim output of the Go SDK's CostTracker.Serialize()
// (sdk/go/agent/cost_tracker.go) for the same three canonical entries (Go's
// only cost source is "provider" — there is no litellm equivalent). Go
// marshals map keys alphabetically, so key order differs from the Python
// fixture; the parser is map-based and order-insensitive. Regenerate by
// recording the three entries from TestCostTrackerSerializeGolden's setup and
// printing the Serialize() result.
const goldenGoUsage = `{
  "entries": [
    {"cache_creation_tokens": 64, "cache_read_tokens": 2048, "cost_source": "provider", "cost_usd": 0.0123, "harness": null, "input_tokens": 100, "model": "claude-opus-4-8", "output_tokens": 50, "provider": "anthropic", "reasoner": "summarize", "source": "llm", "total_tokens": 150},
    {"cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_source": null, "cost_usd": null, "harness": null, "input_tokens": 500, "model": "openrouter/qwen/qwen3-coder", "output_tokens": 200, "provider": "openrouter", "reasoner": "code", "source": "llm", "total_tokens": 700},
    {"cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_source": "provider", "cost_usd": 0.5, "harness": "claude_code", "input_tokens": 0, "model": "claude-opus-4-8", "output_tokens": 0, "provider": "anthropic", "reasoner": null, "source": "harness", "total_tokens": 12345}
  ],
  "total_cost_usd": 0.5123, "total_input_tokens": 600, "total_output_tokens": 250, "total_tokens": 13195
}`

func TestUsageGoldenFromGoSDK(t *testing.T) {
	assertGoldenUsageRows(t, goldenGoUsage, "provider")
}

// goldenTypeScriptUsage is the verbatim output of the TypeScript SDK's
// CostTracker.serialize() (sdk/typescript/src/usage/costTracker.ts) for the
// same three canonical entries (provider-native cost only, like Go).
const goldenTypeScriptUsage = `{"total_cost_usd":0.5123,"total_input_tokens":600,"total_output_tokens":250,"total_tokens":13195,"entries":[{"source":"llm","provider":"anthropic","model":"claude-opus-4-8","harness":null,"reasoner":"summarize","input_tokens":100,"output_tokens":50,"cache_read_tokens":2048,"cache_creation_tokens":64,"total_tokens":150,"cost_usd":0.0123,"cost_source":"provider"},{"source":"llm","provider":"openrouter","model":"openrouter/qwen/qwen3-coder","harness":null,"reasoner":"code","input_tokens":500,"output_tokens":200,"cache_read_tokens":0,"cache_creation_tokens":0,"total_tokens":700,"cost_usd":null,"cost_source":null},{"source":"harness","provider":"anthropic","model":"claude-opus-4-8","harness":"claude_code","reasoner":null,"input_tokens":0,"output_tokens":0,"cache_read_tokens":0,"cache_creation_tokens":0,"total_tokens":12345,"cost_usd":0.5,"cost_source":"provider"}]}`

func TestUsageGoldenFromTypeScriptSDK(t *testing.T) {
	assertGoldenUsageRows(t, goldenTypeScriptUsage, "provider")
}

// TestUsageGoldenSyncBody exercises the sync-200 shape: the SDK merges the
// usage object under the reserved "__agentfield_usage__" envelope key as a
// sibling into the result dict, and the control plane strips it back out
// without disturbing the rest of the result — including a user-owned "usage"
// key, which is payload, not transport.
func TestUsageGoldenSyncBody(t *testing.T) {
	body := []byte(`{"answer": 42, "detail": {"ok": true}, "usage": {"user": "data"}, "__agentfield_usage__": ` + goldenPythonUsage + `}`)

	usageRaw, stripped := extractUsageFromResult(body)
	if usageRaw == nil {
		t.Fatal("usage not extracted from sync body")
	}
	if rows := parseUsageEntries(sampleExec(), usageRaw); len(rows) != 3 {
		t.Fatalf("expected 3 rows from sync body usage, got %d", len(rows))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(stripped, &result); err != nil {
		t.Fatalf("stripped result does not decode: %v", err)
	}
	if _, leaked := result[usageEnvelopeKey]; leaked {
		t.Error("usage envelope key leaked into stored result")
	}
	if result["answer"] != float64(42) {
		t.Errorf("result content disturbed: %+v", result)
	}
	userUsage, ok := result["usage"].(map[string]interface{})
	if !ok || userUsage["user"] != "data" {
		t.Errorf("user-owned usage key was not preserved: %+v", result)
	}
}
