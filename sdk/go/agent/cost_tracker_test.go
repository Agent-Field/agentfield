package agent

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// goldenGoUsage is the wire-contract serialization for three entries: a
// priced LLM call, an unpriced OpenRouter call, and a Claude Code harness
// run. It mirrors the control plane's goldenPythonUsage fixture
// (control-plane/internal/handlers/usage_contract_golden_test.go) except the
// priced entry's cost_source is "provider" — Go has no litellm pricing
// database, so it only ever emits "provider" or null.
const goldenGoUsage = `{
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
      "cost_source": "provider"
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

// newGoldenTracker records the three golden entries in order.
func newGoldenTracker() *CostTracker {
	tracker := NewCostTracker()
	llmCost := 0.0123
	tracker.Record(CostEntry{
		Model:               "claude-opus-4-8",
		Provider:            "anthropic",
		Reasoner:            "summarize",
		InputTokens:         100,
		OutputTokens:        50,
		CacheReadTokens:     2048,
		CacheCreationTokens: 64,
		TotalTokens:         150,
		CostUSD:             &llmCost,
		CostSource:          "provider",
	})
	tracker.Record(CostEntry{
		Model:        "openrouter/qwen/qwen3-coder",
		Reasoner:     "code",
		InputTokens:  500,
		OutputTokens: 200,
		TotalTokens:  700,
	})
	harnessCost := 0.5
	tracker.Record(CostEntry{
		Model:       "claude-opus-4-8",
		Provider:    "anthropic",
		Source:      "harness",
		Harness:     "claude_code",
		TotalTokens: 12345,
		CostUSD:     &harnessCost,
		CostSource:  "provider",
	})
	return tracker
}

// TestCostTrackerSerializeGolden maps to the contract: Serialize() emits the
// exact cross-language wire shape — unset strings and unknown costs as JSON
// null, totals summed, total_cost_usd rounded to 6 decimals, and the
// OpenRouter provider derived from the model slug.
func TestCostTrackerSerializeGolden(t *testing.T) {
	got, err := json.Marshal(newGoldenTracker().Serialize())
	require.NoError(t, err)
	assert.JSONEq(t, goldenGoUsage, string(got))
}

// TestDeriveProvider maps to the contract: provider is the lowercased first
// slug segment when a "/" is present, null (empty) for bare model names.
func TestDeriveProvider(t *testing.T) {
	tests := []struct {
		model string
		want  string
	}{
		{"anthropic/claude-opus-4-8", "anthropic"},
		{"OpenRouter/anthropic/claude", "openrouter"},
		{"gpt-4o", ""},
		{"", ""},
		{"  openai/gpt-4o  ", "openai"},
		{"/weird", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, deriveProvider(tt.model), "model %q", tt.model)
	}
}

// TestCostTrackerNullCostAndTotalsFallback maps to the contract: when no
// entry has a known cost total_cost_usd is null (not 0), and a zero
// total_tokens falls back to input+output per entry.
func TestCostTrackerNullCostAndTotalsFallback(t *testing.T) {
	tracker := NewCostTracker()
	tracker.Record(CostEntry{Model: "gpt-4o", InputTokens: 10, OutputTokens: 5})

	usage := tracker.Serialize()
	assert.Nil(t, usage["total_cost_usd"])
	assert.Equal(t, 15, usage["total_tokens"])

	entries := usage["entries"].([]map[string]any)
	require.Len(t, entries, 1)
	assert.Equal(t, 15, entries[0]["total_tokens"])
	assert.Nil(t, entries[0]["cost_usd"])
	assert.Nil(t, entries[0]["cost_source"])
	assert.Nil(t, entries[0]["provider"], "bare model slug derives no provider")
	assert.Nil(t, entries[0]["harness"])
	assert.Nil(t, entries[0]["reasoner"])
	assert.Equal(t, "llm", entries[0]["source"], "source defaults to llm")
}

// TestCostTrackerConcurrentRecord maps to the contract: parallel LLM calls
// within one execution can record into the same tracker safely.
func TestCostTrackerConcurrentRecord(t *testing.T) {
	tracker := NewCostTracker()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tracker.Record(CostEntry{Model: "m", InputTokens: 1, OutputTokens: 1})
		}()
	}
	wg.Wait()

	usage := tracker.Serialize()
	assert.Equal(t, 50, usage["total_input_tokens"])
	assert.Equal(t, 100, usage["total_tokens"])
	assert.Len(t, usage["entries"], 50)
}

// TestNilTrackerIsSafe maps to the contract: contexts without a bound tracker
// never panic — recording and summarizing are no-ops.
func TestNilTrackerIsSafe(t *testing.T) {
	var tracker *CostTracker
	tracker.Record(CostEntry{Model: "m"})
	assert.False(t, tracker.HasEntries())
	assert.Zero(t, tracker.TotalCostUSD())
	assert.Nil(t, usageSummaryOrNone(tracker))
	assert.Nil(t, CostTrackerFrom(context.Background()))
}

// TestRecordHarnessUsage maps to the contract: a harness run's usage lands in
// the current tracker as a source="harness" entry with the provider name
// underscore-mapped, cost_source="provider" when a cost is present, and an
// entry is recorded even when only cost or only tokens are known — but never
// when neither is.
func TestRecordHarnessUsage(t *testing.T) {
	newAgentForTest := func(t *testing.T) *Agent {
		t.Helper()
		a, err := New(Config{NodeID: "node-1", Version: "1.0.0"})
		require.NoError(t, err)
		return a
	}

	cost := 0.5
	tests := []struct {
		name      string
		result    *harness.Result
		wantEntry bool
		check     func(t *testing.T, entry map[string]any)
	}{
		{
			name: "tokens and cost",
			result: &harness.Result{
				InputTokens: 100, OutputTokens: 50, CacheReadTokens: 7,
				CacheCreationTokens: 3, TotalTokens: 150, CostUSD: &cost,
			},
			wantEntry: true,
			check: func(t *testing.T, entry map[string]any) {
				assert.Equal(t, "harness", entry["source"])
				assert.Equal(t, "claude_code", entry["harness"])
				assert.Equal(t, "sonnet-model", entry["model"])
				assert.Equal(t, 100, entry["input_tokens"])
				assert.Equal(t, 50, entry["output_tokens"])
				assert.Equal(t, 7, entry["cache_read_tokens"])
				assert.Equal(t, 3, entry["cache_creation_tokens"])
				assert.Equal(t, 150, entry["total_tokens"])
				assert.Equal(t, 0.5, entry["cost_usd"])
				assert.Equal(t, "provider", entry["cost_source"])
			},
		},
		{
			name:      "cost only still records",
			result:    &harness.Result{CostUSD: &cost},
			wantEntry: true,
			check: func(t *testing.T, entry map[string]any) {
				assert.Equal(t, 0.5, entry["cost_usd"])
				assert.Equal(t, "provider", entry["cost_source"])
				assert.Equal(t, 0, entry["total_tokens"])
			},
		},
		{
			name:      "tokens only still records with null cost",
			result:    &harness.Result{InputTokens: 10, OutputTokens: 2},
			wantEntry: true,
			check: func(t *testing.T, entry map[string]any) {
				assert.Nil(t, entry["cost_usd"])
				assert.Nil(t, entry["cost_source"])
				assert.Equal(t, 12, entry["total_tokens"], "total falls back to input+output")
			},
		},
		{
			name:      "no tokens and no cost records nothing",
			result:    &harness.Result{},
			wantEntry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := newAgentForTest(t)
			tracker := NewCostTracker()
			ctx := contextWithCostTracker(context.Background(), tracker)
			ctx = contextWithExecution(ctx, ExecutionContext{ReasonerName: "builder"})

			a.recordHarnessUsage(ctx, tt.result, harness.Options{Provider: "claude-code", Model: "sonnet-model"})

			if !tt.wantEntry {
				assert.False(t, tracker.HasEntries())
				return
			}
			usage := tracker.Serialize()
			entries := usage["entries"].([]map[string]any)
			require.Len(t, entries, 1)
			assert.Equal(t, "builder", entries[0]["reasoner"])
			tt.check(t, entries[0])
		})
	}

	t.Run("no tracker in context is a no-op", func(t *testing.T) {
		a := newAgentForTest(t)
		a.recordHarnessUsage(context.Background(), &harness.Result{CostUSD: &cost}, harness.Options{Provider: "claude-code"})
	})

	t.Run("provider falls back to agent harness config", func(t *testing.T) {
		a, err := New(Config{
			NodeID:  "node-1",
			Version: "1.0.0",
			HarnessConfig: &HarnessConfig{
				Provider: "open-code",
				Model:    "openrouter/qwen/qwen3-coder",
			},
		})
		require.NoError(t, err)
		tracker := NewCostTracker()
		ctx := contextWithCostTracker(context.Background(), tracker)

		a.recordHarnessUsage(ctx, &harness.Result{InputTokens: 1}, harness.Options{})

		entries := tracker.Serialize()["entries"].([]map[string]any)
		require.Len(t, entries, 1)
		assert.Equal(t, "open_code", entries[0]["harness"])
		assert.Equal(t, "openrouter/qwen/qwen3-coder", entries[0]["model"])
		assert.Equal(t, "openrouter", entries[0]["provider"])
	})
}
