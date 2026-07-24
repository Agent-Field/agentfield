package agent

import (
	"context"
	"math"
	"strings"
	"sync"
)

// UsageEnvelopeKey is the reserved key under which the SDK attaches the
// serialized usage summary to a synchronous 200 result body. It is namespaced
// so it can never collide with user data: a plain "usage" key in an agent's
// own result object is user payload and must never be touched. The control
// plane strips exactly this key back out (see the control plane's
// extractUsageFromResult). "__agentfield_"-prefixed keys are reserved for
// SDK<->control-plane transport.
const UsageEnvelopeKey = "__agentfield_usage__"

// CostEntry is a single LLM (or harness) call usage record.
//
// Unset string fields (empty "") serialize as JSON null, and a nil CostUSD
// serializes as JSON null — matching the Python SDK's CostEntry, whose
// serialized form is the cross-language wire contract.
type CostEntry struct {
	// Model is the model slug the call ran against (e.g. "openai/gpt-4o").
	Model string

	InputTokens  int
	OutputTokens int
	// TotalTokens falls back to InputTokens+OutputTokens at serialization
	// time when left at zero.
	TotalTokens int

	// CostUSD may be unknown (provider gave no figure) — tokens are recorded
	// regardless, so cost is optional and never gates them. nil means
	// "unknown", not "free".
	CostUSD *float64

	// Reasoner is the reasoner name the call executed under, if any.
	Reasoner string

	// Source is "llm" for direct completions, "harness" for coding-agent
	// runs. Defaults to "llm" when empty at Record time.
	Source string

	// Provider is e.g. "anthropic", "openrouter", "openai" — derived from the
	// model slug at Record time when empty.
	Provider string

	// Harness is e.g. "claude_code" for harness-originated entries; empty for
	// plain LLM calls.
	Harness string

	CacheReadTokens     int
	CacheCreationTokens int

	// CostSource records where CostUSD came from: "provider" when the LLM
	// provider returned a native cost, empty when the cost is unknown. (The
	// Python SDK can additionally emit "litellm"; Go has no litellm pricing
	// database, so it only ever emits "provider" or null.)
	CostSource string
}

// deriveProvider returns a best-effort provider name from a model slug.
//
//	"anthropic/claude-opus-4-8"   -> "anthropic"
//	"openrouter/anthropic/claude" -> "openrouter"
//	"gpt-4o" (no prefix)          -> "" (serialized as null)
//
// Mirrors the Python SDK's derive_provider.
func deriveProvider(model string) string {
	slug := strings.TrimSpace(model)
	if slug == "" {
		return ""
	}
	if idx := strings.Index(slug, "/"); idx >= 0 {
		return strings.ToLower(slug[:idx])
	}
	return ""
}

// CostTracker accumulates LLM/harness usage for a single execution run. It is
// safe for concurrent use: parallel LLM calls within one execution may record
// into the same tracker.
type CostTracker struct {
	mu      sync.Mutex
	entries []CostEntry
}

// NewCostTracker returns an empty tracker.
func NewCostTracker() *CostTracker {
	return &CostTracker{}
}

// Record appends a single call's usage. When entry.Provider is empty it is
// derived from the model slug; when entry.Source is empty it defaults to
// "llm". Cost is optional: a call with known token counts but an unknown
// price is still recorded (CostUSD nil) so tokens are never discarded.
func (t *CostTracker) Record(entry CostEntry) {
	if t == nil {
		return
	}
	if entry.Provider == "" {
		entry.Provider = deriveProvider(entry.Model)
	}
	if entry.Source == "" {
		entry.Source = "llm"
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.entries = append(t.entries, entry)
}

// HasEntries reports whether any usage has been recorded.
func (t *CostTracker) HasEntries() bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.entries) > 0
}

// TotalCostUSD returns the accumulated cost in USD (unknown costs count as
// zero).
func (t *CostTracker) TotalCostUSD() float64 {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	total := 0.0
	for _, e := range t.entries {
		if e.CostUSD != nil {
			total += *e.CostUSD
		}
	}
	return total
}

// Serialize returns the transport contract form attached to execution
// envelopes. Shape (the token/cost usage contract, byte-compatible with the
// Python SDK's CostTracker.serialize()):
//
//	{
//	  "total_cost_usd": number|null,
//	  "total_input_tokens": int,
//	  "total_output_tokens": int,
//	  "total_tokens": int,
//	  "entries": [ {source, provider, model, harness, reasoner,
//	                input_tokens, output_tokens, cache_read_tokens,
//	                cache_creation_tokens, total_tokens, cost_usd,
//	                cost_source} ]
//	}
//
// total_cost_usd is null when no entry carried a known cost, otherwise the
// sum rounded to 6 decimals. Unset string fields and unknown costs serialize
// as JSON null, never "".
func (t *CostTracker) Serialize() map[string]any {
	t.mu.Lock()
	defer t.mu.Unlock()

	entries := make([]map[string]any, 0, len(t.entries))
	totalInput := 0
	totalOutput := 0
	totalTokens := 0
	totalCost := 0.0
	anyCost := false

	for _, e := range t.entries {
		entryTotal := e.TotalTokens
		if entryTotal == 0 {
			entryTotal = e.InputTokens + e.OutputTokens
		}
		entries = append(entries, map[string]any{
			"source":                nullableString(e.Source),
			"provider":              nullableString(e.Provider),
			"model":                 nullableString(e.Model),
			"harness":               nullableString(e.Harness),
			"reasoner":              nullableString(e.Reasoner),
			"input_tokens":          e.InputTokens,
			"output_tokens":         e.OutputTokens,
			"cache_read_tokens":     e.CacheReadTokens,
			"cache_creation_tokens": e.CacheCreationTokens,
			"total_tokens":          entryTotal,
			"cost_usd":              nullableFloat(e.CostUSD),
			"cost_source":           nullableString(e.CostSource),
		})
		totalInput += e.InputTokens
		totalOutput += e.OutputTokens
		totalTokens += entryTotal
		if e.CostUSD != nil {
			totalCost += *e.CostUSD
			anyCost = true
		}
	}

	var totalCostField any
	if anyCost {
		totalCostField = math.Round(totalCost*1e6) / 1e6
	}

	return map[string]any{
		"total_cost_usd":      totalCostField,
		"total_input_tokens":  totalInput,
		"total_output_tokens": totalOutput,
		"total_tokens":        totalTokens,
		"entries":             entries,
	}
}

// nullableString maps Go's empty-string "unset" convention onto the wire
// contract's JSON null.
func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func nullableFloat(f *float64) any {
	if f == nil {
		return nil
	}
	return *f
}

// ---------------------------------------------------------------------------
// Per-execution current tracker (context-scoped).
//
// A single CostTracker per agent would cross-contaminate concurrent
// executions and leak across runs. Instead each handler entrypoint binds a
// fresh tracker into the request context; nested LLM/harness calls made with
// that context record into it, and the transport layer reads it back after
// the reasoner completes. Concurrent executions each carry their own context,
// so they never share a tracker. This is the Go analogue of the Python SDK's
// contextvars-based current tracker.
// ---------------------------------------------------------------------------

type costTrackerContextKey struct{}

// contextWithCostTracker binds tracker as the current-execution tracker.
func contextWithCostTracker(ctx context.Context, tracker *CostTracker) context.Context {
	return context.WithValue(ctx, costTrackerContextKey{}, tracker)
}

func costTrackerFromContext(ctx context.Context) *CostTracker {
	if ctx == nil {
		return nil
	}
	if t, ok := ctx.Value(costTrackerContextKey{}).(*CostTracker); ok {
		return t
	}
	return nil
}

// CostTrackerFrom returns the CostTracker bound to the current execution, or
// nil when the context carries none. Reasoner code may use it to record
// custom usage entries alongside the SDK's automatic LLM/harness capture.
func CostTrackerFrom(ctx context.Context) *CostTracker {
	return costTrackerFromContext(ctx)
}

// usageSummaryOrNone returns the transport "usage" object, or nil when there
// is nothing to report. Omitting the key entirely when there are no entries
// is part of the usage contract, so callers must skip attaching usage on nil.
func usageSummaryOrNone(tracker *CostTracker) map[string]any {
	if tracker == nil || !tracker.HasEntries() {
		return nil
	}
	return tracker.Serialize()
}
