package agent

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/ai"
	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// wrapSyncResultWithUsage attaches the execution's usage summary to a
// synchronous 200 body.
//
// The control plane stores the whole sync body as the result and pulls usage
// back out by stripping the reserved UsageEnvelopeKey (see the control
// plane's extractUsageFromResult). The key is namespaced so a user result
// that legitimately contains its own "usage" key is never touched. Usage is
// merged as a sibling key into the result *object* — NOT wrapped in a
// {"result": ...} envelope, which would double-nest the stored result.
//
// Only results that marshal to a JSON object can carry usage this way;
// non-object results (arrays, scalars, null) are returned unchanged and their
// usage flows via the async status-callback path instead (the production
// path). No-usage results are also returned unchanged (backward compatible).
func wrapSyncResultWithUsage(result any, tracker *CostTracker) any {
	usage := usageSummaryOrNone(tracker)
	if usage == nil {
		return result
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return result
	}
	var obj map[string]any
	if err := json.Unmarshal(encoded, &obj); err != nil || obj == nil {
		// Non-object result: cannot merge a top-level usage key without
		// changing the result's type/shape.
		return result
	}
	obj[UsageEnvelopeKey] = usage
	return obj
}

// attachUsageToPayload merges the execution's usage summary into an async
// status-callback payload under the "usage" key. When there is nothing to
// report the payload is left untouched — omitting the key entirely is part of
// the usage contract.
func attachUsageToPayload(payload map[string]any, tracker *CostTracker) {
	if usage := usageSummaryOrNone(tracker); usage != nil {
		payload["usage"] = usage
	}
}

// recordAIUsage records one LLM response's token/cost usage into the current
// execution's cost tracker. No-op when the context carries no tracker or the
// response has no usage data.
func (a *Agent) recordAIUsage(ctx context.Context, resp *ai.Response) {
	if resp == nil {
		return
	}
	a.recordLLMUsage(ctx, resp.Model, resp.Usage)
}

// recordLLMUsage records a single LLM call's usage. Token recording is
// decoupled from pricing: token counts are recorded whenever available, with
// a nil cost when the price is unknown. cost_source is "provider" only when
// the LLM provider returned a native cost (e.g. OpenRouter usage accounting).
func (a *Agent) recordLLMUsage(ctx context.Context, model string, usage *ai.Usage) {
	if usage == nil {
		return
	}
	tracker := costTrackerFromContext(ctx)
	if tracker == nil {
		return
	}
	if model == "" && a.aiClient != nil {
		model = a.aiClient.Model()
	}
	if model == "" {
		model = "unknown"
	}
	var cost *float64
	costSource := ""
	if usage.Cost != nil {
		c := *usage.Cost
		cost = &c
		costSource = "provider"
	}
	tracker.Record(CostEntry{
		Model:               model,
		InputTokens:         usage.PromptTokens,
		OutputTokens:        usage.CompletionTokens,
		TotalTokens:         usage.TotalTokens,
		CostUSD:             cost,
		Reasoner:            executionContextFrom(ctx).ReasonerName,
		Source:              "llm",
		CacheReadTokens:     usage.CacheReadTokens(),
		CacheCreationTokens: usage.CacheCreationTokens(),
		CostSource:          costSource,
	})
}

// recordToolLoopUsage records the usage of every LLM call a tool-call loop
// made. The trace carries per-turn usage (intermediate tool-calling turns
// included); when it has none — e.g. a provider that reports no usage — the
// final response's usage is recorded instead so nothing is double-counted.
func (a *Agent) recordToolLoopUsage(ctx context.Context, resp *ai.Response, trace *ai.ToolCallTrace) {
	if trace != nil && len(trace.Usage) > 0 {
		for _, turn := range trace.Usage {
			a.recordLLMUsage(ctx, turn.Model, turn.Usage)
		}
		return
	}
	a.recordAIUsage(ctx, resp)
}

// recordHarnessUsage records a harness run's token/cost usage into the
// current execution's cost tracker, mirroring the Python SDK's
// _record_harness_usage. No-op when the harness reported neither tokens nor
// cost (the common case for providers that don't expose usage) so empty
// entries are never emitted. Cost is threaded even when tokens are unknown,
// and vice versa.
func (a *Agent) recordHarnessUsage(ctx context.Context, result *harness.Result, opts harness.Options) {
	if result == nil {
		return
	}
	noTokens := result.InputTokens == 0 && result.OutputTokens == 0 &&
		result.CacheReadTokens == 0 && result.CacheCreationTokens == 0
	if noTokens && result.CostUSD == nil {
		return
	}
	tracker := costTrackerFromContext(ctx)
	if tracker == nil {
		return
	}

	provider := opts.Provider
	if provider == "" && a.cfg.HarnessConfig != nil {
		provider = a.cfg.HarnessConfig.Provider
	}
	harnessName := strings.ReplaceAll(provider, "-", "_")

	model := opts.Model
	if model == "" && a.cfg.HarnessConfig != nil {
		model = a.cfg.HarnessConfig.Model
	}
	// Attribute usage to the BASE model: a "#variant" reasoning-effort
	// suffix is a harness dispatch detail, not part of the model id (the
	// Python SDK likewise records the provider-reported base model).
	model, _ = harness.SplitModelVariant(model)
	if model == "" {
		model = provider
	}
	if model == "" {
		model = "harness"
	}

	total := result.TotalTokens
	if total == 0 {
		total = result.InputTokens + result.OutputTokens
	}

	var cost *float64
	costSource := ""
	if result.CostUSD != nil {
		c := *result.CostUSD
		cost = &c
		costSource = "provider"
	}

	tracker.Record(CostEntry{
		Model:               model,
		InputTokens:         result.InputTokens,
		OutputTokens:        result.OutputTokens,
		TotalTokens:         total,
		CostUSD:             cost,
		Reasoner:            executionContextFrom(ctx).ReasonerName,
		Source:              "harness",
		Harness:             harnessName,
		CacheReadTokens:     result.CacheReadTokens,
		CacheCreationTokens: result.CacheCreationTokens,
		CostSource:          costSource,
	})
}
