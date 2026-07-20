package handlers

import (
	"context"
	"encoding/json"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// maxUsageEntriesPerExecution caps how many usage rows are persisted for a
// single execution. The "usage" object is untrusted agent-supplied input, so
// we bound it defensively.
const maxUsageEntriesPerExecution = 500

// usageEnvelopeKey is the reserved key under which SDKs attach the serialized
// usage summary to a synchronous 200 result body. It is namespaced so it can
// never collide with user data: a plain "usage" key in an agent's own result
// dict is user payload and must pass through untouched. "__agentfield_"-
// prefixed keys are reserved for SDK↔control-plane transport.
const usageEnvelopeKey = "__agentfield_usage__"

// usageWriter is the narrow storage capability the ingest path needs. The
// concrete *storage.LocalStorage satisfies it; test doubles that don't
// implement it simply skip usage persistence.
type usageWriter interface {
	CreateExecutionUsage(ctx context.Context, rows []*types.ExecutionUsage) error
}

// ingestUsage parses an untrusted "usage" object from an agent result envelope
// and persists per-entry rows tied to the execution's identifiers. It never
// returns an error: usage ingestion is best-effort and must not fail the
// execution. Failures are logged and swallowed.
func (c *executionController) ingestUsage(ctx context.Context, exec *types.Execution, usageRaw map[string]interface{}) {
	if exec == nil || len(usageRaw) == 0 {
		return
	}

	rows := parseUsageEntries(exec, usageRaw)
	if len(rows) == 0 {
		return
	}

	writer, ok := c.store.(usageWriter)
	if !ok {
		return
	}

	if err := writer.CreateExecutionUsage(ctx, rows); err != nil {
		logger.Logger.Warn().
			Err(err).
			Str("execution_id", exec.ExecutionID).
			Int("entries", len(rows)).
			Msg("failed to persist execution usage; continuing")
	}
}

// parseUsageEntries converts the untrusted usage object into storage rows,
// clamping negative token counts to zero and capping the number of entries.
func parseUsageEntries(exec *types.Execution, usageRaw map[string]interface{}) []*types.ExecutionUsage {
	entriesRaw, ok := usageRaw["entries"].([]interface{})
	if !ok || len(entriesRaw) == 0 {
		return nil
	}

	rows := make([]*types.ExecutionUsage, 0, len(entriesRaw))
	for _, e := range entriesRaw {
		if len(rows) >= maxUsageEntriesPerExecution {
			break
		}
		entry, ok := e.(map[string]interface{})
		if !ok {
			continue
		}

		input := clampNonNegative(usageNumber(entry["input_tokens"]))
		output := clampNonNegative(usageNumber(entry["output_tokens"]))
		cacheRead := clampNonNegative(usageNumber(entry["cache_read_tokens"]))
		cacheCreation := clampNonNegative(usageNumber(entry["cache_creation_tokens"]))
		total := clampNonNegative(usageNumber(entry["total_tokens"]))
		if total == 0 {
			total = input + output
		}

		row := &types.ExecutionUsage{
			ExecutionID:         exec.ExecutionID,
			WorkflowID:          exec.RunID,
			AgentNodeID:         exec.AgentNodeID,
			Reasoner:            usageString(entry["reasoner"]),
			Source:              usageString(entry["source"]),
			Provider:            usageString(entry["provider"]),
			Model:               usageString(entry["model"]),
			Harness:             usageString(entry["harness"]),
			InputTokens:         input,
			OutputTokens:        output,
			CacheReadTokens:     cacheRead,
			CacheCreationTokens: cacheCreation,
			TotalTokens:         total,
			CostUSD:             usageCost(entry["cost_usd"]),
			CostSource:          usageString(entry["cost_source"]),
		}
		if row.Reasoner == "" {
			row.Reasoner = exec.ReasonerID
		}
		rows = append(rows, row)
	}
	return rows
}

// extractUsageFromResult decodes an agent result body and returns the reserved
// top-level usage envelope object (if present) along with a copy of the result
// with that key stripped so it does not leak into the stored/returned payload.
// Only the namespaced usageEnvelopeKey is recognized — a user result carrying
// its own "usage" key is user data and passes through untouched. When the body
// is not a JSON object or carries no envelope key, the original bytes are
// returned unchanged and usage is nil. As a defensive measure it also checks a
// top-level "result" object for a nested envelope key.
func extractUsageFromResult(result []byte) (usageRaw map[string]interface{}, stripped []byte) {
	if len(result) == 0 {
		return nil, result
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(result, &decoded); err != nil {
		// Not a JSON object (array, scalar, or malformed) — nothing to strip.
		return nil, result
	}

	if u, ok := decoded[usageEnvelopeKey].(map[string]interface{}); ok {
		delete(decoded, usageEnvelopeKey)
		return u, remarshalOrOriginal(decoded, result)
	}

	// Defensive: the SDK could nest the envelope inside a "result" wrapper.
	if inner, ok := decoded["result"].(map[string]interface{}); ok {
		if u, ok := inner[usageEnvelopeKey].(map[string]interface{}); ok {
			delete(inner, usageEnvelopeKey)
			decoded["result"] = inner
			return u, remarshalOrOriginal(decoded, result)
		}
	}

	return nil, result
}

func remarshalOrOriginal(decoded map[string]interface{}, original []byte) []byte {
	out, err := json.Marshal(decoded)
	if err != nil {
		return original
	}
	return out
}

// usageNumber coerces a JSON-decoded value to int64, tolerating float64,
// json.Number, and numeric strings. Non-numeric values yield 0.
func usageNumber(v interface{}) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i
		}
		if f, err := n.Float64(); err == nil {
			return int64(f)
		}
	}
	return 0
}

// usageCost coerces a JSON-decoded value to a *float64. Missing/null/non-numeric
// values yield nil. Negative costs are clamped to zero.
func usageCost(v interface{}) *float64 {
	var f float64
	switch n := v.(type) {
	case float64:
		f = n
	case int64:
		f = float64(n)
	case int:
		f = float64(n)
	case json.Number:
		parsed, err := n.Float64()
		if err != nil {
			return nil
		}
		f = parsed
	default:
		return nil
	}
	if f < 0 {
		f = 0
	}
	return &f
}

func usageString(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func clampNonNegative(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}
