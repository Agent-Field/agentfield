/**
 * Per-execution LLM cost and token-usage tracking.
 *
 * Mirrors the Python SDK's `agentfield/cost_tracker.py`: a `CostTracker`
 * accumulates one `CostEntry` per LLM (or coding-agent harness) call made
 * during a single reasoner execution, and `serialize()` emits the
 * cross-language wire contract consumed by the control plane's usage-ingest
 * path (see the Go `parseUsageEntries`).
 */

/**
 * Reserved envelope key used to attach the serialized usage summary to a
 * synchronous 200 result body. Namespaced so it cannot collide with user data:
 * a plain "usage" key in an agent's own result object is user payload and must
 * never be touched. The control plane strips exactly this key back out (see
 * the Go `extractUsageFromResult`). `__agentfield_`-prefixed keys are reserved
 * for SDK<->control-plane transport.
 */
export const USAGE_ENVELOPE_KEY = '__agentfield_usage__';

/** A single LLM (or harness) call usage record. */
export interface CostEntry {
  model: string;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  /**
   * Cost may be unknown (provider gave no figure) — tokens are recorded
   * regardless, so cost is nullable and never gates them.
   */
  costUsd: number | null;
  reasonerName: string | null;
  /** "llm" for direct model calls, "harness" for coding-agent runs. */
  source: 'llm' | 'harness';
  /** e.g. "anthropic", "openrouter" — derived from the model slug if unset. */
  provider: string | null;
  /** e.g. "claude_code" for harness-originated entries; null for plain LLM. */
  harness: string | null;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  /** Where costUsd came from: "provider" | null. */
  costSource: string | null;
}

/** Input accepted by {@link CostTracker.record}. */
export interface CostEntryInit {
  model: string;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  costUsd?: number | null;
  reasonerName?: string | null;
  source?: 'llm' | 'harness';
  provider?: string | null;
  harness?: string | null;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
  costSource?: string | null;
}

/** Wire form of a single usage entry (snake_case cross-language contract). */
export interface UsageEntryWire {
  source: string;
  provider: string | null;
  model: string;
  harness: string | null;
  reasoner: string | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  cost_source: string | null;
}

/** Wire form of the usage summary attached to execution envelopes. */
export interface UsageSummaryWire {
  /** Null when no entry had a known cost; otherwise sum rounded to 6 decimals. */
  total_cost_usd: number | null;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  entries: UsageEntryWire[];
}

/**
 * Best-effort provider name from a model slug.
 *
 * `anthropic/claude-opus-4-8` -> `anthropic`,
 * `openrouter/anthropic/claude` -> `openrouter`,
 * a bare `gpt-4o` (no provider prefix) -> `null`.
 */
export function deriveProvider(model: string | null | undefined): string | null {
  if (!model) return null;
  const slug = String(model).trim();
  const slash = slug.indexOf('/');
  if (slash < 0) return null;
  return slug.slice(0, slash).toLowerCase() || null;
}

/** Coerce an optional count to a non-negative-ish integer (best effort). */
function toCount(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
  return Math.trunc(value);
}

function toCost(value: number | null | undefined): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/** Accumulates LLM/harness usage for a single execution run. */
export class CostTracker {
  private entries: CostEntry[] = [];

  /**
   * Record a single call's usage.
   *
   * Cost is optional: a call with known token counts but unknown price is
   * still recorded (`costUsd: null`) so tokens are never discarded.
   */
  record(init: CostEntryInit): void {
    this.entries.push({
      model: init.model,
      inputTokens: toCount(init.inputTokens),
      outputTokens: toCount(init.outputTokens),
      totalTokens: toCount(init.totalTokens),
      costUsd: toCost(init.costUsd),
      reasonerName: init.reasonerName ?? null,
      source: init.source ?? 'llm',
      provider: init.provider ?? deriveProvider(init.model),
      harness: init.harness ?? null,
      cacheReadTokens: toCount(init.cacheReadTokens),
      cacheCreationTokens: toCount(init.cacheCreationTokens),
      costSource: init.costSource ?? null
    });
  }

  /** Total accumulated cost in USD (unknown costs count as zero). */
  get totalCostUsd(): number {
    return this.entries.reduce((sum, e) => sum + (e.costUsd ?? 0), 0);
  }

  /** Total tokens used across all calls (per-entry total, no fallback). */
  get totalTokens(): number {
    return this.entries.reduce((sum, e) => sum + e.totalTokens, 0);
  }

  /** Number of calls tracked. */
  get callCount(): number {
    return this.entries.length;
  }

  get hasEntries(): boolean {
    return this.entries.length > 0;
  }

  /**
   * Return the transport contract form attached to execution envelopes.
   *
   * Matches the Python SDK's `CostTracker.serialize()` byte-for-byte in shape:
   * unset string fields serialize as null, `total_cost_usd` is null when no
   * entry had a known cost, and a per-entry `total_tokens` of zero falls back
   * to input + output.
   */
  serialize(): UsageSummaryWire {
    const entries: UsageEntryWire[] = [];
    let totalInput = 0;
    let totalOutput = 0;
    let totalTokens = 0;
    let totalCost = 0;
    let anyCost = false;

    for (const e of this.entries) {
      const entryTotal = e.totalTokens || e.inputTokens + e.outputTokens;
      entries.push({
        source: e.source,
        provider: e.provider,
        model: e.model,
        harness: e.harness,
        reasoner: e.reasonerName,
        input_tokens: e.inputTokens,
        output_tokens: e.outputTokens,
        cache_read_tokens: e.cacheReadTokens,
        cache_creation_tokens: e.cacheCreationTokens,
        total_tokens: entryTotal,
        cost_usd: e.costUsd,
        cost_source: e.costSource
      });
      totalInput += e.inputTokens;
      totalOutput += e.outputTokens;
      totalTokens += entryTotal;
      if (e.costUsd !== null) {
        totalCost += e.costUsd;
        anyCost = true;
      }
    }

    return {
      total_cost_usd: anyCost ? round6(totalCost) : null,
      total_input_tokens: totalInput,
      total_output_tokens: totalOutput,
      total_tokens: totalTokens,
      entries
    };
  }

  /** Clear all tracked entries. */
  reset(): void {
    this.entries = [];
  }
}

/**
 * Return the transport `usage` object, or null when there is nothing to
 * report. Zero entries means no usage key at all per the contract, so callers
 * must skip attaching `usage` on null.
 */
export function usageSummaryOrNull(tracker: CostTracker | undefined | null): UsageSummaryWire | null {
  if (!tracker || !tracker.hasEntries) return null;
  return tracker.serialize();
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Attach usage to a synchronous 200 result body.
 *
 * The summary is merged as a sibling under the reserved
 * {@link USAGE_ENVELOPE_KEY}; the control plane strips exactly that key back
 * out, so a user result that legitimately contains its own "usage" key is
 * never touched. Only plain-object results can carry usage this way —
 * non-object results (arrays, scalars, class instances, null) are returned
 * unchanged and their usage flows via the async status-callback path instead
 * (the production path). No-usage trackers leave any result unchanged.
 */
export function attachUsageToSyncResult(result: unknown, tracker: CostTracker | undefined | null): unknown {
  const usage = usageSummaryOrNull(tracker);
  if (usage === null || !isPlainObject(result)) return result;
  return { ...result, [USAGE_ENVELOPE_KEY]: usage };
}
