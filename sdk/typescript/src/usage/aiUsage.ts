/**
 * Bridges Vercel AI SDK v6 results into the per-execution {@link CostTracker}.
 *
 * The AI SDK reports `LanguageModelUsage` objects
 * (`{ inputTokens, inputTokenDetails: { cacheReadTokens, cacheWriteTokens },
 * outputTokens, totalTokens, raw }`). `raw` carries the provider's untouched
 * usage payload — for OpenRouter (which is OpenAI-wire-compatible) that is
 * where the native `cost` figure appears when usage accounting is requested.
 * Note `totalUsage` (summed across steps) drops `raw`, so cost is looked up
 * per step.
 */

import { ExecutionContext } from '../context/ExecutionContext.js';
import type { CostTracker } from './costTracker.js';

interface TokenCounts {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
}

/** The slice of an AI SDK generate/stream result that usage capture reads. */
export interface AiSdkUsageSource {
  usage?: unknown;
  totalUsage?: unknown;
  steps?: Array<{ usage?: unknown }>;
  providerMetadata?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function num(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Map an AI SDK `LanguageModelUsage` object to contract token counts. */
export function extractTokenUsage(usage: unknown): TokenCounts {
  if (!isRecord(usage)) {
    return { inputTokens: 0, outputTokens: 0, totalTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };
  }
  const inputDetails = isRecord(usage.inputTokenDetails) ? usage.inputTokenDetails : undefined;
  return {
    inputTokens: num(usage.inputTokens),
    outputTokens: num(usage.outputTokens),
    totalTokens: num(usage.totalTokens),
    // `cachedInputTokens` is the deprecated pre-details alias — keep it as a
    // fallback so older provider adapters still report cache reads.
    cacheReadTokens: num(inputDetails?.cacheReadTokens) || num(usage.cachedInputTokens),
    cacheCreationTokens: num(inputDetails?.cacheWriteTokens)
  };
}

/** Read a provider-native cost figure from a usage object's raw payload. */
function costFromUsageRaw(usage: unknown): number | null {
  if (!isRecord(usage) || !isRecord(usage.raw)) return null;
  const cost = usage.raw.cost;
  return typeof cost === 'number' && Number.isFinite(cost) && cost >= 0 ? cost : null;
}

/** Defensive: a dedicated OpenRouter provider surfaces cost in providerMetadata. */
function costFromProviderMetadata(providerMetadata: unknown): number | null {
  if (!isRecord(providerMetadata)) return null;
  const openrouter = providerMetadata.openrouter;
  if (!isRecord(openrouter) || !isRecord(openrouter.usage)) return null;
  const cost = openrouter.usage.cost;
  return typeof cost === 'number' && Number.isFinite(cost) && cost >= 0 ? cost : null;
}

/**
 * Best-effort provider-native cost for a whole generate result.
 *
 * Multi-step results (tool loops) sum per-step raw costs; single-step results
 * read the final usage's raw payload. Null when the provider reported nothing.
 */
export function extractProviderCostUsd(source: AiSdkUsageSource): number | null {
  if (Array.isArray(source.steps) && source.steps.length > 0) {
    let sum = 0;
    let any = false;
    for (const step of source.steps) {
      const cost = costFromUsageRaw(isRecord(step) ? step.usage : undefined);
      if (cost !== null) {
        sum += cost;
        any = true;
      }
    }
    if (any) return sum;
  }
  return (
    costFromUsageRaw(source.totalUsage) ??
    costFromUsageRaw(source.usage) ??
    costFromProviderMetadata(source.providerMetadata)
  );
}

/**
 * Record an AI SDK call's usage into the current execution's cost tracker.
 *
 * No-op when there is no bound tracker (call made outside an execution), or
 * when the result carries neither token counts nor a cost figure (e.g. mocked
 * results without a usage object) so empty entries are never emitted. Never
 * throws — usage capture must never fail the call it observes.
 */
export function recordAiSdkUsage(params: {
  source: AiSdkUsageSource;
  model: string;
  provider?: string | null;
  tracker?: CostTracker;
  reasonerName?: string | null;
}): void {
  try {
    const current = ExecutionContext.getCurrent();
    const tracker = params.tracker ?? current?.costTracker;
    if (!tracker) return;

    const tokens = extractTokenUsage(params.source.totalUsage ?? params.source.usage);
    const cost = extractProviderCostUsd(params.source);
    const hasTokens =
      tokens.inputTokens > 0 ||
      tokens.outputTokens > 0 ||
      tokens.totalTokens > 0 ||
      tokens.cacheReadTokens > 0 ||
      tokens.cacheCreationTokens > 0;
    if (!hasTokens && cost === null) return;

    tracker.record({
      model: params.model,
      ...tokens,
      costUsd: cost,
      costSource: cost !== null ? 'provider' : null,
      reasonerName:
        params.reasonerName !== undefined
          ? params.reasonerName
          : current?.metadata.reasonerId ?? null,
      source: 'llm',
      provider: params.provider ?? undefined
    });
  } catch {
    // Best effort — usage capture must never break the AI call itself.
  }
}
