/**
 * OpenRouter usage-accounting opt-in.
 *
 * OpenRouter only includes its native `cost` figure in a response's `usage`
 * object when the request body carries `usage: { include: true }`
 * (https://openrouter.ai/docs/use-cases/usage-accounting). The SDK reaches
 * OpenRouter through `@ai-sdk/openai` (it is OpenAI-wire-compatible), whose
 * provider options schema strips unknown keys — so the flag cannot be threaded
 * via `providerOptions`. Instead OpenRouter-bound models are built with this
 * fetch wrapper, which injects the flag into outgoing JSON chat-completion
 * bodies. The provider preserves the raw response usage (including `cost`) at
 * `result.usage.raw`, where the usage tracker picks it up.
 */

type FetchLike = (input: any, init?: any) => Promise<any>;

/**
 * Wrap a fetch implementation so JSON request bodies gain
 * `usage: { include: true }`. A body that already has a `usage` member, is not
 * a JSON object, or fails to parse passes through untouched — the wrapper must
 * never break the underlying request.
 */
export function withOpenRouterUsageInclude(baseFetch?: FetchLike): FetchLike {
  return (input: any, init?: any) => {
    // Resolve the underlying fetch lazily so test-time global fetch mocks
    // installed after model construction are still honored.
    const impl: FetchLike = baseFetch ?? ((...args) => globalThis.fetch(...(args as [any, any?])));
    try {
      if (init && typeof init.body === 'string') {
        const parsed: unknown = JSON.parse(init.body);
        if (
          typeof parsed === 'object' &&
          parsed !== null &&
          !Array.isArray(parsed) &&
          (parsed as Record<string, unknown>).usage === undefined
        ) {
          (parsed as Record<string, unknown>).usage = { include: true };
          return impl(input, { ...init, body: JSON.stringify(parsed) });
        }
      }
    } catch {
      // Malformed/non-JSON body — fall through to the untouched request.
    }
    return impl(input, init);
  };
}
