/**
 * Separator for the `provider/model#variant` model-string syntax. `#` is
 * safe because provider model ids never contain it (`:` is taken by
 * OpenRouter suffixes like `:free`, `@` by Vertex-style ids).
 */
export const MODEL_VARIANT_SEP = '#';

export interface ModelVariant {
  model?: string;
  variant?: string;
}

/**
 * Split a `provider/model#variant` string into `{ model, variant }`.
 *
 * The `#variant` suffix carries a provider-specific reasoning-effort
 * variant (e.g. `high`, `minimal`) through config surfaces that only
 * hold a model string. A bare model string passes through as
 * `{ model }`; non-string or empty input yields `{}`.
 */
export function splitModelVariant(model: unknown): ModelVariant {
  if (typeof model !== 'string' || !model.trim()) {
    return {};
  }
  const sepIndex = model.indexOf(MODEL_VARIANT_SEP);
  const base = sepIndex === -1 ? model : model.slice(0, sepIndex);
  const variant = sepIndex === -1 ? '' : model.slice(sepIndex + 1);
  return {
    model: base.trim() || undefined,
    variant: variant.trim() || undefined,
  };
}

/**
 * Resolve `{ model, variant }` from harness options.
 *
 * An explicit `options.variant` wins over a `#variant` suffix on
 * `options.model`; the returned model never carries the suffix.
 */
export function resolveModelAndVariant(options: Record<string, unknown>): ModelVariant {
  const { model, variant } = splitModelVariant(options.model);
  const explicit = options.variant;
  if (typeof explicit === 'string' && explicit.trim()) {
    return { model, variant: explicit.trim() };
  }
  return { model, variant };
}
