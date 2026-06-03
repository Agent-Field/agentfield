export function formatContext(context?: string[]) {
  if (!context || context.length === 0) return 'No additional context provided.';
  return context.map((c) => `- ${c}`).join('\n');
}

import type { ZodSchema } from 'zod';

function extractJsonCandidate(raw: string): string | undefined {
  const trimmed = raw.trim();
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) return fenceMatch[1].trim();

  // Try to grab from the first { or [ to the last matching brace/bracket.
  const firstBrace = trimmed.indexOf('{');
  const firstBracket = trimmed.indexOf('[');
  let start = -1;
  if (firstBrace !== -1 && firstBracket !== -1) {
    start = Math.min(firstBrace, firstBracket);
  } else {
    start = Math.max(firstBrace, firstBracket);
  }
  if (start === -1) return undefined;
  const candidate = trimmed.slice(start);
  return candidate;
}

export function parseWithSchema<T>(
  value: unknown,
  schema: ZodSchema<T>,
  label: string,
  fallback?: () => T,
  strict = false
): T | undefined {
  let parsedInput = value;
  if (typeof value === 'string') {
    const candidate = extractJsonCandidate(value) ?? value;
    try {
      parsedInput = JSON.parse(candidate);
    } catch (err) {
      console.warn(`⚠️ ${label} JSON parse failed`, err);
      parsedInput = candidate;
    }
  }

  const parsed = schema.safeParse(parsedInput);
  if (parsed.success) return parsed.data;

  console.warn(`⚠️ ${label} schema mismatch`, parsed.error?.errors ?? parsed.error);
  if (strict) {
    throw new Error(`${label} schema validation failed`);
  }
  if (fallback) return fallback();
  return undefined;
}

export function flattenAttributes(obj: Record<string, any>, prefix = ''): Record<string, any> {
  return Object.entries(obj).reduce<Record<string, any>>((acc, [key, val]) => {
    const fullKey = prefix ? `${prefix}_${key}` : key;
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      Object.assign(acc, flattenAttributes(val as Record<string, any>, fullKey));
    } else {
      acc[fullKey] = val;
    }
    return acc;
  }, {});
}
