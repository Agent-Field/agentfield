import { describe, it, expect, expectTypeOf } from 'vitest';
import { z } from 'zod';
import { z as z3 } from 'zod/v3';
import * as z4 from 'zod/v4';

import type { AIClient, AIRequestOptions, ReasonerContext, ZodSchema } from '../src/index.js';
import { toJsonSchema } from '../src/utils/schema.js';

describe('toJsonSchema', () => {
  it('converts zod objects to json schema', () => {
    expect(toJsonSchema(z.object({ name: z.string() }))).toMatchObject({
      type: 'object',
      properties: {
        name: {
          type: 'string',
        },
      },
    });
  });

  it('omits the $schema key from zod conversions', () => {
    const result = toJsonSchema(z.object({ name: z.string() }));

    expect(result).not.toHaveProperty('$schema');
  });

  it('converts zod 4 objects with the native converter', () => {
    expect(toJsonSchema(z4.object({ name: z4.string(), count: z4.number().optional() }))).toEqual({
      type: 'object',
      properties: {
        name: { type: 'string' },
        count: { type: 'number' },
      },
      required: ['name'],
      additionalProperties: false,
    });
  });

  it('converts zod 3 objects with the legacy converter', () => {
    expect(toJsonSchema(z3.object({ name: z3.string(), count: z3.number().optional() }))).toEqual({
      type: 'object',
      properties: {
        name: { type: 'string' },
        count: { type: 'number' },
      },
      required: ['name'],
      additionalProperties: false,
    });
  });

  it('preserves zod 4 output inference through the public API', () => {
    const schema = z.object({ a: z.string() });
    const generate = (ai: AIClient) => ai.generate('prompt', { schema });
    const contextAi = (context: ReasonerContext) => context.ai('prompt', { schema });

    expectTypeOf(schema).toExtend<ZodSchema<{ a: string }>>();
    expectTypeOf({ schema }).toExtend<AIRequestOptions>();
    expectTypeOf(generate).returns.toEqualTypeOf<Promise<{ a: string }>>();
    expectTypeOf(contextAi).returns.toEqualTypeOf<Promise<{ a: string }>>();
  });

  it('keeps zod 3 schemas assignable through the public API', () => {
    const schema = z3.object({ a: z3.string() });
    const generate = (ai: AIClient) => ai.generate('prompt', { schema });

    expectTypeOf(schema).toExtend<ZodSchema<{ a: string }>>();
    expectTypeOf(generate).returns.toEqualTypeOf<Promise<{ a: string }>>();
  });

  it('returns plain json schema objects unchanged', () => {
    const schema = { type: 'string' };

    expect(toJsonSchema(schema)).toEqual(schema);
  });

  it('returns an empty object for null input', () => {
    expect(toJsonSchema(null)).toEqual({});
  });

  it('returns an empty object for undefined input', () => {
    expect(toJsonSchema(undefined)).toEqual({});
  });
});
