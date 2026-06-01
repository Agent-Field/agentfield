import { describe, it, expect } from 'vitest';
import { z } from 'zod';

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

  it('converts zod objects with nested fields', () => {
    const schema = z.object({
      name: z.string(),
      age: z.number().optional(),
      tags: z.array(z.string()),
    });
    const result = toJsonSchema(schema);
    expect(result).toMatchObject({
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'number' },
        tags: {
          type: 'array',
          items: { type: 'string' },
        },
      },
    });
    expect(result).not.toHaveProperty('$schema');
  });

  it('converts zod enums', () => {
    const schema = z.enum(['a', 'b', 'c']);
    const result = toJsonSchema(schema);
    expect(result).toMatchObject({
      type: 'string',
      enum: ['a', 'b', 'c'],
    });
  });

  it('handles non-object non-zod values', () => {
    expect(toJsonSchema(42)).toEqual({});
    expect(toJsonSchema('string')).toEqual({});
    expect(toJsonSchema(true)).toEqual({});
  });

  it('preserves $schema on plain objects', () => {
    const schema = { type: 'string', $schema: 'http://json-schema.org/draft-07/schema#' };
    expect(toJsonSchema(schema)).toEqual(schema);
  });

  it('converts zod objects with default values', () => {
    const schema = z.object({
      name: z.string().default('hello'),
      count: z.number().default(0),
    });
    const result = toJsonSchema(schema);
    expect(result).toMatchObject({
      type: 'object',
      properties: {
        name: { type: 'string', default: 'hello' },
        count: { type: 'number', default: 0 },
      },
    });
  });
});
