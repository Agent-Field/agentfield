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
});
