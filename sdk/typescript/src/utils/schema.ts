import { zodToJsonSchema } from 'zod-to-json-schema';
import type { ZodType } from 'zod';

/**
 * Check if a value is a Zod schema by looking for Zod's internal structure.
 */
function isZodSchema(value: unknown): value is ZodType {
  if (!value || typeof value !== 'object') return false;
  const obj = value as Record<string, unknown>;
  // Zod schemas have _def with typeName, or ~standard with vendor: 'zod'
  return (
    ('_def' in obj && typeof obj._def === 'object') ||
    ('~standard' in obj && (obj['~standard'] as Record<string, unknown>)?.vendor === 'zod')
  );
}

/**
 * Convert a schema to JSON Schema format.
 * If the input is a Zod schema, converts it using zod-to-json-schema.
 * If the input is already a plain object (assumed to be JSON Schema), returns it as-is.
 * If the input is undefined/null, returns an empty object.
 */
export function toJsonSchema(schema: unknown): Record<string, unknown> {
  if (schema === undefined || schema === null) {
    return {};
  }

  if (isZodSchema(schema)) {
    // Convert Zod schema to JSON Schema
    // Use 'openApi3' target for better compatibility with tool calling
    const jsonSchema = zodToJsonSchema(schema, {
      target: 'openApi3',
      $refStrategy: 'none', // Inline all definitions instead of using $ref
    });

    // Remove the $schema property as it's not needed for tool calling
    if (typeof jsonSchema === 'object' && jsonSchema !== null) {
      const { $schema, ...rest } = jsonSchema as Record<string, unknown>;
      return rest;
    }
    return jsonSchema as Record<string, unknown>;
  }

  // Assume it's already a JSON Schema or plain object
  if (typeof schema === 'object') {
    return schema as Record<string, unknown>;
  }

  return {};
}
