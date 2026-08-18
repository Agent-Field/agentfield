import { createRequire } from 'node:module';

type JsonSchemaRecord = Record<string, unknown>;
type JsonSchemaFactory = (schema: unknown) => JsonSchemaRecord;

type ZodModule = {
  toJSONSchema?: JsonSchemaFactory;
};

let zod4Converter: JsonSchemaFactory | null | undefined;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isZod4Schema(value: unknown): boolean {
  if (!isRecord(value) || !('_zod' in value) || !isRecord(value._zod)) {
    return false;
  }
  return isRecord(value._zod.def);
}

function getZod4Converter(): JsonSchemaFactory | null {
  if (zod4Converter !== undefined) {
    return zod4Converter;
  }

  const require = createRequire(import.meta.url);
  for (const moduleName of ['zod', 'zod/v4']) {
    try {
      const mod = require(moduleName) as ZodModule;
      if (typeof mod.toJSONSchema === 'function') {
        zod4Converter = mod.toJSONSchema;
        return zod4Converter;
      }
    } catch {
      // Try the next Zod 4 entry point.
    }
  }

  zod4Converter = null;
  return zod4Converter;
}

export function zod4ToJsonSchema(schema: unknown): JsonSchemaRecord {
  const converter = getZod4Converter();
  if (converter === null) {
    throw new TypeError(
      'Cannot convert a Zod 4 schema: install the "zod" package with its "zod/v4" entry point available.',
    );
  }
  return converter(schema);
}
