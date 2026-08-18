import { createRequire } from 'node:module';
import path from 'node:path';

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

/**
 * Resolve zod 4's native `toJSONSchema`.
 *
 * Order matters: the application's own zod copy (resolved from the working
 * directory) is tried first because it is the copy that built the schema
 * instance. zod 4 keeps `.describe()` metadata in a per-copy registry, so a
 * *different* copy (e.g. the SDK's nested zod 3.25 exposing `zod/v4`) still
 * converts the structure but silently drops descriptions and some
 * refinements (`.int()` → `number`). The SDK-relative copies are the fallback.
 */
function getZod4Converter(): JsonSchemaFactory | null {
  if (zod4Converter !== undefined) {
    return zod4Converter;
  }

  const requires: NodeJS.Require[] = [];
  try {
    requires.push(createRequire(path.join(process.cwd(), 'package.json')));
  } catch {
    // No resolvable application root; fall through to the SDK's own tree.
  }
  requires.push(createRequire(import.meta.url));

  for (const req of requires) {
    for (const moduleName of ['zod', 'zod/v4']) {
      try {
        const mod = req(moduleName) as ZodModule;
        if (typeof mod.toJSONSchema === 'function') {
          zod4Converter = mod.toJSONSchema;
          return zod4Converter;
        }
      } catch {
        // Try the next Zod 4 entry point.
      }
    }
  }

  zod4Converter = null;
  return zod4Converter;
}

/** Test hook: forget the cached converter so resolution runs again. */
export function resetZod4ConverterCache(): void {
  zod4Converter = undefined;
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
