import { z, type ZodTypeAny } from "zod";
import type { JsonSchema } from "@/types/execution";

function toArray<T>(value: T | T[] | undefined): T[] {
  if (Array.isArray(value)) {
    return value;
  }
  if (value === undefined) {
    return [];
  }
  return [value];
}

function buildField(schema?: JsonSchema): ZodTypeAny {
  if (!schema) {
    return z.any();
  }

  const types = toArray(schema.type);
  const nullable = types.includes("null");

  if (schema.enum && schema.enum.length) {
    const enumValues = schema.enum.map((value) => String(value)) as [string, ...string[]];
    const baseEnum = z.enum(enumValues);
    return nullable ? baseEnum.nullable() : baseEnum;
  }

  if (types.includes("number") || types.includes("integer")) {
    const numberSchema = z.number();
    return nullable ? numberSchema.nullable() : numberSchema;
  }

  if (types.includes("boolean")) {
    const boolSchema = z.boolean();
    return nullable ? boolSchema.nullable() : boolSchema;
  }

  if (types.includes("array")) {
    const itemSchema = buildField(Array.isArray(schema.items) ? schema.items[0] : schema.items);
    const arraySchema = z.array(itemSchema);
    return nullable ? arraySchema.nullable() : arraySchema;
  }

  if (types.includes("object") && schema.properties) {
    const objectSchema = jsonSchemaToZodObject(schema);
    return nullable ? objectSchema.nullable() : objectSchema;
  }

  const stringSchema = z.string();
  return nullable ? stringSchema.nullable() : stringSchema;
}

export function jsonSchemaToZodObject(schema: JsonSchema): z.ZodObject<any> {
  const properties = schema.properties ?? {};
  const required = new Set(schema.required ?? []);

  const shape: Record<string, ZodTypeAny> = {};

  for (const [key, propertySchema] of Object.entries(properties)) {
    const fieldSchema = buildField(propertySchema);
    shape[key] = required.has(key) ? fieldSchema : fieldSchema.optional();
  }

  const baseObject = z.object(shape).passthrough();

  if (schema.type && !toArray(schema.type).includes("object")) {
    return z.object({ value: buildField(schema) }).passthrough();
  }

  return baseObject;
}
