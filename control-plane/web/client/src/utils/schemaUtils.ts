import type { JsonSchema, FormField } from "../types/execution";
import { ZodError } from "zod";
import { jsonSchemaToZodObject } from "./jsonSchemaToZod";

const TYPE_PRIORITY = ["object", "array", "string", "number", "integer", "boolean", "null"];

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null;
}

function toArray<T>(value: T | T[] | undefined): T[] {
  if (Array.isArray(value)) {
    return value;
  }
  if (value === undefined) {
    return [];
  }
  return [value];
}

function resolvePrimaryType(schema: JsonSchema | undefined): string | undefined {
  if (!schema) {
    return undefined;
  }

  const candidates = toArray(schema.type);
  if (candidates.length === 0) {
    if (schema.properties) {
      return "object";
    }
    if (schema.items) {
      return "array";
    }
    if (schema.enum || schema.const || schema.default !== undefined) {
      return "string";
    }
    return undefined;
  }

  for (const preferred of TYPE_PRIORITY) {
    if (candidates.includes(preferred)) {
      return preferred;
    }
  }

  return candidates[0];
}

function deepClone<T>(value: T): T {
  if (value === undefined) {
    return value;
  }

  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }

  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

function getDefaultValue(schema: JsonSchema): any {
  if (schema.default !== undefined) {
    return deepClone(schema.default);
  }

  const examples = schema.examples ?? toArray(schema.example);
  if (examples.length > 0) {
    return deepClone(examples[0]);
  }

  if (schema.const !== undefined) {
    return deepClone(schema.const);
  }

  if (schema.enum && schema.enum.length > 0) {
    return deepClone(schema.enum[0]);
  }

  return undefined;
}

function extractArrayMetadata(schema: JsonSchema) {
  const items = schema.items;
  const tupleSchemas = Array.isArray(items) ? items : undefined;
  const itemSchema = !Array.isArray(items) && isRecord(items) ? items : null;

  return {
    itemSchema,
    tupleSchemas,
    minItems: schema.minItems,
    maxItems: schema.maxItems,
  };
}

function extractCombinatorMetadata(schema: JsonSchema) {
  const combinators: Array<{
    type: "oneOf" | "anyOf" | "allOf";
    variants: JsonSchema[];
  }> = [];

  if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
    combinators.push({ type: "oneOf", variants: schema.oneOf });
  }
  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    combinators.push({ type: "anyOf", variants: schema.anyOf });
  }
  if (Array.isArray(schema.allOf) && schema.allOf.length > 0) {
    combinators.push({ type: "allOf", variants: schema.allOf });
  }

  return combinators[0] ?? null;
}

function buildField(
  fieldPath: string,
  key: string,
  schema: JsonSchema,
  required: boolean
): FormField {
  const primaryType = resolvePrimaryType(schema);
  const baseType = getFieldType(schema);

  const field: FormField = {
    name: fieldPath,
    label: formatFieldLabel(key),
    type: baseType,
    required,
    description: schema.description,
    placeholder: getPlaceholder(schema),
    options: schema.enum?.map((option) =>
      typeof option === "string" ? option : JSON.stringify(option)
    ),
    enumValues: schema.enum ? deepClone(schema.enum) : undefined,
    schema,
    defaultValue: getDefaultValue(schema),
    examples: schema.examples ?? (schema.example !== undefined ? [schema.example] : undefined),
    format: schema.format,
  };

  if (primaryType === "array") {
    const metadata = extractArrayMetadata(schema);
    field.itemSchema = metadata.itemSchema;
    field.tupleSchemas = metadata.tupleSchemas;
    field.minItems = metadata.minItems;
    field.maxItems = metadata.maxItems;
  }

  const combinator = extractCombinatorMetadata(schema);
  if (combinator) {
    field.combinator = combinator.type;
    field.variantSchemas = combinator.variants;
    field.variantTitles = combinator.variants.map(
      (variant, index) => variant.title ?? variant.description ?? `Variant ${index + 1}`
    );
  }

  return field;
}

/**
 * Convert a JSON schema to form fields for dynamic form generation
 */
export function schemaToFormFields(
  schema: JsonSchema,
  parentPath: string = ""
): FormField[] {
  const fields: FormField[] = [];

  if (!isRecord(schema)) {
    console.warn("schemaToFormFields received invalid schema:", schema);
    return fields;
  }

  const primaryType = resolvePrimaryType(schema);
  if (primaryType !== "object" || !schema.properties) {
    return fields;
  }

  try {
    const requiredSet = new Set(schema.required ?? []);

    Object.entries(schema.properties).forEach(([key, propSchema]) => {
      if (!isRecord(propSchema)) {
        console.warn(`Invalid property schema for key "${key}":`, propSchema);
        return;
      }

      const fieldPath = parentPath ? `${parentPath}.${key}` : key;
      const isRequired = requiredSet.has(key);
      fields.push(buildField(fieldPath, key, propSchema, isRequired));
    });
  } catch (error) {
    console.error("Error in schemaToFormFields:", error, "schema:", schema);
  }

  return fields;
}

/**
 * Get the appropriate form field type from JSON schema type
 */
function getFieldType(schema: JsonSchema): FormField["type"] {
  if (schema.enum) {
    return "select";
  }

  const type = resolvePrimaryType(schema);
  switch (type) {
    case "string":
      return "string";
    case "number":
    case "integer":
      return "number";
    case "boolean":
      return "boolean";
    case "object":
      return "object";
    case "array":
      return "array";
    default:
      return "string";
  }
}

/**
 * Format field name to a human-readable label
 */
function formatFieldLabel(fieldName: string): string {
  if (!fieldName || typeof fieldName !== "string") {
    console.warn("formatFieldLabel received invalid fieldName:", fieldName);
    return "Field";
  }

  try {
    return fieldName
      .replace(/([A-Z])/g, " $1")
      .replace(/[_-]/g, " ")
      .replace(/\b\w/g, (match) =>
        match && typeof match === "string" ? match.toUpperCase() : match
      )
      .trim();
  } catch (error) {
    console.error("Error formatting field label:", error, "fieldName:", fieldName);
    return fieldName || "Field";
  }
}

/**
 * Generate placeholder text from schema
 */
function getPlaceholder(schema: JsonSchema): string {
  const defaultValue = getDefaultValue(schema);
  if (defaultValue !== undefined) {
    if (typeof defaultValue === "string") {
      return defaultValue;
    }
    try {
      return JSON.stringify(defaultValue);
    } catch {
      return "Enter value...";
    }
  }

  if (schema.format) {
    switch (schema.format) {
      case "email":
        return "user@example.com";
      case "uri":
      case "url":
        return "https://example.com";
      case "date":
        return "YYYY-MM-DD";
      case "date-time":
        return "YYYY-MM-DDTHH:mm:ssZ";
      case "uuid":
        return "123e4567-e89b-12d3-a456-426614174000";
      default:
        return `Enter ${schema.format}`;
    }
  }

  const type = resolvePrimaryType(schema);
  switch (type) {
    case "string":
      return "Enter text...";
    case "number":
    case "integer":
      return "Enter number...";
    case "boolean":
      return "";
    case "array":
      return "Add items...";
    case "object":
      return "Configure object...";
    default:
      return "Enter value...";
  }
}

/**
 * Validate form data against JSON schema
 */
export function validateFormData(
  data: any,
  schema: JsonSchema
): { isValid: boolean; errors: string[] } {
  if (!isRecord(schema)) {
    return { isValid: true, errors: [] };
  }

  const input = data ?? {};

  try {
    const zodSchema = jsonSchemaToZodObject(schema);
    const result = zodSchema.safeParse(input);

    if (result.success) {
      return { isValid: true, errors: [] };
    }

    const errors = result.error.issues.map((issue) => {
      const path =
        issue.path && issue.path.length > 0
          ? issue.path.map((segment) => formatFieldLabel(String(segment))).join(" › ")
          : "Input";
      return `${path}: ${issue.message}`;
    });

    return { isValid: false, errors };
  } catch (error) {
    if (error instanceof ZodError) {
      const errors = error.issues.map((issue) => {
        const path =
          issue.path && issue.path.length > 0
            ? issue.path.map((segment) => formatFieldLabel(String(segment))).join(" › ")
            : "Input";
        return `${path}: ${issue.message}`;
      });
      return { isValid: false, errors };
    }

    console.warn("validateFormData: falling back to permissive validation", error);
    return { isValid: true, errors: [] };
  }
}

function validateAgainstVariants(
  value: any,
  variants: JsonSchema[],
  fieldLabel: string,
  combinator: "oneOf" | "anyOf" | "allOf"
): string[] {
  const variantResults = variants.map((variant) => ({
    errors: validateValue(value, variant, fieldLabel),
  }));

  if (combinator === "allOf") {
    return variantResults.flatMap((result) => result.errors);
  }

  const successfulVariants = variantResults.filter((result) => result.errors.length === 0);

  if (combinator === "anyOf") {
    if (successfulVariants.length > 0) {
      return [];
    }
    return variantResults[0]?.errors ?? [`${fieldLabel} does not match any valid variant`];
  }

  if (combinator === "oneOf") {
    if (successfulVariants.length === 1) {
      return [];
    }
    if (successfulVariants.length > 1) {
      return [`${fieldLabel} matches multiple variants. Please choose one.`];
    }
    return variantResults[0]?.errors ?? [`${fieldLabel} does not match any valid variant`];
  }

  return [];
}

/**
 * Validate a single value against its schema
 */
function validateValue(value: any, schema: JsonSchema, fieldLabel: string): string[] {
  const errors: string[] = [];

  if (!isRecord(schema)) {
    return errors;
  }

  if (value === undefined || value === null || value === "") {
    return errors;
  }

  if (schema.const !== undefined && value !== schema.const) {
    errors.push(`${fieldLabel} must be exactly ${schema.const}`);
    return errors;
  }

  const combinator = extractCombinatorMetadata(schema);
  if (combinator) {
    const variantErrors = validateAgainstVariants(value, combinator.variants, fieldLabel, combinator.type);
    errors.push(...variantErrors);
    if (variantErrors.length === 0) {
      return errors;
    }
  }

  const type = resolvePrimaryType(schema);

  switch (type) {
    case "string": {
      if (typeof value !== "string") {
        errors.push(`${fieldLabel} must be a string`);
        return errors;
      }

      if (schema.minLength !== undefined && value.length < schema.minLength) {
        errors.push(`${fieldLabel} must be at least ${schema.minLength} characters`);
      }

      if (schema.maxLength !== undefined && value.length > schema.maxLength) {
        errors.push(`${fieldLabel} must be no more than ${schema.maxLength} characters`);
      }

      if (schema.pattern && !new RegExp(schema.pattern).test(value)) {
        errors.push(`${fieldLabel} format is invalid`);
      }
      break;
    }

    case "number":
    case "integer": {
      const numValue = Number(value);
      if (Number.isNaN(numValue)) {
        errors.push(`${fieldLabel} must be a number`);
        return errors;
      }

      if (schema.minimum !== undefined && numValue < schema.minimum) {
        errors.push(`${fieldLabel} must be at least ${schema.minimum}`);
      }

      if (schema.maximum !== undefined && numValue > schema.maximum) {
        errors.push(`${fieldLabel} must be no more than ${schema.maximum}`);
      }

      if (type === "integer" && !Number.isInteger(numValue)) {
        errors.push(`${fieldLabel} must be an integer`);
      }
      break;
    }

    case "boolean": {
      if (typeof value !== "boolean") {
        errors.push(`${fieldLabel} must be true or false`);
      }
      break;
    }

    case "array": {
      if (!Array.isArray(value)) {
        errors.push(`${fieldLabel} must be an array`);
        return errors;
      }

      if (schema.minItems !== undefined && value.length < schema.minItems) {
        errors.push(`${fieldLabel} must contain at least ${schema.minItems} items`);
      }

      if (schema.maxItems !== undefined && value.length > schema.maxItems) {
        errors.push(`${fieldLabel} must contain no more than ${schema.maxItems} items`);
      }

      if (Array.isArray(schema.items)) {
        schema.items.forEach((itemSchema, index) => {
          if (!isRecord(itemSchema)) {
            return;
          }

          const itemValue = value[index];
          if (itemValue === undefined) {
            errors.push(`${fieldLabel}[${index}] is required`);
            return;
          }

          const itemErrors = validateValue(itemValue, itemSchema, `${fieldLabel}[${index}]`);
          errors.push(...itemErrors);
        });

        if (
          schema.additionalItems === false &&
          value.length > schema.items.length
        ) {
          errors.push(`${fieldLabel} has too many items`);
        }
      } else if (isRecord(schema.items)) {
        value.forEach((item, index) => {
          const itemErrors = validateValue(item, schema.items as JsonSchema, `${fieldLabel}[${index}]`);
          errors.push(...itemErrors);
        });
      }
      break;
    }

    case "object": {
      if (!isRecord(value) || Array.isArray(value)) {
        errors.push(`${fieldLabel} must be an object`);
        return errors;
      }

      const requiredFields = schema.required ?? [];
      requiredFields.forEach((childField) => {
        const childValue = (value as Record<string, any>)[childField];
        if (childValue === undefined || childValue === null || childValue === "") {
          errors.push(`${formatFieldLabel(childField)} is required`);
        }
      });

      if (schema.properties) {
        Object.entries(schema.properties).forEach(([key, propSchema]) => {
          if (!isRecord(propSchema)) {
            return;
          }

          const childValue = (value as Record<string, any>)[key];
          const childErrors = validateValue(childValue, propSchema, `${fieldLabel}.${formatFieldLabel(key)}`);
          errors.push(...childErrors);
        });
      }
      break;
    }
  }

  if (schema.enum && !schema.enum.some((option) => option === value)) {
    errors.push(`${fieldLabel} must be one of: ${schema.enum.join(", ")}`);
  }

  return errors;
}

export function validateValueAgainstSchema(value: any, schema: JsonSchema): string[] {
  return validateValue(value, schema, "Value");
}

/**
 * Generate example data from JSON schema
 */
export function generateExampleData(schema: JsonSchema): any {
  if (!isRecord(schema)) {
    return null;
  }

  const defaultValue = getDefaultValue(schema);
  if (defaultValue !== undefined) {
    return defaultValue;
  }

  const combinator = extractCombinatorMetadata(schema);
  if (combinator && combinator.variants.length > 0) {
    return generateExampleData(combinator.variants[0]);
  }

  const type = resolvePrimaryType(schema);

  switch (type) {
    case "string":
      if (schema.format === "email") {
        return "user@example.com";
      }
      if (schema.format === "uri" || schema.format === "url") {
        return "https://example.com";
      }
      if (schema.format === "uuid") {
        return "123e4567-e89b-12d3-a456-426614174000";
      }
      return "example";

    case "number":
    case "integer":
      if (schema.minimum !== undefined) {
        return schema.minimum;
      }
      if (schema.maximum !== undefined) {
        return schema.maximum;
      }
      return type === "integer" ? 1 : 1.0;

    case "boolean":
      return true;

    case "array": {
      if (Array.isArray(schema.items)) {
        return schema.items.map((item) => generateExampleData(item));
      }
      if (isRecord(schema.items)) {
        return [generateExampleData(schema.items)];
      }
      return [];
    }

    case "object": {
      const result: Record<string, any> = {};
      if (schema.properties) {
        Object.entries(schema.properties).forEach(([key, propSchema]) => {
          if (isRecord(propSchema)) {
            result[key] = generateExampleData(propSchema);
          }
        });
      }
      return result;
    }

    default:
      if (schema.enum && schema.enum.length > 0) {
        return deepClone(schema.enum[0]);
      }
      return null;
  }
}
