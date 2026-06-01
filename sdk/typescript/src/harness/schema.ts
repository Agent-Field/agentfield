import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

export const OUTPUT_FILENAME = '.agentfield_output.json';
export const SCHEMA_FILENAME = '.agentfield_schema.json';
export const LARGE_SCHEMA_TOKEN_THRESHOLD = 4000;

type JsonSchemaRecord = Record<string, unknown>;

type JsonSchemaFactory = (schema: unknown) => JsonSchemaRecord;

type JsonSchemaModule = {
  zodToJsonSchema?: JsonSchemaFactory;
  default?: JsonSchemaFactory;
};

type JsonSchemaProvider = {
  jsonSchema: () => JsonSchemaRecord;
};

type ParseSchema = {
  parse: (value: unknown) => unknown;
};

let zodConverter: JsonSchemaFactory | null | undefined;

function getZodConverter(): JsonSchemaFactory | null {
  if (zodConverter !== undefined) {
    return zodConverter;
  }

  try {
    const require = createRequire(import.meta.url);
    const mod = require('zod-to-json-schema') as JsonSchemaModule;
    zodConverter = mod.zodToJsonSchema ?? mod.default ?? null;
  } catch {
    zodConverter = null;
  }

  return zodConverter;
}

function isRecord(value: unknown): value is JsonSchemaRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasJsonSchema(value: unknown): value is JsonSchemaProvider {
  return isRecord(value) && typeof value.jsonSchema === 'function';
}

function hasParse(value: unknown): value is ParseSchema {
  return isRecord(value) && typeof value.parse === 'function';
}

function estimateTokens(text: string): number {
  return Math.floor(text.length / 4);
}

function writeSchemaFile(schemaJson: string, cwd: string): string {
  const filePath = getSchemaPath(cwd);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const fd = fs.openSync(filePath, 'w', 0o600);
  try {
    fs.writeFileSync(fd, schemaJson, 'utf8');
  } finally {
    fs.closeSync(fd);
  }
  return filePath;
}

function validateAgainstSchema(data: unknown, schema: unknown): unknown {
  if (hasParse(schema)) {
    return schema.parse(data);
  }
  return data;
}

export function getOutputPath(cwd: string): string {
  return path.join(cwd, OUTPUT_FILENAME);
}

export function getSchemaPath(cwd: string): string {
  return path.join(cwd, SCHEMA_FILENAME);
}

export function schemaToJsonSchema(schema: unknown): JsonSchemaRecord {
  if (isRecord(schema)) {
    if ('type' in schema || 'properties' in schema || '$schema' in schema) {
      return schema;
    }
    if (hasJsonSchema(schema)) {
      return schema.jsonSchema();
    }
  }

  const converter = getZodConverter();
  if (converter !== null) {
    return converter(schema);
  }

  throw new TypeError('Unsupported schema type. Expected a Zod schema, JSON schema object, or jsonSchema() provider.');
}

export function isLargeSchema(schemaJson: string): boolean {
  return estimateTokens(schemaJson) > LARGE_SCHEMA_TOKEN_THRESHOLD;
}

export function buildPromptSuffix(schema: unknown, cwd: string): string {
  const jsonSchema = schemaToJsonSchema(schema);
  const schemaJson = JSON.stringify(jsonSchema, null, 2);
  const outputPath = getOutputPath(cwd);

  if (isLargeSchema(schemaJson)) {
    const schemaPath = writeSchemaFile(schemaJson, cwd);
    return (
      '\n\n---\n' +
      'OUTPUT REQUIREMENTS:\n' +
      `Read the JSON Schema at: ${schemaPath}\n` +
      `Write your final answer as valid JSON conforming to that schema to: ${outputPath}\n` +
      'Do not include any text outside the JSON in that file. Do not wrap in markdown fences.'
    );
  }

  return (
    '\n\n---\n' +
    'OUTPUT REQUIREMENTS:\n' +
    `Write your final answer as valid JSON to the file: ${outputPath}\n` +
    'The JSON must conform to this schema:\n' +
    `${schemaJson}\n` +
    'Do not include any text outside the JSON in that file. Do not wrap in markdown fences.'
  );
}

export function cosmeticRepair(raw: string): string {
  let text = raw.trim();

  const fenceMatch = text.match(/^```(?:json)?\s*\n([\s\S]*?)```\s*$/);
  if (fenceMatch) {
    text = fenceMatch[1].trim();
  }

  if (text.length > 0 && text[0] !== '{' && text[0] !== '[') {
    const firstJsonCharIndex = [...text].findIndex((char) => char === '{' || char === '[');
    if (firstJsonCharIndex >= 0) {
      text = text.slice(firstJsonCharIndex);
    }
  }

  text = text.replace(/,\s*([}\]])/g, '$1');

  const openBraces = (text.match(/{/g)?.length ?? 0) - (text.match(/}/g)?.length ?? 0);
  const openBrackets = (text.match(/\[/g)?.length ?? 0) - (text.match(/\]/g)?.length ?? 0);

  if (openBraces > 0 || openBrackets > 0) {
    text += ']'.repeat(openBrackets) + '}'.repeat(openBraces);
  }

  return text;
}

export function readAndParse(filePath: string): unknown | null {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    if (content.trim().length === 0) {
      return null;
    }
    return JSON.parse(content);
  } catch {
    return null;
  }
}

export function readRepairAndParse(filePath: string): unknown | null {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    if (content.trim().length === 0) {
      return null;
    }
    return JSON.parse(cosmeticRepair(content));
  } catch {
    return null;
  }
}

export function parseAndValidate(filePath: string, schema: unknown): unknown | null {
  const parsed = readAndParse(filePath);
  if (parsed !== null) {
    try {
      return validateAgainstSchema(parsed, schema);
    } catch {
    }
  }

  const repaired = readRepairAndParse(filePath);
  if (repaired !== null) {
    try {
      return validateAgainstSchema(repaired, schema);
    } catch {
      return null;
    }
  }

  return null;
}

export function cleanupTempFiles(cwd: string): void {
  for (const filename of [OUTPUT_FILENAME, SCHEMA_FILENAME]) {
    try {
      fs.unlinkSync(path.join(cwd, filename));
    } catch {
    }
  }
}

export function buildFollowupPrompt(errorMessage: string, cwd: string): string {
  const outputPath = getOutputPath(cwd);
  return (
    `The JSON at ${outputPath} failed validation: ${errorMessage}\n` +
    'Please rewrite the corrected, valid JSON to the same file.'
  );
}
