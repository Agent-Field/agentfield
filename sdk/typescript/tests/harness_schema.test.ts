import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { z } from 'zod';

import {
  LARGE_SCHEMA_TOKEN_THRESHOLD,
  buildFollowupPrompt,
  buildPromptSuffix,
  cleanupTempFiles,
  cosmeticRepair,
  getOutputPath,
  getSchemaPath,
  isLargeSchema,
  parseAndValidate,
  readAndParse,
  readRepairAndParse,
  schemaToJsonSchema,
} from '../src/harness/schema.js';

const tempDirs: string[] = [];

function makeTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-schema-'));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0, tempDirs.length)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe('harness schema', () => {
  it('returns deterministic output path', () => {
    const cwd = makeTempDir();
    expect(getOutputPath(cwd)).toBe(path.join(cwd, '.agentfield_output.json'));
  });

  it('returns deterministic schema path', () => {
    const cwd = makeTempDir();
    expect(getSchemaPath(cwd)).toBe(path.join(cwd, '.agentfield_schema.json'));
  });

  it('converts zod schema and plain object to json schema', () => {
    const zodSchema = z.object({ name: z.string(), count: z.number() });
    const jsonSchema = schemaToJsonSchema(zodSchema);
    expect(jsonSchema).toHaveProperty('type', 'object');

    const plain = { type: 'object', properties: { name: { type: 'string' } } };
    expect(schemaToJsonSchema(plain)).toEqual(plain);
  });

  it('detects large schema threshold', () => {
    const small = 'a'.repeat(LARGE_SCHEMA_TOKEN_THRESHOLD * 4);
    const large = 'a'.repeat((LARGE_SCHEMA_TOKEN_THRESHOLD + 1) * 4);
    expect(isLargeSchema(small)).toBe(false);
    expect(isLargeSchema(large)).toBe(true);
  });

  it('builds small schema prompt suffix inline', () => {
    const cwd = makeTempDir();
    const suffix = buildPromptSuffix(z.object({ name: z.string() }), cwd);
    expect(suffix).toContain('OUTPUT REQUIREMENTS');
    expect(suffix).toContain('The JSON must conform to this schema');
    expect(suffix).toContain(getOutputPath(cwd));
    expect(suffix).not.toContain(getSchemaPath(cwd));
    expect(fs.existsSync(getSchemaPath(cwd))).toBe(false);
  });

  it('builds large schema prompt suffix with schema file reference', () => {
    const cwd = makeTempDir();
    const largeSchema = {
      type: 'object',
      properties: {
        payload: {
          type: 'string',
          description: 'x'.repeat(20000),
        },
      },
    };

    const suffix = buildPromptSuffix(largeSchema, cwd);
    expect(suffix).toContain('Read the JSON Schema at');
    expect(suffix).toContain(getSchemaPath(cwd));
    expect(fs.existsSync(getSchemaPath(cwd))).toBe(true);
  });

  it('cosmeticRepair strips json markdown fences', () => {
    expect(cosmeticRepair('```json\n{"a": 1}\n```')).toBe('{"a": 1}');
  });

  it('cosmeticRepair strips plain markdown fences', () => {
    expect(cosmeticRepair('```\n{"a": 1}\n```')).toBe('{"a": 1}');
  });

  it('cosmeticRepair removes trailing commas', () => {
    expect(cosmeticRepair('{"a": 1,}')).toBe('{"a": 1}');
  });

  it('cosmeticRepair fixes truncated json', () => {
    expect(cosmeticRepair('{"a": 1')).toBe('{"a": 1}');
  });

  it('cosmeticRepair passes valid json unchanged', () => {
    expect(cosmeticRepair('{"a": 1}')).toBe('{"a": 1}');
  });

  it('cosmeticRepair handles json preceded by text', () => {
    expect(cosmeticRepair('Result below:\n{"a": 1}')).toBe('{"a": 1}');
  });

  it('readAndParse handles valid, missing, and invalid json files', () => {
    const cwd = makeTempDir();
    const validPath = path.join(cwd, 'valid.json');
    fs.writeFileSync(validPath, '{"name":"a","count":1}', 'utf8');

    expect(readAndParse(validPath)).toEqual({ name: 'a', count: 1 });
    expect(readAndParse(path.join(cwd, 'missing.json'))).toBeNull();

    const invalidPath = path.join(cwd, 'invalid.json');
    fs.writeFileSync(invalidPath, '{"name":"a",}', 'utf8');
    expect(readAndParse(invalidPath)).toBeNull();
  });

  it('readRepairAndParse handles markdown-fenced json', () => {
    const cwd = makeTempDir();
    const filePath = path.join(cwd, 'fenced.json');
    fs.writeFileSync(filePath, '```json\n{"name":"a","count":1}\n```', 'utf8');
    expect(readRepairAndParse(filePath)).toEqual({ name: 'a', count: 1 });
  });

  it('parseAndValidate runs layer 1 direct parse', () => {
    const cwd = makeTempDir();
    const filePath = path.join(cwd, 'layer1.json');
    fs.writeFileSync(filePath, '{"name":"a","count":1}', 'utf8');
    const schema = z.object({ name: z.string(), count: z.number() });
    expect(parseAndValidate(filePath, schema)).toEqual({ name: 'a', count: 1 });
  });

  it('parseAndValidate runs layer 2 repair fallback', () => {
    const cwd = makeTempDir();
    const filePath = path.join(cwd, 'layer2.json');
    fs.writeFileSync(filePath, '```json\n{"name":"a","count":1,}\n```', 'utf8');
    const schema = z.object({ name: z.string(), count: z.number() });
    expect(parseAndValidate(filePath, schema)).toEqual({ name: 'a', count: 1 });
  });

  it('cleanupTempFiles removes files and is safe when absent', () => {
    const cwd = makeTempDir();
    fs.writeFileSync(path.join(cwd, '.agentfield_output.json'), '{}', 'utf8');
    fs.writeFileSync(path.join(cwd, '.agentfield_schema.json'), '{}', 'utf8');

    cleanupTempFiles(cwd);
    expect(fs.existsSync(path.join(cwd, '.agentfield_output.json'))).toBe(false);
    expect(fs.existsSync(path.join(cwd, '.agentfield_schema.json'))).toBe(false);

    expect(() => cleanupTempFiles(cwd)).not.toThrow();
  });

  it('buildFollowupPrompt includes error message and output path', () => {
    const cwd = makeTempDir();
    const prompt = buildFollowupPrompt('count is required', cwd);
    expect(prompt).toContain('count is required');
    expect(prompt).toContain(path.join(cwd, '.agentfield_output.json'));
  });
});
