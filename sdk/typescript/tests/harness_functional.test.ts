/**
 * Live functional tests for harness providers.
 *
 * These tests invoke REAL coding agents and make real API calls.
 * They are NOT included in the default `vitest run` — run explicitly:
 *
 *   npx vitest run tests/harness_functional.test.ts --timeout=300000
 *
 * Run a single provider:
 *   npx vitest run tests/harness_functional.test.ts -t "Codex" --timeout=300000
 *   npx vitest run tests/harness_functional.test.ts -t "OpenCode" --timeout=300000
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { HarnessRunner } from '../src/harness/runner.js';

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

function hasBinary(name: string): boolean {
  try {
    execSync(`which ${name}`, { encoding: 'utf8', stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

const HAS_CODEX = hasBinary('codex');
const HAS_OPENCODE = hasBinary('opencode');

/** Plain JSON Schema — no Zod dependency needed. */
const simpleSchema = {
  type: 'object' as const,
  properties: {
    greeting: { type: 'string' },
    number: { type: 'integer' },
  },
  required: ['greeting', 'number'],
};

const codeReviewSchema = {
  type: 'object' as const,
  properties: {
    summary: { type: 'string' },
    score: { type: 'integer' },
    suggestions: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'score', 'suggestions'],
};

// ────────────────────────────────────────────────────────────────────────
// Temp dir helpers
// ────────────────────────────────────────────────────────────────────────

let workDir: string;

function createWorkDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-harness-test-'));
  // Codex requires a git repo — init one in the temp dir
  execSync('git init', { cwd: dir, stdio: 'pipe' });
  fs.writeFileSync(path.join(dir, '.gitkeep'), '');
  execSync('git add . && git commit -m init --allow-empty', { cwd: dir, stdio: 'pipe' });
  return dir;
}

function cleanupWorkDir(dir: string): void {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

// ════════════════════════════════════════════════════════════════════════
// CODEX
// ════════════════════════════════════════════════════════════════════════

describe.skipIf(!HAS_CODEX)('Codex Functional', () => {
  beforeEach(() => {
    workDir = createWorkDir();
  });
  afterEach(() => {
    cleanupWorkDir(workDir);
  });

  it('basic prompt returns text', async () => {
    const { CodexProvider } = await import('../src/harness/providers/codex.js');
    const provider = new CodexProvider();
    const result = await provider.execute(
      'What is 2+2? Reply with ONLY the number, nothing else.',
      { cwd: workDir, permissionMode: 'auto' },
    );

    expect(result.isError).toBe(false);
    expect(result.result).toBeDefined();
    expect(result.result).toContain('4');
  }, 120_000);

  it('schema pipeline: prompt → file write → parse', async () => {
    const runner = new HarnessRunner();
    const result = await runner.run(
      'Return exactly: greeting="Hello from Codex" and number=42. Follow the OUTPUT REQUIREMENTS below precisely.',
      {
        provider: 'codex',
        schema: simpleSchema,
        cwd: workDir,
        permissionMode: 'auto',
        maxRetries: 1,
      },
    );

    expect(result.isError).toBe(false);
    expect(result.parsed).not.toBeNull();
    expect(result.parsed).toBeDefined();

    const parsed = result.parsed as { greeting: string; number: number };
    expect(typeof parsed.greeting).toBe('string');
    expect(parsed.greeting.length).toBeGreaterThan(0);
    expect(typeof parsed.number).toBe('number');
  }, 120_000);

  it('temp files are cleaned up after schema run', async () => {
    const runner = new HarnessRunner();
    await runner.run(
      'Return greeting="cleanup" and number=0. Follow the OUTPUT REQUIREMENTS below.',
      {
        provider: 'codex',
        schema: simpleSchema,
        cwd: workDir,
        permissionMode: 'auto',
        maxRetries: 1,
      },
    );

    expect(fs.existsSync(path.join(workDir, '.agentfield_output.json'))).toBe(false);
  }, 120_000);
});

// ════════════════════════════════════════════════════════════════════════
// OPENCODE
// ════════════════════════════════════════════════════════════════════════

/**
 * OpenCode Functional tests.
 *
 * All tests are marked as `.todo` because OpenCode v1.2.10 has a known
 * upstream bug in headless mode — `opencode run` returns "Session not found"
 * regardless of context.
 * See: https://github.com/anomalyco/opencode/issues/13851
 */
describe.skipIf(!HAS_OPENCODE)('OpenCode Functional (xfail: upstream bug)', () => {
  beforeEach(() => {
    workDir = createWorkDir();
  });
  afterEach(() => {
    cleanupWorkDir(workDir);
  });

  it.fails('basic prompt returns text (OpenCode v1.2.10 headless bug)', async () => {
    const { OpenCodeProvider } = await import('../src/harness/providers/opencode.js');
    const provider = new OpenCodeProvider();
    const result = await provider.execute(
      'What is 2+2? Reply with ONLY the number, nothing else.',
      { cwd: workDir },
    );

    expect(result.isError).toBe(false);
    expect(result.result).toBeDefined();
  }, 120_000);

  it.fails('schema pipeline (OpenCode v1.2.10 headless bug)', async () => {
    const runner = new HarnessRunner();
    const result = await runner.run(
      'Return exactly: greeting="Hello from OpenCode" and number=42. Follow the OUTPUT REQUIREMENTS below precisely.',
      {
        provider: 'opencode',
        schema: simpleSchema,
        cwd: workDir,
        maxRetries: 1,
      },
    );

    expect(result.isError).toBe(false);
    expect(result.parsed).not.toBeNull();
    expect(result.parsed).toBeDefined();

    const parsed = result.parsed as { greeting: string; number: number };
    expect(typeof parsed.greeting).toBe('string');
    expect(parsed.greeting.length).toBeGreaterThan(0);
    expect(typeof parsed.number).toBe('number');
  }, 120_000);
});

// ════════════════════════════════════════════════════════════════════════
// CROSS-PROVIDER
// ════════════════════════════════════════════════════════════════════════

// NOTE: OpenCode excluded from cross-provider until upstream headless bug is fixed.
// See: https://github.com/anomalyco/opencode/issues/13851
describe.skipIf(!HAS_CODEX)('Cross-Provider Consistency', () => {
  beforeEach(() => {
    workDir = createWorkDir();
  });
  afterEach(() => {
    cleanupWorkDir(workDir);
  });

  it('codex returns data matching the schema', async () => {
    const runner = new HarnessRunner();
    const subDir = path.join(workDir, 'codex');
    fs.mkdirSync(subDir, { recursive: true });

    const result = await runner.run(
      'Return greeting="hello" and number=42. Follow the OUTPUT REQUIREMENTS below.',
      {
        provider: 'codex',
        schema: simpleSchema,
        cwd: subDir,
        permissionMode: 'auto',
        maxRetries: 1,
      },
    );

    expect(result.isError).toBe(false);
    expect(result.parsed).not.toBeNull();
    const parsed = result.parsed as { greeting: string; number: number };
    expect(typeof parsed.greeting).toBe('string');
    expect(typeof parsed.number).toBe('number');
  }, 120_000);
});
