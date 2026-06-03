import express from 'express';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';

import { ProcessLogRing, registerAgentfieldLogsRoute } from '../src/agent/processLogs.js';

const ENV_KEYS = [
  'AGENTFIELD_LOGS_ENABLED',
  'AGENTFIELD_LOG_BUFFER_BYTES',
  'AGENTFIELD_LOG_MAX_LINE_BYTES',
  'AGENTFIELD_LOG_MAX_TAIL_LINES',
  'AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN'
] as const;

afterEach(() => {
  for (const key of ENV_KEYS) {
    delete process.env[key];
  }
});

function applyEnv(overrides: Partial<Record<(typeof ENV_KEYS)[number], string>>): void {
  for (const key of ENV_KEYS) {
    delete process.env[key];
  }
  Object.assign(process.env, overrides);
}

function parseNdjson(text: string): Array<Record<string, unknown>> {
  return text
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

async function withLogsServer(
  ring: ProcessLogRing,
  callback: (baseUrl: string) => Promise<void>
): Promise<void> {
  const app = express();
  registerAgentfieldLogsRoute(app, ring);

  const server = await new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const instance = app.listen(0, () => resolve(instance));
  });

  try {
    const address = server.address() as AddressInfo;
    await callback(`http://127.0.0.1:${address.port}`);
  } finally {
    server.closeIdleConnections?.();
    server.closeAllConnections?.();
    await new Promise<void>((resolve, reject) => {
      server.close((error?: Error) => (error ? reject(error) : resolve()));
    });
  }
}

describe('ProcessLogRing', () => {
  it('tails and snapshots logs while trimming to the configured buffer size', () => {
    applyEnv({ AGENTFIELD_LOG_BUFFER_BYTES: '1024' });
    const ring = new ProcessLogRing();
    const longLine = 'x'.repeat(400);

    ring.append('stdout', `${longLine}-first`, false);
    ring.append('stderr', `${longLine}-second`, true);
    ring.append('custom', `${longLine}-third`, false);

    expect(ring.tail(0)).toEqual([]);

    const entries = ring.tail(10);
    expect(entries).toHaveLength(2);
    expect(entries.map((entry) => entry.line)).toEqual([
      `${longLine}-second`,
      `${longLine}-third`
    ]);
    expect(entries[0]?.level).toBe('error');
    expect(entries[0]?.truncated).toBe(true);
    expect(entries[1]?.level).toBe('log');

    expect(ring.snapshotAfter(1, null).map((entry) => entry.line)).toEqual([
      `${longLine}-second`,
      `${longLine}-third`
    ]);
    expect(ring.snapshotAfter(1, 1).map((entry) => entry.line)).toEqual([`${longLine}-third`]);
  });
});

describe('registerAgentfieldLogsRoute', () => {
  it('returns 404 when process logs are disabled', async () => {
    applyEnv({ AGENTFIELD_LOGS_ENABLED: 'false' });
    const ring = new ProcessLogRing();

    await withLogsServer(ring, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/agentfield/v1/logs`);

      expect(response.status).toBe(404);
      await expect(response.json()).resolves.toEqual({
        error: 'logs_disabled',
        message: 'Process logs API is disabled'
      });
    });
  });

  it('enforces the internal bearer token and tail_lines cap', async () => {
    applyEnv({
      AGENTFIELD_LOGS_ENABLED: 'true',
      AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN: 'secret-token',
      AGENTFIELD_LOG_MAX_TAIL_LINES: '1'
    });
    const ring = new ProcessLogRing();
    ring.append('stdout', 'first line', false);

    await withLogsServer(ring, async (baseUrl) => {
      const unauthorized = await fetch(`${baseUrl}/agentfield/v1/logs`);
      expect(unauthorized.status).toBe(401);
      await expect(unauthorized.json()).resolves.toEqual({
        error: 'unauthorized',
        message: 'Valid Authorization Bearer required'
      });

      const tooLarge = await fetch(`${baseUrl}/agentfield/v1/logs?tail_lines=2`, {
        headers: { Authorization: 'Bearer secret-token' }
      });
      expect(tooLarge.status).toBe(413);
      await expect(tooLarge.json()).resolves.toEqual({
        error: 'tail_too_large',
        message: 'tail_lines exceeds max 1'
      });
    });
  });

  it('returns ndjson tail output and supports since_seq filtering', async () => {
    applyEnv({
      AGENTFIELD_LOGS_ENABLED: 'true',
      AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN: 'secret-token'
    });
    const ring = new ProcessLogRing();
    ring.append('stdout', 'first line', false);
    ring.append('stderr', 'second line', true);

    await withLogsServer(ring, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/agentfield/v1/logs`, {
        headers: { Authorization: 'Bearer secret-token' }
      });

      expect(response.status).toBe(200);
      expect(response.headers.get('content-type')).toContain('application/x-ndjson');
      expect(response.headers.get('cache-control')).toBe('no-store');

      const initial = parseNdjson(await response.text());
      expect(initial.map((entry) => entry.line)).toEqual(['first line', 'second line']);
      expect(initial[1]?.level).toBe('error');
      expect(initial[1]?.truncated).toBe(true);

      const sinceResponse = await fetch(`${baseUrl}/agentfield/v1/logs?since_seq=1&tail_lines=1`, {
        headers: { Authorization: 'Bearer secret-token' }
      });

      expect(sinceResponse.status).toBe(200);
      expect(parseNdjson(await sinceResponse.text()).map((entry) => entry.line)).toEqual([
        'second line'
      ]);
    });
  });
});
