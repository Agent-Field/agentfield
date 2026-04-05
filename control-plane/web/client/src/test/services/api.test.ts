import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  setGlobalApiKey,
  getGlobalApiKey,
  setGlobalAdminToken,
  getGlobalAdminToken,
  parseNodeLogsNDJSON,
} from '@/services/api';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeFetchMock(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
    body: null,
  });
}

// ---------------------------------------------------------------------------
// API key management
// ---------------------------------------------------------------------------

describe('API key management', () => {
  afterEach(() => {
    setGlobalApiKey(null);
  });

  it('getGlobalApiKey returns null initially (no stored key)', () => {
    setGlobalApiKey(null);
    expect(getGlobalApiKey()).toBeNull();
  });

  it('setGlobalApiKey + getGlobalApiKey round-trip', () => {
    setGlobalApiKey('my-test-key');
    expect(getGlobalApiKey()).toBe('my-test-key');
  });

  it('setGlobalApiKey(null) clears the key', () => {
    setGlobalApiKey('temp-key');
    setGlobalApiKey(null);
    expect(getGlobalApiKey()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Admin token management
// ---------------------------------------------------------------------------

describe('Admin token management', () => {
  afterEach(() => {
    setGlobalAdminToken(null);
  });

  it('getGlobalAdminToken returns null when not set', () => {
    setGlobalAdminToken(null);
    expect(getGlobalAdminToken()).toBeNull();
  });

  it('setGlobalAdminToken + getGlobalAdminToken round-trip', () => {
    setGlobalAdminToken('admin-secret-token');
    expect(getGlobalAdminToken()).toBe('admin-secret-token');
  });

  it('setGlobalAdminToken(null) clears the token', () => {
    setGlobalAdminToken('some-token');
    setGlobalAdminToken(null);
    expect(getGlobalAdminToken()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Auth header injection — verify X-API-Key is forwarded in requests
// ---------------------------------------------------------------------------

describe('Auth header injection', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
  });

  it('includes X-API-Key header when a global key is set', async () => {
    setGlobalApiKey('injected-key');

    const capturedHeaders: Record<string, string> = {};
    globalThis.fetch = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      const h = new Headers(init?.headers);
      h.forEach((v, k) => { capturedHeaders[k] = v; });
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ nodes: [], count: 0 }),
        text: () => Promise.resolve(''),
        body: null,
      } as any);
    });

    // Import lazily so the module picks up the mocked fetch
    const { getNodesSummary } = await import('@/services/api');
    await getNodesSummary();

    expect(capturedHeaders['x-api-key']).toBe('injected-key');
  });

  it('does not include X-API-Key header when no key is set', async () => {
    setGlobalApiKey(null);

    const capturedHeaders: Record<string, string> = {};
    globalThis.fetch = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      const h = new Headers(init?.headers);
      h.forEach((v, k) => { capturedHeaders[k] = v; });
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ nodes: [], count: 0 }),
        text: () => Promise.resolve(''),
        body: null,
      } as any);
    });

    const { getNodesSummary } = await import('@/services/api');
    await getNodesSummary();

    expect(capturedHeaders['x-api-key']).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Error response parsing
// ---------------------------------------------------------------------------

describe('Error response parsing', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
  });

  it('throws an error with the message from the JSON body on non-OK response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ message: 'Bad request — missing field' }),
      text: () => Promise.resolve(''),
      body: null,
    } as any);

    const { getNodesSummary } = await import('@/services/api');
    await expect(getNodesSummary()).rejects.toThrow('Bad request — missing field');
  });

  it('falls back to generic status message when JSON body has no message', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.resolve({}),
      text: () => Promise.resolve(''),
      body: null,
    } as any);

    const { getNodesSummary } = await import('@/services/api');
    await expect(getNodesSummary()).rejects.toThrow('503');
  });

  it('throws AbortError-derived message on timeout', async () => {
    globalThis.fetch = vi.fn().mockImplementation(() => {
      const err = new Error('The operation was aborted');
      err.name = 'AbortError';
      return Promise.reject(err);
    });

    const { getNodesSummary } = await import('@/services/api');
    await expect(getNodesSummary()).rejects.toThrow(/timeout/i);
  });
});

// ---------------------------------------------------------------------------
// parseNodeLogsNDJSON — pure function, no fetch needed
// ---------------------------------------------------------------------------

describe('parseNodeLogsNDJSON', () => {
  it('parses valid NDJSON lines', () => {
    const text = [
      JSON.stringify({ v: 1, seq: 0, ts: '2026-01-01T00:00:00Z', stream: 'stdout', line: 'hello' }),
      JSON.stringify({ v: 1, seq: 1, ts: '2026-01-01T00:00:01Z', stream: 'stderr', line: 'world' }),
    ].join('\n');

    const entries = parseNodeLogsNDJSON(text);
    expect(entries).toHaveLength(2);
    expect(entries[0].line).toBe('hello');
    expect(entries[1].stream).toBe('stderr');
  });

  it('skips blank lines', () => {
    const text = '\n\n' + JSON.stringify({ v: 1, seq: 0, ts: 't', stream: 's', line: 'ok' }) + '\n\n';
    const entries = parseNodeLogsNDJSON(text);
    expect(entries).toHaveLength(1);
  });

  it('skips malformed JSON lines without throwing', () => {
    const text = [
      'not-valid-json',
      JSON.stringify({ v: 1, seq: 0, ts: 't', stream: 's', line: 'good' }),
    ].join('\n');

    const entries = parseNodeLogsNDJSON(text);
    expect(entries).toHaveLength(1);
    expect(entries[0].line).toBe('good');
  });

  it('returns an empty array for an empty string', () => {
    expect(parseNodeLogsNDJSON('')).toEqual([]);
  });
});
