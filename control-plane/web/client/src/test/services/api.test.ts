import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  bulkNodeStatus,
  fetchNodeLogsText,
  getAgentConfigurationSchema,
  getAgentEnvironmentVariables,
  setGlobalApiKey,
  getGlobalApiKey,
  setGlobalAdminToken,
  getGlobalAdminToken,
  getNodeDetailsWithPackageInfo,
  getNodeLogProxySettings,
  getNodeStatus,
  parseNodeLogsNDJSON,
  putNodeLogProxySettings,
  refreshNodeStatus,
  registerServerlessAgent,
  startAgentWithStatus,
  stopAgentWithStatus,
  streamNodeEvents,
  streamNodeLogsEntries,
  subscribeToUnifiedStatusEvents,
  updateAgentEnvironmentVariables,
  updateNodeStatus,
} from '@/services/api';
import {
  installEventSourceMock,
  mockJsonResponse,
  mockTextResponse,
  textStream,
} from '@/test/serviceTestUtils';

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

describe('EventSource helpers', () => {
  afterEach(() => {
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it('builds stream URLs with the current API key when present', () => {
    setGlobalApiKey('stream-key');
    const mock = installEventSourceMock();

    streamNodeEvents();
    subscribeToUnifiedStatusEvents();

    expect(mock.instances.map((entry) => entry.url)).toEqual([
      '/api/ui/v1/nodes/events?api_key=stream-key',
      '/api/ui/v1/nodes/events?api_key=stream-key',
    ]);

    mock.restore();
  });
});

describe('Environment and status APIs', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it('targets environment, status, and serverless endpoints correctly', async () => {
    setGlobalApiKey('node-key');
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockJsonResponse(200, { variables: {} }))
      .mockResolvedValueOnce(mockJsonResponse(200, { message: 'updated' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { schema: {} }))
      .mockResolvedValueOnce(mockJsonResponse(200, { id: 'node-1' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { state: 'running' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { state: 'running' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { 'node-1': { state: 'running' } }))
      .mockResolvedValueOnce(mockJsonResponse(200, { state: 'stopped' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { state: 'running' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { state: 'stopped' }))
      .mockResolvedValueOnce(mockJsonResponse(200, { success: true, node: { id: 'svless', version: '1.0.0' } }));

    await getAgentEnvironmentVariables('agent-1', 'pkg-1');
    await updateAgentEnvironmentVariables('agent-1', 'pkg-1', { KEY: 'VALUE' });
    await getAgentConfigurationSchema('agent-1', 'pkg-1');
    await getNodeDetailsWithPackageInfo('node-1', 'admin');
    await getNodeStatus('node-1');
    await refreshNodeStatus('node-1');
    await bulkNodeStatus(['node-1']);
    await updateNodeStatus('node-1', { desired_state: 'stopped' } as never);
    await startAgentWithStatus('node-1');
    await stopAgentWithStatus('node-1');
    await registerServerlessAgent('https://example.com/invoke');

    const calls = vi.mocked(globalThis.fetch).mock.calls as [string, RequestInit][];
    expect(calls[0][0]).toBe('/api/ui/v1/agents/agent-1/env?packageId=pkg-1');
    expect(calls[1][0]).toBe('/api/ui/v1/agents/agent-1/env?packageId=pkg-1');
    expect(calls[1][1].method).toBe('PUT');
    expect(JSON.parse(String(calls[1][1].body))).toEqual({ variables: { KEY: 'VALUE' } });
    expect(calls[2][0]).toBe('/api/ui/v1/agents/agent-1/config/schema?packageId=pkg-1');
    expect(calls[3][0]).toBe('/api/ui/v1/nodes/node-1/details?mode=admin');
    expect(calls[4][0]).toBe('/api/ui/v1/nodes/node-1/status');
    expect(calls[5][0]).toBe('/api/ui/v1/nodes/node-1/status/refresh');
    expect(calls[6][0]).toBe('/api/ui/v1/nodes/status/bulk');
    expect(JSON.parse(String(calls[6][1].body))).toEqual({ node_ids: ['node-1'] });
    expect(calls[7][0]).toBe('/api/ui/v1/nodes/node-1/status');
    expect(calls[7][1].method).toBe('PUT');
    expect(calls[8][0]).toBe('/api/ui/v1/nodes/node-1/start');
    expect(calls[9][0]).toBe('/api/ui/v1/nodes/node-1/stop');
    expect(calls[10][0]).toBe('/api/v1/nodes/register-serverless');
    expect(new Headers(calls[10][1].headers).get('X-API-Key')).toBe('node-key');
    expect(JSON.parse(String(calls[10][1].body))).toEqual({ invocation_url: 'https://example.com/invoke' });
  });
});

describe('Node log APIs', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it('fetches log text and node log settings with auth headers', async () => {
    setGlobalApiKey('logs-key');
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockTextResponse(200, '{"v":1}\n'))
      .mockResolvedValueOnce(mockJsonResponse(200, { effective: { max_tail_lines: 100 } }))
      .mockResolvedValueOnce(mockJsonResponse(200, { effective: { max_tail_lines: 200 } }));

    await expect(fetchNodeLogsText('node/1', { tail_lines: '5', since_seq: '3' })).resolves.toContain('{"v":1}');
    await expect(getNodeLogProxySettings()).resolves.toMatchObject({ effective: { max_tail_lines: 100 } });
    await expect(putNodeLogProxySettings({ max_tail_lines: 200 })).resolves.toMatchObject({ effective: { max_tail_lines: 200 } });

    const calls = vi.mocked(globalThis.fetch).mock.calls as [string, RequestInit][];
    expect(calls[0][0]).toBe('/api/ui/v1/nodes/node%2F1/logs?tail_lines=5&since_seq=3');
    expect(new Headers(calls[0][1].headers).get('X-API-Key')).toBe('logs-key');
    expect(calls[1][0]).toBe('/api/ui/v1/settings/node-log-proxy');
    expect(calls[2][0]).toBe('/api/ui/v1/settings/node-log-proxy');
    expect(calls[2][1].method).toBe('PUT');
    expect(JSON.parse(String(calls[2][1].body))).toEqual({ max_tail_lines: 200 });
  });

  it('streams NDJSON log entries and surfaces structured HTTP errors', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        body: textStream([
          '{"v":1,"seq":1,"ts":"2026-04-07T12:00:00Z","stream":"stdout","line":"one"}\n',
          'not-json\n',
          '{"v":1,"seq":2,"ts":"2026-04-07T12:00:01Z","stream":"stderr","line":"two"}\n',
        ]),
      } as Response)
      .mockResolvedValueOnce(mockJsonResponse(401, { message: 'token required' }));

    const signal = new AbortController().signal;
    const entries: Array<{ line: string }> = [];
    for await (const entry of streamNodeLogsEntries('node-1', { follow: '1' }, signal)) {
      entries.push({ line: entry.line });
    }

    expect(entries).toEqual([{ line: 'one' }, { line: 'two' }]);

    await expect(fetchNodeLogsText('node-1', { tail_lines: '10' })).rejects.toThrow('token required');
  });
});
