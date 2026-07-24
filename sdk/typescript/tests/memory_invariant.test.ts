/**
 * Behavioral invariant tests for MemoryClient / MemoryClientBase.
 *
 * These tests verify structural properties (header mapping, set/get roundtrip,
 * delete consistency, key independence) that must always hold.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { MemoryClient, MemoryClientBase } from '../src/memory/MemoryClient.js';
import type { MemoryScope } from '../src/types/agent.js';

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Scope-to-header mapping ───────────────────────────────────────────────────

describe('INVARIANT: scope header mapping stability', () => {
  /**
   * The MemoryClientBase.buildHeaders method maps scopes to specific HTTP
   * headers.  This mapping is part of the public API contract (the control
   * plane reads these headers).  It must never silently change.
   */

  function buildHeadersFor(scope: MemoryScope, scopeId: string) {
    const client = new MemoryClientBase('http://localhost:8080');
    return (client as any).buildHeaders({ scope, scopeId });
  }

  it('workflow scope → X-Workflow-ID header', () => {
    const headers = buildHeadersFor('workflow', 'wf-001');
    expect(headers['X-Workflow-ID']).toBe('wf-001');
    expect(headers['X-Session-ID']).toBeUndefined();
    expect(headers['X-Actor-ID']).toBeUndefined();
  });

  it('session scope → X-Session-ID header', () => {
    const headers = buildHeadersFor('session', 'sess-abc');
    expect(headers['X-Session-ID']).toBe('sess-abc');
    expect(headers['X-Workflow-ID']).toBeUndefined();
    expect(headers['X-Actor-ID']).toBeUndefined();
  });

  it('actor scope → X-Actor-ID header', () => {
    const headers = buildHeadersFor('actor', 'actor-x');
    expect(headers['X-Actor-ID']).toBe('actor-x');
    expect(headers['X-Workflow-ID']).toBeUndefined();
    expect(headers['X-Session-ID']).toBeUndefined();
  });

  it('global scope → no extra scope-specific header beyond defaults', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const headers = (client as any).buildHeaders({ scope: 'global' as MemoryScope });
    // global scope does NOT set X-Workflow-ID / X-Session-ID / X-Actor-ID
    expect(headers['X-Workflow-ID']).toBeUndefined();
    expect(headers['X-Session-ID']).toBeUndefined();
    expect(headers['X-Actor-ID']).toBeUndefined();
  });

  it('no scope → no scope-specific headers', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const headers = (client as any).buildHeaders({});
    expect(headers['X-Workflow-ID']).toBeUndefined();
    expect(headers['X-Session-ID']).toBeUndefined();
    expect(headers['X-Actor-ID']).toBeUndefined();
  });

  it('metadata workflowId maps to X-Workflow-ID', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const headers = (client as any).buildHeaders({
      metadata: { workflowId: 'wf-meta-001' }
    });
    expect(headers['X-Workflow-ID']).toBe('wf-meta-001');
  });

  it('metadata sessionId maps to X-Session-ID', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const headers = (client as any).buildHeaders({
      metadata: { sessionId: 'sess-meta-001' }
    });
    expect(headers['X-Session-ID']).toBe('sess-meta-001');
  });

  it('metadata actorId maps to X-Actor-ID', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const headers = (client as any).buildHeaders({
      metadata: { actorId: 'actor-meta-001' }
    });
    expect(headers['X-Actor-ID']).toBe('actor-meta-001');
  });
});

// ── HTTP-layer invariants (mocked axios) ─────────────────────────────────────

function makeMemoryClient() {
  const client = new MemoryClient('http://localhost:8080');
  return client;
}

function mockAxiosOn(client: MemoryClient, method: 'post' | 'get', handler: (url: string, ...args: any[]) => any) {
  vi.spyOn((client as any).http, method).mockImplementation(handler);
}

describe('INVARIANT: set-get roundtrip', () => {
  it('set stores data that get retrieves', async () => {
    const client = makeMemoryClient();
    const store = new Map<string, any>();

    mockAxiosOn(client, 'post', async (url: string, payload: any) => {
      if (url === '/api/v1/memory/set') {
        store.set(payload.key, payload.data);
        return { data: { ok: true } };
      }
      if (url === '/api/v1/memory/get') {
        const val = store.get(payload.key);
        if (val === undefined) throw Object.assign(new Error('not found'), { response: { status: 404 } });
        return { data: { data: val } };
      }
    });

    await client.set('my-key', { answer: 42 });
    const result = await client.get('my-key');
    expect(result).toEqual({ answer: 42 });
  });

  it('set with scope stores and get with same scope retrieves', async () => {
    const client = makeMemoryClient();
    const store = new Map<string, any>();

    mockAxiosOn(client, 'post', async (url: string, payload: any) => {
      if (url === '/api/v1/memory/set') {
        store.set(`${payload.scope ?? 'none'}:${payload.key}`, payload.data);
        return { data: { ok: true } };
      }
      if (url === '/api/v1/memory/get') {
        const val = store.get(`${payload.scope ?? 'none'}:${payload.key}`);
        if (val === undefined) {
          throw Object.assign(new Error('not found'), { isAxiosError: true, response: { status: 404 } });
        }
        return { data: { data: val } };
      }
    });

    await client.set('scoped-key', 'hello', { scope: 'session' });
    const result = await client.get('scoped-key', { scope: 'session' });
    expect(result).toBe('hello');
  });
});

describe('INVARIANT: delete-get consistency', () => {
  it('delete followed by get returns undefined (404 → undefined)', async () => {
    const client = makeMemoryClient();
    const store = new Map<string, any>();
    store.set('my-key', 'initial');

    mockAxiosOn(client, 'post', async (url: string, payload: any) => {
      if (url === '/api/v1/memory/delete') {
        store.delete(payload.key);
        return { data: { ok: true } };
      }
      if (url === '/api/v1/memory/get') {
        if (!store.has(payload.key)) {
          throw Object.assign(new Error('not found'), { isAxiosError: true, response: { status: 404 } });
        }
        return { data: { data: store.get(payload.key) } };
      }
    });

    await client.delete('my-key');
    const result = await client.get('my-key');
    expect(result).toBeUndefined();
  });
});

describe('INVARIANT: key independence', () => {
  it('set(scope, "a", 1) does not affect get(scope, "b")', async () => {
    const client = makeMemoryClient();
    const store = new Map<string, any>();

    mockAxiosOn(client, 'post', async (url: string, payload: any) => {
      if (url === '/api/v1/memory/set') {
        store.set(payload.key, payload.data);
        return { data: { ok: true } };
      }
      if (url === '/api/v1/memory/get') {
        if (!store.has(payload.key)) {
          throw Object.assign(new Error('not found'), { isAxiosError: true, response: { status: 404 } });
        }
        return { data: { data: store.get(payload.key) } };
      }
    });

    await client.set('key-a', 'value-a');
    // key-b was never set
    const result = await client.get('key-b');
    expect(result).toBeUndefined();
  });

  it('updating key-a does not change key-b', async () => {
    const client = makeMemoryClient();
    const store = new Map<string, any>();
    store.set('key-b', 'original-b');

    mockAxiosOn(client, 'post', async (url: string, payload: any) => {
      if (url === '/api/v1/memory/set') {
        store.set(payload.key, payload.data);
        return { data: { ok: true } };
      }
      if (url === '/api/v1/memory/get') {
        if (!store.has(payload.key)) {
          throw Object.assign(new Error('not found'), { isAxiosError: true, response: { status: 404 } });
        }
        return { data: { data: store.get(payload.key) } };
      }
    });

    await client.set('key-a', 'new-value-a');
    const bResult = await client.get('key-b');
    expect(bResult).toBe('original-b');
  });
});

describe('INVARIANT: buildHeaders is pure — same inputs always produce same header keys', () => {
  it('same scope and metadata produce identical header keys on each call', () => {
    const client = new MemoryClientBase('http://localhost:8080');
    const options = {
      scope: 'workflow' as MemoryScope,
      scopeId: 'wf-123',
      metadata: { sessionId: 'sess-abc', actorId: 'user-1' }
    };

    const h1 = (client as any).buildHeaders(options);
    const h2 = (client as any).buildHeaders(options);

    expect(Object.keys(h1).sort()).toEqual(Object.keys(h2).sort());
    expect(h1).toEqual(h2);
  });
});
