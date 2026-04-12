import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { MemoryClient, MemoryClientBase } from '../src/memory/MemoryClient.js';

// ---------------------------------------------------------------------------
// Module-level axios mock (mirrors the pattern used in memory_and_discovery.test.ts)
// ---------------------------------------------------------------------------

vi.mock('axios', () => {
  const create = vi.fn(() => ({
    post: vi.fn(),
    get: vi.fn()
  }));

  const isAxiosError = (err: any) => Boolean(err?.isAxiosError);

  return {
    default: { create, isAxiosError },
    create,
    isAxiosError
  };
});

/** Returns the most-recently created axios instance */
function getHttpMock() {
  const mockCreate = (axios as any).create as ReturnType<typeof vi.fn>;
  const last = mockCreate.mock.results.at(-1);
  return last?.value as { post: ReturnType<typeof vi.fn>; get: ReturnType<typeof vi.fn> };
}

/** Produce a minimal axios-like error with a response status */
function axiosError(status: number) {
  const err: any = new Error(`Request failed with status ${status}`);
  err.isAxiosError = true;
  err.response = { status, data: {} };
  return err;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MemoryClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // set
  // -------------------------------------------------------------------------
  describe('set()', () => {
    it('POSTs to /api/v1/memory/set with key and data', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('my-key', { value: 42 });

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/set',
        expect.objectContaining({ key: 'my-key', data: { value: 42 } }),
        expect.any(Object)
      );
    });

    it('includes scope in payload when provided', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', { scope: 'session' });

      const [, body] = http.post.mock.calls[0];
      expect(body.scope).toBe('session');
    });

    it('does not include scope in payload when omitted', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v');

      const [, body] = http.post.mock.calls[0];
      expect(body.scope).toBeUndefined();
    });

    it('propagates errors from HTTP layer', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockRejectedValue(new Error('network error'));

      await expect(client.set('k', 'v')).rejects.toThrow('network error');
    });
  });

  // -------------------------------------------------------------------------
  // get
  // -------------------------------------------------------------------------
  describe('get()', () => {
    it('returns data on success', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { data: { answer: 42 } } });

      const result = await client.get('my-key');
      expect(result).toEqual({ answer: 42 });
    });

    it('POSTs to /api/v1/memory/get', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { data: 'ok' } });

      await client.get('target-key', { scope: 'workflow' });

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/get',
        expect.objectContaining({ key: 'target-key', scope: 'workflow' }),
        expect.any(Object)
      );
    });

    it('returns undefined when server responds 404', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockRejectedValue(axiosError(404));

      const result = await client.get('missing-key');
      expect(result).toBeUndefined();
    });

    it('re-throws non-404 errors', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockRejectedValue(axiosError(500));

      await expect(client.get('k')).rejects.toMatchObject({ response: { status: 500 } });
    });

    it('re-throws non-axios errors', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockRejectedValue(new Error('unexpected'));

      await expect(client.get('k')).rejects.toThrow('unexpected');
    });
  });

  // -------------------------------------------------------------------------
  // delete
  // -------------------------------------------------------------------------
  describe('delete()', () => {
    it('POSTs to /api/v1/memory/delete with key', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.delete('del-key');

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/delete',
        expect.objectContaining({ key: 'del-key' }),
        expect.any(Object)
      );
    });

    it('includes scope when provided', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.delete('k', { scope: 'actor' });

      const [, body] = http.post.mock.calls[0];
      expect(body.scope).toBe('actor');
    });
  });

  // -------------------------------------------------------------------------
  // listKeys
  // -------------------------------------------------------------------------
  describe('listKeys()', () => {
    it('GETs /api/v1/memory/list and extracts key strings', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.get.mockResolvedValue({ data: [{ key: 'a' }, { key: 'b' }] });

      const keys = await client.listKeys('global');
      expect(keys).toEqual(['a', 'b']);
    });

    it('returns empty array when response is null', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.get.mockResolvedValue({ data: null });

      const keys = await client.listKeys('session');
      expect(keys).toEqual([]);
    });

    it('filters out items without a key', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.get.mockResolvedValue({ data: [{ key: 'x' }, {}, null] });

      const keys = await client.listKeys('workflow');
      expect(keys).toEqual(['x']);
    });
  });

  // -------------------------------------------------------------------------
  // exists
  // -------------------------------------------------------------------------
  describe('exists()', () => {
    it('returns true when get() returns a value', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { data: 'some-value' } });

      expect(await client.exists('k')).toBe(true);
    });

    it('returns false when get() returns undefined (404)', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockRejectedValue(axiosError(404));

      expect(await client.exists('k')).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // Scope → header resolution
  // -------------------------------------------------------------------------
  describe('scope header resolution', () => {
    it('adds X-Workflow-ID header for workflow scope', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', { scope: 'workflow', metadata: { workflowId: 'wf-99' } });

      const [, , config] = http.post.mock.calls[0];
      expect(config.headers['X-Workflow-ID']).toBe('wf-99');
    });

    it('adds X-Session-ID header for session scope', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', { scope: 'session', metadata: { sessionId: 'sess-7' } });

      const [, , config] = http.post.mock.calls[0];
      expect(config.headers['X-Session-ID']).toBe('sess-7');
    });

    it('adds X-Actor-ID header for actor scope', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', { scope: 'actor', metadata: { actorId: 'actor-3' } });

      const [, , config] = http.post.mock.calls[0];
      expect(config.headers['X-Actor-ID']).toBe('actor-3');
    });

    it('uses explicit scopeId over metadata-derived value', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', {
        scope: 'workflow',
        scopeId: 'explicit-id',
        metadata: { workflowId: 'meta-id' }
      });

      const [, , config] = http.post.mock.calls[0];
      expect(config.headers['X-Workflow-ID']).toBe('explicit-id');
    });

    it('populates X-Execution-ID from metadata', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.set('k', 'v', { metadata: { executionId: 'exec-55' } });

      const [, , config] = http.post.mock.calls[0];
      expect(config.headers['X-Execution-ID']).toBe('exec-55');
    });
  });

  // -------------------------------------------------------------------------
  // Vector methods
  // -------------------------------------------------------------------------
  describe('setVector()', () => {
    it('POSTs to /api/v1/memory/vector/set with embedding', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.setVector('vec-key', [0.1, 0.2, 0.3]);

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/vector/set',
        expect.objectContaining({ key: 'vec-key', embedding: [0.1, 0.2, 0.3] }),
        expect.any(Object)
      );
    });

    it('includes metadata when provided', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.setVector('k', [1, 2], { label: 'test' });

      const [, body] = http.post.mock.calls[0];
      expect(body.metadata).toEqual({ label: 'test' });
    });
  });

  describe('deleteVector()', () => {
    it('POSTs to /api/v1/memory/vector/delete', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      await client.deleteVector('vec-key');

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/vector/delete',
        expect.objectContaining({ key: 'vec-key' }),
        expect.any(Object)
      );
    });
  });

  describe('searchVector()', () => {
    it('POSTs to /api/v1/memory/vector/search and returns results', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      const mockResults = [{ key: 'k1', score: 0.9, scope: 'global', scopeId: 'global' }];
      http.post.mockResolvedValue({ data: mockResults });

      const results = await client.searchVector([0.1, 0.2], { topK: 5 });

      expect(http.post).toHaveBeenCalledWith(
        '/api/v1/memory/vector/search',
        expect.objectContaining({ query_embedding: [0.1, 0.2], top_k: 5 }),
        expect.any(Object)
      );
      expect(results).toEqual(mockResults);
    });

    it('defaults topK to 10', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: [] });

      await client.searchVector([0.1]);

      const [, body] = http.post.mock.calls[0];
      expect(body.top_k).toBe(10);
    });

    it('returns empty array when response data is null', async () => {
      const client = new MemoryClient('http://localhost:8080');
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: null });

      const results = await client.searchVector([0.1]);
      expect(results).toEqual([]);
    });
  });
});

// ---------------------------------------------------------------------------
// MemoryClientBase – buildHeaders (white-box unit tests via subclass)
// ---------------------------------------------------------------------------
describe('MemoryClientBase – buildHeaders', () => {
  class TestBase extends MemoryClientBase {
    headers(options = {}) {
      return this.buildHeaders(options);
    }
  }

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns an object with no required headers by default', () => {
    const base = new TestBase('http://localhost:8080');
    const h = base.headers({});
    expect(typeof h).toBe('object');
  });

  it('maps runId to X-Workflow-ID as fallback', () => {
    const base = new TestBase('http://localhost:8080');
    const h = base.headers({ metadata: { runId: 'run-fallback' } });
    expect(h['X-Workflow-ID']).toBe('run-fallback');
  });

  it('workflowId takes precedence over runId for X-Workflow-ID', () => {
    const base = new TestBase('http://localhost:8080');
    const h = base.headers({ metadata: { workflowId: 'wf-main', runId: 'run-fallback' } });
    expect(h['X-Workflow-ID']).toBe('wf-main');
  });
});
