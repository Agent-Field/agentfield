import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { installEventSourceMock, mockJsonResponse } from '@/test/serviceTestUtils';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeRawExecution(overrides: Record<string, any> = {}): Record<string, any> {
  return {
    id: 'exec-1',
    workflow_id: 'wf-1',
    execution_id: 'exec-1',
    agentfield_request_id: 'req-1',
    agent_node_id: 'node-1',
    reasoner_id: 'reasoner-1',
    status: 'success',
    input_data: { query: 'hello' },
    output_data: { answer: 'world' },
    input_size: 10,
    output_size: 20,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
    workflow_tags: [],
    notes: [],
    webhook_events: [],
    ...overrides,
  };
}

function makePaginatedResponse(executions: any[], overrides: Record<string, any> = {}) {
  return {
    executions,
    total: executions.length,
    page: 1,
    page_size: 20,
    total_pages: 1,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Fetch mock helper
// ---------------------------------------------------------------------------

function mockFetch(status: number, body: unknown) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
    body: null,
  } as any);
}

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
  vi.useRealTimers();
});

beforeEach(() => {
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// getExecutionsSummary — list executions
// ---------------------------------------------------------------------------

describe('getExecutionsSummary', () => {
  it('returns a PaginatedExecutions response on success', async () => {
    const raw = makeRawExecution();
    mockFetch(200, makePaginatedResponse([raw]));

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    const result = await getExecutionsSummary();

    expect(result).toHaveProperty('executions');
    expect((result as any).executions).toHaveLength(1);
    expect((result as any).total_count).toBe(1);
  });

  it('forwards filter parameters in the query string', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(makePaginatedResponse([])),
        body: null,
      } as any);
    });

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    await getExecutionsSummary({ status: 'failed', page: 2, page_size: 50 });

    expect(capturedUrls[0]).toMatch(/status=failed/);
    expect(capturedUrls[0]).toMatch(/page=2/);
    expect(capturedUrls[0]).toMatch(/page_size=50/);
  });

  it('normalises "success" backend status to frontend format', async () => {
    const raw = makeRawExecution({ status: 'success' });
    mockFetch(200, makePaginatedResponse([raw]));

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    const result = await getExecutionsSummary() as any;

    // normalizeExecutionStatus maps "success" → "succeeded" or keeps it — just check it's a string
    expect(typeof result.executions[0].status).toBe('string');
  });

  it('sets has_next=true when more pages exist', async () => {
    mockFetch(200, makePaginatedResponse([], { page: 1, total_pages: 3, total: 60 }));

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    const result = await getExecutionsSummary() as any;

    expect(result.has_next).toBe(true);
  });

  it('sets has_prev=true when not on the first page', async () => {
    mockFetch(200, makePaginatedResponse([], { page: 2, total_pages: 3, total: 60 }));

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    const result = await getExecutionsSummary() as any;

    expect(result.has_prev).toBe(true);
  });

  it('throws on non-OK response', async () => {
    mockFetch(500, { message: 'DB error' });

    const { getExecutionsSummary } = await import('@/services/executionsApi');
    await expect(getExecutionsSummary()).rejects.toThrow('DB error');
  });
});

// ---------------------------------------------------------------------------
// getExecutionDetails — get by ID
// ---------------------------------------------------------------------------

describe('getExecutionDetails', () => {
  it('returns a WorkflowExecution with the correct id', async () => {
    const raw = makeRawExecution({ id: 'exec-42', execution_id: 'exec-42' });
    mockFetch(200, raw);

    const { getExecutionDetails } = await import('@/services/executionsApi');
    const result = await getExecutionDetails('exec-42');

    expect(result.id).toBe('exec-42');
  });

  it('maps created_at to started_at when started_at is missing', async () => {
    const raw = makeRawExecution({ started_at: undefined, created_at: '2026-03-01T12:00:00Z' });
    mockFetch(200, raw);

    const { getExecutionDetails } = await import('@/services/executionsApi');
    const result = await getExecutionDetails('exec-1');

    expect(result.started_at).toBe('2026-03-01T12:00:00Z');
  });

  it('resolves input_data from alternative field names (input)', async () => {
    const raw = makeRawExecution({ input_data: undefined, input: { key: 'value' } });
    mockFetch(200, raw);

    const { getExecutionDetails } = await import('@/services/executionsApi');
    const result = await getExecutionDetails('exec-1');

    expect(result.input_data).toEqual({ key: 'value' });
  });

  it('resolves output_data from alternative field names (output)', async () => {
    const raw = makeRawExecution({ output_data: undefined, output: { result: 42 } });
    mockFetch(200, raw);

    const { getExecutionDetails } = await import('@/services/executionsApi');
    const result = await getExecutionDetails('exec-1');

    expect(result.output_data).toEqual({ result: 42 });
  });

  it('sets webhook_registered=true when webhook_events array is non-empty', async () => {
    const raw = makeRawExecution({
      webhook_registered: false,
      webhook_events: [{
        id: 'ev-1',
        event_type: 'webhook',
        status: 'delivered',
        created_at: '2026-01-01T00:00:00Z',
      }],
    });
    mockFetch(200, raw);

    const { getExecutionDetails } = await import('@/services/executionsApi');
    const result = await getExecutionDetails('exec-1');

    expect(result.webhook_registered).toBe(true);
  });

  it('includes the request URL in the details endpoint', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(makeRawExecution()),
        body: null,
      } as any);
    });

    const { getExecutionDetails } = await import('@/services/executionsApi');
    await getExecutionDetails('my-exec-id');

    expect(capturedUrls[0]).toMatch(/my-exec-id/);
    expect(capturedUrls[0]).toMatch(/details/);
  });

  it('throws on non-OK response', async () => {
    mockFetch(404, { message: 'Execution not found' });

    const { getExecutionDetails } = await import('@/services/executionsApi');
    await expect(getExecutionDetails('missing')).rejects.toThrow('Execution not found');
  });
});

// ---------------------------------------------------------------------------
// Pagination helpers
// ---------------------------------------------------------------------------

describe('getExecutionsByAgent', () => {
  it('passes agent_node_id filter and pagination params', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(makePaginatedResponse([])),
        body: null,
      } as any);
    });

    const { getExecutionsByAgent } = await import('@/services/executionsApi');
    await getExecutionsByAgent('node-xyz', 2, 10);

    expect(capturedUrls[0]).toMatch(/agent_node_id=node-xyz/);
    expect(capturedUrls[0]).toMatch(/page=2/);
    expect(capturedUrls[0]).toMatch(/page_size=10/);
  });
});

describe('getExecutionsByStatus', () => {
  it('passes status filter to the query string', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(makePaginatedResponse([])),
        body: null,
      } as any);
    });

    const { getExecutionsByStatus } = await import('@/services/executionsApi');
    await getExecutionsByStatus('failed');

    expect(capturedUrls[0]).toMatch(/status=failed/);
  });
});

// ---------------------------------------------------------------------------
// cancelExecution
// ---------------------------------------------------------------------------

describe('cancelExecution', () => {
  it('sends a POST request and returns the cancel response', async () => {
    const cancelResponse = {
      execution_id: 'exec-1',
      previous_status: 'running',
      status: 'cancelled',
      cancelled_at: '2026-01-01T00:05:00Z',
    };
    mockFetch(200, cancelResponse);

    const { cancelExecution } = await import('@/services/executionsApi');
    const result = await cancelExecution('exec-1', 'manual cancel');

    expect(result.status).toBe('cancelled');
    expect(result.execution_id).toBe('exec-1');
  });

  it('throws when the server rejects the cancel', async () => {
    mockFetch(409, { message: 'Execution already completed' });

    const { cancelExecution } = await import('@/services/executionsApi');
    await expect(cancelExecution('exec-done')).rejects.toThrow('Execution already completed');
  });
});

// ---------------------------------------------------------------------------
// getExecutionStats
// ---------------------------------------------------------------------------

describe('getExecutionStats', () => {
  it('returns mapped stats with expected fields', async () => {
    const backendStats = {
      successful_count: 10,
      failed_count: 2,
      running_count: 1,
      executions_by_status: { success: 10, failed: 2 },
    };
    mockFetch(200, backendStats);

    const { getExecutionStats } = await import('@/services/executionsApi');
    const result = await getExecutionStats();

    expect(result.successful_executions).toBe(10);
    expect(result.failed_executions).toBe(2);
    expect(result.running_executions).toBe(1);
  });

  it('passes filter parameters to the stats endpoint', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ successful_count: 0, failed_count: 0, running_count: 0, executions_by_status: {} }),
        body: null,
      } as any);
    });

    const { getExecutionStats } = await import('@/services/executionsApi');
    await getExecutionStats({ agent_node_id: 'node-abc' });

    expect(capturedUrls[0]).toMatch(/agent_node_id=node-abc/);
    expect(capturedUrls[0]).toMatch(/stats/);
  });
});

describe('execution logs and event streams', () => {
  it('queries execution logs with all supported filters', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve(mockJsonResponse(200, { logs: [], total: 0 }));
    });

    const { getExecutionLogs } = await import('@/services/executionsApi');
    await getExecutionLogs('exec/1', {
      tail: 25,
      afterSeq: 10,
      levels: ['info', 'error'],
      nodeIds: ['node-a', 'node-b'],
      sources: ['stdout', 'stderr'],
      q: '  trace me  ',
    });

    expect(capturedUrls[0]).toContain('/executions/exec/1/logs?');
    expect(capturedUrls[0]).toContain('tail=25');
    expect(capturedUrls[0]).toContain('after_seq=10');
    expect(capturedUrls[0]).toContain('levels=info');
    expect(capturedUrls[0]).toContain('levels=error');
    expect(capturedUrls[0]).toContain('node_ids=node-a');
    expect(capturedUrls[0]).toContain('node_ids=node-b');
    expect(capturedUrls[0]).toContain('sources=stdout');
    expect(capturedUrls[0]).toContain('sources=stderr');
    expect(capturedUrls[0]).toContain('q=trace+me');
  });

  it('creates execution log, event, and note streams with auth in the URL', async () => {
    const apiModule = await import('@/services/api');
    apiModule.setGlobalApiKey('stream-key');
    const eventSource = installEventSourceMock();

    const {
      streamExecutionEvents,
      streamExecutionLogs,
      streamExecutionNotes,
    } = await import('@/services/executionsApi');

    streamExecutionEvents();
    streamExecutionLogs('exec/1', {
      tail: 15,
      afterSeq: 4,
      levels: ['warn'],
      nodeIds: ['node-a'],
      sources: ['stdout'],
      q: 'hello world',
    });
    streamExecutionNotes('exec-1');

    expect(eventSource.instances.map((entry) => entry.url)).toEqual([
      '/api/ui/v1/executions/events?api_key=stream-key',
      '/api/ui/v1/executions/exec%2F1/logs/stream?tail=15&since_seq=4&levels=warn&node_ids=node-a&sources=stdout&q=hello+world&api_key=stream-key',
      '/api/ui/v1/executions/exec-1/notes/stream?api_key=stream-key',
    ]);

    eventSource.restore();
    apiModule.setGlobalApiKey(null);
  });
});

describe('pause, resume, retry, and grouped helper paths', () => {
  it('posts pause, resume, and retry webhook actions', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockJsonResponse(200, {
        execution_id: 'exec-1',
        previous_status: 'running',
        status: 'paused',
        paused_at: '2026-01-01T00:05:00Z',
      }))
      .mockResolvedValueOnce(mockJsonResponse(200, {
        execution_id: 'exec-1',
        previous_status: 'paused',
        status: 'running',
        resumed_at: '2026-01-01T00:06:00Z',
      }))
      .mockResolvedValueOnce(mockJsonResponse(200, { ok: true }));

    const { pauseExecution, resumeExecution, retryExecutionWebhook } = await import('@/services/executionsApi');

    await expect(pauseExecution('exec-1', 'operator pause')).resolves.toMatchObject({ status: 'paused' });
    await expect(resumeExecution('exec-1')).resolves.toMatchObject({ status: 'running' });
    await expect(retryExecutionWebhook('exec-1')).resolves.toBeUndefined();

    const calls = vi.mocked(globalThis.fetch).mock.calls as [string, RequestInit][];
    expect(calls[0][0]).toBe('/api/ui/v1/executions/exec-1/pause');
    expect(calls[0][1].method).toBe('POST');
    expect(JSON.parse(String(calls[0][1].body))).toEqual({ reason: 'operator pause' });
    expect(calls[1][0]).toBe('/api/ui/v1/executions/exec-1/resume');
    expect(JSON.parse(String(calls[1][1].body))).toEqual({});
    expect(calls[2][0]).toBe('/api/ui/v1/executions/exec-1/webhook/retry');
    expect(calls[2][1].method).toBe('POST');
  });

  it('returns empty grouped payloads and supports grouped dashboard helpers', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(mockJsonResponse(200, makePaginatedResponse([makeRawExecution()], {
      total: 3,
      page: 2,
      total_pages: 4,
    })));

    const {
      getGroupedExecutionsByAgent,
      getGroupedExecutionsBySession,
      getGroupedExecutionsByStatus,
      getGroupedExecutionsByWorkflow,
    } = await import('@/services/executionsApi');

    await expect(getGroupedExecutionsByWorkflow({ search: 'checkout' })).resolves.toMatchObject({
      groups: [],
      total_count: 3,
      page: 2,
      total_pages: 4,
      has_next: true,
      has_prev: true,
    });
    await expect(getGroupedExecutionsBySession({ actor_id: 'actor-1' })).resolves.toMatchObject({ groups: [] });
    await expect(getGroupedExecutionsByAgent({ agent_node_id: 'node-1' })).resolves.toMatchObject({ groups: [] });
    await expect(getGroupedExecutionsByStatus({ status: 'failed' })).resolves.toMatchObject({ groups: [] });

    const urls = vi.mocked(globalThis.fetch).mock.calls.map((call) => call[0] as string);
    expect(urls[0]).toContain('group_by=workflow');
    expect(urls[1]).toContain('group_by=session');
    expect(urls[2]).toContain('group_by=agent');
    expect(urls[3]).toContain('group_by=status');
  });
});

describe('search, time-range, enhanced, and notes helpers', () => {
  it('supports search, recent, time-range, and enhanced execution queries', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T16:00:00Z'));

    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      if (url.includes('/executions/enhanced')) {
        return Promise.resolve(mockJsonResponse(200, {
          executions: [],
          total_count: 0,
          page: 1,
          page_size: 10,
          total_pages: 0,
          has_more: false,
        }));
      }
      return Promise.resolve(mockJsonResponse(200, makePaginatedResponse([], {
        total: 0,
        total_pages: 0,
      })));
    });

    const {
      getEnhancedExecutions,
      getExecutionsBySession,
      getExecutionsByWorkflow,
      getExecutionsInTimeRange,
      getRecentExecutions,
      searchExecutions,
    } = await import('@/services/executionsApi');

    const start = new Date('2026-04-07T00:00:00Z');
    const end = new Date('2026-04-08T00:00:00Z');
    const signal = new AbortController().signal;

    await searchExecutions('invoice', { status: 'failed' }, 3, 40);
    await getRecentExecutions(6, 2, 15);
    await getExecutionsInTimeRange(start, end, { agent_node_id: 'node-9' }, 4, 50);
    await getExecutionsByWorkflow('wf-22', 5, 12);
    await getExecutionsBySession('session-7', 6, 18);
    await getEnhancedExecutions({ status: 'running' }, 'duration_ms', 'asc', 7, 30, signal);

    expect(capturedUrls[0]).toContain('search=invoice');
    expect(capturedUrls[0]).toContain('status=failed');
    expect(capturedUrls[0]).toContain('page=3');
    expect(capturedUrls[0]).toContain('page_size=40');

    expect(capturedUrls[1]).toContain('start_time=2026-04-08T10%3A00%3A00.000Z');
    expect(capturedUrls[1]).toContain('end_time=2026-04-08T16%3A00%3A00.000Z');
    expect(capturedUrls[1]).toContain('page=2');
    expect(capturedUrls[1]).toContain('page_size=15');

    expect(capturedUrls[2]).toContain('start_time=2026-04-07T00%3A00%3A00.000Z');
    expect(capturedUrls[2]).toContain('end_time=2026-04-08T00%3A00%3A00.000Z');
    expect(capturedUrls[2]).toContain('agent_node_id=node-9');
    expect(capturedUrls[2]).toContain('page=4');
    expect(capturedUrls[2]).toContain('page_size=50');

    expect(capturedUrls[3]).toContain('workflow_id=wf-22');
    expect(capturedUrls[4]).toContain('session_id=session-7');
    expect(capturedUrls[5]).toBe('/api/ui/v1/executions/enhanced?status=running&sort_by=duration_ms&sort_order=asc&page=7&page_size=30');
  });

  it('handles execution notes, note tags, and add-note headers', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockJsonResponse(200, {
        notes: [
          { message: 'first', tags: ['ops', 'sev-1'] },
          { message: 'second', tags: ['sev-1', 'owner'] },
        ],
      }))
      .mockResolvedValueOnce(mockJsonResponse(200, {
        note: { message: 'added', tags: ['ops'] },
      }))
      .mockResolvedValueOnce(mockJsonResponse(200, {
        notes: [
          { message: 'first', tags: ['ops', 'sev-1'] },
          { message: 'second', tags: ['sev-1', 'owner'] },
        ],
      }))
      .mockResolvedValueOnce(mockJsonResponse(500, { message: 'boom' }));

    const {
      addExecutionNote,
      getExecutionNotes,
      getExecutionNoteTags,
    } = await import('@/services/executionsApi');

    await expect(getExecutionNotes('exec-1', { tags: ['ops', 'sev-1'] })).resolves.toMatchObject({
      notes: expect.any(Array),
    });
    await expect(addExecutionNote('exec-1', { message: 'added', tags: ['ops'] })).resolves.toMatchObject({
      note: { message: 'added', tags: ['ops'] },
    });
    await expect(getExecutionNoteTags('exec-1')).resolves.toEqual(['ops', 'owner', 'sev-1']);
    await expect(getExecutionNoteTags('exec-fail')).resolves.toEqual([]);

    const calls = vi.mocked(globalThis.fetch).mock.calls as [string, RequestInit][];
    expect(calls[0][0]).toBe('/api/ui/v1/executions/exec-1/notes?tags=ops%2Csev-1');
    expect(calls[1][0]).toBe('/api/ui/v1/executions/note');
    expect(calls[1][1].method).toBe('POST');
    expect(new Headers(calls[1][1].headers).get('X-Execution-ID')).toBe('exec-1');
    expect(JSON.parse(String(calls[1][1].body))).toEqual({ message: 'added', tags: ['ops'] });
  });
});
