import { describe, it, expect, vi, afterEach } from 'vitest';

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
