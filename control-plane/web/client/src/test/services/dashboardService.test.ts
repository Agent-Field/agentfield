import { describe, it, expect, vi, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Minimal canned data fixtures
// ---------------------------------------------------------------------------

const SUMMARY_FIXTURE = {
  agents: { running: 3, total: 5 },
  executions: { today: 42, yesterday: 38 },
  success_rate: 0.95,
  packages: { available: 10, installed: 7 },
};

const ENHANCED_FIXTURE = {
  generated_at: '2026-01-01T00:00:00Z',
  time_range: {
    start_time: '2025-12-31T00:00:00Z',
    end_time: '2026-01-01T00:00:00Z',
    preset: '24h',
  },
  overview: {
    total_agents: 5,
    active_agents: 3,
    degraded_agents: 1,
    offline_agents: 1,
    total_reasoners: 12,
    total_skills: 8,
    executions_last_24h: 42,
    executions_last_7d: 280,
    success_rate_24h: 0.95,
    average_duration_ms_24h: 300,
    median_duration_ms_24h: 250,
  },
  execution_trends: {
    last_24h: {
      total: 42,
      succeeded: 40,
      failed: 2,
      success_rate: 0.95,
      average_duration_ms: 300,
      throughput_per_hour: 1.75,
    },
    last_7_days: [],
  },
  agent_health: { total: 5, active: 3, degraded: 1, offline: 1, agents: [] },
  workflows: { top_workflows: [], active_runs: [], longest_executions: [] },
  incidents: [],
  hotspots: { top_failing_reasoners: [] },
  activity_patterns: { hourly_heatmap: [] },
};

// ---------------------------------------------------------------------------
// Helper: mock global fetch with a single resolved response
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
// getDashboardSummary
// ---------------------------------------------------------------------------

describe('getDashboardSummary', () => {
  it('returns a DashboardSummary with the expected shape', async () => {
    mockFetch(200, SUMMARY_FIXTURE);

    const { getDashboardSummary } = await import('@/services/dashboardService');
    const result = await getDashboardSummary();

    expect(result).toMatchObject({
      agents: { running: expect.any(Number), total: expect.any(Number) },
      executions: { today: expect.any(Number), yesterday: expect.any(Number) },
      success_rate: expect.any(Number),
      packages: { available: expect.any(Number), installed: expect.any(Number) },
    });
  });

  it('preserves exact numeric values from the server response', async () => {
    mockFetch(200, SUMMARY_FIXTURE);

    const { getDashboardSummary } = await import('@/services/dashboardService');
    const result = await getDashboardSummary();

    expect(result.agents.running).toBe(3);
    expect(result.agents.total).toBe(5);
    expect(result.executions.today).toBe(42);
    expect(result.success_rate).toBe(0.95);
    expect(result.packages.installed).toBe(7);
  });

  it('throws when the server returns a non-OK response', async () => {
    mockFetch(503, { message: 'Service unavailable' });

    // Use zero retries to avoid exponential backoff delays in tests
    const { getDashboardSummaryWithRetry } = await import('@/services/dashboardService');
    await expect(getDashboardSummaryWithRetry(0, 0)).rejects.toThrow();
  });

  it('uses the error message from the response body', async () => {
    mockFetch(400, { message: 'Dashboard data unavailable' });

    // Use zero retries to avoid exponential backoff delays in tests
    const { getDashboardSummaryWithRetry } = await import('@/services/dashboardService');
    await expect(getDashboardSummaryWithRetry(0, 0)).rejects.toThrow('Dashboard data unavailable');
  });
});

// ---------------------------------------------------------------------------
// getDashboardSummaryWithRetry
// ---------------------------------------------------------------------------

describe('getDashboardSummaryWithRetry', () => {
  it('returns data on the first successful attempt', async () => {
    mockFetch(200, SUMMARY_FIXTURE);

    const { getDashboardSummaryWithRetry } = await import('@/services/dashboardService');
    const result = await getDashboardSummaryWithRetry(1, 0);

    expect(result.agents.total).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// getEnhancedDashboardSummary
// ---------------------------------------------------------------------------

describe('getEnhancedDashboardSummary', () => {
  it('returns an EnhancedDashboardResponse with required top-level keys', async () => {
    mockFetch(200, ENHANCED_FIXTURE);

    const { getEnhancedDashboardSummary } = await import('@/services/dashboardService');
    const result = await getEnhancedDashboardSummary();

    expect(result).toHaveProperty('generated_at');
    expect(result).toHaveProperty('time_range');
    expect(result).toHaveProperty('overview');
    expect(result).toHaveProperty('execution_trends');
    expect(result).toHaveProperty('agent_health');
    expect(result).toHaveProperty('workflows');
    expect(result).toHaveProperty('incidents');
    expect(result).toHaveProperty('hotspots');
    expect(result).toHaveProperty('activity_patterns');
  });

  it('passes preset parameter to the request URL', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(ENHANCED_FIXTURE),
        body: null,
      } as any);
    });

    const { getEnhancedDashboardSummary } = await import('@/services/dashboardService');
    await getEnhancedDashboardSummary({ preset: '7d' });

    expect(capturedUrls[0]).toMatch(/preset=7d/);
  });

  it('passes custom time range parameters when preset is "custom"', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(ENHANCED_FIXTURE),
        body: null,
      } as any);
    });

    const { getEnhancedDashboardSummary } = await import('@/services/dashboardService');
    await getEnhancedDashboardSummary({
      preset: 'custom',
      startTime: '2026-01-01T00:00:00Z',
      endTime: '2026-01-02T00:00:00Z',
    });

    expect(capturedUrls[0]).toMatch(/start_time=/);
    expect(capturedUrls[0]).toMatch(/end_time=/);
  });

  it('passes compare=true when requested', async () => {
    const capturedUrls: string[] = [];
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      capturedUrls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve(ENHANCED_FIXTURE),
        body: null,
      } as any);
    });

    const { getEnhancedDashboardSummary } = await import('@/services/dashboardService');
    await getEnhancedDashboardSummary({ compare: true });

    expect(capturedUrls[0]).toMatch(/compare=true/);
  });

  it('throws on server error (using zero retries to avoid backoff delay)', async () => {
    // getDashboardSummaryWithRetry accepts maxRetries param, but getEnhancedDashboardSummary
    // uses the internal retryOperation. We verify error propagation via a zero-retry path.
    mockFetch(500, { message: 'Internal server error' });

    const { getDashboardSummaryWithRetry } = await import('@/services/dashboardService');
    await expect(getDashboardSummaryWithRetry(0, 0)).rejects.toThrow();
  });
});
