import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import {
  ReasonersApiError,
  reasonersApi,
} from "@/services/reasonersApi";

function mockResponse(status: number, body: unknown, statusText = "OK") {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(
      typeof body === "string" ? body : JSON.stringify(body)
    ),
  } as unknown as Response;
}

describe("reasonersApi", () => {
  const originalFetch = globalThis.fetch;
  const originalEventSource = globalThis.EventSource;

  beforeEach(() => {
    setGlobalApiKey(null);
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    globalThis.EventSource = originalEventSource;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it("fetches all reasoners with filters, auth headers, and payload validation", async () => {
    setGlobalApiKey("secret");

    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponse(200, {
        reasoners: "bad-shape",
        total: "bad",
        online_count: 2,
      })
    );

    const result = await reasonersApi.getAllReasoners({
      status: "online",
      search: "planner",
      limit: 10,
      offset: 20,
    });

    expect(result).toEqual({
      reasoners: [],
      total: 0,
      online_count: 2,
      offline_count: 0,
      nodes_count: 0,
    });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    const headers = new Headers(init.headers);
    expect(url).toContain("/api/ui/v1/reasoners/all?");
    expect(url).toContain("status=online");
    expect(url).toContain("search=planner");
    expect(url).toContain("limit=10");
    expect(url).toContain("offset=20");
    expect(headers.get("X-API-Key")).toBe("secret");
  });

  it("wraps network errors when fetching all reasoners", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("socket hang up"));

    await expect(reasonersApi.getAllReasoners()).rejects.toMatchObject({
      name: "ReasonersApiError",
      message: "Network error: socket hang up",
    });
  });

  it("handles reasoner details and specialized 404 errors", async () => {
    globalThis.fetch = vi.fn().mockResolvedValueOnce(
      mockResponse(200, {
        reasoner_id: "node.main",
        name: "Main",
        description: "desc",
        node_id: "node",
        node_status: "active",
        node_version: "1.0.0",
        input_schema: {},
        output_schema: {},
        memory_config: {
          auto_inject: [],
          memory_retention: "1h",
          cache_results: false,
        },
        last_updated: "2026-04-07T11:00:00Z",
      })
    );
    await expect(reasonersApi.getReasonerDetails("node.main")).resolves.toMatchObject({
      reasoner_id: "node.main",
    });

    globalThis.fetch = vi.fn().mockResolvedValueOnce(
      mockResponse(404, { message: "missing" }, "Not Found")
    );
    await expect(reasonersApi.getReasonerDetails("missing")).rejects.toEqual(
      new ReasonersApiError("Reasoner not found", 404)
    );
  });

  it("posts execute requests and validates malformed sync and async responses", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          execution_id: "exec-1",
          result: { ok: true },
          duration_ms: 20,
          status: "succeeded",
          timestamp: "2026-04-07T11:00:00Z",
          node_id: "node-1",
          type: "reasoner",
          target: "agent",
          workflow_id: "wf-1",
        })
      )
      .mockResolvedValueOnce(mockResponse(200, null))
      .mockResolvedValueOnce(
        mockResponse(200, {
          execution_id: "",
        })
      );

    await expect(
      reasonersApi.executeReasoner("core", { input: { prompt: "hello" } })
    ).resolves.toMatchObject({
      execution_id: "exec-1",
      result: { ok: true },
    });

    const [, init] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ input: { prompt: "hello" } }));
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");

    await expect(
      reasonersApi.executeReasoner("core", { input: { prompt: "bad" } })
    ).rejects.toThrow("Invalid response format from server");
    await expect(
      reasonersApi.executeReasonerAsync("core", { input: { prompt: "bad" } })
    ).rejects.toThrow("Invalid async response format from server");
  });

  it("handles execution status and helper fetch endpoints", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          execution_id: "exec-1",
          workflow_id: "wf-1",
          status: "running",
          target: "agent",
          type: "reasoner",
          progress: 10,
          started_at: "2026-04-07T11:00:00Z",
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          avg_response_time_ms: 12,
          success_rate: 0.9,
          total_executions: 4,
          executions_last_24h: 2,
          error_rate: 0.1,
          recent_executions: [],
          performance_trend: [],
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          executions: [],
          total: 0,
          page: 1,
          limit: 20,
        })
      )
      .mockResolvedValueOnce(mockResponse(200, []))
      .mockResolvedValueOnce(
        mockResponse(200, {
          id: "template-1",
          name: "Quickstart",
          input: {},
          created_at: "2026-04-07T11:00:00Z",
        })
      );

    await expect(reasonersApi.getExecutionStatus("exec-1")).resolves.toMatchObject({
      execution_id: "exec-1",
    });
    await expect(reasonersApi.getPerformanceMetrics("core")).resolves.toMatchObject({
      total_executions: 4,
    });
    await expect(reasonersApi.getExecutionHistory("core", 2, 50)).resolves.toMatchObject({
      page: 1,
    });
    await expect(reasonersApi.getExecutionTemplates("core")).resolves.toEqual([]);
    await expect(
      reasonersApi.saveExecutionTemplate("core", {
        name: "Quickstart",
        input: {},
      })
    ).resolves.toMatchObject({ id: "template-1" });

    const historyUrl = vi.mocked(globalThis.fetch).mock.calls[2]?.[0] as string;
    expect(historyUrl).toContain("page=2");
    expect(historyUrl).toContain("limit=50");
  });

  it("surfaces detailed errors for metrics, history, templates, and save-template calls", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockResponse(500, { message: 'ignored' }, 'Bad Gateway'))
      .mockResolvedValueOnce(mockResponse(503, { message: 'ignored' }, 'Service Unavailable'))
      .mockResolvedValueOnce(mockResponse(404, { message: 'ignored' }, 'Missing'))
      .mockResolvedValueOnce(mockResponse(422, { message: 'ignored' }, 'Unprocessable Entity'));

    const checks = [
      [() => reasonersApi.getPerformanceMetrics('core'), 'Failed to fetch performance metrics: Bad Gateway', 500],
      [() => reasonersApi.getExecutionHistory('core'), 'Failed to fetch execution history: Service Unavailable', 503],
      [() => reasonersApi.getExecutionTemplates('core'), 'Failed to fetch execution templates: Missing', 404],
      [() => reasonersApi.saveExecutionTemplate('core', { name: 'Bad', input: {} }), 'Failed to save execution template: Unprocessable Entity', 422],
    ] as const;

    for (const [run, message, status] of checks) {
      try {
        await run();
        throw new Error('expected helper to reject');
      } catch (error) {
        expect(error).toBeInstanceOf(ReasonersApiError);
        expect((error as ReasonersApiError).message).toBe(message);
        expect((error as ReasonersApiError).status).toBe(status);
      }
    }
  });

  it("wraps generic network failures for downstream reasoner helpers", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('network down'));

    await expect(reasonersApi.getPerformanceMetrics('core')).rejects.toMatchObject({
      name: 'ReasonersApiError',
      message: 'Network error: network down',
    });
    await expect(reasonersApi.getExecutionHistory('core')).rejects.toMatchObject({
      name: 'ReasonersApiError',
      message: 'Network error: network down',
    });
    await expect(reasonersApi.getExecutionTemplates('core')).rejects.toMatchObject({
      name: 'ReasonersApiError',
      message: 'Network error: network down',
    });
    await expect(
      reasonersApi.saveExecutionTemplate('core', { name: 'Retry', input: {} })
    ).rejects.toMatchObject({
      name: 'ReasonersApiError',
      message: 'Network error: network down',
    });
  });

  it("creates and closes event streams with callback handling", () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      url: string;
      readyState = 1;
      closed = false;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockEventSource.instances.push(this);
      }

      close() {
        this.closed = true;
      }
    }

    globalThis.EventSource = MockEventSource as unknown as typeof EventSource;
    setGlobalApiKey("secret");
    const onEvent = vi.fn();
    const onError = vi.fn();
    const onConnect = vi.fn();

    const stream = reasonersApi.createEventStream(onEvent, onError, onConnect);
    const instance = MockEventSource.instances[0];

    expect(instance.url).toContain("api_key=secret");

    instance.onopen?.();
    instance.onmessage?.({ data: JSON.stringify({ ok: true }) } as MessageEvent);
    instance.onmessage?.({ data: "{bad-json" } as MessageEvent);
    instance.onerror?.(new Event("error"));

    expect(onConnect).toHaveBeenCalled();
    expect(onEvent).toHaveBeenCalledWith({ ok: true });
    expect(onError).toHaveBeenNthCalledWith(1, new Error("Failed to parse event data"));
    expect(onError).toHaveBeenNthCalledWith(
      2,
      new Error("SSE connection error - readyState: 1")
    );

    reasonersApi.closeEventStream(stream);
    expect(instance.closed).toBe(true);
  });

  it("creates event streams without an API key and keeps custom errors intact", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      url: string;
      readyState = 0;
      closed = false;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockEventSource.instances.push(this);
      }

      close() {
        this.closed = true;
      }
    }

    globalThis.EventSource = MockEventSource as unknown as typeof EventSource;
    const onError = vi.fn();
    const customError = new ReasonersApiError('custom', 418);

    const stream = reasonersApi.createEventStream(vi.fn(), onError);
    const instance = MockEventSource.instances[0];

    expect(instance.url).toBe('/api/ui/v1/reasoners/events');
    instance.onerror?.(new Event('error'));
    expect(onError).toHaveBeenCalledWith(new Error('SSE connection error - readyState: 0'));

    globalThis.fetch = vi.fn().mockRejectedValue(customError);
    await expect(reasonersApi.getReasonerDetails('core')).rejects.toBe(customError);

    reasonersApi.closeEventStream(stream);
    expect(instance.closed).toBe(true);
  });
});
