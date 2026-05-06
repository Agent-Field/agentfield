import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  cancelWorkflowTree,
  deleteWorkflow,
  getExecutionViewStats,
  getFilterOptions,
  getWorkflowDetails,
  getWorkflowRunDetail,
  searchExecutionData,
} from "@/services/workflowsApi";

function mockResponse(status: number, body: unknown, statusText = "OK") {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

describe("workflowsApi extras", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T12:00:00Z"));
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("fetches details, filter options, and encoded view stats", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockResponse(200, { workflow_id: "wf/1", name: "Primary" }))
      .mockResolvedValueOnce(
        mockResponse(200, {
          agents: [],
          workflows: ["wf/1"],
          sessions: [],
          statuses: ["running"],
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          total_count: 3,
          status_breakdown: { running: 2, failed: 1 },
          recent_activity: 1,
        })
      );

    await expect(getWorkflowDetails("wf/1")).resolves.toMatchObject({ workflow_id: "wf/1" });
    await expect(getFilterOptions()).resolves.toMatchObject({ statuses: ["running"] });
    await expect(
      getExecutionViewStats("workflows", {
        search: "hello world/ops",
        status: "running",
        workflow: "wf/1",
      } as any)
    ).resolves.toMatchObject({ total_count: 3 });

    expect(vi.mocked(globalThis.fetch).mock.calls[0]?.[0]).toBe(
      "/api/ui/v1/workflows/wf/1/details"
    );
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe(
      "/api/ui/v1/executions/filter-options"
    );

    const statsUrl = new URL(
      vi.mocked(globalThis.fetch).mock.calls[2]?.[0] as string,
      "http://localhost"
    );
    expect(statsUrl.pathname).toBe("/api/ui/v1/executions/view-stats");
    expect(statsUrl.searchParams.get("view_mode")).toBe("workflows");
    expect(statsUrl.searchParams.get("search")).toBe("hello world/ops");
    expect(statsUrl.searchParams.get("workflow")).toBe("wf/1");
  });

  it("fetches workflow run detail and surfaces JSON error messages", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          run: {
            run_id: "run-1",
            root_workflow_id: "wf-1",
            status: "running",
            total_steps: 3,
            completed_steps: 1,
            failed_steps: 0,
            created_at: "2026-04-08T10:00:00Z",
            updated_at: "2026-04-08T10:05:00Z",
          },
          executions: [],
        })
      )
      .mockResolvedValueOnce(mockResponse(404, { message: "workflow run missing" }, "Not Found"));

    await expect(getWorkflowRunDetail("run-1")).resolves.toMatchObject({
      run: { run_id: "run-1" },
    });
    await expect(getWorkflowRunDetail("run-missing")).rejects.toThrow("workflow run missing");
  });

  it("routes search requests through workflow summaries and handles delete variants", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          runs: [],
          total_count: 0,
          page: 1,
          page_size: 20,
          has_more: false,
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          workflow_id: "wf-1",
          dry_run: true,
          deleted_records: { executions: 2 },
          freed_space_bytes: 64,
          duration_ms: 5,
          success: true,
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          workflow_id: "wf-2",
          dry_run: false,
          deleted_records: { executions: 2 },
          freed_space_bytes: 64,
          duration_ms: 5,
          success: true,
        })
      );

    await expect(
      searchExecutionData("triage/ops", "workflows", { status: "all" }, 1, 20)
    ).resolves.toMatchObject({ total_count: 0 });
    await expect(deleteWorkflow("wf-1", true)).resolves.toMatchObject({ dry_run: true });
    await expect(deleteWorkflow("wf-2", false)).resolves.toMatchObject({ dry_run: false });

    const searchUrl = new URL(
      vi.mocked(globalThis.fetch).mock.calls[0]?.[0] as string,
      "http://localhost"
    );
    expect(searchUrl.pathname).toBe("/api/ui/v2/workflow-runs");
    expect(searchUrl.searchParams.get("search")).toBe("triage/ops");
    expect(searchUrl.searchParams.get("status")).toBeNull();

    const [dryRunUrl, dryRunInit] = vi.mocked(globalThis.fetch).mock.calls[1] as [string, RequestInit];
    expect(dryRunUrl).toBe("/api/ui/v1/workflows/wf-1/cleanup?dry_run=true&confirm=true");
    expect(dryRunInit.method).toBe("DELETE");

    const [deleteUrl, deleteInit] = vi.mocked(globalThis.fetch).mock.calls[2] as [string, RequestInit];
    expect(deleteUrl).toBe("/api/ui/v1/workflows/wf-2/cleanup?confirm=true");
    expect(deleteInit.method).toBe("DELETE");
  });

  it("posts to cancel-tree with the supplied reason", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          run_id: "run-1",
          total_nodes: 3,
          cancelled_count: 2,
          skipped_count: 1,
          error_count: 0,
          nodes: [],
          cancelled_at: "2026-04-08T12:00:00Z",
        })
      );
    globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;

    await expect(cancelWorkflowTree("run-1", "user clicked cancel")).resolves.toMatchObject({
      run_id: "run-1",
      cancelled_count: 2,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/ui/v1/workflows/run-1/cancel-tree");
    expect(init.method).toBe("POST");
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ reason: "user clicked cancel" }));
  });

  it("defaults the reason to empty string when omitted", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          run_id: "run-2",
          total_nodes: 0,
          cancelled_count: 0,
          skipped_count: 0,
          error_count: 0,
          nodes: [],
          cancelled_at: "2026-04-08T12:00:00Z",
        })
      );
    globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;

    await cancelWorkflowTree("run-2");
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.body).toBe(JSON.stringify({ reason: "" }));
  });
});
