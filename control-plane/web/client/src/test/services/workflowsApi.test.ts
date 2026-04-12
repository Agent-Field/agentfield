import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import {
  deleteWorkflows,
  getEnhancedExecutions,
  getExecutionsByViewMode,
  getWorkflowDAGLightweight,
  getWorkflowRunSummary,
  getWorkflowsSummary,
  mapWorkflowSortKeyToApi,
} from "@/services/workflowsApi";

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

describe("workflowsApi", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-07T12:00:00Z"));
    setGlobalApiKey(null);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("maps workflow sort keys", () => {
    expect(mapWorkflowSortKeyToApi("status")).toBe("status");
    expect(mapWorkflowSortKeyToApi("nodes")).toBe("total_steps");
    expect(mapWorkflowSortKeyToApi("issues")).toBe("failed_steps");
    expect(mapWorkflowSortKeyToApi("started_at")).toBe("created_at");
    expect(mapWorkflowSortKeyToApi("unknown")).toBe("updated_at");
  });

  it("fetches workflow summaries with normalized filters and mapped runs", async () => {
    setGlobalApiKey("secret");
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponse(200, {
        runs: [
          {
            run_id: "run-1",
            workflow_id: "wf-1",
            root_execution_id: "exec-1",
            status: "success",
            display_name: "Main workflow",
            current_task: "",
            root_reasoner: "planner",
            agent_id: "node-1",
            session_id: "session-1",
            actor_id: "actor-1",
            total_executions: 4,
            max_depth: 2,
            active_executions: 1,
            status_counts: { success: 4 },
            started_at: "2026-04-07T10:00:00Z",
            updated_at: "2026-04-07T11:00:00Z",
            latest_activity: "",
            completed_at: null,
            duration_ms: 1200,
            terminal: false,
          },
        ],
        total_count: 11,
        page: 2,
        page_size: 5,
        has_more: true,
      })
    );

    const result = await getWorkflowsSummary(
      {
        status: "success",
        workflow: "wf-1",
        session: "session-1",
        timeRange: "24h",
        search: "planner",
      },
      2,
      5,
      "nodes",
      "asc"
    );

    expect(result).toMatchObject({
      total_count: 11,
      page: 2,
      page_size: 5,
      total_pages: 3,
      has_more: true,
    });
    expect(result.workflows[0]).toMatchObject({
      run_id: "run-1",
      status: "succeeded",
      current_task: "planner",
      latest_activity: "2026-04-07T11:00:00Z",
      agent_name: "node-1",
    });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    const parsed = new URL(url, "http://localhost");
    expect(parsed.pathname).toBe("/api/ui/v2/workflow-runs");
    expect(parsed.searchParams.get("page")).toBe("2");
    expect(parsed.searchParams.get("page_size")).toBe("5");
    expect(parsed.searchParams.get("sort_by")).toBe("total_steps");
    expect(parsed.searchParams.get("status")).toBe("succeeded");
    expect(parsed.searchParams.get("workflow_id")).toBe("wf-1");
    expect(parsed.searchParams.get("session_id")).toBe("session-1");
    expect(parsed.searchParams.get("since")).toBe("2026-04-06T12:00:00.000Z");
    expect(parsed.searchParams.get("search")).toBe("planner");
    expect(new Headers(init.headers).get("X-API-Key")).toBe("secret");
  });

  it("returns null when a workflow run summary is missing", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponse(200, {
        runs: [],
        total_count: 0,
        page: 1,
        page_size: 1,
        has_more: false,
      })
    );

    await expect(getWorkflowRunSummary("run-missing")).resolves.toBeNull();
  });

  it("fetches enhanced executions and lightweight DAGs", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          executions: [],
          total_count: 0,
          page: 1,
          page_size: 20,
          total_pages: 0,
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          root_workflow_id: "wf-1",
          workflow_status: "running",
          workflow_name: "Main workflow",
          total_nodes: 0,
          max_depth: 0,
          timeline: [],
          mode: "lightweight",
        })
      );

    await expect(
      getEnhancedExecutions(
        {
          workflow: "wf-1",
          session: "session-1",
          status: "running",
        },
        3,
        50,
        "when",
        "desc"
      )
    ).resolves.toMatchObject({ total_count: 0 });
    await expect(getWorkflowDAGLightweight("wf-1")).resolves.toMatchObject({
      mode: "lightweight",
    });

    const enhancedUrl = new URL(
      vi.mocked(globalThis.fetch).mock.calls[0]?.[0] as string,
      "http://localhost"
    );
    expect(enhancedUrl.pathname).toBe("/api/ui/v1/executions/enhanced");
    expect(enhancedUrl.searchParams.get("workflow_id")).toBe("wf-1");
    expect(enhancedUrl.searchParams.get("session_id")).toBe("session-1");
    expect(enhancedUrl.searchParams.get("status")).toBe("running");

    const dagUrl = vi.mocked(globalThis.fetch).mock.calls[1]?.[0] as string;
    expect(dagUrl).toContain("/workflows/wf-1/dag?mode=lightweight");
  });

  it("routes view-mode requests and batches workflow cleanup results", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          runs: [],
          total_count: 0,
          page: 1,
          page_size: 20,
          total_pages: 0,
          has_more: false,
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          executions: [],
          total_count: 0,
          page: 1,
          page_size: 20,
          total_pages: 0,
        })
      )
      .mockImplementation((url: string) => {
        if (url.includes("wf-good")) {
          return Promise.resolve(
            mockResponse(200, {
              workflow_id: "wf-good",
              dry_run: false,
              deleted_records: { executions: 3 },
              freed_space_bytes: 1024,
              duration_ms: 20,
              success: true,
            })
          );
        }

        return Promise.resolve(
          mockResponse(500, { message: "cleanup failed" }, "Server Error")
        );
      });

    await expect(getExecutionsByViewMode("workflows")).resolves.toMatchObject({
      workflows: [],
    });
    await expect(getExecutionsByViewMode("executions")).resolves.toMatchObject({
      executions: [],
    });

    const results = await deleteWorkflows(["wf-good", "wf-bad", "wf-good", ""]);
    expect(results).toEqual([
      {
        workflow_id: "wf-good",
        dry_run: false,
        deleted_records: { executions: 3 },
        freed_space_bytes: 1024,
        duration_ms: 20,
        success: true,
      },
      {
        workflow_id: "wf-bad",
        dry_run: false,
        deleted_records: {},
        freed_space_bytes: 0,
        duration_ms: 0,
        success: false,
        error_message: "cleanup failed",
      },
    ]);
  });
});
