import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLLMHealth, useQueueStatus } from "@/hooks/queries/useSystemHealth";

const apiState = vi.hoisted(() => ({
  getGlobalApiKey: vi.fn(),
}));

vi.mock("@/services/api", () => apiState);
vi.mock("@/hooks/useSSEQuerySync", () => ({
  useSSESync: () => ({ execConnected: true }),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};

describe("useQueueStatus", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    apiState.getGlobalApiKey.mockReturnValue("test-key");
  });

  it("returns safe defaults when queue endpoint fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
    } as Response);

    const { result } = renderHook(() => useQueueStatus(), { wrapper });

    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data).toEqual({
      enabled: false,
      max_per_agent: 0,
      agents: [],
      total_running: 0,
    });
    expect(fetch).toHaveBeenCalledWith("/api/ui/v1/queue/status", {
      headers: { "X-API-Key": "test-key" },
    });
  });

  it("normalizes legacy map-based queue payloads", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        max_per_agent: 3,
        agents: {
          "agent-a": { running: 2, max_concurrent: 3 },
          "agent-b": { running: 1, max_concurrent: 3 },
        },
      }),
    } as Response);

    const { result } = renderHook(() => useQueueStatus(), { wrapper });

    await waitFor(() => expect(result.current.data?.agents?.length).toBe(2));
    expect(result.current.data).toEqual({
      enabled: true,
      max_per_agent: 3,
      total_running: 3,
      agents: [
        { agent_node_id: "agent-a", running: 2, max: 3, available: 1 },
        { agent_node_id: "agent-b", running: 1, max: 3, available: 2 },
      ],
    });
  });

  it("normalizes array payloads and filters invalid agent ids", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        enabled: true,
        max_per_agent: 4,
        agents: [
          { agent_node_id: "node-1", running: 2, max: 4 },
          { agent_node_id: "", running: 1, max: 4, available: 3 },
        ],
      }),
    } as Response);

    const { result } = renderHook(() => useQueueStatus(), { wrapper });

    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data).toEqual({
      enabled: true,
      max_per_agent: 4,
      total_running: 2,
      agents: [{ agent_node_id: "node-1", running: 2, max: 4, available: 2 }],
    });
  });
});

describe("useLLMHealth", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    apiState.getGlobalApiKey.mockReturnValue("test-key");
  });

  it("returns disabled defaults when health endpoint fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
    } as Response);

    const { result } = renderHook(() => useLLMHealth(), { wrapper });

    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data).toEqual({
      enabled: false,
      healthy: false,
      endpoints: [],
    });
    expect(fetch).toHaveBeenCalledWith("/api/ui/v1/llm/health", {
      headers: { "X-API-Key": "test-key" },
    });
  });

  it("normalizes modern llm health payload with circuit state fields", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        enabled: true,
        healthy: false,
        checked_at: "2026-04-08T10:06:00Z",
        endpoints: [
          {
            name: "litellm",
            healthy: false,
            circuit_state: "open",
            consecutive_failures: 3,
            last_error: "unhealthy status code: 500",
            last_success: "2026-04-08T09:58:00Z",
            last_checked: "2026-04-08T10:06:00Z",
            total_checks: 12,
            total_failures: 4,
            url: "http://127.0.0.1:4000/health",
          },
        ],
      }),
    } as Response);

    const { result } = renderHook(() => useLLMHealth(), { wrapper });

    await waitFor(() => expect(result.current.data?.endpoints?.length).toBe(1));
    expect(result.current.data).toEqual({
      enabled: true,
      healthy: false,
      checked_at: "2026-04-08T10:06:00Z",
      endpoints: [
        {
          name: "litellm",
          healthy: false,
          circuit_state: "open",
          consecutive_failures: 3,
          last_error: "unhealthy status code: 500",
          last_success: "2026-04-08T09:58:00Z",
          last_checked: "2026-04-08T10:06:00Z",
          total_checks: 12,
          total_failures: 4,
          url: "http://127.0.0.1:4000/health",
          model: undefined,
          latency_ms: undefined,
          error: undefined,
        },
      ],
    });
  });

  it("backfills circuit state and failure count from legacy payload fields", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        enabled: true,
        healthy: true,
        endpoints: [
          {
            name: "legacy-healthy",
            healthy: true,
            consecutiveFailures: 0,
            lastError: "",
          },
          {
            name: "legacy-unhealthy",
            healthy: false,
            error: "request failed",
            consecutiveFailures: 2,
          },
          {
            name: "",
            healthy: true,
          },
        ],
      }),
    } as Response);

    const { result } = renderHook(() => useLLMHealth(), { wrapper });

    await waitFor(() => expect(result.current.data?.endpoints?.length).toBe(2));
    expect(result.current.data?.endpoints).toEqual([
      {
        name: "legacy-healthy",
        healthy: true,
        circuit_state: "closed",
        consecutive_failures: 0,
        last_error: "",
        last_success: undefined,
        last_checked: undefined,
        total_checks: undefined,
        total_failures: undefined,
        url: undefined,
        model: undefined,
        latency_ms: undefined,
        error: undefined,
      },
      {
        name: "legacy-unhealthy",
        healthy: false,
        circuit_state: "open",
        consecutive_failures: 2,
        last_error: "request failed",
        last_success: undefined,
        last_checked: undefined,
        total_checks: undefined,
        total_failures: undefined,
        url: undefined,
        model: undefined,
        latency_ms: undefined,
        error: "request failed",
      },
    ]);
  });
});
