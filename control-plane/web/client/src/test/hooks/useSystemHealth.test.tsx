import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useQueueStatus } from "@/hooks/queries/useSystemHealth";

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
