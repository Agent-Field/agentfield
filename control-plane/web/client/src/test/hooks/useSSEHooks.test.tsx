// @ts-nocheck
import React from "react";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  useNodeEventsSSE,
  useNodeUnifiedStatusSSE,
  useSSE,
  useUnifiedStatusSSE,
} from "@/hooks/useSSE";
import { SSESyncProvider, useSSESync } from "@/hooks/useSSEQuerySync";

const apiState = vi.hoisted(() => ({
  getGlobalApiKey: vi.fn(),
}));

const sseSyncState = vi.hoisted(() => ({
  callIndex: 0,
  responses: [] as Array<any>,
}));

vi.mock("@/services/api", () => apiState);

vi.mock("@/hooks/useSSE", async () => {
  const actual = await vi.importActual<typeof import("@/hooks/useSSE")>("@/hooks/useSSE");

  return {
    ...actual,
    useSSE: vi.fn((...args: any[]) => {
      const response = sseSyncState.responses[sseSyncState.callIndex];
      sseSyncState.callIndex += 1;
      return response ?? actual.useSSE(...args);
    }),
  };
});

class MockEventSource {
  static instances: MockEventSource[] = [];
  static OPEN = 1;
  static CLOSED = 2;

  url: string;
  readyState = 0;
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  close = vi.fn(() => {
    this.readyState = MockEventSource.CLOSED;
  });
  listeners = new Map<string, Array<(event: Event) => void>>();

  constructor(url: string | URL) {
    this.url = String(url);
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: Event) => void) {
    const existing = this.listeners.get(type) ?? [];
    existing.push(listener);
    this.listeners.set(type, existing);
  }

  emitOpen() {
    this.readyState = MockEventSource.OPEN;
    this.onopen?.(new Event("open"));
  }

  emitError() {
    this.readyState = MockEventSource.CLOSED;
    this.onerror?.(new Event("error"));
  }

  emitMessage(data: unknown, lastEventId = "evt-1") {
    this.onmessage?.(
      new MessageEvent("message", {
        data: JSON.stringify(data),
        lastEventId,
      })
    );
  }

  emitCustom(type: string, data: unknown) {
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }
}

describe("SSE hooks", () => {
  const originalEventSource = globalThis.EventSource;

  beforeEach(() => {
    vi.useFakeTimers();
    MockEventSource.instances = [];
    apiState.getGlobalApiKey.mockReturnValue("stream-key");
    globalThis.EventSource = MockEventSource as any;
    sseSyncState.callIndex = 0;
    sseSyncState.responses = [];
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    globalThis.EventSource = originalEventSource;
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("useSSE connects, parses events, filters by type, reconnects, and disconnects", async () => {
    const onConnectionChange = vi.fn();
    const onError = vi.fn();
    const { result, unmount } = renderHook(() =>
      useSSE("/events", {
        eventTypes: ["custom"],
        trackEvents: true,
        maxReconnectAttempts: 2,
        reconnectDelayMs: 100,
        exponentialBackoff: true,
        onConnectionChange,
        onError,
      })
    );

    const instance = MockEventSource.instances[0];
    expect(instance.url).toBe("/events?api_key=stream-key");

    await act(async () => {
      instance.emitOpen();
      instance.emitMessage({ type: "execution_started", execution_id: "exec-1" });
      instance.emitCustom("custom", { custom: true });
    });

    expect(result.current.connected).toBe(true);
    expect(result.current.latestEvent).toEqual(
      expect.objectContaining({ type: "custom", data: { custom: true } })
    );
    expect(result.current.eventCount).toBe(2);
    expect(result.current.getEventsByType("execution_started")).toHaveLength(1);
    expect(onConnectionChange).toHaveBeenCalledWith(true);

    await act(async () => {
      instance.emitError();
    });
    expect(onError).toHaveBeenCalled();
    expect(result.current.reconnecting).toBe(true);

    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    expect(MockEventSource.instances).toHaveLength(2);

    await act(async () => {
      result.current.clearEvents();
      result.current.disconnect();
      result.current.reconnect();
    });
    expect(result.current.eventCount).toBe(0);
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(3);

    unmount();
    expect(MockEventSource.instances.at(-1)?.close).toHaveBeenCalled();
  });

  it("useSSE ignores invalid payloads and closes when url becomes null", async () => {
    const { result, rerender } = renderHook(
      ({ url }) => useSSE(url, { trackEvents: true }),
      { initialProps: { url: "/events" as string | null } }
    );

    const instance = MockEventSource.instances[0];

    act(() => {
      instance.onmessage?.(new MessageEvent("message", { data: "" }));
      instance.onmessage?.(new MessageEvent("message", { data: JSON.stringify("bad") }));
    });

    expect(result.current.eventCount).toBe(0);

    rerender({ url: null });
    expect(instance.close).toHaveBeenCalled();
    expect(result.current.connected).toBe(false);
  });

  it("specialized SSE hooks use the expected endpoints", () => {
    renderHook(() => useNodeEventsSSE());
    renderHook(() => useUnifiedStatusSSE());
    renderHook(() => useNodeUnifiedStatusSSE("node-1"));

    expect(MockEventSource.instances.map((instance) => instance.url)).toEqual([
      "/api/ui/v1/nodes/events?api_key=stream-key",
      "/api/ui/v1/nodes/events?api_key=stream-key",
      "/api/ui/v1/nodes/events?api_key=stream-key",
    ]);
  });

  it("useSSESync invalidates live queries and exposes combined connection state", async () => {
    const queryClient = new QueryClient();
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    sseSyncState.responses = [
      {
        latestEvent: {
          type: "execution_updated",
          data: { workflow_id: "run-1", execution_id: "exec-1", type: "execution_updated" },
        },
        connected: true,
        reconnecting: false,
      },
      {
        latestEvent: { type: "node_online", data: { type: "node_online" } },
        connected: false,
        reconnecting: true,
      },
      {
        latestEvent: { type: "reasoner_registered", data: { type: "reasoner_registered" } },
        connected: false,
        reconnecting: false,
      },
    ];

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <SSESyncProvider>{children}</SSESyncProvider>
      </QueryClientProvider>
    );

    const { result } = renderHook(() => useSSESync(), { wrapper });

    expect(result.current.execConnected).toBe(true);
    expect(result.current.anyConnected).toBe(true);
    expect(result.current.reconnecting).toBe(true);
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["runs"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["dashboard-summary"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["run-dag", "run-1"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["step-detail", "exec-1"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["agents"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["reasoners"] });

    await act(async () => {
      result.current.refreshAllLiveQueries();
    });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["llm-health"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["queue-status"] });
  });

  it("useSSESync skips heartbeat-only invalidation", async () => {
    const queryClient = new QueryClient();
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    sseSyncState.responses = [
      {
        latestEvent: { type: "message", data: { type: "heartbeat" } },
        connected: true,
        reconnecting: false,
      },
      { latestEvent: null, connected: false, reconnecting: false },
      { latestEvent: null, connected: false, reconnecting: false },
    ];

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <SSESyncProvider>{children}</SSESyncProvider>
      </QueryClientProvider>
    );

    renderHook(() => useSSESync(), { wrapper });

    expect(invalidateQueries).not.toHaveBeenCalled();
  });
});