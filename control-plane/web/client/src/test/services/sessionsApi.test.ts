import { beforeEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import { sessionsApi } from "@/services/sessionsApi";

describe("sessionsApi", () => {
  beforeEach(() => {
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it("starts a session with auth headers and encoded target", async () => {
    setGlobalApiKey("secret");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: "sess-1",
        target: "support.voice",
        provider: "openai",
        transport: "webrtc",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const response = await sessionsApi.startSession("support.voice", { voice: "marin" });

    expect(response.session_id).toBe("sess-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/session-targets/support.voice/start",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ voice: "marin" }),
      }),
    );
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(new Headers(init.headers).get("X-API-Key")).toBe("secret");
  });

  it("invokes tools and surfaces API errors", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ run_id: "run-1", status: "queued" }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: "bad tool" }),
      });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      sessionsApi.invokeTool("sess/1", "resolve voice", {
        target: "support.resolve_voice_turn",
        input: { text: "hi" },
      }),
    ).resolves.toEqual({ run_id: "run-1", status: "queued" });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/v1/session-instances/sess%2F1/tools/resolve%20voice",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          target: "support.resolve_voice_turn",
          input: { text: "hi" },
        }),
      }),
    );

    await expect(
      sessionsApi.invokeTool("sess-1", "missing", { input: {} }),
    ).rejects.toThrow("bad tool");
  });
});
