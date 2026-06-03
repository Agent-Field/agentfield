import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Info, PauseCircle } from "lucide-react";

import { statusTone } from "@/lib/theme";
import {
  NotificationProvider,
  getNotificationGlyph,
  resolveEventKind,
  useDIDNotifications,
  useNotifications,
  useRunNotification,
  useVCNotifications,
  useWarningNotification,
} from "@/components/ui/notification";

const {
  toastSuccess,
  toastError,
  toastWarning,
  toastInfo,
  toastDismiss,
  toasterSpy,
  useThemeMock,
} = vi.hoisted(() => ({
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
  toastWarning: vi.fn(),
  toastInfo: vi.fn(),
  toastDismiss: vi.fn(),
  toasterSpy: vi.fn(),
  useThemeMock: vi.fn(),
}));

vi.mock("next-themes", () => ({
  useTheme: () => useThemeMock(),
}));

vi.mock("sonner", () => ({
  Toaster: (props: { theme: string }) => {
    toasterSpy(props);
    return <div data-testid="toaster" data-theme={props.theme} />;
  },
  toast: {
    success: toastSuccess,
    error: toastError,
    warning: toastWarning,
    info: toastInfo,
    dismiss: toastDismiss,
  },
}));

function wrapper({ children }: { children: ReactNode }) {
  return <NotificationProvider>{children}</NotificationProvider>;
}

describe("notification helpers", () => {
  it("resolves fallback event kinds and returns matching glyph metadata", () => {
    expect(resolveEventKind({ type: "success" })).toBe("complete");
    expect(resolveEventKind({ type: "warning" })).toBe("info");
    expect(resolveEventKind({ type: "info", eventKind: "pause" })).toBe("pause");

    const pauseGlyph = getNotificationGlyph({
      type: "success",
      eventKind: "pause",
    });
    expect(pauseGlyph.Icon).toBe(PauseCircle);
    expect(pauseGlyph.iconClass).toBe(statusTone.warning.accent);

    const infoGlyph = getNotificationGlyph({ type: "warning" });
    expect(infoGlyph.Icon).toBe(Info);
    expect(infoGlyph.iconClass).toBe(statusTone.info.accent);
  });

  it("requires the notification provider", () => {
    expect(() => renderHook(() => useNotifications())).toThrow(
      "useNotifications must be used within a NotificationProvider",
    );
  });
});

describe("NotificationProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    useThemeMock.mockReturnValue({ resolvedTheme: "dark" });
    document.title = "Client UI";
  });

  it("tracks notification state, configures the toaster, and wires toast actions", async () => {
    vi.stubEnv("VITE_BASE_PATH", "/console");

    const assignMock = vi.fn();
    const customAction = vi.fn();

    useThemeMock.mockReturnValue({ resolvedTheme: "light" });
    vi.stubGlobal("location", {
      ...window.location,
      assign: assignMock,
    });

    const { result } = renderHook(() => useNotifications(), { wrapper });

    expect(toasterSpy).toHaveBeenCalledWith(
      expect.objectContaining({ theme: "light" }),
    );

    let runId = "";
    let pauseId = "";
    act(() => {
      runId = result.current.addNotification({
        type: "success",
        title: "Run completed",
        runId: "run-123",
      });
      pauseId = result.current.addNotification({
        type: "warning",
        eventKind: "pause",
        title: "Run paused",
        message: "Waiting for approval",
        persistent: true,
        action: {
          label: "Resume",
          onClick: customAction,
        },
      });
    });

    expect(result.current.notifications).toHaveLength(2);
    expect(result.current.notifications[0]?.title).toBe("Run paused");
    expect(result.current.notifications[1]?.title).toBe("Run completed");
    expect(result.current.unreadCount).toBe(2);
    expect(document.title).toBe("(2) Client UI");

    const runToastOptions = toastSuccess.mock.calls[0]?.[1] as {
      action?: { label: string; onClick: () => void };
      duration: number;
    };
    expect(runToastOptions.duration).toBe(5000);
    expect(runToastOptions.action?.label).toBe("View run");

    act(() => {
      runToastOptions.action?.onClick();
    });
    expect(assignMock).toHaveBeenCalledWith("/console/runs/run-123");

    const pauseToastOptions = toastWarning.mock.calls[0]?.[1] as {
      action?: { label: string; onClick: () => void };
      duration: number;
    };
    expect(pauseToastOptions.duration).toBe(Infinity);
    expect(pauseToastOptions.action?.label).toBe("Resume");

    act(() => {
      pauseToastOptions.action?.onClick();
    });
    expect(customAction).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.markRead(runId);
    });
    expect(result.current.unreadCount).toBe(1);
    expect(document.title).toBe("(1) Client UI");

    act(() => {
      result.current.markAllRead();
    });
    expect(result.current.notifications.every((item) => item.read)).toBe(true);
    expect(result.current.unreadCount).toBe(0);
    expect(document.title).toBe("Client UI");

    act(() => {
      result.current.removeNotification(pauseId);
    });
    expect(toastDismiss).toHaveBeenCalledWith(pauseId);
    expect(result.current.notifications).toHaveLength(1);

    act(() => {
      result.current.clearAll();
    });
    expect(result.current.notifications).toHaveLength(0);
    expect(toastDismiss).toHaveBeenLastCalledWith();

    await waitFor(() => {
      expect(result.current.notifications).toEqual([]);
    });
  });

  it("keeps the notification log bounded to the newest 50 entries", () => {
    const { result } = renderHook(() => useNotifications(), { wrapper });

    act(() => {
      for (let index = 0; index < 52; index += 1) {
        result.current.addNotification({
          type: "info",
          title: `Notice ${index}`,
        });
      }
    });

    expect(result.current.notifications).toHaveLength(50);
    expect(result.current.notifications[0]?.title).toBe("Notice 51");
    expect(result.current.notifications[49]?.title).toBe("Notice 2");
    expect(result.current.unreadCount).toBe(50);
  });
});

describe("notification hooks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.stubEnv("VITE_BASE_PATH", "/console");
    useThemeMock.mockReturnValue({ resolvedTheme: "dark" });
    document.title = "Client UI";
  });

  it("emits warning, run, DID, and VC notifications with the expected semantics", () => {
    const { result } = renderHook(
      () => ({
        context: useNotifications(),
        warning: useWarningNotification(),
        run: useRunNotification(),
        did: useDIDNotifications(),
        vc: useVCNotifications(),
      }),
      { wrapper },
    );

    act(() => {
      result.current.warning("Careful", "Check the inputs");
      result.current.run({
        type: "error",
        eventKind: "error",
        title: "Run failed",
        message: "Worker exited",
        runId: "run-9",
        runLabel: "demo.run",
      });
      result.current.did.didCopied();
      result.current.did.didRegistered("node-7");
      result.current.did.didError("bad did");
      result.current.did.didRefreshed();
      result.current.vc.vcCopied();
      result.current.vc.vcDownloaded("proof.json");
      result.current.vc.vcVerified(true);
      result.current.vc.vcVerified(false);
      result.current.vc.vcError("bad vc");
      result.current.vc.vcChainLoaded(3);
    });

    expect(toastWarning).toHaveBeenCalledWith(
      "Careful",
      expect.objectContaining({
        description: "Check the inputs",
        duration: 5000,
      }),
    );
    expect(toastError).toHaveBeenCalledWith(
      "Run failed",
      expect.objectContaining({
        description: "Worker exited",
        duration: 6000,
      }),
    );

    expect(result.current.context.notifications).toHaveLength(12);
    expect(result.current.context.notifications.map((item) => item.title)).toEqual([
      "VC Chain Loaded",
      "VC Operation Failed",
      "VC Verification Failed",
      "VC Verified",
      "VC Downloaded",
      "VC Copied",
      "DID Data Refreshed",
      "DID Operation Failed",
      "DID Registered",
      "DID Copied",
      "Run failed",
      "Careful",
    ]);
    expect(result.current.context.notifications[0]).toMatchObject({
      title: "VC Chain Loaded",
      message: "Loaded 3 verification credentials",
    });
    expect(result.current.context.notifications[1]).toMatchObject({
      title: "VC Operation Failed",
      type: "error",
    });
    expect(result.current.context.notifications[2]).toMatchObject({
      title: "VC Verification Failed",
      type: "error",
    });
    expect(result.current.context.notifications[4]).toMatchObject({
      title: "VC Downloaded",
      message: "Downloaded as proof.json",
    });
    expect(result.current.context.notifications[7]).toMatchObject({
      title: "DID Operation Failed",
      type: "error",
    });
    expect(result.current.context.notifications[8]).toMatchObject({
      title: "DID Registered",
      message: "DID identity registered for node node-7",
    });
    expect(result.current.context.notifications[10]).toMatchObject({
      title: "Run failed",
      href: "/console/runs/run-9",
      runLabel: "demo.run",
      eventKind: "error",
    });
    expect(result.current.context.notifications[11]).toMatchObject({
      title: "Careful",
      type: "warning",
    });
  });
});
