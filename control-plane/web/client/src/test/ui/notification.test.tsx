import { act, renderHook } from "@testing-library/react";
import { type PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const notificationState = vi.hoisted(() => ({
  dismiss: vi.fn<(id?: string) => void>(),
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ resolvedTheme: "light" }),
}));

vi.mock("sonner", () => ({
  Toaster: ({ children }: PropsWithChildren) => <div data-testid="toaster">{children}</div>,
  toast: {
    dismiss: notificationState.dismiss,
    success: notificationState.success,
    error: notificationState.error,
    warning: notificationState.warning,
    info: notificationState.info,
  },
}));

import {
  NotificationProvider,
  getNotificationGlyph,
  resolveEventKind,
  useNotifications,
  useRunNotification,
  useSuccessNotification,
} from "@/components/ui/notification";

function wrapper({ children }: PropsWithChildren) {
  return <NotificationProvider>{children}</NotificationProvider>;
}

describe("notification helpers", () => {
  beforeEach(() => {
    notificationState.dismiss.mockReset();
    notificationState.success.mockReset();
    notificationState.error.mockReset();
    notificationState.warning.mockReset();
    notificationState.info.mockReset();
    document.title = "AgentField";
  });

  it("resolves event kinds and returns the mapped glyph", () => {
    expect(resolveEventKind({ type: "success" })).toBe("complete");
    expect(resolveEventKind({ type: "warning", eventKind: "pause" })).toBe("pause");

    const glyph = getNotificationGlyph({ type: "info", eventKind: "resume" });
    expect(glyph.Icon).toBeTruthy();
    expect(glyph.iconClass).toContain("text-");
  });

  it("stores notifications, updates unread state, and dismisses removed entries", () => {
    const { result } = renderHook(() => useNotifications(), { wrapper });

    let id = "";
    act(() => {
      id = result.current.addNotification({
        type: "success",
        title: "Saved",
        message: "Configuration updated",
      });
    });

    expect(result.current.notifications).toHaveLength(1);
    expect(result.current.unreadCount).toBe(1);
    expect(document.title).toBe("(1) AgentField");
    expect(notificationState.success).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.markRead(id);
    });
    expect(result.current.unreadCount).toBe(0);
    expect(document.title).toBe("AgentField");

    act(() => {
      result.current.removeNotification(id);
    });
    expect(result.current.notifications).toHaveLength(0);
    expect(notificationState.dismiss).toHaveBeenCalledWith(id);
  });

  it("creates run-scoped notifications with a synthesized view action and clears the log", () => {
    const assign = vi.fn<(href: string) => void>();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { assign },
    });

    const { result } = renderHook(
      () => ({
        notifications: useNotifications(),
        showRun: useRunNotification(),
        showSuccess: useSuccessNotification(),
      }),
      { wrapper },
    );

    act(() => {
      result.current.showRun({
        type: "warning",
        eventKind: "pause",
        title: "Run paused",
        runId: "run-42",
      });
      result.current.showSuccess("Saved again", "Done");
    });

    expect(result.current.notifications.notifications).toHaveLength(2);
    expect(notificationState.warning).toHaveBeenCalledTimes(1);
    expect(notificationState.success).toHaveBeenCalledTimes(1);

    const options = notificationState.warning.mock.calls[0]?.[1] as {
      action?: { label?: string; onClick: () => void };
      duration: number;
    };
    expect(options.duration).toBe(5000);
    expect(options.action?.label).toBe("View run");

    act(() => {
      options.action?.onClick();
    });
    expect(assign).toHaveBeenCalledWith("/ui/runs/run-42");

    act(() => {
      result.current.notifications.markAllRead();
    });
    expect(result.current.notifications.unreadCount).toBe(0);

    act(() => {
      result.current.notifications.clearAll();
    });
    expect(result.current.notifications.notifications).toHaveLength(0);
    expect(notificationState.dismiss).toHaveBeenCalledWith();
  });
});
