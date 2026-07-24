// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationBell } from "@/components/NotificationBell";

const state = vi.hoisted(() => ({
  notifications: [] as Array<{
    id: string;
    type: "info" | "success" | "warning" | "error";
    title: string;
    message?: string;
    createdAt: number;
    read: boolean;
    runId?: string;
    runLabel?: string;
  }>,
  unreadCount: 0,
  markRead: vi.fn<(id: string) => void>(),
  markAllRead: vi.fn<() => void>(),
  removeNotification: vi.fn<(id: string) => void>(),
  clearAll: vi.fn<() => void>(),
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  PopoverContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <hr />,
}));

vi.mock("@/components/ui/notification", () => ({
  useNotifications: () => ({
    notifications: state.notifications,
    unreadCount: state.unreadCount,
    markRead: state.markRead,
    markAllRead: state.markAllRead,
    removeNotification: state.removeNotification,
    clearAll: state.clearAll,
  }),
  getNotificationGlyph: () => ({
    Icon: ({ className }: { className?: string }) => (
      <svg data-testid="notification-glyph" className={className} />
    ),
    iconClass: "text-muted",
  }),
}));

describe("NotificationBell", () => {
  beforeEach(() => {
    state.notifications = [];
    state.unreadCount = 0;
    state.markRead.mockReset();
    state.markAllRead.mockReset();
    state.removeNotification.mockReset();
    state.clearAll.mockReset();
  });

  it("renders the empty state with a neutral bell label", () => {
    render(
      <MemoryRouter>
        <NotificationBell />
      </MemoryRouter>,
    );

    expect(screen.getByRole("button", { name: "Notifications" })).toBeInTheDocument();
    expect(screen.getByText("No notifications yet")).toBeInTheDocument();
    expect(screen.getByText(/run events, errors, and actions/i)).toBeInTheDocument();
  });

  it("renders grouped notifications and handles read, dismiss, and clear actions", async () => {
    const user = userEvent.setup();

    state.notifications = [
      {
        id: "run-latest",
        type: "info",
        title: "Latest run event",
        message: "Run completed",
        createdAt: Date.now() - 5_000,
        read: false,
        runId: "run-1",
        runLabel: "demo.run",
      },
      {
        id: "run-older",
        type: "info",
        title: "Earlier run event",
        createdAt: Date.now() - 60_000,
        read: true,
        runId: "run-1",
        runLabel: "demo.run",
      },
      {
        id: "single-1",
        type: "warning",
        title: "Single notification",
        message: "Needs attention",
        createdAt: Date.now() - 120_000,
        read: false,
      },
    ];
    state.unreadCount = 2;

    render(
      <MemoryRouter>
        <NotificationBell />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: "Notifications, 2 unread" }),
    ).toBeInTheDocument();
    expect(screen.getByText("2 new")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "demo.run" })).toHaveAttribute("href", "/runs/run-1");

    await user.click(screen.getByText("Latest run event"));
    expect(state.markRead).toHaveBeenCalledWith("run-latest");

    await user.click(screen.getByRole("button", { name: "Mark all as read" }));
    expect(state.markAllRead).toHaveBeenCalledTimes(1);

    await user.click(screen.getAllByRole("button", { name: "Dismiss notification" }).at(-1)!);
    expect(state.removeNotification).toHaveBeenCalledWith("single-1");

    await user.click(screen.getByRole("button", { name: "Clear all notifications" }));
    expect(state.clearAll).toHaveBeenCalledTimes(1);
  });
});