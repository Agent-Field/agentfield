import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { EnhancedNodesHeader } from "@/components/nodes/EnhancedNodesHeader";

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    ArrowClockwise: Icon,
    Plus: Icon,
    WifiHigh: Icon,
    WifiSlash: Icon,
  };
});

vi.mock("@/utils/node-status", () => ({
  getNodeStatusPresentation: (canonical?: string, secondaryCanonical?: string) => ({
    theme: { indicatorClass: `${canonical}-${secondaryCanonical ?? "none"}` },
    shouldPulse: canonical === "starting" || canonical === "degraded",
  }),
}));

describe("EnhancedNodesHeader", () => {
  it("renders live counts, updated badge, actions, and add button", async () => {
    const user = userEvent.setup();
    const onAddServerless = vi.fn();

    render(
      <EnhancedNodesHeader
        totalNodes={8}
        onlineCount={5}
        offlineCount={2}
        degradedCount={1}
        startingCount={1}
        lastUpdated={new Date(Date.now() - 2 * 60 * 1000)}
        isConnected
        onAddServerless={onAddServerless}
        actions={<button type="button">custom-action</button>}
      />
    );

    expect(screen.getByText("8 total")).toBeInTheDocument();
    expect(screen.getByText("5 online")).toBeInTheDocument();
    expect(screen.getByText("1 starting")).toBeInTheDocument();
    expect(screen.getByText("1 degraded")).toBeInTheDocument();
    expect(screen.getByText("2 offline")).toBeInTheDocument();
    expect(screen.getByText("Live updates")).toBeInTheDocument();
    expect(screen.getByText("Updated 2m ago")).toBeInTheDocument();
    expect(screen.getByText("custom-action")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add serverless agent/i }));
    expect(onAddServerless).toHaveBeenCalledTimes(1);
  });

  it("renders reconnect state and custom subtitle", async () => {
    const user = userEvent.setup();
    const onReconnect = vi.fn();

    render(
      <EnhancedNodesHeader
        totalNodes={1}
        onlineCount={0}
        offlineCount={1}
        degradedCount={0}
        startingCount={0}
        isConnected={false}
        isReconnecting={false}
        onReconnect={onReconnect}
        subtitle="Custom subtitle"
      />
    );

    expect(screen.getByText("Custom subtitle")).toBeInTheDocument();
    expect(screen.getByText("Disconnected")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /reconnect/i }));
    expect(onReconnect).toHaveBeenCalledTimes(1);
  });
});
