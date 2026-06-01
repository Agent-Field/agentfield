// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EnhancedNodeDetailHeader } from "@/components/nodes/EnhancedNodeDetailHeader";

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

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    ArrowLeft: Icon,
    Maximize: Icon,
    Minimize: Icon,
  };
});

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/node-status", () => ({
  getNodeStatusPresentation: (lifecycleStatus?: string, healthStatus?: string) => ({
    label: `${lifecycleStatus ?? "unknown"}:${healthStatus ?? "none"}`,
    shouldPulse: lifecycleStatus === "starting",
    theme: {
      indicatorClass: "indicator",
      textClass: "text-class",
    },
  }),
}));

describe("EnhancedNodeDetailHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders node status details and handles actions", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    const onFullscreenChange = vi.fn();

    render(
      <EnhancedNodeDetailHeader
        nodeId="node-123"
        lifecycleStatus="starting"
        healthStatus="healthy"
        lastHeartbeat="2026-04-08T12:00:00Z"
        onBack={onBack}
        focusMode={false}
        onFocusModeChange={vi.fn()}
        isFullscreen={false}
        onFullscreenChange={onFullscreenChange}
        viewMode="standard"
        onViewModeChange={vi.fn()}
        rightActions={<button type="button">Inspect</button>}
        statusBadges={<span>status-badge</span>}
        liveStatusBadge={<span>live</span>}
      />
    );

    expect(screen.getByText("starting:healthy")).toBeInTheDocument();
    expect(screen.getByText("node-123")).toBeInTheDocument();
    expect(screen.getByText(/Last heartbeat/)).toBeInTheDocument();
    expect(screen.getByText("status-badge")).toBeInTheDocument();
    expect(screen.getByText("live")).toBeInTheDocument();
    expect(screen.getByText("Inspect")).toBeInTheDocument();

    await user.click(screen.getByTitle("Back to nodes"));
    expect(onBack).toHaveBeenCalledTimes(1);

    await user.click(screen.getByTitle("Enter fullscreen"));
    expect(onFullscreenChange).toHaveBeenCalledWith(true);
  });

  it("renders without heartbeat for invalid values and shows exit fullscreen", () => {
    render(
      <EnhancedNodeDetailHeader
        nodeId="node-456"
        lifecycleStatus="offline"
        healthStatus="unhealthy"
        lastHeartbeat="not-a-date"
        focusMode={false}
        onFocusModeChange={vi.fn()}
        isFullscreen={true}
        onFullscreenChange={vi.fn()}
        viewMode="debug"
        onViewModeChange={vi.fn()}
      />
    );

    expect(screen.queryByText(/Last heartbeat/)).toBeNull();
    expect(screen.getByTitle("Exit fullscreen")).toBeInTheDocument();
  });
});
