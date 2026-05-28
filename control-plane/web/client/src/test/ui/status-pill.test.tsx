import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  LifecycleDot,
  LifecycleIcon,
  LifecyclePill,
  StatusDot,
  StatusIcon,
  StatusPill,
} from "@/components/ui/status-pill";

describe("status-pill", () => {
  it("renders execution status dot, icon, and pill variants", () => {
    const { container } = render(
      <>
        <StatusDot status="running" size="lg" className="dot-wrap" />
        <StatusDot status=" success " label={false} />
        <StatusIcon status="running" size="md" className="icon-wrap" />
        <StatusPill status="completed" size="sm" />
        <StatusPill status="paused" showIcon={false} />
        <StatusPill status="failed" showLabel={false} />
      </>,
    );

    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Succeeded" })).toBeInTheDocument();
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(screen.getByText("Paused")).toBeInTheDocument();
    expect(container.querySelector(".motion-safe\\:animate-ping")).not.toBeNull();
    expect(container.querySelector(".motion-safe\\:animate-spin")).not.toBeNull();
    expect(container.querySelector(".icon-wrap")).not.toBeNull();
  });

  it("renders lifecycle status variants and motion states", () => {
    const { container } = render(
      <>
        <LifecycleDot status="running" />
        <LifecycleDot status="ready" label={false} />
        <LifecycleIcon status="starting" size="lg" className="starting-icon" />
        <LifecycleIcon status="degraded" />
        <LifecyclePill status="offline" size="lg" />
        <LifecyclePill status={null} showIcon={false} />
        <LifecyclePill status="running" showLabel={false} />
      </>,
    );

    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Ready" })).toBeInTheDocument();
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getByText("Unknown")).toBeInTheDocument();
    expect(container.querySelector(".motion-safe\\:animate-pulse")).not.toBeNull();
    expect(container.querySelector(".starting-icon")).not.toBeNull();
    expect(container.querySelector("[data-status='offline']")).not.toBeNull();
  });
});
