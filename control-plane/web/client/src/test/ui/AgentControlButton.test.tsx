import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => (
    <svg data-testid="agent-control-icon" className={className} />
  );

  return {
    Play: Icon,
    Stop: Icon,
    InProgress: Icon,
    CheckmarkFilled: Icon,
    WarningFilled: Icon,
    Restart: Icon,
  };
});

import { AgentControlButton } from "@/components/ui/AgentControlButton";

describe("AgentControlButton", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("starts an agent, shows transient success feedback, and hides it after the timer", async () => {
    vi.useFakeTimers();
    const onToggle = vi.fn().mockResolvedValue(undefined);

    const { container } = render(
      <AgentControlButton
        agentId="agent-1"
        currentState="stopped"
        onToggle={onToggle}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start agent-1" }));
    });

    expect(onToggle).toHaveBeenCalledWith("start");
    expect(container.querySelector(".text-status-success")).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1200);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.querySelector(".text-status-success")).toBeFalsy();
  });

  it("renders label mode for reconcile and prevents interaction while disabled or transitional", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn().mockResolvedValue(undefined);

    const { rerender } = render(
      <AgentControlButton
        agentId="agent-2"
        currentState="error"
        onToggle={onToggle}
        showLabel
        size="lg"
      />,
    );

    expect(screen.getByText("Reconcile")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Reconcile agent-2" }));
    expect(onToggle).toHaveBeenCalledWith("reconcile");

    rerender(
      <AgentControlButton
        agentId="agent-2"
        currentState="starting"
        onToggle={onToggle}
        disabled
      />,
    );

    const button = screen.getByRole("button", { name: "Starting... agent-2" });
    expect(button).toBeDisabled();
    expect(button.className).toContain("opacity-50");
  });

  it("logs toggle failures and leaves success feedback hidden", async () => {
    const user = userEvent.setup();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const onToggle = vi.fn().mockRejectedValue(new Error("nope"));

    const { container } = render(
      <AgentControlButton
        agentId="agent-3"
        currentState="running"
        onToggle={onToggle}
        variant="minimal"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Stop agent-3" }));

    expect(onToggle).toHaveBeenCalledWith("stop");
    expect(errorSpy).toHaveBeenCalledWith(
      "Failed to toggle agent action",
      expect.objectContaining({ action: "stop", agentId: "agent-3" }),
    );
    expect(container.querySelector(".text-status-success")).toBeFalsy();
  });
});
