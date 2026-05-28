// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => (
    <svg data-testid="formatter-icon" className={className} />
  );

  return {
    Time: Icon,
    DataBase: Icon,
  };
});

import {
  AgentCapabilitySummary,
  DataSizeComparison,
  DurationDisplay,
  ExecutionIdDisplay,
  FileSizeDisplay,
  LiveElapsedDuration,
  MetricDisplay,
  PercentageDisplay,
  StatusHealthDisplay,
  TimestampDisplay,
  formatDurationHumanReadable,
  highlightSearchText,
} from "@/components/ui/data-formatters";

describe("data-formatters", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("formats human readable durations across thresholds", () => {
    expect(formatDurationHumanReadable(undefined)).toBe("—");
    expect(formatDurationHumanReadable(0)).toBe("—");
    expect(formatDurationHumanReadable(999)).toBe("999ms");
    expect(formatDurationHumanReadable(12_300)).toBe("12s");
    expect(formatDurationHumanReadable(185_000)).toBe("3m 5s");
    expect(formatDurationHumanReadable(3_780_000)).toBe("1h 3m");
    expect(formatDurationHumanReadable(93_600_000)).toBe("1d 2h");
  });

  it("highlights escaped search text matches without breaking regex characters", () => {
    const result = highlightSearchText("alpha (beta) alpha", "(beta)");
    const rendered = render(<>{result}</>);

    expect(rendered.container.querySelectorAll("mark")).toHaveLength(1);
    expect(screen.getByText("(beta)").tagName).toBe("MARK");
  });

  it("renders formatter components with expected text and icon states", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T12:00:00Z"));
    const clipboardWrite = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: { writeText: clipboardWrite },
    });

    render(
      <div>
        <DurationDisplay durationMs={1500} showIcon />
        <DurationDisplay durationMs={undefined} />
        <TimestampDisplay timestamp="2026-04-08T11:59:00Z" />
        <TimestampDisplay timestamp="2026-03-01T00:00:00Z" format="absolute" />
        <TimestampDisplay timestamp="2026-04-07T12:00:00Z" format="smart" />
        <TimestampDisplay timestamp={null} />
        <FileSizeDisplay bytes={1536} showIcon />
        <FileSizeDisplay bytes={0} />
        <ExecutionIdDisplay executionId="execution-id-1234567890" showCopy />
        <DataSizeComparison inputSize={1024} outputSize={1048576} showIcons />
        <AgentCapabilitySummary reasonerCount={2} skillCount={3} />
        <AgentCapabilitySummary reasonerCount={2} skillCount={3} format="compact" />
        <AgentCapabilitySummary reasonerCount={2} skillCount={3} format="minimal" />
        <PercentageDisplay value={0.1234} decimals={2} />
        <MetricDisplay label="Runs" value={1250} format="abbreviated" icon={<span>@</span>} />
        <MetricDisplay label="Name" value="stable" />
        <StatusHealthDisplay status="running" healthPercentage={81} />
        <StatusHealthDisplay status="idle" healthPercentage={50} />
        <StatusHealthDisplay status="unknown" />
        <LiveElapsedDuration startedAt="2026-04-08T11:59:58Z" className="live" />
      </div>,
    );

    expect(screen.getByText("1.5s")).toBeInTheDocument();
    expect(screen.getAllByText("N/A")).toHaveLength(2);
    expect(screen.getByText("1m ago")).toBeInTheDocument();
    expect(screen.getByText(new Date("2026-03-01T00:00:00Z").toLocaleDateString())).toBeInTheDocument();
    expect(screen.getByText("1d ago")).toBeInTheDocument();
    expect(screen.getByText("Never")).toBeInTheDocument();
    expect(screen.getByText("1.5KB")).toBeInTheDocument();
    expect(screen.getByText("executio...7890")).toBeInTheDocument();
    expect(screen.getByText("1.0KB")).toBeInTheDocument();
    expect(screen.getByText("1.0MB")).toBeInTheDocument();
    expect(screen.getByText("2 reasoners, 3 skills")).toBeInTheDocument();
    expect(screen.getByText("2R/3S")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("12.34%")).toBeInTheDocument();
    expect(screen.getByText("1.3K")).toBeInTheDocument();
    expect(screen.getByText("stable")).toBeInTheDocument();
    expect(screen.getByText("(81%)").className).toContain("text-green-600");
    expect(screen.getByText("(50%)").className).toContain("text-red-600");
    expect(screen.getByText("2s").className).toContain("live");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "📋" }));
    });
    expect(clipboardWrite).toHaveBeenCalledWith("execution-id-1234567890");

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText("3s")).toBeInTheDocument();
    expect(screen.getAllByTestId("formatter-icon").length).toBeGreaterThan(0);
  });
});