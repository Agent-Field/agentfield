// @ts-nocheck
import React from "react";
import { fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, afterEach } from "vitest";

import {
  FastTableSearch,
  createSearchMatcher,
  useOptimizedSearch,
} from "@/components/ui/FastTableSearch";
import { MetricCard } from "@/components/ui/MetricCard";
import {
  JsonHighlightedPre,
  formatTruncatedFormattedJson,
  highlightJsonText,
  jsonStringifyFormatted,
} from "@/components/ui/json-syntax-highlight";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    Search: Icon,
    Close: Icon,
  };
});

describe("ui extras search, json, and metric helpers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    // @ts-expect-error test cleanup for optional window API
    delete window.requestIdleCallback;
  });

  it("highlights JSON tokens, escapes html, and renders truncated suffixes", () => {
    const highlighted = highlightJsonText('{"tag":"<safe>","count":12,"ok":true,"none":null}');
    expect(highlighted).toContain('class="json-hl-key"');
    expect(highlighted).toContain("&lt;safe&gt;");
    expect(highlighted).toContain('class="json-hl-number"');
    expect(highlighted).toContain('class="json-hl-boolean"');
    expect(highlighted).toContain('class="json-hl-null"');

    expect(jsonStringifyFormatted({ alpha: 1 })).toBe('{\n  "alpha": 1\n}');

    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(jsonStringifyFormatted(circular)).toMatch(/\[object Object\]/);

    const truncated = formatTruncatedFormattedJson({ text: "x".repeat(20) }, 10);
    expect(truncated).toContain("… truncated");
    expect(formatTruncatedFormattedJson("", 5)).toBe("—");

    const { container, rerender } = render(
      <JsonHighlightedPre text='{"alpha":"beta"}' className="viewer" />,
    );
    const pre = container.querySelector("pre");
    expect(pre?.className).toContain("viewer");
    expect(pre?.innerHTML).toContain('class="json-hl-key"');

    rerender(<JsonHighlightedPre text={truncated} />);
    expect(pre?.textContent).toContain("… truncated");

    rerender(<JsonHighlightedPre text="—" />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("debounces FastTableSearch, supports keyboard shortcuts, and clears visible results", async () => {
    const onSearch = vi.fn();

    Object.defineProperty(navigator, "platform", {
      configurable: true,
      value: "Win32",
    });

    render(
      <FastTableSearch
        onSearch={onSearch}
        placeholder="Search rows"
        resultCount={2}
        totalCount={5}
      />,
    );

    const input = screen.getByPlaceholderText("Search rows");
    expect(screen.getByText("Ctrl K")).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    expect(input).toHaveFocus();

    fireEvent.change(input, { target: { value: "agent" } });
    await waitFor(() => {
      expect(onSearch).toHaveBeenLastCalledWith("agent");
    });
    expect(screen.getByText('2 of 5 results')).toBeInTheDocument();
    expect(screen.getByText(/for "/)).toHaveTextContent('for "agent"');

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => {
      expect(onSearch).toHaveBeenLastCalledWith("");
    });

    fireEvent.change(input, { target: { value: "run" } });
    await waitFor(() => {
      expect(onSearch).toHaveBeenLastCalledWith("run");
    });
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() => {
      expect(onSearch).toHaveBeenLastCalledWith("");
    });
    expect(input).toHaveFocus();
  });

  it("matches nested fields and optimizes search for enabled, disabled, small, and large datasets", () => {
    const searchFields = ["agent.name", "status"];
    const matcher = createSearchMatcher(["agent.name", "status"]);
    expect(matcher({ agent: { name: "Alpha Agent" }, status: "running" }, "alpha run")).toBe(true);
    expect(matcher({ agent: { name: "Alpha Agent" }, status: "running" }, "beta")).toBe(false);

    const data = [
      { id: 1, agent: { name: "Alpha" }, status: "running" },
      { id: 2, agent: { name: "Beta" }, status: "failed" },
    ];

    const { result, rerender } = renderHook(
      ({ query, enabled }) => useOptimizedSearch(data, query, searchFields, enabled),
      { initialProps: { query: "alpha", enabled: true } },
    );

    expect(result.current).toEqual([data[0]]);

    rerender({ query: "failed", enabled: false });
    expect(result.current).toEqual(data);

    const idleCallback = vi.fn((cb: () => void) => {
      cb();
      return 1;
    });
    // @ts-expect-error optional browser api
    window.requestIdleCallback = idleCallback;

    const largeData = Array.from({ length: 1001 }, (_, index) => ({
      id: index,
      agent: { name: index === 1000 ? "Late Match" : `Agent ${index}` },
      status: index === 1000 ? "running" : "idle",
    }));

    const largeHook = renderHook(() =>
      useOptimizedSearch(largeData, "late running", searchFields),
    );

    expect(idleCallback).toHaveBeenCalled();
    expect(largeHook.result.current).toHaveLength(1);
    expect(largeHook.result.current[0]?.agent.name).toBe("Late Match");
  });

  it("renders MetricCard variants, loading state, and click handlers", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    const Icon = ({ className }: { className?: string }) => <svg data-testid="metric-icon" className={className} />;

    const { rerender, container } = render(
      <MetricCard
        label="Executions"
        value="42"
        delta="+5%"
        icon={Icon}
        size="lg"
        variant="default"
        onClick={onClick}
      />,
    );

    expect(screen.getByText("Executions")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("+5%")).toBeInTheDocument();
    expect(screen.getByTestId("metric-icon")).toBeInTheDocument();

    await user.click(screen.getByText("42"));
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".rounded-xl")).toBeTruthy();

    rerender(
      <MetricCard
        label="Latency"
        value="120ms"
        delta="-10%"
        size="sm"
        variant="compact"
        loading
      />,
    );

    expect(screen.getByText("Latency")).toBeInTheDocument();
    expect(screen.queryByText("120ms")).not.toBeInTheDocument();
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
  });
});