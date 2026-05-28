// @ts-nocheck
import React from "react";
import { fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FilterMultiCombobox } from "@/components/ui/filter-multi-combobox";
import { ChipInput } from "@/components/ui/chip-input";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { ResizableSplitPane, useResponsiveSplitPane } from "@/components/ui/ResizableSplitPane";
import { UnifiedDataPanel } from "@/components/ui/UnifiedDataPanel";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    ArrowDown: Icon,
    ArrowUp: Icon,
    Database: Icon,
    Maximize: Icon,
    InProgress: Icon,
    ChevronLeft: Icon,
    ChevronRight: Icon,
    Search: Icon,
    X: Icon,
  };
});

vi.mock("@/components/ui/copy-button", () => ({
  CopyButton: ({
    value,
    tooltip,
    onClick,
  }: {
    value: string;
    tooltip?: string;
    onClick?: React.MouseEventHandler<HTMLButtonElement>;
  }) => (
    <button type="button" aria-label={tooltip ?? "Copy"} data-value={value} onClick={onClick}>
      copy
    </button>
  ),
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({
    data,
    maxHeight,
    searchable,
    showHeader,
    className,
  }: {
    data: unknown;
    maxHeight?: string;
    searchable?: boolean;
    showHeader?: boolean;
    className?: string;
  }) => (
    <div
      data-testid="json-viewer"
      data-max-height={maxHeight}
      data-searchable={String(searchable)}
      data-show-header={String(showHeader)}
      className={className}
    >
      {JSON.stringify(data)}
    </div>
  ),
}));

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", ResizeObserverMock);
Element.prototype.scrollIntoView ??= () => {};

describe("ui extras panels and inputs", () => {
  it("renders UnifiedDataPanel data, copies content, and opens the modal from header and button", async () => {
    const user = userEvent.setup();
    const onModalOpen = vi.fn();

    render(
      <UnifiedDataPanel
        title="Input payload"
        type="input"
        data={{ alpha: 1, beta: true }}
        size={1536}
        onModalOpen={onModalOpen}
      />,
    );

    expect(screen.getByText("Input payload")).toBeInTheDocument();
    expect(screen.getByText("1.5 KB")).toBeInTheDocument();
    expect(screen.getByText("2 keys")).toBeInTheDocument();
    expect(screen.getByTestId("json-viewer")).toHaveAttribute("data-searchable", "true");
    expect(screen.getByRole("button", { name: "Copy data" })).toHaveAttribute(
      "data-value",
      JSON.stringify({ alpha: 1, beta: true }, null, 2),
    );

    await user.click(screen.getByTitle("Click to expand in modal"));
    await user.click(screen.getByRole("button", { name: "Expand in modal (or click header)" }));

    expect(onModalOpen).toHaveBeenCalledTimes(2);
  });

  it("renders UnifiedDataPanel loading, error, and empty states", () => {
    const { rerender } = render(
      <UnifiedDataPanel title="Output" type="output" data={null} isLoading />,
    );
    expect(screen.getByText("Loading data...")).toBeInTheDocument();

    rerender(
      <UnifiedDataPanel
        title="Output"
        type="output"
        data={null}
        error="network failed"
      />,
    );
    expect(screen.getByText("Failed to load data")).toBeInTheDocument();
    expect(screen.getByText("network failed")).toBeInTheDocument();

    rerender(
      <UnifiedDataPanel
        title="Output"
        type="output"
        data={{}}
        emptyStateConfig={{ title: "Nothing here", description: "No records yet" }}
      />,
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
    expect(screen.getByText("No records yet")).toBeInTheDocument();
  });

  it("resizes and collapses a horizontal split pane", () => {
    const onSizeChange = vi.fn();
    const { container } = render(
      <ResizableSplitPane
        defaultSizePercent={30}
        minSizePercent={20}
        maxSizePercent={60}
        collapsible
        onSizeChange={onSizeChange}
      >
        <div>left</div>
        <div>right</div>
      </ResizableSplitPane>,
    );

    const root = container.firstElementChild as HTMLDivElement;
    const leftPanel = root.firstElementChild as HTMLDivElement;
    const separator = screen.getByRole("separator");

    vi.spyOn(root, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      bottom: 300,
      right: 1000,
      width: 1000,
      height: 300,
      toJSON: () => ({}),
    });

    fireEvent.mouseDown(separator);
    fireEvent.mouseMove(document, { clientX: 800 });
    fireEvent.mouseUp(document);

    expect(onSizeChange).toHaveBeenCalledWith(60);
    expect(leftPanel.style.width).toBe("60%");

    fireEvent.click(screen.getByTitle("Collapse panel"));
    expect(leftPanel.style.width).toBe("48%");
    expect(screen.getByTitle("Expand panel")).toBeInTheDocument();
  });

  it("resizes a vertical split pane and tracks responsive breakpoint changes", () => {
    const onSizeChange = vi.fn();
    const { container } = render(
      <ResizableSplitPane orientation="vertical" defaultSizePercent={40} onSizeChange={onSizeChange}>
        <div>top</div>
        <div>bottom</div>
      </ResizableSplitPane>,
    );

    const root = container.firstElementChild as HTMLDivElement;
    const topPanel = root.firstElementChild as HTMLDivElement;

    vi.spyOn(root, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      bottom: 400,
      right: 800,
      width: 800,
      height: 400,
      toJSON: () => ({}),
    });

    fireEvent.mouseDown(screen.getByRole("separator"));
    fireEvent.mouseMove(document, { clientY: 100 });
    fireEvent.mouseUp(document);

    expect(onSizeChange).toHaveBeenCalledWith(25);
    expect(topPanel.style.height).toBe("25%");

    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 500 });
    const { result } = renderHook(() => useResponsiveSplitPane(600));
    expect(result.current.isSmallScreen).toBe(true);

    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 700 });
    fireEvent(window, new Event("resize"));
    expect(result.current.isSmallScreen).toBe(false);
  });

  it("adds, selects, and removes chips from ChipInput", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [tags, setTags] = React.useState<string[]>(["beta"]);
      return (
        <div>
          <ChipInput
            value={tags}
            onChange={setTags}
            suggestions={["alpha", "beta", "gamma"]}
            placeholder="Add tags"
          />
          <button type="button">outside</button>
        </div>
      );
    }

    render(<Harness />);
    const input = screen.getByRole("textbox");

    await user.click(input);
    await user.type(input, "ga");
    await user.keyboard("{ArrowDown}{Enter}");
    expect(screen.getByText("gamma")).toBeInTheDocument();

    await user.type(screen.getByRole("textbox"), "alpha,");
    expect(screen.getByText("alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Remove gamma" }));
    expect(screen.queryByRole("button", { name: "Remove gamma" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("textbox"));
    await user.click(screen.getByRole("button", { name: "outside" }));
    expect(screen.queryByRole("button", { name: "Remove gamma" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("textbox"));
    await user.keyboard("{Backspace}");
    expect(screen.queryByRole("button", { name: "Remove alpha" })).not.toBeInTheDocument();
  });

  it("filters command items and renders dialog wrappers", async () => {
    const user = userEvent.setup();

    render(
      <div>
        <Command>
          <CommandInput placeholder="Command search" />
          <CommandList>
            <CommandEmpty>No results</CommandEmpty>
            <CommandGroup heading="Actions">
              <CommandItem>Open Dashboard</CommandItem>
              <CommandItem>
                Run Checks
                <CommandShortcut>⌘R</CommandShortcut>
              </CommandItem>
            </CommandGroup>
            <CommandSeparator />
          </CommandList>
        </Command>

        <CommandDialog open onOpenChange={() => undefined}>
          <CommandInput placeholder="Inside dialog" />
          <CommandList>
            <CommandItem>Dialog item</CommandItem>
          </CommandList>
        </CommandDialog>
      </div>,
    );

    fireEvent.input(screen.getByPlaceholderText("Command search"), {
      target: { value: "checks" },
    });

    expect(screen.getByText("Run Checks")).toBeInTheDocument();
    expect(screen.getByText("⌘R")).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Dialog item")).toBeInTheDocument();
  });

  it("toggles selections in FilterMultiCombobox", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [selected, setSelected] = React.useState<Set<string>>(new Set());
      return (
        <FilterMultiCombobox
          label="Status filter"
          emptyLabel="All statuses"
          pluralLabel={(count) => `${count} statuses`}
          options={[
            { value: "ok", label: "Healthy", leading: <span data-testid="ok-dot" /> },
            { value: "err", label: "Errored" },
          ]}
          selected={selected}
          onSelectedChange={(updater) => setSelected((prev) => updater(prev))}
        />
      );
    }

    render(<Harness />);

    const trigger = screen.getByRole("combobox", { name: "Status filter" });

    await user.click(trigger);
    await user.click(await screen.findByText("Healthy"));
    expect(trigger).toHaveTextContent("Healthy");

  });
});