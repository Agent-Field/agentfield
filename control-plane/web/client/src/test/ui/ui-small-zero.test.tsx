// @ts-nocheck
import * as React from "react";
import { forwardRef } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, afterEach, describe, expect, it, vi } from "vitest";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { AutoExpandingTextarea } from "@/components/ui/auto-expanding-textarea";
import { CopyButton } from "@/components/ui/copy-button";
import {
  CopyIdentifierChip,
  truncateIdMiddle,
  truncateIdTail,
} from "@/components/ui/copy-identifier-chip";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { FilterCombobox } from "@/components/ui/filter-combobox";
import { FilterSelect } from "@/components/ui/FilterSelect";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { Sparkline } from "@/components/ui/Sparkline";
import { TextInput } from "@/components/ui/TextInput";
import { TrendMetricCard } from "@/components/ui/TrendMetricCard";
import { TooltipProvider } from "@/components/ui/tooltip";

vi.mock("vaul", async () => {
  const ReactModule = await import("react");

  const DrawerContext = ReactModule.createContext<{
    open: boolean;
    setOpen: (open: boolean) => void;
  } | null>(null);

  function useDrawerContext() {
    const context = ReactModule.useContext(DrawerContext);
    if (!context) {
      throw new Error("Drawer context missing");
    }
    return context;
  }

  const Root = ({
    open,
    defaultOpen = false,
    onOpenChange,
    children,
  }: React.PropsWithChildren<{
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
  }>) => {
    const [internalOpen, setInternalOpen] = ReactModule.useState(defaultOpen);
    const isControlled = open !== undefined;
    const resolvedOpen = isControlled ? open : internalOpen;

    const setOpen = (nextOpen: boolean) => {
      if (!isControlled) {
        setInternalOpen(nextOpen);
      }
      onOpenChange?.(nextOpen);
    };

    return (
      <DrawerContext.Provider value={{ open: resolvedOpen, setOpen }}>
        <div data-testid="mock-drawer-root">
          {children}
        </div>
      </DrawerContext.Provider>
    );
  };

  const Trigger = ({ asChild, children }: React.PropsWithChildren<{ asChild?: boolean }>) => {
    const { setOpen, open } = useDrawerContext();
    if (asChild && ReactModule.isValidElement(children)) {
      return ReactModule.cloneElement(children, {
        onClick: (event: React.MouseEvent) => {
          children.props.onClick?.(event);
          setOpen(true);
        },
        "aria-expanded": open,
      });
    }
    return <button type="button" onClick={() => setOpen(true)}>{children}</button>;
  };

  const Close = ({ asChild, children }: React.PropsWithChildren<{ asChild?: boolean }>) => {
    const { setOpen } = useDrawerContext();
    if (asChild && ReactModule.isValidElement(children)) {
      return ReactModule.cloneElement(children, {
        onClick: (event: React.MouseEvent) => {
          children.props.onClick?.(event);
          setOpen(false);
        },
      });
    }
    return <button type="button" onClick={() => setOpen(false)}>{children}</button>;
  };

  const Portal = ({ children }: React.PropsWithChildren) => <>{children}</>;

  const Overlay = ReactModule.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    (props, ref) => {
      const { open } = useDrawerContext();
      if (!open) return null;
      return <div ref={ref} data-testid="mock-drawer-overlay" {...props} />;
    },
  );
  Overlay.displayName = "MockDrawerOverlay";

  const Content = ReactModule.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => {
      const { open } = useDrawerContext();
      if (!open) return null;
      return (
        <div ref={ref} role="dialog" {...props}>
          {children}
        </div>
      );
    },
  );
  Content.displayName = "MockDrawerContent";

  const Title = ReactModule.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
    (props, ref) => <h2 ref={ref} {...props} />,
  );
  Title.displayName = "MockDrawerTitle";

  const Description = ReactModule.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
    (props, ref) => <p ref={ref} {...props} />,
  );
  Description.displayName = "MockDrawerDescription";

  return {
    Drawer: {
      Root,
      Trigger,
      Portal,
      Close,
      Overlay,
      Content,
      Title,
      Description,
    },
  };
});

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(
    ({ className }, ref) => <svg ref={ref} data-testid="mock-icon" className={className} />,
  );
  Icon.displayName = "MockIcon";

  return {
    Check: Icon,
    Copy: Icon,
    Search: Icon,
  };
});

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => {};
  Element.prototype.releasePointerCapture ??= () => {};
  Element.prototype.scrollIntoView ??= () => {};
  window.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  window.matchMedia ??= ((query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList);
});

beforeEach(() => {
  vi.useRealTimers();
  vi.stubGlobal("navigator", {
    clipboard: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ui small zero coverage", () => {
  it("renders pagination pieces and handles button clicks", async () => {
    const user = userEvent.setup();
    const onPrevious = vi.fn();
    const onNext = vi.fn();
    const onPage = vi.fn();

    render(
      <Pagination className="custom-pagination">
        <PaginationContent>
          <PaginationPrevious onClick={onPrevious} />
          <PaginationLink isActive onClick={onPage}>
            1
          </PaginationLink>
          <PaginationLink size="default">2</PaginationLink>
          <PaginationEllipsis data-testid="ellipsis" />
          <PaginationNext onClick={onNext} />
        </PaginationContent>
      </Pagination>,
    );

    expect(screen.getByRole("navigation", { name: "Pagination" }).className).toContain(
      "custom-pagination",
    );
    expect(screen.getByRole("button", { name: "1" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "2" }).className).toContain("h-8");
    expect(screen.getByTestId("ellipsis")).toBeInTheDocument();
    expect(screen.getByText("More pages")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Go to previous page" }));
    await user.click(screen.getByRole("button", { name: "1" }));
    await user.click(screen.getByRole("button", { name: "Go to next page" }));

    expect(onPrevious).toHaveBeenCalledTimes(1);
    expect(onPage).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("opens and closes an alert dialog through trigger and action buttons", async () => {
    const user = userEvent.setup();

    render(
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <button type="button">Open alert</button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader className="header-class">
            <AlertDialogTitle>Delete run</AlertDialogTitle>
            <AlertDialogDescription>Permanent action.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="footer-class">
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction>Confirm</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>,
    );

    await user.click(screen.getByRole("button", { name: "Open alert" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Delete run")).toBeInTheDocument();
    expect(screen.getByText("Permanent action.")).toBeInTheDocument();
    expect(screen.getByText("Confirm").className).toContain("bg-destructive");
    expect(screen.getByText("Cancel").className).toContain("border");

    await user.click(screen.getByRole("button", { name: "Confirm" }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("opens and closes a drawer and renders its sections", async () => {
    const user = userEvent.setup();

    render(
      <Drawer>
        <DrawerTrigger asChild>
          <button type="button">Open drawer</button>
        </DrawerTrigger>
        <DrawerContent className="drawer-shell">
          <DrawerHeader className="drawer-header">
            <DrawerTitle>Filters</DrawerTitle>
            <DrawerDescription>Adjust the visible data.</DrawerDescription>
          </DrawerHeader>
          <DrawerFooter className="drawer-footer">
            <DrawerClose asChild>
              <button type="button">Done</button>
            </DrawerClose>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>,
    );

    await user.click(screen.getByRole("button", { name: "Open drawer" }));
    expect(screen.getByText("Filters")).toBeInTheDocument();
    expect(screen.getByText("Adjust the visible data.")).toBeInTheDocument();
    expect(screen.getByText("Filters").className).toContain("text-xl");
    expect(document.querySelector(".drawer-shell")).toBeTruthy();
    expect(document.querySelector(".drawer-header")).toBeTruthy();
    expect(document.querySelector(".drawer-footer")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() => {
      expect(screen.queryByText("Adjust the visible data.")).not.toBeInTheDocument();
    });
  });

  it("selects options in filter combobox and supports disabled non-searchable mode", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    const { rerender } = render(
      <FilterCombobox
        label="Time range"
        value=""
        onValueChange={onValueChange}
        placeholder="Pick range"
        searchPlaceholder="Search ranges"
        emptyMessage="No ranges"
        options={[
          { value: "1h", label: "Last hour" },
          { value: "24h", label: "Last day" },
        ]}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Time range" }));
    expect(screen.getByPlaceholderText("Search ranges")).toBeInTheDocument();
    await user.click(screen.getByText("Last day"));

    expect(onValueChange).toHaveBeenCalledWith("24h");
    await waitFor(() => {
      expect(screen.queryByPlaceholderText("Search ranges")).not.toBeInTheDocument();
    });

    rerender(
      <FilterCombobox
        label="Status"
        value="1h"
        onValueChange={onValueChange}
        searchable={false}
        disabled
        className="narrow-trigger"
        options={[{ value: "1h", label: "Last hour" }]}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Status" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Status" }).className).toContain("narrow-trigger");
  });

  it("copies values with copy button, supports prevented clicks, and resets after timeout", async () => {
    const onCopied = vi.fn();
    const onClick = vi.fn();

    const { rerender } = render(
      <TooltipProvider delayDuration={0}>
        <CopyButton
          value="alpha-123"
          tooltip="Copy value"
          copiedTooltip="Done"
          onCopied={onCopied}
          onClick={onClick}
        />
      </TooltipProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy value" }));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("alpha-123");
    expect(onClick).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(onCopied).toHaveBeenCalledWith("alpha-123");
      expect(screen.getByRole("button", { name: "Done" })).toBeInTheDocument();
    });

    rerender(
      <TooltipProvider delayDuration={0}>
        <CopyButton
          key="prevented"
          value="beta-456"
          onClick={(event) => event.preventDefault()}
        >
          {(copied) => <span>{copied ? "copied" : "ready"}</span>}
        </CopyButton>
      </TooltipProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy to clipboard" }));
    expect(screen.getByText("ready")).toBeInTheDocument();
    expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
  });

  it("renders filter select variations and notifies value changes", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    const { rerender } = render(
      <FilterSelect
        id="state-filter"
        label="State"
        helperText="Choose a status"
        value="closed"
        onValueChange={onValueChange}
        options={[
          { value: "open", label: "Open" },
          { value: "closed", label: "Closed" },
        ]}
      />,
    );

    const select = screen.getByLabelText("State");
    expect(select).toHaveAttribute("aria-describedby", "state-filter-description");
    expect(screen.getByText("Choose a status")).toBeInTheDocument();

    await user.selectOptions(select, "open");
    expect(onValueChange).toHaveBeenCalledWith("open");

    rerender(
      <FilterSelect
        hideLabel
        orientation="stacked"
        disabled
        className="stacked-select"
        value="open"
        onValueChange={onValueChange}
        options={[
          { value: "open", label: "Open" },
          { value: "closed", label: "Closed", disabled: true },
        ]}
      />,
    );

    expect(screen.queryByText("State")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeDisabled();
    expect(screen.getByRole("combobox").className).toContain("stacked-select");
  });

  it("renders sparkline fallback and svg paths", () => {
    const { rerender, container } = render(<Sparkline data={[5]} width={80} height={24} />);

    expect(screen.getByText("—")).toBeInTheDocument();

    rerender(
      <Sparkline
        data={[2, 4, 3, 8]}
        width={80}
        height={24}
        color="rgb(1 2 3)"
        showArea={false}
        className="sparkline-chart"
      />,
    );

    const svg = container.querySelector("svg");
    const paths = container.querySelectorAll("path");

    expect(svg).toHaveAttribute("viewBox", "0 0 80 24");
    expect(svg?.className.baseVal).toContain("sparkline-chart");
    expect(svg).toHaveStyle({ color: "rgb(1 2 3)" });
    expect(paths).toHaveLength(1);
    expect(paths[0]?.getAttribute("d")).toContain("M");
  });

  it("renders trend metric card states including loading, subtitle, trend, and sparkline", () => {
    const TrendIcon = forwardRef<SVGSVGElement, { className?: string }>(({ className }, ref) => (
      <svg ref={ref} data-testid="trend-icon" className={className} />
    ));
    TrendIcon.displayName = "TrendIcon";

    const { rerender, container } = render(
      <TrendMetricCard label="Latency" value="125 ms" loading />,
    );

    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);

    rerender(
      <TrendMetricCard
        label="Requests"
        value="10.2k"
        currentValue={120}
        previousValue={100}
        trendPolarity="up-is-good"
        sparklineData={[10, 11, 13, 12]}
        icon={TrendIcon}
      />,
    );

    expect(screen.getByText("Requests")).toBeInTheDocument();
    expect(screen.getByText("10.2k")).toBeInTheDocument();
    expect(screen.getByText("↑ +20.0% vs prev")).toBeInTheDocument();
    expect(screen.getByTestId("trend-icon")).toBeInTheDocument();
    expect(container.querySelector("svg")).toBeTruthy();

    rerender(
      <TrendMetricCard
        label="Errors"
        value="7"
        subtitle="No comparison yet"
      />,
    );

    expect(screen.getByText("No comparison yet")).toBeInTheDocument();
    expect(screen.queryByText(/vs prev/)).not.toBeInTheDocument();
  });

  it("renders text input metadata and updates value", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    const { rerender } = render(
      <TextInput
        id="query"
        label="Query"
        description="Use exact match."
        helperText="Press enter to search."
        errorText="Required field"
        value=""
        onChange={onChange}
      />,
    );

    const input = screen.getByLabelText("Query");
    expect(input).toHaveAttribute(
      "aria-describedby",
      "query-description query-helper query-error",
    );
    expect(screen.getByText("Use exact match.")).toBeInTheDocument();
    expect(screen.getByText("Press enter to search.")).toBeInTheDocument();
    expect(screen.getByText("Required field")).toBeInTheDocument();
    expect(input.className).toContain("border-status-error");

    await user.type(input, "abc");
    expect(onChange).toHaveBeenCalled();

    rerender(<TextInput hideLabel label="Query" aria-label="Hidden query" value="x" onChange={onChange} />);
    expect(screen.queryByText("Query")).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Hidden query" })).toHaveValue("x");
  });

  it("renders copy identifier chip states and truncation helpers", () => {
    expect(truncateIdTail("abcdef", 3)).toBe("…def");
    expect(truncateIdTail("abc", 0)).toBe("…c");
    expect(truncateIdMiddle("abcdefghijklmnop", 8)).toBe("abcde…nop");
    expect(truncateIdMiddle("short", 20)).toBe("short");

    const { rerender } = render(
      <TooltipProvider delayDuration={0}>
        <CopyIdentifierChip
          label="Run"
          value="run_123456789"
          tooltip="Copy run id"
          formatDisplay={(value) => value.toUpperCase()}
        />
      </TooltipProvider>,
    );

    expect(screen.getByText("Run")).toBeInTheDocument();
    expect(screen.getByText("RUN_123456789")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy run id" })).toBeInTheDocument();

    rerender(
      <CopyIdentifierChip
        value={undefined}
        tooltip="Copy"
        noValueMessage="No id available"
        noValueTitle="Missing id"
      />,
    );

    expect(screen.getByText("No id available")).toHaveAttribute("title", "Missing id");

    rerender(<CopyIdentifierChip value={undefined} tooltip="Copy" />);
    expect(screen.queryByText("No id available")).not.toBeInTheDocument();
  });

  it("auto-expands textarea height, sets overflow, and forwards events and refs", () => {
    const onInput = vi.fn();
    const onChange = vi.fn();
    const ref = React.createRef<HTMLTextAreaElement>();
    let currentScrollHeight = 140;
    vi.spyOn(HTMLTextAreaElement.prototype, "scrollHeight", "get").mockImplementation(
      () => currentScrollHeight,
    );

    const { unmount } = render(
      <AutoExpandingTextarea
        ref={ref}
        defaultValue="hello"
        maxHeight={90}
        onInput={onInput}
        onChange={onChange}
      />,
    );

    const textarea = screen.getByRole("textbox");
    expect(textarea.style.height).toBe("90px");
    expect(textarea.style.overflowY).toBe("auto");

    fireEvent.input(textarea, { target: { value: "line 1\nline 2" } });
    expect(onInput).toHaveBeenCalledTimes(1);
    expect(textarea.style.height).toBe("90px");
    expect(textarea.style.overflowY).toBe("auto");

    onChange.mockClear();
    fireEvent.change(textarea, { target: { value: "short" } });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(ref.current).toBe(textarea);

    unmount();

    currentScrollHeight = 60;

    render(
      <AutoExpandingTextarea
        defaultValue="short"
        maxHeight={90}
      />,
    );

    const shortTextarea = screen.getByRole("textbox");
    expect(shortTextarea.style.height).toBe("60px");
    expect(shortTextarea.style.overflowY).toBe("hidden");
  });
});
