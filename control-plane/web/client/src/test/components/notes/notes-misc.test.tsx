// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NoteCard, NoteCardSkeleton } from "@/components/notes/NoteCard";
import { TagFilter } from "@/components/notes/TagFilter";
import type { ExecutionNote } from "@/types/notes";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    Checkmark: Icon,
    ChevronDown: Icon,
    Filter: Icon,
    Time: Icon,
    Copy: Icon,
    Chat: Icon,
  };
});

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
}));

vi.mock("@/components/notes/TagBadge", () => ({
  TagBadge: ({
    tag,
    removable,
    onRemove,
    onClick,
  }: {
    tag: string;
    removable?: boolean;
    onRemove?: (tag: string) => void;
    onClick?: (tag: string) => void;
  }) => (
    <div>
      <button type="button" onClick={() => onClick?.(tag)}>
        tag:{tag}
      </button>
      {removable && (
        <button type="button" aria-label={`remove-${tag}`} onClick={() => onRemove?.(tag)}>
          remove
        </button>
      )}
    </div>
  ),
}));

describe("notes misc components", () => {
  const originalClipboard = navigator.clipboard;
  let clipboardWriteText: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T12:00:00Z"));
    clipboardWriteText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWriteText },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: originalClipboard,
    });
    vi.restoreAllMocks();
  });

  it("filters, selects, removes, clears, and closes the tag filter", () => {
    const onTagsChange = vi.fn();

    const { rerender } = render(
      <TagFilter
        availableTags={["alpha", "beta", "gamma"]}
        selectedTags={["alpha"]}
        onTagsChange={onTagsChange}
      />
    );

    expect(screen.getByText("1 tag selected")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /1 tag selected/i }));
    const input = screen.getByPlaceholderText("Search tags...");
    expect(input).toHaveFocus();

    fireEvent.change(input, { target: { value: "be" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onTagsChange).toHaveBeenCalledWith(["alpha", "beta"]);

    rerender(
      <TagFilter
        availableTags={["alpha", "beta", "gamma"]}
        selectedTags={["alpha", "beta"]}
        onTagsChange={onTagsChange}
      />
    );

    expect(screen.getByText("2 tags selected")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /clear all/i }));
    expect(onTagsChange).toHaveBeenCalledWith([]);

    fireEvent.click(screen.getByRole("button", { name: /remove-alpha/i }));
    expect(onTagsChange).toHaveBeenCalledWith(["beta"]);

    const maybeOpenInput = screen.queryByPlaceholderText("Search tags...");
    if (!maybeOpenInput) {
      fireEvent.click(screen.getByRole("button", { name: /2 tags selected/i }));
    }
    fireEvent.change(screen.getByPlaceholderText("Search tags..."), {
      target: { value: "zzz" },
    });
    expect(screen.getByText("No tags found")).toBeInTheDocument();

    fireEvent.mouseDown(document.body);
    expect(screen.queryByPlaceholderText("Search tags...")).not.toBeInTheDocument();
  });

  it("shows disabled state when no tags are available", () => {
    render(<TagFilter availableTags={[]} selectedTags={[]} onTagsChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: /filter by tags/i })).toBeDisabled();
  });

  it("renders a note card, copies content, and handles tag clicks", async () => {
    const onTagClick = vi.fn();
    const note: ExecutionNote = {
      message: "**Important** note body",
      tags: ["ops", "ui"],
      timestamp: "2026-04-08T10:30:00Z",
    };

    render(<NoteCard note={note} onTagClick={onTagClick} />);

    expect(screen.getByText("1h ago")).toBeInTheDocument();
    expect(screen.getByText("Important")).toBeInTheDocument();

    fireEvent.click(screen.getByTitle("Copy note"));
    expect(clipboardWriteText).toHaveBeenCalledWith(
      expect.stringContaining("Tags: ops, ui")
    );

    fireEvent.click(screen.getByRole("button", { name: "tag:ops" }));
    expect(onTagClick).toHaveBeenCalledWith("ops");

    act(() => {
      vi.advanceTimersByTime(2000);
    });
  });

  it("renders older dates and the note skeleton", () => {
    render(
      <>
        <NoteCard
          note={{
            message: "plain text",
            tags: [],
            timestamp: "2025-02-01T10:00:00Z",
          }}
        />
        <NoteCardSkeleton />
      </>
    );

    expect(screen.getByText("Feb 1, 2025")).toBeInTheDocument();
    expect(screen.getAllByTestId("skeleton").length).toBeGreaterThan(0);
  });
});