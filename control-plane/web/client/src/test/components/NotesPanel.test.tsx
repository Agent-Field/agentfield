import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotesPanel } from "@/components/notes/NotesPanel";
import type { ExecutionNote } from "@/types/notes";

const state = vi.hoisted(() => ({
  getExecutionNotes: vi.fn(),
  getExecutionNoteTags: vi.fn(),
}));

vi.mock("@/services/executionsApi", () => ({
  getExecutionNotes: (...args: unknown[]) => state.getExecutionNotes(...args),
  getExecutionNoteTags: (...args: unknown[]) => state.getExecutionNoteTags(...args),
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

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <section className={className}>{children}</section>
  ),
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <h2 className={className}>{children}</h2>
  ),
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
}));

vi.mock("@/components/notes/NoteCard", () => ({
  NoteCard: ({
    note,
    onTagClick,
  }: {
    note: ExecutionNote;
    onTagClick?: (tag: string) => void;
  }) => (
    <div data-testid="note-card">
      <span>{note.message}</span>
      {note.tags[0] ? (
        <button type="button" onClick={() => onTagClick?.(note.tags[0])}>
          Select {note.tags[0]}
        </button>
      ) : null}
    </div>
  ),
  NoteCardSkeleton: () => <div>Loading note skeleton</div>,
}));

vi.mock("@/components/notes/TagFilter", () => ({
  TagFilter: ({
    availableTags,
    selectedTags,
    onTagsChange,
  }: {
    availableTags: string[];
    selectedTags: string[];
    onTagsChange: (tags: string[]) => void;
  }) => (
    <div>
      <div>Selected tags: {selectedTags.join(",") || "none"}</div>
      <button
        type="button"
        onClick={() => onTagsChange(availableTags[0] ? [availableTags[0]] : [])}
      >
        Apply first tag
      </button>
      <button type="button" onClick={() => onTagsChange([])}>
        Reset tags
      </button>
    </div>
  ),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string; size?: number }) => <svg data-testid="icon" className={className} />;
  return {
    ArrowDown: Icon,
    ArrowUp: Icon,
    Document: Icon,
    Renew: Icon,
    Chat: Icon,
  };
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("NotesPanel", () => {
  beforeEach(() => {
    state.getExecutionNotes.mockReset();
    state.getExecutionNoteTags.mockReset();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("renders loading first and then the empty state", async () => {
    const notesRequest = deferred<{ execution_id: string; notes: ExecutionNote[]; total: number }>();
    state.getExecutionNotes.mockReturnValue(notesRequest.promise);
    state.getExecutionNoteTags.mockResolvedValue([]);

    render(<NotesPanel executionId="exec-1" />);

    expect(screen.getAllByText("Loading note skeleton")).toHaveLength(3);

    notesRequest.resolve({
      execution_id: "exec-1",
      notes: [],
      total: 0,
    });

    expect(await screen.findByText("No notes yet")).toBeInTheDocument();
    expect(screen.getByText(/Use app.note\(\)/i)).toBeInTheDocument();
  });

  it("renders an error state and retries successfully", async () => {
    const user = userEvent.setup();

    state.getExecutionNotes.mockRejectedValueOnce(new Error("notes failed"));
    state.getExecutionNoteTags.mockResolvedValue([]);

    render(<NotesPanel executionId="exec-2" />);

    expect(await screen.findByText("Failed to load notes")).toBeInTheDocument();
    expect(screen.getByText("notes failed")).toBeInTheDocument();

    state.getExecutionNotes.mockResolvedValueOnce({
      execution_id: "exec-2",
      notes: [],
      total: 0,
    });

    await user.click(screen.getByRole("button", { name: /Try Again/i }));

    expect(await screen.findByText("No notes yet")).toBeInTheDocument();
  });

  it("renders notes, applies tag filters, toggles sort order, and refreshes", async () => {
    const user = userEvent.setup();

    const notes: ExecutionNote[] = [
      {
        message: "newer note",
        tags: ["alpha"],
        timestamp: "2026-04-08T11:00:00Z",
      },
      {
        message: "older note",
        tags: ["beta"],
        timestamp: "2026-04-08T10:00:00Z",
      },
    ];

    state.getExecutionNotes.mockResolvedValue({
      execution_id: "exec-3",
      notes,
      total: notes.length,
    });
    state.getExecutionNoteTags.mockResolvedValue(["alpha", "beta"]);

    render(<NotesPanel executionId="exec-3" />);

    expect(await screen.findByText("newer note")).toBeInTheDocument();
    expect(screen.getByText("(2 of 2)")).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/i)).toBeInTheDocument();

    const beforeFilterCards = screen.getAllByTestId("note-card");
    expect(beforeFilterCards[0]).toHaveTextContent("newer note");
    expect(beforeFilterCards[1]).toHaveTextContent("older note");

    await user.click(screen.getByRole("button", { name: "Apply first tag" }));
    await waitFor(() => {
      expect(state.getExecutionNotes).toHaveBeenLastCalledWith("exec-3", { tags: ["alpha"] });
    });
    expect(screen.getByText("(1 of 2)")).toBeInTheDocument();
    expect(screen.getByText("newer note")).toBeInTheDocument();
    expect(screen.queryByText("older note")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Reset tags" }));
    expect(await screen.findByText("(2 of 2)")).toBeInTheDocument();

    await user.click(screen.getByTitle("Sort oldest first"));
    const ascendingCards = screen.getAllByTestId("note-card");
    expect(ascendingCards[0]).toHaveTextContent("older note");
    expect(ascendingCards[1]).toHaveTextContent("newer note");

    await user.click(screen.getByTitle("Refresh notes"));
    expect(state.getExecutionNotes).toHaveBeenCalledTimes(4);
  });
});
