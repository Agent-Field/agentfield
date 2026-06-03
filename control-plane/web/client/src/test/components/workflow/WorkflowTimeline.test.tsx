// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { WorkflowTimeline } from "@/components/workflow/WorkflowTimeline";
import type { WorkflowTimelineNode } from "@/types/workflows";

const state = vi.hoisted(() => ({
  getExecutionNotes: vi.fn<(executionId: string) => Promise<{ execution_id: string; notes: Array<{ message: string; tags: string[]; timestamp: string }>; total: number }>>(),
}));

vi.mock("@/services/executionsApi", () => ({
  getExecutionNotes: (executionId: string, filters: unknown) => state.getExecutionNotes(executionId, filters),
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
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/notes/TagBadge", () => ({
  TagBadge: ({
    tag,
    onClick,
  }: {
    tag: string;
    onClick?: () => void;
  }) => (
    <button type="button" onClick={onClick}>
      {`tag:${tag}`}
    </button>
  ),
}));

vi.mock("@/components/workflow/TimelineNodeCard", () => ({
  TimelineNodeCard: ({
    node,
    notes,
    onClick,
    onTagClick,
  }: {
    node: WorkflowTimelineNode;
    notes: Array<{ tags: string[] }>;
    onClick?: () => void;
    onTagClick?: (tag: string) => void;
  }) => (
    <div>
      <button type="button" onClick={onClick}>
        {`node:${node.execution_id}:${notes.length}`}
      </button>
      <button type="button" onClick={() => onTagClick?.(notes[0]?.tags[0] ?? "none")}>
        {`tag-action:${node.execution_id}`}
      </button>
    </div>
  ),
  TimelineNodeCardSkeleton: () => <div>timeline-loading</div>,
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  ArrowDown: () => <span>down</span>,
  ArrowUp: () => <span>up</span>,
  CollapseAll: () => <span>collapse</span>,
  Events: () => <span>events</span>,
  ExpandAll: () => <span>expand</span>,
  Filter: () => <span>filter</span>,
}));

function createNode(overrides: Partial<WorkflowTimelineNode> = {}): WorkflowTimelineNode {
  return {
    workflow_id: "wf-1",
    execution_id: "exec-1",
    agent_node_id: "agent-1",
    reasoner_id: "planner",
    status: "running",
    started_at: "2026-04-08T10:00:00Z",
    completed_at: "2026-04-08T10:01:00Z",
    duration_ms: 60000,
    workflow_depth: 0,
    agent_name: "Agent One",
    task_name: "Plan",
    ...overrides,
  };
}

describe("WorkflowTimeline", () => {
  beforeEach(() => {
    state.getExecutionNotes.mockReset();
    state.getExecutionNotes.mockImplementation(async (executionId: string) => ({
      execution_id: executionId,
      notes:
        executionId === "exec-1"
          ? [{ message: "Ops note", tags: ["ops"], timestamp: "2026-04-08T10:02:00Z" }]
          : [{ message: "QA note", tags: ["qa"], timestamp: "2026-04-08T10:03:00Z" }],
      total: 1,
    }));
  });

  it("fetches notes, renders tag filters, and wires node and filter actions", async () => {
    const onNodeSelect = vi.fn();
    const onTagFilter = vi.fn();
    const onSortOrderChange = vi.fn();
    const onExpandAllChange = vi.fn();
    const nodes = [
      createNode({ execution_id: "exec-1", started_at: "2026-04-08T10:00:00Z" }),
      createNode({ execution_id: "exec-2", started_at: "2026-04-08T10:05:00Z", reasoner_id: "review" }),
    ];

    render(
      <WorkflowTimeline
        nodes={nodes}
        onNodeSelect={onNodeSelect}
        onTagFilter={onTagFilter}
        sortOrder="asc"
        onSortOrderChange={onSortOrderChange}
        expandAll={false}
        onExpandAllChange={onExpandAllChange}
      />
    );

    expect(screen.getAllByText("timeline-loading").length).toBeGreaterThan(0);

    await waitFor(() => {
      expect(screen.getByText("Workflow Timeline")).toBeInTheDocument();
    });

    expect(screen.getByText("(2 notes)")).toBeInTheDocument();
    expect(screen.getByText("tag:ops")).toBeInTheDocument();
    expect(screen.getByText("tag:qa")).toBeInTheDocument();
    expect(screen.getByText("node:exec-1:1")).toBeInTheDocument();
    expect(screen.getByText("node:exec-2:1")).toBeInTheDocument();

    fireEvent.click(screen.getByTitle("Sort newest first"));
    expect(onSortOrderChange).toHaveBeenCalledWith("desc");

    fireEvent.click(screen.getByTitle("Expand all notes"));
    expect(onExpandAllChange).toHaveBeenCalledWith(true);

    fireEvent.click(screen.getByText("tag:ops"));
    expect(onTagFilter).toHaveBeenCalledWith(["ops"]);

    fireEvent.click(screen.getByText("node:exec-1:1"));
    expect(onNodeSelect).toHaveBeenCalledWith(expect.objectContaining({ execution_id: "exec-1" }));

    fireEvent.click(screen.getByText("tag-action:exec-2"));
    expect(onTagFilter).toHaveBeenCalledWith(["qa"]);
  });

  it("shows the filtered empty state and clears filters", async () => {
    const onTagFilter = vi.fn();
    const nodes = [createNode({ execution_id: "exec-1" })];

    render(
      <WorkflowTimeline
        nodes={nodes}
        selectedTags={["missing"]}
        onTagFilter={onTagFilter}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("No nodes match the selected filters.")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Clear filters"));
    expect(onTagFilter).toHaveBeenCalledWith([]);
  });
});