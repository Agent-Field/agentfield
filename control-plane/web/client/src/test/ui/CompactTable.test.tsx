// @ts-nocheck
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CompactTable } from "@/components/ui/CompactTable";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    CaretDown: Icon,
    CaretRight: Icon,
    CaretUp: Icon,
    SpinnerGap: Icon,
  };
});

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count, estimateSize }: { count: number; estimateSize: () => number }) => {
    const size = estimateSize();
    return {
      getVirtualItems: () =>
        Array.from({ length: count }, (_, index) => ({
          index,
          key: `row-${index}`,
          size,
          start: index * size,
        })),
      getTotalSize: () => count * size,
    };
  },
}));

type Row = { id: string; name: string; status: string };

const columns = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    render: (row: Row) => row.name,
  },
  {
    key: "status",
    header: "Status",
    align: "right" as const,
    render: (row: Row) => row.status,
  },
];

describe("CompactTable", () => {
  it("renders loading and empty states with actions", async () => {
    const user = userEvent.setup();
    const retry = vi.fn();
    const secondary = vi.fn();
    const { rerender } = render(
      <CompactTable<Row>
        data={[]}
        loading
        hasMore={false}
        isFetchingMore={false}
        sortBy="name"
        sortOrder="asc"
        onSortChange={vi.fn()}
        columns={columns}
        gridTemplate="1fr 1fr"
        getRowKey={(row) => row.id}
      />,
    );

    expect(screen.getAllByText((_, element) => element?.className.includes("animate-pulse") ?? false).length).toBeGreaterThan(0);

    rerender(
      <CompactTable<Row>
        data={[]}
        loading={false}
        hasMore={false}
        isFetchingMore={false}
        sortBy="name"
        sortOrder="asc"
        onSortChange={vi.fn()}
        columns={columns}
        gridTemplate="1fr 1fr"
        getRowKey={(row) => row.id}
        emptyState={{
          title: "Nothing here",
          description: "Try again later",
          action: { label: "Retry", onClick: retry },
          secondaryAction: { label: "Dismiss", onClick: secondary },
        }}
      />,
    );

    expect(screen.getByText("Nothing here")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(retry).toHaveBeenCalledTimes(1);
    expect(secondary).toHaveBeenCalledTimes(1);
  });

  it("renders rows, sorts, loads more, and handles row clicks", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    const onLoadMore = vi.fn();
    const onRowClick = vi.fn();

    render(
      <CompactTable<Row>
        data={[
          { id: "1", name: "Alpha", status: "ready" },
          { id: "2", name: "Beta", status: "paused" },
        ]}
        loading={false}
        hasMore
        isFetchingMore={false}
        sortBy="name"
        sortOrder="asc"
        onSortChange={onSortChange}
        onLoadMore={onLoadMore}
        onRowClick={onRowClick}
        columns={columns}
        gridTemplate="1fr 1fr"
        getRowKey={(row) => row.id}
      />,
    );

    await waitFor(() => {
      expect(onLoadMore).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.getByText("Loading more…")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /name/i }));
    expect(onSortChange).toHaveBeenCalledWith("name");

    await user.click(screen.getByText("Beta"));
    expect(onRowClick).toHaveBeenCalledWith({ id: "2", name: "Beta", status: "paused" });
  });
});