import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";

describe("ResponsiveGrid", () => {
  it("applies variant defaults and composed utility classes", () => {
    render(
      <ResponsiveGrid variant="metrics" flow="dense" className="custom-grid">
        <div>One</div>
        <div>Two</div>
      </ResponsiveGrid>,
    );

    const grid = screen.getByText("One").parentElement;
    expect(grid).toHaveClass("grid");
    expect(grid?.className).toContain("gap-4");
    expect(grid?.className).toContain("items-center");
    expect(grid?.className).toContain("grid-flow-dense");
    expect(grid?.className).toContain("sm:grid-cols-2");
    expect(grid?.className).toContain("lg:grid-cols-4");
    expect(grid?.className).toContain("custom-grid");
  });

  it("prefers explicit columns, gap, and align props over preset defaults and forwards refs", () => {
    const ref = React.createRef<HTMLDivElement>();

    render(
      <ResponsiveGrid
        ref={ref}
        preset="quarters"
        columns={{ base: 2, xl: 5 }}
        gap="none"
        align="start"
        flow="col"
        data-testid="grid"
      >
        <div>Child</div>
      </ResponsiveGrid>,
    );

    const grid = screen.getByTestId("grid");
    expect(ref.current).toBe(grid);
    expect(grid.className).toContain("gap-0");
    expect(grid.className).toContain("items-start");
    expect(grid.className).toContain("grid-flow-col");
    expect(grid.className).toContain("grid-cols-2");
    expect(grid.className).toContain("xl:grid-cols-5");
    expect(grid.className).not.toContain("sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4");
  });

  it("renders grid items with element overrides and responsive spans", () => {
    render(
      <ResponsiveGrid>
        <ResponsiveGrid.Item
          as="section"
          span={{ base: 2, md: 3, xl: 4 }}
          className="item-extra"
          data-testid="grid-item"
        >
          Item
        </ResponsiveGrid.Item>
      </ResponsiveGrid>,
    );

    const item = screen.getByTestId("grid-item");
    expect(item.tagName).toBe("SECTION");
    expect(item.className).toContain("col-span-2");
    expect(item.className).toContain("md:col-span-3");
    expect(item.className).toContain("xl:col-span-4");
    expect(item.className).toContain("item-extra");
  });
});
