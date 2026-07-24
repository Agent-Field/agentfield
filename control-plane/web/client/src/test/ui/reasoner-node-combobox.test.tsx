// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { ReasonerNodeCombobox } from "@/components/ui/reasoner-node-combobox";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    AgentNodeIcon: Icon,
    Check: Icon,
    ChevronDown: Icon,
    ReasonerIcon: Icon,
    Search: Icon,
  };
});

const reasoners = [
  {
    node_id: "node-a",
    name: "Alpha Skill",
    reasoner_id: "alpha.skill",
    description: "First skill",
  },
  {
    node_id: "node-a",
    name: "Beta Skill",
    reasoner_id: "beta.skill",
    description: "Second skill",
  },
  {
    node_id: "node-b",
    name: "Gamma Skill",
    reasoner_id: "gamma.skill",
    description: "Searchable alternative",
  },
] as const;

beforeAll(() => {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  vi.stubGlobal("ResizeObserver", ResizeObserverMock);
});

describe("ReasonerNodeCombobox", () => {
  it("renders loading and disabled states", () => {
    render(
      <ReasonerNodeCombobox
        reasoners={[]}
        value={null}
        onValueChange={vi.fn()}
        loading
      />,
    );

    expect(screen.getByRole("combobox", { name: "Select agent node and skill" })).toBeDisabled();
    expect(screen.getByText("Loading skills…")).toBeInTheDocument();
  });

  it("opens, filters nodes and skills, and selects a reasoner", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    render(
      <ReasonerNodeCombobox
        reasoners={[...reasoners]}
        value={null}
        onValueChange={onValueChange}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Select agent node and skill" }));
    expect(await screen.findByText("node-a")).toBeInTheDocument();
    expect(screen.getByText("2 skills")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("Search nodes, skills, or full id…"), "gamma");
    expect(screen.getByText("node-b")).toBeInTheDocument();
    expect(screen.getByText("1 skill")).toBeInTheDocument();

    await user.click(screen.getByText("Gamma Skill"));
    expect(onValueChange).toHaveBeenCalledWith("gamma.skill");
  });

  it("shows the selected node and skill in the trigger", () => {
    render(
      <ReasonerNodeCombobox
        reasoners={[...reasoners]}
        value="beta.skill"
        onValueChange={vi.fn()}
      />,
    );

    expect(screen.getByText("node-a")).toBeInTheDocument();
    expect(screen.getByText("Beta Skill")).toBeInTheDocument();
  });
});