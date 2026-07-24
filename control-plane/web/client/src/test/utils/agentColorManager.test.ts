import { beforeEach, describe, expect, it } from "vitest";

import { agentColorManager } from "@/utils/agentColorManager";

describe("agentColorManager", () => {
  beforeEach(() => {
    agentColorManager.clearCache();
  });

  it("returns stable cached colors and tracks cached agents", () => {
    const first = agentColorManager.getAgentColor("Research Agent");
    const second = agentColorManager.getAgentColor("Research Agent");

    expect(second).toBe(first);
    expect(first).toMatchObject({
      name: "Research Agent",
    });
    expect(first.primary).toMatch(/^hsl\(\d+, \d+%, \d+%\)$/);
    expect(first.background).toMatch(/^hsla\(/);
    expect(first.border).toMatch(/^hsla\(/);
    expect(first.text).toMatch(/^hsl\(/);
    expect(agentColorManager.getAllAgentColors()).toEqual([first]);
  });

  it("falls back to agent id for generic agent names", () => {
    const first = agentColorManager.getAgentColor("agent_12", "node-1");
    const second = agentColorManager.getAgentColor("agent-999", "node-1");

    expect(second).toBe(first);
    expect(agentColorManager.getAllAgentColors()).toHaveLength(1);
  });

  it("cleans up unused agents and rebuilds the cache", () => {
    const alpha = agentColorManager.getAgentColor("Alpha");
    agentColorManager.getAgentColor("Beta");
    agentColorManager.cleanupUnusedAgents(["Alpha"]);

    expect(agentColorManager.getAllAgentColors()).toEqual([alpha]);
  });

  it("derives initials from camel case, separators, single words, and empty names", () => {
    expect(agentColorManager.getAgentInitials("DataReviewerAgent")).toBe("DR");
    expect(agentColorManager.getAgentInitials("data-reviewer")).toBe("DR");
    expect(agentColorManager.getAgentInitials("qa")).toBe("QA");
    expect(agentColorManager.getAgentInitials("x")).toBe("X");
    expect(agentColorManager.getAgentInitials("")).toBe("?");
  });
});
