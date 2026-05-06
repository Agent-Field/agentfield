import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type {
    AgentState,
    HealthStatus,
    LifecycleStatus,
} from "@/types/agentfield";
import {
    StatusBadge,
    getHealthScoreColor,
    getHealthScoreBadgeVariant,
} from "@/components/status/StatusBadge";

describe("StatusBadge", () => {
    it("renders Unknown when no props are passed", () => {
        render(<StatusBadge />);
        expect(screen.getByText("Unknown")).toBeInTheDocument();
    });

    describe("state prop", () => {
        const cases: [AgentState, string][] = [
            ["active", "Active"],
            ["inactive", "Inactive"],
            ["starting", "Starting"],
            ["stopping", "Stopping"],
            ["error", "Error"],
        ];

        it.each(cases)("state='%s' → label '%s'", (state, expectedLabel) => {
            render(<StatusBadge state={state} />);
            expect(screen.getByText(expectedLabel)).toBeInTheDocument();
        });
    });

    describe("healthStatus prop", () => {
        const cases: [HealthStatus, string][] = [
            ["active", "Healthy"],
            ["inactive", "Unhealthy"],
            ["starting", "Starting"],
            ["ready", "Ready"],
            ["degraded", "Degraded"],
            ["offline", "Offline"],
            ["unknown", "Unknown"],
        ];

        it.each(cases)(
            "healthStatus='%s' → label '%s'",
            (healthStatus, expectedLabel) => {
                render(<StatusBadge healthStatus={healthStatus} />);
                expect(screen.getByText(expectedLabel)).toBeInTheDocument();
            },
        );
    });

    describe("lifecycleStatus prop", () => {
        const cases: [LifecycleStatus, string][] = [
            ["starting", "Starting"],
            ["ready", "Ready"],
            ["degraded", "Degraded"],
            ["offline", "Offline"],
            ["running", "Running"],
            ["stopped", "Stopped"],
            ["error", "Error"],
            ["unknown", "Unknown"],
        ];

        it.each(cases)(
            "lifecycleStatus='%s' → label '%s'",
            (lifecycleStatus, expectedLabel) => {
                render(<StatusBadge lifecycleStatus={lifecycleStatus} />);
                expect(screen.getByText(expectedLabel)).toBeInTheDocument();
            },
        );
    });

    describe("priority: state takes precedence over healthStatus and lifecycleStatus", () => {
        it("shows state label when both state and healthStatus are provided", () => {
            render(<StatusBadge state="active" healthStatus="degraded" />);
            expect(screen.getByText("Active")).toBeInTheDocument();
            expect(screen.queryByText("Degraded")).not.toBeInTheDocument();
        });

        it("shows healthStatus label when state is absent but lifecycleStatus is also provided", () => {
            render(
                <StatusBadge
                    healthStatus="degraded"
                    lifecycleStatus="running"
                />,
            );
            expect(screen.getByText("Degraded")).toBeInTheDocument();
            expect(screen.queryByText("Running")).not.toBeInTheDocument();
        });
    });

    describe("showIcon prop", () => {
        it("renders fewer icons when showIcon is false", () => {
            const { container: withIcon } = render(
                <StatusBadge state="active" showIcon={true} />,
            );
            const { container: withoutIcon } = render(
                <StatusBadge state="active" showIcon={false} />,
            );
            expect(withoutIcon.querySelectorAll("svg").length).toBeLessThan(
                withIcon.querySelectorAll("svg").length,
            );
        });
    });

    describe("size prop", () => {
        it.each(["sm", "md", "lg"] as const)(
            "renders without error at size='%s'",
            (size) => {
                render(<StatusBadge state="active" size={size} />);
                expect(screen.getByText("Active")).toBeInTheDocument();
            },
        );
    });

    describe("status prop (AgentStatus object)", () => {
        it("renders state from status.state", () => {
            render(<StatusBadge status={{ status: "ok", state: "active" }} />);
            expect(screen.getByText("Active")).toBeInTheDocument();
        });

        it("appends health_score when showHealthScore=true", () => {
            render(
                <StatusBadge
                    status={{ status: "ok", state: "active", health_score: 85 }}
                    showHealthScore={true}
                />,
            );
            expect(screen.getByText("Active (85%)")).toBeInTheDocument();
        });

        it("does not append health_score when showHealthScore=false (default)", () => {
            render(
                <StatusBadge
                    status={{ status: "ok", state: "active", health_score: 85 }}
                />,
            );
            expect(screen.getByText("Active")).toBeInTheDocument();
            expect(screen.queryByText(/85%/)).not.toBeInTheDocument();
        });

        it("renders state_transition arrow with target state label", () => {
            render(
                <StatusBadge
                    status={{
                        status: "ok",
                        state: "starting",
                        state_transition: { from: "inactive", to: "active" },
                    }}
                />,
            );
            expect(screen.getByText("Starting")).toBeInTheDocument();
            expect(screen.getByText(/→ Active/)).toBeInTheDocument();
        });

        it("applies animate-pulse class during transition", () => {
            const { container } = render(
                <StatusBadge
                    status={{
                        status: "ok",
                        state: "starting",
                        state_transition: { from: "inactive", to: "active" },
                    }}
                />,
            );
            expect(container.querySelector(".animate-pulse")).not.toBeNull();
        });
    });
});

import {
    AgentStateBadge,
    HealthStatusBadge,
    LifecycleStatusBadge,
} from "@/components/status/StatusBadge";

describe("AgentStateBadge", () => {
    it("renders the correct label for a given state", () => {
        render(<AgentStateBadge state="active" />);
        expect(screen.getByText("Active")).toBeInTheDocument();
    });
});

describe("HealthStatusBadge", () => {
    it("renders the correct label for a given healthStatus", () => {
        render(<HealthStatusBadge healthStatus="degraded" />);
        expect(screen.getByText("Degraded")).toBeInTheDocument();
    });
});

describe("LifecycleStatusBadge", () => {
    it("renders the correct label for a given lifecycleStatus", () => {
        render(<LifecycleStatusBadge lifecycleStatus="running" />);
        expect(screen.getByText("Running")).toBeInTheDocument();
    });
});

describe("getHealthScoreColor", () => {
    it("returns success accent for score >= 90", () => {
        expect(getHealthScoreColor(90)).toBe(getHealthScoreColor(100));
        expect(getHealthScoreColor(90)).not.toBe(getHealthScoreColor(89));
    });

    it("returns info accent for score >= 70 and < 90", () => {
        expect(getHealthScoreColor(70)).toBe(getHealthScoreColor(89));
        expect(getHealthScoreColor(70)).not.toBe(getHealthScoreColor(90));
    });

    it("returns warning accent for score >= 50 and < 70", () => {
        expect(getHealthScoreColor(50)).toBe(getHealthScoreColor(69));
        expect(getHealthScoreColor(50)).not.toBe(getHealthScoreColor(70));
    });

    it("returns error accent for score < 50", () => {
        expect(getHealthScoreColor(0)).toBe(getHealthScoreColor(49));
        expect(getHealthScoreColor(0)).not.toBe(getHealthScoreColor(50));
    });
});

describe("getHealthScoreBadgeVariant", () => {
    it("returns 'success' for score >= 90", () => {
        expect(getHealthScoreBadgeVariant(90)).toBe("success");
        expect(getHealthScoreBadgeVariant(100)).toBe("success");
    });

    it("returns 'running' for score >= 70 and < 90", () => {
        expect(getHealthScoreBadgeVariant(70)).toBe("running");
        expect(getHealthScoreBadgeVariant(89)).toBe("running");
    });

    it("returns 'pending' for score >= 50 and < 70", () => {
        expect(getHealthScoreBadgeVariant(50)).toBe("pending");
        expect(getHealthScoreBadgeVariant(69)).toBe("pending");
    });

    it("returns 'failed' for score < 50", () => {
        expect(getHealthScoreBadgeVariant(49)).toBe("failed");
        expect(getHealthScoreBadgeVariant(0)).toBe("failed");
    });
});
