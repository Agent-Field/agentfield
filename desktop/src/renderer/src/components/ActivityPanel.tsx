import { Fragment, useEffect, useRef, useState } from "react";
import type { ReactElement } from "react";
import { createPortal } from "react-dom";
import { m, useReducedMotion } from "motion/react";
import type { ExecutionsResult, ExecutionSummary } from "../../../shared/types";
import { EmptyState } from "./EmptyMark";
import {
    formatDuration,
    formatElapsed,
    formatFullStarted,
    formatRelativeStarted,
} from "../../../shared/timeFormat";
import { COMMUNITY_LINKS } from "./communityLinks";

/** Native OS tooltips feel ~1s late — show our tip almost immediately. */
const TIP_SHOW_MS = 40;

type ActivityFilter = "all" | "running" | "succeeded" | "failed";

/** Rows rendered before the first "Show more" click, and per click after. */
const PAGE_SIZE = 50;

/** Optional per-run usage fields — present when the control plane exposes them. */
type ExecutionUsageFields = {
    totalTokens?: number;
    costUsd?: number;
};

interface ActivityPanelProps {
    executions: ExecutionsResult | null;
    controlPlaneUp: boolean;
    /** Optional header note; unused by App today — kept optional so callers don't break. */
    usage?: unknown;
}

/** Keep relative / live elapsed labels fresh between snapshot polls. */
function useTickingNow(intervalMs = 1000): number {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        const id = window.setInterval(() => setNow(Date.now()), intervalMs);
        return () => window.clearInterval(id);
    }, [intervalMs]);
    return now;
}

function formatTokens(n: number): string {
    if (n >= 10_000) return `${Math.round(n / 1000)}k tok`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k tok`;
    return `${n} tok`;
}

function formatCost(usd: number): string {
    if (usd >= 0.01) return `$${usd.toFixed(2)}`;
    return `$${usd.toFixed(4)}`;
}

function readUsage(run: ExecutionSummary): ExecutionUsageFields {
    const row = run as ExecutionSummary & ExecutionUsageFields;
    return {
        totalTokens:
            typeof row.totalTokens === "number" ? row.totalTokens : undefined,
        costUsd: typeof row.costUsd === "number" ? row.costUsd : undefined,
    };
}

function isFailed(run: ExecutionSummary): boolean {
    return run.status === "failed" || run.status === "timeout";
}

function statusBadgeClass(
    status: string,
    live: boolean,
    failed: boolean,
): string {
    if (live || status === "succeeded") return "running";
    if (failed) return "warn";
    return "stopped";
}

const STATUS_GLYPH: Record<string, string> = {
    succeeded: "✓",
    failed: "✕",
    cancelled: "⊘",
    timeout: "⏱",
};

const STATUS_LABEL: Record<string, string> = {
    succeeded: "Succeeded",
    failed: "Failed",
    cancelled: "Cancelled",
    timeout: "Timed out",
    running: "Running",
};

/** True when the timestamp falls on the current local date. */
function isToday(iso: string): boolean {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return false;
    const d = new Date(t);
    const now = new Date();
    return (
        d.getFullYear() === now.getFullYear() &&
        d.getMonth() === now.getMonth() &&
        d.getDate() === now.getDate()
    );
}

/** Feed group label for a finished run: Today / Yesterday / "Jul 20". */
function dateGroupLabel(iso: string): string {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "Earlier";
    const d = new Date(t);
    const now = new Date();
    const startOfDay = (x: Date): number =>
        new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diffDays = Math.round((startOfDay(now) - startOfDay(d)) / 86_400_000);
    if (diffDays <= 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

interface FeedEntry {
    run: ExecutionSummary;
    live: boolean;
}

interface FeedSection {
    label: string;
    entries: FeedEntry[];
}

export function ActivityPanel({
    executions,
    controlPlaneUp,
}: ActivityPanelProps): ReactElement {
    const [filter, setFilter] = useState<ActivityFilter>("all");
    const [agentFilter, setAgentFilter] = useState<string>("");
    const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
    const now = useTickingNow(1000);
    const reducedMotion = useReducedMotion();

    // Diff-only entrances (DESIGN.md §5.2): run IDs seen in the previous
    // snapshot render. Rows whose runId is NOT in this set animate in;
    // everything else — including the entire initial paint (null) — renders
    // settled. Updated after every render so filter flips never re-animate.
    const prevRunIdsRef = useRef<Set<string> | null>(null);
    const prevRunIds = prevRunIdsRef.current;
    useEffect(() => {
        if (executions !== null) {
            prevRunIdsRef.current = new Set(
                [...executions.running, ...executions.recent].map(
                    (r) => r.runId,
                ),
            );
        }
    });
    const isNewRow = (runId: string): boolean =>
        !reducedMotion && prevRunIds !== null && !prevRunIds.has(runId);

    if (!controlPlaneUp || executions === null) {
        return (
            <div className="panel">
                <EmptyState
                    size="hero"
                    variant="pulse"
                    title="Waiting for the server"
                    description="Activity will begin collecting here as soon as the AgentField server is running."
                />
            </div>
        );
    }

    const empty =
        executions.running.length === 0 && executions.recent.length === 0;
    if (empty) {
        return (
            <div className="panel">
                <EmptyState
                    size="hero"
                    variant="pulse"
                    title="The signal is quiet"
                    description="Runs will gather here when an agent is called — from Claude Code, Codex, or any client."
                    action={
                        <a
                            className="link-button"
                            href={COMMUNITY_LINKS.docs}
                            target="_blank"
                            rel="noreferrer"
                        >
                            See how agents get called
                        </a>
                    }
                />
            </div>
        );
    }

    // Summary strip — computed from the unfiltered prop.
    const runningCount = executions.running.length;
    const failedToday = executions.recent.filter(
        (r) => isFailed(r) && isToday(r.startedAt),
    ).length;
    const runs24h = executions.running.length + executions.recent.length;

    const agents = Array.from(
        new Set(
            [...executions.running, ...executions.recent]
                .map((r) => r.agentId)
                .filter((id) => id !== ""),
        ),
    ).sort();

    const matchesAgent = (run: ExecutionSummary): boolean =>
        agentFilter === "" || run.agentId === agentFilter;

    const liveRuns =
        filter === "all" || filter === "running"
            ? executions.running.filter(matchesAgent)
            : [];
    const doneRuns =
        filter === "running"
            ? []
            : executions.recent.filter(
                  (r) =>
                      matchesAgent(r) &&
                      (filter === "all" ||
                          (filter === "succeeded"
                              ? r.status === "succeeded"
                              : isFailed(r))),
              );

    // Time-grouped sections: Running first, then finished runs bucketed by date
    // (the feed arrives newest-first, so consecutive grouping = per-date grouping).
    const sections: FeedSection[] = [];
    if (liveRuns.length > 0) {
        sections.push({
            label: "Running",
            entries: liveRuns.map((run) => ({ run, live: true })),
        });
    }
    for (const run of doneRuns) {
        const label = dateGroupLabel(run.startedAt);
        const last = sections[sections.length - 1];
        if (last && last.label === label && !last.entries[0]?.live) {
            last.entries.push({ run, live: false });
        } else {
            sections.push({ label, entries: [{ run, live: false }] });
        }
    }

    const totalRows = liveRuns.length + doneRuns.length;
    const visibleSections: FeedSection[] = [];
    let budget = visibleCount;
    for (const section of sections) {
        if (budget <= 0) break;
        const entries = section.entries.slice(0, budget);
        visibleSections.push({ label: section.label, entries });
        budget -= entries.length;
    }
    const hiddenCount = Math.max(0, totalRows - visibleCount);

    const applyFilter = (next: ActivityFilter): void => {
        setFilter(next);
        setVisibleCount(PAGE_SIZE);
    };
    const applyAgentFilter = (next: string): void => {
        setAgentFilter(next);
        setVisibleCount(PAGE_SIZE);
    };

    return (
        <div className="activity-view">
            <p className="activity-summary">
                {runningCount} running · {failedToday} failed today · {runs24h}{" "}
                runs (24h)
            </p>
            <div className="filter-bar activity-filters">
                {(
                    [
                        ["all", "All"],
                        ["running", "Running"],
                        ["succeeded", "Succeeded"],
                        ["failed", "Failed"],
                    ] as const
                ).map(([id, label]) => (
                    <button
                        key={id}
                        type="button"
                        className={`filter-chip${filter === id ? " active" : ""}`}
                        aria-pressed={filter === id}
                        onClick={() => applyFilter(id)}
                    >
                        {label}
                    </button>
                ))}
                {agents.length > 1 && (
                    <select
                        className="activity-agent-select"
                        aria-label="Filter by agent"
                        value={agentFilter}
                        onChange={(e) => applyAgentFilter(e.target.value)}
                    >
                        <option value="">All agents</option>
                        {agents.map((id) => (
                            <option key={id} value={id}>
                                {id}
                            </option>
                        ))}
                    </select>
                )}
            </div>
            <div className="panel">
                {totalRows === 0 ? (
                    <div className="empty secondary">
                        No runs match this filter.
                    </div>
                ) : (
                    <>
                        <ul className="feed-list">
                            {visibleSections.map((section, i) => (
                                <Fragment key={`${section.label}-${i}`}>
                                    <li className="feed-group-label">
                                        {section.label}
                                    </li>
                                    {section.entries.map(({ run, live }) => (
                                        <FeedRow
                                            key={run.runId}
                                            run={run}
                                            live={live}
                                            now={now}
                                            entering={isNewRow(run.runId)}
                                        />
                                    ))}
                                </Fragment>
                            ))}
                        </ul>
                        {hiddenCount > 0 && (
                            <button
                                type="button"
                                className="action-button ghost feed-more"
                                onClick={() =>
                                    setVisibleCount((c) => c + PAGE_SIZE)
                                }
                            >
                                Show more ({hiddenCount} remaining)
                            </button>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

/** Dense single-line feed row (§4.12) — 34px, glyph · name · agent · mono meta. */
function FeedRow({
    run,
    live,
    now,
    entering,
}: {
    run: ExecutionSummary;
    live: boolean;
    now: number;
    /** Diff-only entrance (§5.2): true only for runs new since the last poll. */
    entering: boolean;
}): ReactElement {
    const failed = !live && isFailed(run);
    const usage = readUsage(run);
    const duration = live
        ? formatElapsed(run.startedAt, now)
        : formatDuration(run.durationMs);
    const statusLabel = live
        ? "Running"
        : (STATUS_LABEL[run.status] ?? run.status);

    return (
        <m.li
            className="feed-row"
            initial={entering ? { opacity: 0, y: -4 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            title={failed && run.errorMessage ? run.errorMessage : undefined}
        >
            {live ? (
                <span
                    className="row-dot running pulse"
                    role="img"
                    aria-label="Running"
                    title="Running"
                />
            ) : (
                <span
                    className={`feed-glyph ${run.status}`}
                    role="img"
                    aria-label={statusLabel}
                    title={statusLabel}
                >
                    {STATUS_GLYPH[run.status] ?? "·"}
                </span>
            )}
            <span className="feed-name">{run.displayName}</span>
            <span className="feed-agent">{run.agentId}</span>
            <span className="feed-meta">
                <StartedTime iso={run.startedAt} now={now} />
                {!live && duration ? (
                    <>
                        <span className="feed-meta-sep" aria-hidden="true">
                            ·
                        </span>
                        <span className="feed-duration" title="Run duration">
                            {duration}
                        </span>
                    </>
                ) : null}
                {live && duration ? (
                    <>
                        <span className="feed-meta-sep" aria-hidden="true">
                            ·
                        </span>
                        <span className="feed-duration" title="Elapsed">
                            {duration}
                        </span>
                    </>
                ) : null}
                {usage.totalTokens !== undefined ? (
                    <>
                        <span className="feed-meta-sep" aria-hidden="true">
                            ·
                        </span>
                        {formatTokens(usage.totalTokens)}
                    </>
                ) : null}
                {usage.costUsd !== undefined ? (
                    <>
                        <span className="feed-meta-sep" aria-hidden="true">
                            ·
                        </span>
                        {formatCost(usage.costUsd)}
                    </>
                ) : null}
            </span>
            <button
                className="action-button icon run-open"
                title="Open this run in the AgentField web UI"
                onClick={() =>
                    void window.agentfield.openWebUI(
                        `/ui/runs/${encodeURIComponent(run.runId)}`,
                    )
                }
            >
                ↗
            </button>
        </m.li>
    );
}

/**
 * Relative start in plain English. Hover shows a fast, on-brand tip with the
 * compact absolute time (not the slow native OS tooltip).
 */
function StartedTime({
    iso,
    now,
}: {
    iso: string;
    now: number;
}): ReactElement | null {
    const [tip, setTip] = useState<{ left: number; top: number } | null>(null);
    const showTimer = useRef<number | null>(null);

    const clearTip = (): void => {
        if (showTimer.current !== null) {
            window.clearTimeout(showTimer.current);
            showTimer.current = null;
        }
        setTip(null);
    };

    useEffect(
        () => () => {
            if (showTimer.current !== null) {
                window.clearTimeout(showTimer.current);
            }
        },
        [],
    );

    if (!iso || Number.isNaN(Date.parse(iso))) return null;

    const label = formatRelativeStarted(iso, now);
    const exact = formatFullStarted(iso, now);

    const showTip = (el: HTMLElement): void => {
        const rect = el.getBoundingClientRect();
        clearTip();
        showTimer.current = window.setTimeout(() => {
            setTip({
                left: rect.left + rect.width / 2,
                top: rect.top,
            });
        }, TIP_SHOW_MS);
    };

    return (
        <>
            <span
                className="feed-time"
                aria-label={`${label}, ${exact}`}
                onMouseEnter={(e) => showTip(e.currentTarget)}
                onMouseLeave={clearTip}
            >
                {label}
            </span>
            {tip
                ? createPortal(
                      <div
                          className="af-tooltip"
                          role="tooltip"
                          style={{ left: tip.left, top: tip.top }}
                      >
                          {exact}
                      </div>,
                      document.body,
                  )
                : null}
        </>
    );
}

export function ExecutionRow({
    run,
    live = false,
}: {
    run: ExecutionSummary;
    live?: boolean;
}) {
    const now = useTickingNow(live ? 1000 : 15_000);
    const failed = !live && isFailed(run);
    const usage = readUsage(run);
    const duration = live
        ? formatElapsed(run.startedAt, now)
        : formatDuration(run.durationMs);

    const metaTail = [
        !live && duration ? duration : null,
        usage.totalTokens !== undefined
            ? formatTokens(usage.totalTokens)
            : null,
        usage.costUsd !== undefined ? formatCost(usage.costUsd) : null,
    ].filter(Boolean);

    const statusLabel = live
        ? "Running"
        : (STATUS_LABEL[run.status] ?? run.status);

    return (
        <li className={`row ${live ? "" : "row-past"}`}>
            {live ? (
                <span className="row-dot running pulse" aria-hidden="true" />
            ) : (
                <span className={`run-glyph ${run.status}`} aria-hidden="true">
                    {STATUS_GLYPH[run.status] ?? "·"}
                </span>
            )}
            <div className="row-main">
                <span className="row-title">{run.displayName}</span>
                <span className="row-sub">
                    {run.agentId ? (
                        <>
                            {run.agentId}
                            <span className="feed-meta-sep" aria-hidden="true">
                                ·
                            </span>
                        </>
                    ) : null}
                    <StartedTime iso={run.startedAt} now={now} />
                    {metaTail.length > 0 ? (
                        <>
                            <span className="feed-meta-sep" aria-hidden="true">
                                ·
                            </span>
                            {metaTail.join(" · ")}
                        </>
                    ) : null}
                </span>
                {failed && run.errorMessage && (
                    <span
                        className="row-sub error-text"
                        title={run.errorMessage}
                    >
                        {run.errorMessage}
                    </span>
                )}
            </div>
            <div className="row-side">
                <span
                    className={`badge ${statusBadgeClass(run.status, live, failed)}`}
                >
                    <span className="badge-dot" aria-hidden="true" />
                    {statusLabel}
                </span>
                <button
                    className="action-button icon run-open"
                    title="Open this run in the AgentField web UI"
                    onClick={() =>
                        void window.agentfield.openWebUI(
                            `/ui/runs/${encodeURIComponent(run.runId)}`,
                        )
                    }
                >
                    ↗
                </button>
            </div>
        </li>
    );
}
