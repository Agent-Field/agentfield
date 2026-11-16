/**
 * The sections of the developer Home page. Each one renders a slice of the
 * /dashboard/home payload; the arithmetic lives in homeModel.ts.
 */
import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import { Link, useNavigate } from "react-router";
import { AlertTriangle, ArrowRight, Copy, Check } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkline } from "@/components/ui/Sparkline";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { statusTone } from "@/lib/theme";
import type {
  DeveloperHomeResponse,
  HomeEndpoint,
  HomeProblem,
} from "@/types/developerHome";

import {
  DIRECT_SOURCE,
  MAP_NODE_H,
  MAP_NODE_W,
  callSnippet,
  connectSnippet,
  formatAge,
  formatCount,
  formatMs,
  formatPct,
  failedPct,
  layoutCallMap,
  runsFilterHref,
  type Figure,
  type HomeAlert,
  type StatusLine,
} from "./homeModel";

const TONE_TEXT = {
  bad: statusTone.error.fg,
  warn: statusTone.warning.fg,
} as const;

// ─── status line ────────────────────────────────────────────────────────────

export function HomeStatusLine({ status }: { status: StatusLine }) {
  const ok = status.problems === 0;
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <h1 className="flex flex-wrap items-center gap-2.5 text-xl font-semibold tracking-tight">
        <span
          aria-hidden
          className={cn(
            "inline-block size-2.5 rounded-full",
            ok ? statusTone.success.solidBg : statusTone.error.solidBg,
          )}
        />
        <span>
          {ok ? "No problems" : `${status.problems} problem${status.problems === 1 ? "" : "s"}`}
        </span>
        {status.registered > 0 && (
          <span className="font-normal text-muted-foreground">
            · {status.healthy} of {status.registered} agent{status.registered === 1 ? "" : "s"} healthy
          </span>
        )}
      </h1>
      <p className="text-sm text-muted-foreground">{status.detail.join(" · ")}</p>
    </div>
  );
}

// ─── alert banners ──────────────────────────────────────────────────────────

export function HomeAlerts({ alerts }: { alerts: HomeAlert[] }) {
  if (alerts.length === 0) return null;
  return (
    <div className="flex flex-col gap-2" role="list" aria-label="Problems">
      {alerts.map((a) => {
        const tone = a.tone === "bad" ? statusTone.error : statusTone.warning;
        return (
          <div
            key={a.id}
            role="listitem"
            className={cn(
              "flex flex-col gap-3 rounded-lg px-4 py-3 sm:flex-row sm:items-start sm:justify-between",
              tone.bg,
              tone.border,
            )}
          >
            <div className="flex min-w-0 gap-2.5">
              <AlertTriangle className={cn("mt-0.5 size-4 shrink-0", tone.fg)} aria-hidden />
              <div className="flex min-w-0 flex-col gap-0.5">
                <p className="text-sm font-semibold">{a.title}</p>
                <p className="text-sm text-muted-foreground">{a.detail}</p>
                <p className="text-xs text-muted-foreground/80">{a.rule}</p>
              </div>
            </div>
            {a.actions.length > 0 && (
              <div className="flex shrink-0 gap-2">
                {a.actions.map((act) => (
                  <Button key={act.href} asChild variant="outline" size="sm" className="h-7 text-xs">
                    <Link to={act.href}>{act.label}</Link>
                  </Button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── figures ────────────────────────────────────────────────────────────────

function FigureCell({ figure }: { figure: Figure }) {
  const body = (
    <div className="flex h-full flex-col gap-1 px-4 py-3">
      <span className="text-xs text-muted-foreground">{figure.label}</span>
      <div className="flex items-end justify-between gap-2">
        <span
          className={cn(
            "text-2xl font-semibold tabular-nums tracking-tight",
            figure.tone && TONE_TEXT[figure.tone],
          )}
        >
          {figure.value}
        </span>
        {figure.spark && figure.spark.length >= 2 && (
          <Sparkline
            data={figure.spark}
            width={72}
            height={24}
            color={figure.tone === "bad" ? "hsl(var(--status-error))" : "hsl(var(--muted-foreground))"}
            className="mb-1 shrink-0 opacity-80"
          />
        )}
      </div>
      <span className="text-xs text-muted-foreground">
        {[
          figure.comparable === false ? undefined : `usual ${figure.usual ?? "—"}`,
          figure.note,
        ]
          .filter(Boolean)
          .join(" · ") || "\u00a0"}
      </span>
    </div>
  );
  if (figure.href) {
    return (
      <Link
        to={figure.href}
        className="block min-w-0 transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {body}
      </Link>
    );
  }
  return <div className="min-w-0">{body}</div>;
}

export function HomeFigures({ figures }: { figures: Figure[] }) {
  return (
    <Card className="overflow-hidden">
      <div
        className={cn(
          "grid grid-cols-2 divide-border sm:grid-cols-3",
          figures.length > 5 ? "lg:grid-cols-6" : "lg:grid-cols-5",
          "[&>*]:border-b [&>*]:border-r [&>*]:border-border",
        )}
      >
        {figures.map((f) => (
          <FigureCell key={f.key} figure={f} />
        ))}
      </div>
    </Card>
  );
}

// ─── call map ───────────────────────────────────────────────────────────────

const LINK_COLOR = {
  bad: "hsl(var(--status-error))",
  warn: "hsl(var(--status-warning))",
  ok: "hsl(var(--muted-foreground) / 0.45)",
} as const;

function truncateLabel(s: string, max = 20): string {
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/** Width of an element, following resizes, so the map is drawn at 1:1 scale. */
function useElementWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const w = Math.floor(el.getBoundingClientRect().width);
      if (w > 0) setWidth(w);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** The map never draws narrower than this; a narrow screen scrolls it sideways. */
const MAP_MIN_WIDTH = 720;

export function HomeCallMap({ home, phrase }: { home: DeveloperHomeResponse; phrase: string }) {
  const navigate = useNavigate();
  const [boxRef, boxWidth] = useElementWidth<HTMLDivElement>(960);
  const layout = useMemo(
    () => layoutCallMap(home, Math.max(MAP_MIN_WIDTH, boxWidth)),
    [home, boxWidth],
  );
  const hasFlow = layout.links.length > 0;
  const stopped = new Set(home.stopped_agents.map((s) => s.id));

  return (
    <Card>
      <CardHeader className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
        <div>
          <CardTitle className="text-sm font-medium">Call map, {phrase.replace(/^in the /, "")}</CardTitle>
          <CardDescription className="text-xs">Who called whom. Click an agent for its runs.</CardDescription>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <LegendSwatch color={LINK_COLOR.ok} label="calls" />
          <LegendSwatch color={LINK_COLOR.bad} label="failing" />
          <LegendSwatch color={LINK_COLOR.warn} label="held" />
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0">
        <div ref={boxRef}>
        {!hasFlow && layout.nodes.length === 0 ? (
          <p className="py-6 text-center text-xs text-muted-foreground">No calls {phrase}.</p>
        ) : (
          <div className="overflow-x-auto">
            <svg
              role="img"
              aria-label="Call map"
              width={layout.width}
              height={layout.height}
              viewBox={`0 0 ${layout.width} ${layout.height}`}
              className="block"
            >
              {layout.links.map((l) => (
                <path
                  key={l.id}
                  d={l.path}
                  fill="none"
                  stroke={LINK_COLOR[l.tone]}
                  strokeWidth={l.width}
                  strokeLinecap="round"
                >
                  <title>
                    {`${l.from} → ${l.to}: ${formatCount(l.calls)} calls`}
                    {l.failed > 0 ? `, ${formatCount(l.failed)} failed` : ""}
                    {l.held > 0 ? `, ${formatCount(l.held)} held` : ""}
                  </title>
                </path>
              ))}
              {layout.nodes.map((n) => {
                const isStopped = stopped.has(n.id);
                const isSource = n.kind === "source";
                const label = isSource
                  ? n.id === DIRECT_SOURCE
                    ? "direct calls"
                    : n.id.replace(/^trigger:/, "trigger ")
                  : n.label;
                return (
                  <g
                    key={n.id}
                    transform={`translate(${n.x} ${n.y})`}
                    className={cn(!isSource && "cursor-pointer")}
                    onClick={
                      isSource ? undefined : () => navigate(runsFilterHref({ search: n.id }))
                    }
                  >
                    <title>
                      {isSource
                        ? label
                        : `${n.id}${n.status ? ` (${n.status})` : ""}${n.selfCalls > 0 ? `, ${formatCount(n.selfCalls)} calls to itself` : ""}`}
                    </title>
                    <rect
                      width={MAP_NODE_W}
                      height={MAP_NODE_H}
                      rx={6}
                      fill={isSource ? "transparent" : "hsl(var(--card))"}
                      stroke={
                        isStopped
                          ? "hsl(var(--status-error))"
                          : isSource
                            ? "hsl(var(--border))"
                            : "hsl(var(--border))"
                      }
                      strokeDasharray={isSource ? "3 3" : undefined}
                      strokeWidth={isStopped ? 1.5 : 1}
                    />
                    {!isSource && (
                      <circle
                        cx={10}
                        cy={MAP_NODE_H / 2}
                        r={3}
                        fill={
                          isStopped
                            ? "hsl(var(--status-error))"
                            : n.status === "running"
                              ? "hsl(var(--status-success))"
                              : "hsl(var(--status-warning))"
                        }
                      />
                    )}
                    <text
                      x={isSource ? MAP_NODE_W / 2 : 19}
                      y={MAP_NODE_H / 2 + 4}
                      textAnchor={isSource ? "middle" : "start"}
                      fontSize={11}
                      fill={isSource ? "hsl(var(--muted-foreground))" : "hsl(var(--foreground))"}
                      className="font-mono"
                    >
                      {truncateLabel(label, isSource ? 22 : 20)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}
        </div>
        {layout.idle.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            Not called {phrase}: {layout.idle.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="16" height="6" aria-hidden>
        <line x1="0" y1="3" x2="16" y2="3" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      {label}
    </span>
  );
}

// ─── problems by cause ──────────────────────────────────────────────────────

const STATUS_WORD: Record<string, string> = {
  failed: "Failed",
  timeout: "Timed out",
  cancelled: "Cancelled",
};

function problemHref(p: HomeProblem): string {
  if (p.example_run_id) return `/runs/${encodeURIComponent(p.example_run_id)}`;
  return runsFilterHref({ status: p.status, search: p.reasoner_id });
}

export function HomeProblems({
  problems,
  phrase,
  now,
}: {
  problems: HomeProblem[];
  phrase: string;
  now: Date;
}) {
  const navigate = useNavigate();
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between px-4 py-3 space-y-0">
        <div>
          <CardTitle className="text-sm font-medium">Problems, by cause</CardTitle>
          <CardDescription className="text-xs">
            Failed calls grouped by the endpoint that failed and its error, most first.
          </CardDescription>
        </div>
        <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
          <Link to={runsFilterHref({ status: "failed" })}>
            Failed runs <ArrowRight className="size-3.5" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        {problems.length === 0 ? (
          <p className="px-4 pb-4 text-xs text-muted-foreground">No failed calls {phrase}.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="h-8 px-4 text-xs">Cause</TableHead>
                <TableHead className="h-8 w-[90px] px-3 text-right text-xs">Count</TableHead>
                <TableHead className="hidden h-8 w-[110px] px-3 text-xs sm:table-cell">Over time</TableHead>
                <TableHead className="h-8 w-[110px] px-4 text-right text-xs">Last</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {problems.map((p) => (
                <TableRow
                  key={`${p.agent_node_id}.${p.reasoner_id}.${p.status}.${p.message}`}
                  className="cursor-pointer"
                  onClick={() => navigate(problemHref(p))}
                >
                  <TableCell className="max-w-0 px-4 py-2">
                    <Link
                      to={problemHref(p)}
                      onClick={(e) => e.stopPropagation()}
                      className="block truncate font-mono text-xs font-medium hover:underline"
                    >
                      {p.agent_node_id}.{p.reasoner_id}
                    </Link>
                    <p className="truncate text-xs text-muted-foreground" title={p.message || undefined}>
                      <span className={statusTone.error.fg}>{STATUS_WORD[p.status] ?? p.status}</span>
                      {p.message ? ` · ${p.message}` : " · no error message recorded"}
                    </p>
                  </TableCell>
                  <TableCell className="px-3 py-2 text-right font-mono text-xs tabular-nums">
                    {formatCount(p.count)}
                  </TableCell>
                  <TableCell className="hidden px-3 py-2 sm:table-cell">
                    <Sparkline
                      data={p.buckets}
                      width={90}
                      height={20}
                      color="hsl(var(--status-error))"
                    />
                  </TableCell>
                  <TableCell className="whitespace-nowrap px-4 py-2 text-right text-xs text-muted-foreground">
                    {formatAge(p.last_seen, now)} ago
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

// ─── endpoints ──────────────────────────────────────────────────────────────

function calledBy(e: HomeEndpoint): string {
  const parts: string[] = [];
  if (e.direct_calls > 0) parts.push("direct");
  if (e.trigger_calls > 0) parts.push("trigger");
  if (e.caller_agents.length === 1) parts.push(e.caller_agents[0]);
  else if (e.caller_agents.length > 1) parts.push(`${e.caller_agents.length} agents`);
  return parts.join(", ") || "—";
}

/** Endpoints shown before "Show all"; failing ones sort first, so they are always in view. */
export const ENDPOINTS_COLLAPSED = 12;

export function HomeEndpoints({ endpoints, phrase }: { endpoints: HomeEndpoint[]; phrase: string }) {
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? endpoints : endpoints.slice(0, ENDPOINTS_COLLAPSED);
  const hidden = endpoints.length - shown.length;
  return (
    <Card>
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-sm font-medium">Endpoints</CardTitle>
        <CardDescription className="text-xs">
          Every reasoner and skill called {phrase}, failing ones first.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto p-0">
        {endpoints.length === 0 ? (
          <p className="px-4 pb-4 text-xs text-muted-foreground">No calls {phrase}.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="h-8 px-4 text-xs">Endpoint</TableHead>
                <TableHead className="h-8 px-3 text-xs">Called by</TableHead>
                <TableHead className="h-8 w-[80px] px-3 text-right text-xs">Calls</TableHead>
                <TableHead className="h-8 w-[80px] px-3 text-right text-xs">Failed</TableHead>
                <TableHead className="h-8 w-[90px] px-3 text-right text-xs">p50</TableHead>
                <TableHead className="h-8 w-[90px] px-4 text-right text-xs">p95</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((e) => {
                const pct = failedPct(e.failed, e.terminal);
                const href = runsFilterHref({ search: e.reasoner_id });
                return (
                  <TableRow
                    key={`${e.agent_node_id}.${e.reasoner_id}`}
                    className="cursor-pointer"
                    onClick={() => navigate(href)}
                  >
                    <TableCell className="px-4 py-1.5 font-mono text-xs font-medium">
                      <Link to={href} onClick={(ev) => ev.stopPropagation()} className="hover:underline">
                        {e.agent_node_id}.{e.reasoner_id}
                      </Link>
                    </TableCell>
                    <TableCell
                      className="max-w-[220px] truncate px-3 py-1.5 text-xs text-muted-foreground"
                      title={e.caller_agents.join(", ") || undefined}
                    >
                      {calledBy(e)}
                    </TableCell>
                    <TableCell className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">
                      {formatCount(e.calls)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "px-3 py-1.5 text-right font-mono text-xs tabular-nums",
                        e.failed > 0 && statusTone.error.fg,
                      )}
                    >
                      {formatPct(pct)}
                    </TableCell>
                    <TableCell className="px-3 py-1.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                      {formatMs(e.p50_ms)}
                    </TableCell>
                    <TableCell className="px-4 py-1.5 text-right font-mono text-xs tabular-nums">
                      {formatMs(e.p95_ms)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
        {(hidden > 0 || expanded) && endpoints.length > ENDPOINTS_COLLAPSED && (
          <div className="border-t border-border px-4 py-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Show fewer" : `Show all ${formatCount(endpoints.length)} endpoints`}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── waiting on people ──────────────────────────────────────────────────────

export function HomeWaiting({ home, now }: { home: DeveloperHomeResponse; now: Date }) {
  const pendingAgents = home.agents.filter((a) => a.status === "pending_approval");
  if (home.approvals.total === 0 && pendingAgents.length === 0) return null;
  const more = home.approvals.total - home.approvals.items.length;
  return (
    <Card>
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-sm font-medium">Waiting on people</CardTitle>
        <CardDescription className="text-xs">Runs paused for an approval, and agents waiting for tag approval.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col divide-y divide-border p-0">
        {home.approvals.items.map((a) => (
          <Link
            key={a.execution_id}
            to={`/runs/${encodeURIComponent(a.run_id)}`}
            className="flex items-center justify-between gap-3 px-4 py-2 text-xs hover:bg-muted/50"
          >
            <span className="min-w-0 truncate">
              <span className="font-mono font-medium">{a.agent_node_id}.{a.reasoner_id}</span>
              <span className="text-muted-foreground"> · approval requested</span>
              {a.expires_at && (
                <span className="text-muted-foreground"> · expires {formatAge(now.toISOString(), new Date(a.expires_at))} from now</span>
              )}
            </span>
            <span className="shrink-0 text-muted-foreground">{formatAge(a.requested_at, now)}</span>
          </Link>
        ))}
        {more > 0 && (
          <p className="px-4 py-2 text-xs text-muted-foreground">and {formatCount(more)} more</p>
        )}
        {pendingAgents.map((a) => (
          <Link
            key={a.id}
            to="/access"
            className="flex items-center justify-between gap-3 px-4 py-2 text-xs hover:bg-muted/50"
          >
            <span className="min-w-0 truncate">
              <span className="font-mono font-medium">{a.id}</span>
              <span className="text-muted-foreground"> · registered, waiting for tag approval</span>
            </span>
            <span className="shrink-0 text-muted-foreground">Review</span>
          </Link>
        ))}
      </CardContent>
    </Card>
  );
}

// ─── in and out ─────────────────────────────────────────────────────────────

function IOCell({
  title,
  value,
  detail,
  tone,
  href,
}: {
  title: string;
  value: string;
  detail: string;
  tone?: "bad";
  href: string;
}) {
  return (
    <Link to={href} className="flex min-w-0 flex-col gap-0.5 px-4 py-3 hover:bg-muted/50">
      <span className="text-xs text-muted-foreground">{title}</span>
      <span className={cn("text-lg font-semibold tabular-nums", tone && statusTone.error.fg)}>{value}</span>
      <span className="truncate text-xs text-muted-foreground">{detail}</span>
    </Link>
  );
}

export function HomeInOut({ home, phrase }: { home: DeveloperHomeResponse; phrase: string }) {
  const cells: ReactElement[] = [];
  const t = home.triggers;
  if (t) {
    const bits = [`${formatCount(t.enabled)} of ${formatCount(t.configured)} enabled`];
    if (t.failed > 0) bits.push(`${formatCount(t.failed)} dispatch failed`);
    bits.push(`dead-letter queue ${formatCount(t.dead_letters)}`);
    cells.push(
      <IOCell
        key="triggers"
        title="Trigger events in"
        value={formatCount(t.events)}
        detail={bits.join(" · ")}
        tone={t.failed > 0 || t.dead_letters > 0 ? "bad" : undefined}
        href="/triggers"
      />,
    );
  }
  const w = home.webhooks;
  if (w) {
    const first = w.recent_failures[0];
    let host = "";
    if (first) {
      try {
        host = new URL(first.url).host;
      } catch {
        host = first.url;
      }
    }
    cells.push(
      <IOCell
        key="webhooks"
        title="Result webhooks out"
        value={w.failed > 0 ? `${formatCount(w.failed)} failed` : `${formatCount(w.delivered)} delivered`}
        detail={
          w.failed > 0 && host
            ? `latest to ${host}${first.last_error ? `: ${first.last_error}` : ""}`
            : `${formatCount(w.registered)} registered${w.pending > 0 ? ` · ${formatCount(w.pending)} pending` : ""}`
        }
        tone={w.failed > 0 ? "bad" : undefined}
        href={first ? `/runs?search=${encodeURIComponent(first.execution_id)}` : "/runs"}
      />,
    );
  }
  const f = home.forwarder;
  if (f) {
    cells.push(
      <IOCell
        key="forwarder"
        title="Event forwarder"
        value={`${formatCount(f.dead_letters)} dropped`}
        detail={f.enabled ? "events that could not be delivered" : "forwarding is turned off"}
        tone={f.dead_letters > 0 ? "bad" : undefined}
        href="/settings"
      />,
    );
  }
  if (cells.length === 0) return null;
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-sm font-medium">In and out</CardTitle>
        <CardDescription className="text-xs">Events that came in and deliveries that went out {phrase}.</CardDescription>
      </CardHeader>
      <div className="grid grid-cols-1 divide-y divide-border border-t border-border sm:grid-cols-3 sm:divide-x sm:divide-y-0">
        {cells}
      </div>
    </Card>
  );
}

// ─── first agent ────────────────────────────────────────────────────────────

export function HomeFirstAgent({ origin }: { origin: string }) {
  const [copied, setCopied] = useState(false);
  const snippet = connectSnippet(origin);
  const copy = () => {
    void navigator.clipboard?.writeText(snippet).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <Card className="mx-auto w-full max-w-2xl">
      <CardHeader className="px-6 pt-6">
        <CardTitle className="text-lg font-semibold">Connect your first agent</CardTitle>
        <CardDescription>
          Nothing has registered with this control plane yet. Install the Python SDK, save the file below and run it;
          the agent appears here as soon as it sends its first heartbeat.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 px-6 pb-6">
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 px-4 py-3 font-mono text-xs">
          pip install agentfield
        </pre>
        <div className="relative">
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed">
            {snippet}
          </pre>
          <Button
            variant="ghost"
            size="sm"
            className="absolute right-2 top-2 h-7 gap-1.5 text-xs"
            onClick={copy}
          >
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Then call it, adding <code className="font-mono">-H "X-API-Key: …"</code> when the control plane
          requires a key:
        </p>
        <pre className="-mt-2 overflow-x-auto rounded-md border border-border bg-muted/40 px-4 py-3 font-mono text-xs">
          {callSnippet(origin)}
        </pre>
      </CardContent>
    </Card>
  );
}
