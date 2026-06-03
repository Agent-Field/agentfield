import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { SearchBar } from "@/components/ui/SearchBar";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  MoreHorizontal,
  PauseCircle,
  Play,
  RefreshCw,
  Terminal,
} from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";
import {
  fetchNodeLogsText,
  NodeLogsError,
  parseNodeLogsNDJSON,
  streamNodeLogsEntries,
  type NodeLogEntry,
} from "@/services/api";
import { HintIcon } from "@/components/authorization/HintIcon";
import { observabilityStyles } from "@/components/execution/observabilityStyles";

const MAX_BUFFER = 5000;
const DEFAULT_TAIL = "200";

type StreamFilter = "all" | "stdout" | "stderr";
type FormatFilter = "all" | "structured" | "plain";

type ParsedStructuredProcessLog = {
  v?: number | string;
  ts?: string;
  execution_id?: string;
  workflow_id?: string;
  run_id?: string;
  root_workflow_id?: string;
  parent_execution_id?: string;
  agent_node_id?: string;
  reasoner_id?: string;
  level?: string;
  source?: string;
  event_type?: string;
  message?: string;
  attributes?: unknown;
  system_generated?: boolean;
  [key: string]: unknown;
};

function normalizeStream(s: string | undefined): "stdout" | "stderr" | "other" {
  const x = (s ?? "").toLowerCase();
  if (x === "stderr" || x === "err") return "stderr";
  if (x === "stdout" || x === "out") return "stdout";
  return "other";
}

/** Wall time for scanning; falls back to — if missing/invalid. */
function formatLogTime(iso: string | undefined): string {
  if (!iso?.trim()) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function formatLogDate(iso: string | undefined): string | null {
  if (!iso?.trim()) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) return null;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: d.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
  });
}

function levelBadgeVariant(
  level: string | undefined
): "default" | "secondary" | "destructive" | "outline" {
  const l = (level ?? "").toLowerCase();
  if (l === "error" || l === "fatal" || l === "critical") return "destructive";
  if (l === "warn" || l === "warning") return "outline";
  if (l === "info" || l === "log") return "secondary";
  return "outline";
}

/** Hide level pill when it only repeats stdout/stderr semantics (SDK defaults). */
function isRedundantLevel(
  level: string | undefined,
  ns: "stdout" | "stderr" | "other"
): boolean {
  const l = (level ?? "").toLowerCase();
  return (
    (l === "info" && ns === "stdout") ||
    (l === "error" && ns === "stderr")
  );
}

function parseStructuredProcessLog(
  line: string | undefined
): ParsedStructuredProcessLog | null {
  if (!line?.trim()) return null;
  try {
    const parsed = JSON.parse(line) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const candidate = parsed as ParsedStructuredProcessLog;
    const hasStructuredKeys =
      "event_type" in candidate ||
      "execution_id" in candidate ||
      "run_id" in candidate ||
      "message" in candidate ||
      "source" in candidate ||
      "reasoner_id" in candidate;
    return hasStructuredKeys ? candidate : null;
  } catch {
    return null;
  }
}

function compactValue(value: string | number | undefined): string | null {
  if (value == null) return null;
  const text = String(value).trim();
  if (!text) return null;
  if (text.length <= 18) return text;
  return `${text.slice(0, 8)}…${text.slice(-6)}`;
}

function hasStructuredDetails(entry: ParsedStructuredProcessLog | null): boolean {
  if (!entry) return false;
  const { attributes, message, event_type, level, source, ...rest } = entry;
  if (attributes != null) return true;
  return Object.keys(rest).some((key) => {
    const value = rest[key];
    return value != null && String(value).trim() !== "";
  });
}

export interface NodeProcessLogsPanelProps {
  nodeId: string;
  className?: string;
}

function maxSeq(entries: NodeLogEntry[]): number {
  let m = 0;
  for (const e of entries) {
    if (typeof e.seq === "number" && e.seq > m) m = e.seq;
  }
  return m;
}

export function NodeProcessLogsPanel({
  nodeId,
  className,
}: NodeProcessLogsPanelProps) {
  const [entries, setEntries] = useState<NodeLogEntry[]>([]);
  const [filter, setFilter] = useState("");
  const [streamFilter, setStreamFilter] = useState<StreamFilter>("all");
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
  const [live, setLive] = useState(false);
  const [loadingTail, setLoadingTail] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const liveAbortRef = useRef<AbortController | null>(null);
  const sinceSeqRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const stopLive = useCallback(() => {
    liveAbortRef.current?.abort();
    liveAbortRef.current = null;
    setLive(false);
  }, []);

  const loadTail = useCallback(async () => {
    setLoadingTail(true);
    setStreamError(null);
    try {
      const text = await fetchNodeLogsText(nodeId, {
        tail_lines: DEFAULT_TAIL,
      });
      const parsed = parseNodeLogsNDJSON(text);
      setEntries(parsed.slice(-MAX_BUFFER));
    } catch (e) {
      // Treat "no base_url" (agent never ran, no upstream URL yet) and 404
      // (node has no logs endpoint) as expected empty states — not errors.
      // Branch on the structured fields of NodeLogsError rather than
      // string-matching the human message, which is brittle to backend
      // phrasing changes.
      if (
        e instanceof NodeLogsError &&
        (e.status === 404 || e.code === "agent_unreachable")
      ) {
        // Surface to devtools so developers debugging the panel can still
        // see the swallowed error; do not raise a destructive UI alert.
        if (import.meta.env?.DEV) {
          console.debug(
            `[NodeProcessLogsPanel] expected empty state for node ${nodeId}:`,
            e.status,
            e.code ?? e.message,
          );
        }
        setEntries([]);
      } else {
        setStreamError(e instanceof Error ? e.message : "Failed to load logs");
      }
    } finally {
      setLoadingTail(false);
    }
  }, [nodeId]);

  useEffect(() => {
    void loadTail();
  }, [loadTail]);

  useEffect(() => {
    if (!live) return;

    const since = sinceSeqRef.current;
    const ac = new AbortController();
    liveAbortRef.current = ac;

    (async () => {
      try {
        for await (const entry of streamNodeLogsEntries(
          nodeId,
          { follow: "1", since_seq: String(since) },
          ac.signal
        )) {
          setStreamError(null);
          setEntries((prev) => [...prev, entry].slice(-MAX_BUFFER));
        }
      } catch (e) {
        if (e instanceof Error && e.name === "AbortError") return;
        setStreamError(
          e instanceof Error ? e.message : "Log stream interrupted"
        );
      } finally {
        if (liveAbortRef.current === ac) {
          liveAbortRef.current = null;
          setLive(false);
        }
      }
    })();

    return () => {
      ac.abort();
    };
  }, [live, nodeId]);

  const streamCounts = useMemo(() => {
    let stdout = 0;
    let stderr = 0;
    let other = 0;
    for (const e of entries) {
      const k = normalizeStream(e.stream);
      if (k === "stdout") stdout += 1;
      else if (k === "stderr") stderr += 1;
      else other += 1;
    }
    return { all: entries.length, stdout, stderr, other };
  }, [entries]);

  const streamScoped = useMemo(() => {
    if (streamFilter === "all") return entries;
    return entries.filter((e) => normalizeStream(e.stream) === streamFilter);
  }, [entries, streamFilter]);

  const formatScoped = useMemo(() => {
    if (formatFilter === "all") return streamScoped;
    return streamScoped.filter((entry) => {
      const structured = parseStructuredProcessLog(entry.line);
      return formatFilter === "structured" ? structured !== null : structured === null;
    });
  }, [formatFilter, streamScoped]);

  const formatCounts = useMemo(() => {
    let structured = 0;
    let plain = 0;
    for (const entry of streamScoped) {
      if (parseStructuredProcessLog(entry.line)) structured += 1;
      else plain += 1;
    }
    return { structured, plain, all: streamScoped.length };
  }, [streamScoped]);

  const overallFormatCounts = useMemo(() => {
    let structured = 0;
    let plain = 0;
    for (const entry of entries) {
      if (parseStructuredProcessLog(entry.line)) structured += 1;
      else plain += 1;
    }
    return { structured, plain, all: entries.length };
  }, [entries]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return formatScoped;
    return formatScoped.filter((e) => {
      const structured = parseStructuredProcessLog(e.line);
      const line = (e.line ?? "").toLowerCase();
      const stream = (e.stream ?? "").toLowerCase();
      const level = (structured?.level ?? e.level ?? "").toLowerCase();
      const source = (structured?.source ?? e.source ?? "").toLowerCase();
      const seq = String(e.seq ?? "");
      const eventType = (structured?.event_type ?? "").toLowerCase();
      const executionId = (structured?.execution_id ?? "").toLowerCase();
      const runId = (structured?.run_id ?? "").toLowerCase();
      const workflowId = (structured?.workflow_id ?? "").toLowerCase();
      const reasonerId = (structured?.reasoner_id ?? "").toLowerCase();
      const message = (structured?.message ?? "").toLowerCase();
      return (
        line.includes(q) ||
        stream.includes(q) ||
        level.includes(q) ||
        source.includes(q) ||
        seq.includes(q) ||
        eventType.includes(q) ||
        executionId.includes(q) ||
        runId.includes(q) ||
        workflowId.includes(q) ||
        reasonerId.includes(q) ||
        message.includes(q)
      );
    });
  }, [formatScoped, filter]);

  const ndjsonBlob = useMemo(() => {
    return filtered.map((e) => JSON.stringify(e)).join("\n");
  }, [filtered]);

  const copyVisible = useCallback(() => {
    void navigator.clipboard.writeText(ndjsonBlob);
  }, [ndjsonBlob]);

  const downloadVisible = useCallback(() => {
    const blob = new Blob([ndjsonBlob], {
      type: "application/x-ndjson",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${nodeId}-logs.ndjson`;
    a.click();
    URL.revokeObjectURL(url);
  }, [ndjsonBlob, nodeId]);

  useEffect(() => {
    if (!live || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(
      "[data-radix-scroll-area-viewport]"
    );
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [filtered.length, live]);

  return (
    <Card className={cn(observabilityStyles.card, className)}>
      <CardHeader className={observabilityStyles.processHeader}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between lg:gap-6">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <Terminal className="size-4 shrink-0 text-muted-foreground" aria-hidden />
              <CardTitle className="text-base font-semibold leading-none sm:text-sm sm:font-medium">
                Process logs
              </CardTitle>
              <Badge variant="outline" className="font-mono text-[10px]">
                NDJSON
              </Badge>
              <HintIcon label="What process logs show">
                NDJSON from the agent. Structured SDK lines surface correlation fields like
                execution, run, source, and event inline while plain stdout and stderr stay
                available for low-level debugging.
              </HintIcon>
            </div>
          </div>

          {/* Narrow: 2×2 grid + overflow menu. md+: single toolbar row (shadcn button group). */}
          <div className={observabilityStyles.processActions}>
            <div className="hidden w-full grid-cols-2 gap-2 min-[400px]:grid md:hidden">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 w-full justify-center gap-1.5 px-2"
                disabled={loadingTail || live}
                onClick={() => {
                  stopLive();
                  void loadTail();
                }}
              >
                <RefreshCw
                  className={cn("size-3.5 shrink-0", loadingTail && "animate-spin")}
                />
                <span className="text-xs">Refresh</span>
              </Button>
              <Button
                type="button"
                variant={live ? "secondary" : "default"}
                size="sm"
                className="h-9 w-full justify-center gap-1.5 px-2"
                onClick={() => {
                  if (live) {
                    stopLive();
                  } else {
                    sinceSeqRef.current = maxSeq(entries);
                    setStreamError(null);
                    setLive(true);
                  }
                }}
              >
                {live ? (
                  <>
                    <PauseCircle className="size-3.5 shrink-0" />
                    <span className="text-xs">Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="size-3.5 shrink-0" />
                    <span className="text-xs">Live</span>
                  </>
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 w-full justify-center gap-1.5 px-2"
                onClick={copyVisible}
                disabled={filtered.length === 0}
              >
                <Copy className="size-3.5 shrink-0" />
                <span className="text-xs">Copy</span>
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 w-full justify-center gap-1.5 px-2"
                onClick={downloadVisible}
                disabled={filtered.length === 0}
              >
                <Download className="size-3.5 shrink-0" />
                <span className="text-xs">Download</span>
              </Button>
            </div>

            <div className="flex min-[400px]:hidden items-stretch gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 flex-1 gap-1.5"
                disabled={loadingTail || live}
                onClick={() => {
                  stopLive();
                  void loadTail();
                }}
              >
                <RefreshCw
                  className={cn("size-3.5", loadingTail && "animate-spin")}
                />
                <span className="text-xs">Refresh</span>
              </Button>
              <Button
                type="button"
                variant={live ? "secondary" : "default"}
                size="sm"
                className="h-9 flex-1 gap-1.5"
                onClick={() => {
                  if (live) {
                    stopLive();
                  } else {
                    sinceSeqRef.current = maxSeq(entries);
                    setStreamError(null);
                    setLive(true);
                  }
                }}
              >
                {live ? (
                  <>
                    <PauseCircle className="size-3.5" />
                    <span className="text-xs">Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="size-3.5" />
                    <span className="text-xs">Live</span>
                  </>
                )}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="h-9 w-9 shrink-0"
                    aria-label="More log actions"
                  >
                    <MoreHorizontal className="size-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  <DropdownMenuItem
                    disabled={filtered.length === 0}
                    onClick={() => copyVisible()}
                  >
                    <Copy className="mr-2 size-4" />
                    Copy visible
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={filtered.length === 0}
                    onClick={() => downloadVisible()}
                  >
                    <Download className="mr-2 size-4" />
                    Download NDJSON
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <div className="hidden items-center justify-end gap-1.5 md:flex md:flex-nowrap">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 shrink-0 gap-1.5"
                disabled={loadingTail || live}
                onClick={() => {
                  stopLive();
                  void loadTail();
                }}
              >
                <RefreshCw
                  className={cn("size-3.5", loadingTail && "animate-spin")}
                />
                <span className="text-xs">Refresh</span>
              </Button>
              <Button
                type="button"
                variant={live ? "secondary" : "default"}
                size="sm"
                className="h-8 shrink-0 gap-1.5"
                onClick={() => {
                  if (live) {
                    stopLive();
                  } else {
                    sinceSeqRef.current = maxSeq(entries);
                    setStreamError(null);
                    setLive(true);
                  }
                }}
              >
                {live ? (
                  <>
                    <PauseCircle className="size-3.5" />
                    <span className="text-xs">Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="size-3.5" />
                    <span className="text-xs">Live</span>
                  </>
                )}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 gap-1.5"
                onClick={copyVisible}
                disabled={filtered.length === 0}
              >
                <Copy className="size-3.5" />
                <span className="text-xs">Copy</span>
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 gap-1.5"
                onClick={downloadVisible}
                disabled={filtered.length === 0}
              >
                <Download className="size-3.5" />
                <span className="text-xs">Download</span>
              </Button>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className={observabilityStyles.processContent}>
        <div className={observabilityStyles.compactToolbarGrid}>
          <div
            className={observabilityStyles.filterGroup}
            role="group"
            aria-label="Filter by stdout or stderr"
          >
            <div className={observabilityStyles.filterLabelRow}>
              <p className={observabilityStyles.filterLabel}>Stream</p>
              <HintIcon label="What the stream filter does">
                Choose which process stream to inspect. Structured SDK logs usually appear on
                stdout, while plain failures and stack traces often show up on stderr.
              </HintIcon>
            </div>
            <SegmentedControl
              value={streamFilter}
              onValueChange={(v) => setStreamFilter(v as StreamFilter)}
              size="sm"
              className="w-full xl:w-auto"
              options={[
                {
                  value: "all",
                  label: `All${streamCounts.all ? ` (${streamCounts.all})` : ""}`,
                },
                {
                  value: "stdout",
                  label: `Stdout${streamCounts.stdout ? ` (${streamCounts.stdout})` : ""}`,
                },
                {
                  value: "stderr",
                  label: `Stderr${streamCounts.stderr ? ` (${streamCounts.stderr})` : ""}`,
                },
              ]}
            />
          </div>

          <div
            className={observabilityStyles.filterGroup}
            role="group"
            aria-label="Filter by line format"
          >
            <div className={observabilityStyles.filterLabelRow}>
              <p className={observabilityStyles.filterLabel}>Format</p>
              <HintIcon label="What the format filter does">
                Structured lines are SDK-emitted JSON events with execution metadata. Plain lines
                are raw stdout or stderr text from the node process.
              </HintIcon>
            </div>
            <SegmentedControl
              value={formatFilter}
              onValueChange={(v) => setFormatFilter(v as FormatFilter)}
              size="sm"
              className="w-full xl:w-auto"
              options={[
                {
                  value: "all",
                  label: `All${formatCounts.all ? ` (${formatCounts.all})` : ""}`,
                },
                {
                  value: "structured",
                  label: `Structured${formatCounts.structured ? ` (${formatCounts.structured})` : ""}`,
                },
                {
                  value: "plain",
                  label: `Plain${formatCounts.plain ? ` (${formatCounts.plain})` : ""}`,
                },
              ]}
            />
          </div>

          <div className={observabilityStyles.filterGroup}>
            <Label
              htmlFor={`${nodeId}-log-text-filter`}
              className={observabilityStyles.filterLabel}
            >
              Search
            </Label>
            <SearchBar
              id={`${nodeId}-log-text-filter`}
              value={filter}
              onChange={setFilter}
              placeholder="Text, execution, run, event, reasoner, source…"
              size="sm"
              inputClassName="border-border/80 bg-background"
              aria-label="Filter log lines by text"
            />
          </div>
        </div>

        {streamCounts.other > 0 ? (
          <p className={observabilityStyles.helperText}>
            {streamCounts.other} line{streamCounts.other === 1 ? "" : "s"} on other streams, shown
            only in <span className="font-medium">All</span>.
          </p>
        ) : null}

        {filter.trim() !== "" || streamFilter !== "all" || formatFilter !== "all" ? (
          <p className={observabilityStyles.helperText}>
            {filter.trim() !== "" ? (
              <>
                <span className="font-medium tabular-nums text-foreground">
                  {filtered.length}
                </span>{" "}
                match{filtered.length === 1 ? "" : "es"} within{" "}
                <span className="tabular-nums">{formatScoped.length}</span> line
                {formatScoped.length === 1 ? "" : "s"}
                {streamFilter !== "all" ? ` (${streamFilter})` : ""}
                {formatFilter !== "all" ? ` · ${formatFilter}` : ""}
              </>
            ) : (
              <>
                <span className="font-medium tabular-nums text-foreground">
                  {filtered.length}
                </span>{" "}
                line{filtered.length === 1 ? "" : "s"} · stream: {streamFilter}
                {formatFilter !== "all" ? ` · format: ${formatFilter}` : ""}
              </>
            )}
          </p>
        ) : null}

        {formatFilter === "structured" &&
        filtered.length === 0 &&
        overallFormatCounts.structured > 0 ? (
          <Alert>
            <AlertCircle className="size-4" />
            <AlertTitle className="text-sm">Structured logs are available on a different stream</AlertTitle>
            <AlertDescription className="text-xs">
              The current stream filter is hiding them. Try <span className="font-medium">All</span>
              {streamCounts.stdout > 0 ? " or Stdout" : ""} to inspect the structured SDK lines.
            </AlertDescription>
          </Alert>
        ) : null}

        {streamError ? (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle className="text-sm">Logs unavailable</AlertTitle>
            <AlertDescription className="text-xs">{streamError}</AlertDescription>
          </Alert>
        ) : null}

        <ScrollArea
          ref={scrollRef}
          className={observabilityStyles.processScroll}
        >
          <div className={observabilityStyles.processScrollInner}>
            {filtered.length === 0 && !loadingTail ? (
              <p className="px-2 py-6 text-center text-muted-foreground text-xs">
                No log lines yet. Try Refresh, or enable live tail if the agent
                supports streaming.
              </p>
            ) : (
              <div
                role="log"
                aria-label="Process log lines"
                className="min-w-0 select-text font-mono"
              >
                {filtered.map((e, i) => {
                  const ns = normalizeStream(e.stream);
                  const dateHint = formatLogDate(e.ts);
                  const timeStr = formatLogTime(e.ts);
                  const structured = parseStructuredProcessLog(e.line);
                  const title = `${e.ts ?? ""} · seq ${e.seq ?? "?"}${
                    structured?.level ?? e.level ? ` · ${structured?.level ?? e.level}` : ""
                  }${structured?.source ?? e.source ? ` · ${structured?.source ?? e.source}` : ""}`;
                  const showLevel =
                    (structured?.level ?? e.level) &&
                    !isRedundantLevel(structured?.level ?? e.level, ns);
                  const showSource =
                    (structured?.source ?? e.source) &&
                    (structured?.source ?? e.source)?.toLowerCase() !== "process";
                  const primaryMessage =
                    structured?.message?.trim() ||
                    structured?.event_type?.trim() ||
                    e.line;
                  const streamLabel = ns === "other" ? e.stream || "?" : ns;
                  const metadata = [
                    { label: "v", value: compactValue(structured?.v) },
                    { label: "exec", value: compactValue(structured?.execution_id) },
                    { label: "run", value: compactValue(structured?.run_id) },
                    { label: "reasoner", value: compactValue(structured?.reasoner_id) },
                    { label: "event", value: compactValue(structured?.event_type) },
                    {
                      label: "source",
                      value: compactValue(structured?.source ?? e.source),
                    },
                  ].filter((item) => item.value);

                  return structured ? (
                    <Collapsible
                      key={`${e.seq}-${e.ts}-${i}`}
                      title={title}
                      className="border-b border-border/30 last:border-b-0"
                    >
                      <div
                        className={cn(
                          observabilityStyles.processStructuredRow,
                          ns === "stderr" && "bg-destructive/[0.04]"
                        )}
                      >
                        <div className={cn(observabilityStyles.processTimestamp, "sm:w-full sm:max-w-[8.5rem]")}>
                          <time
                            dateTime={e.ts}
                            className="min-w-0 shrink truncate text-muted-foreground"
                          >
                            {dateHint ? (
                              <span className="mr-1 text-[9px] opacity-85">{dateHint}</span>
                            ) : null}
                            <span className="whitespace-nowrap">{timeStr}</span>
                          </time>
                          <span className="shrink-0 text-[9px] text-muted-foreground/65">
                            #{e.seq}
                          </span>
                        </div>

                        <div className="flex flex-wrap items-center gap-0.5 sm:self-start sm:pt-px">
                          <Badge
                            variant={ns === "stderr" ? "destructive" : "secondary"}
                            className="h-4 shrink-0 px-1 py-0 text-[8px] font-normal uppercase leading-none"
                          >
                            {streamLabel}
                          </Badge>
                          {showLevel ? (
                            <Badge
                              variant={levelBadgeVariant(structured?.level ?? e.level)}
                              className="h-4 max-w-[4.5rem] shrink-0 truncate px-1 py-0 text-[8px] font-normal capitalize leading-none"
                            >
                              {structured?.level ?? e.level}
                            </Badge>
                          ) : null}
                        </div>

                        <div className="min-w-0 sm:pt-px">
                          <div className="flex min-w-0 items-baseline gap-2">
                            <span className="truncate select-text font-mono text-[10px] leading-snug text-foreground sm:text-[11px]">
                              {primaryMessage}
                              {e.truncated ? (
                                <span className="text-muted-foreground"> …</span>
                              ) : null}
                            </span>
                          </div>
                          <div className={observabilityStyles.processMeta}>
                            {metadata.map((item) => (
                              <span key={`${e.seq}-${item.label}`} className="font-mono">
                                {item.label}:{item.value}
                              </span>
                            ))}
                          </div>
                        </div>

                        {hasStructuredDetails(structured) ? (
                          <CollapsibleTrigger className={observabilityStyles.detailTrigger}>
                            <ChevronRight className="h-3 w-3 group-data-[state=open]:hidden" />
                            <ChevronDown className="hidden h-3 w-3 group-data-[state=open]:block" />
                            details
                          </CollapsibleTrigger>
                        ) : null}
                      </div>

                      {hasStructuredDetails(structured) ? (
                        <CollapsibleContent className="border-t border-border/30 bg-muted/[0.08] px-2 py-2 sm:pl-[calc(8.5rem+1.25rem)]">
                          <div className="overflow-hidden rounded-md border border-border/50 bg-background/80">
                            <pre className="overflow-x-auto select-text p-2 text-[10px] leading-relaxed text-foreground/85">
                              {JSON.stringify(structured, null, 2)}
                            </pre>
                          </div>
                        </CollapsibleContent>
                      ) : null}
                    </Collapsible>
                  ) : (
                    <div
                      key={`${e.seq}-${e.ts}-${i}`}
                      title={title}
                      className={cn(
                        observabilityStyles.processRow,
                        ns === "stderr" && "bg-destructive/[0.04]"
                      )}
                    >
                      <div className={cn(observabilityStyles.processTimestamp, "sm:w-full sm:max-w-[9rem]")}>
                        <time
                          dateTime={e.ts}
                          className="min-w-0 shrink truncate text-muted-foreground"
                        >
                          {dateHint ? (
                            <span className="mr-1 text-[9px] opacity-85">{dateHint}</span>
                          ) : null}
                          <span className="whitespace-nowrap">{timeStr}</span>
                        </time>
                        <span className="shrink-0 text-[9px] text-muted-foreground/65">
                          #{e.seq}
                        </span>
                      </div>

                      <div className="flex flex-wrap items-center gap-0.5 sm:self-start sm:pt-px">
                        <Badge
                          variant={ns === "stderr" ? "destructive" : "secondary"}
                          className="h-4 shrink-0 px-1 py-0 text-[8px] font-normal uppercase leading-none"
                        >
                          {streamLabel}
                        </Badge>
                        {showLevel ? (
                          <Badge
                            variant={levelBadgeVariant(e.level)}
                            className="h-4 max-w-[4.5rem] shrink-0 truncate px-1 py-0 text-[8px] font-normal capitalize leading-none"
                          >
                            {e.level}
                          </Badge>
                        ) : null}
                      </div>

                      <div className="min-w-0 sm:pt-px">
                        {showSource ? (
                          <span className="mb-0.5 block truncate text-[9px] font-sans text-muted-foreground">
                            {e.source}
                          </span>
                        ) : null}
                        <span className="block select-text whitespace-pre-wrap break-all font-mono text-[10px] leading-snug text-foreground/90 sm:text-[11px]">
                          {e.line}
                          {e.truncated ? (
                            <span className="text-muted-foreground"> …</span>
                          ) : null}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
