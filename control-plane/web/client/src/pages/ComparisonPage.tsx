import { Fragment, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import type { UseQueryResult } from "@tanstack/react-query";
import { useRunDAG, useStepDetail } from "@/hooks/queries";
import { formatDuration } from "@/components/RunTrace";
import { normalizeExecutionStatus } from "@/utils/status";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertTriangle, ArrowLeft, ChevronDown, Equal, ExternalLink, Minus } from "lucide-react";
import type { WorkflowDAGLightweightNode } from "@/types/workflows";
import type { WorkflowExecution } from "@/types/executions";
import type { CanonicalStatus } from "@/utils/status";
import { JsonHighlightedPre } from "@/components/ui/json-syntax-highlight";
import {
  extractReasonerInputLayers,
  formatOutputUsageHint,
} from "@/utils/reasonerCompareExtract";

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Maps workflow status to ui/badge StatusBadge variant (same as workflow identity). */
type UiRunStatusBadge = "success" | "failed" | "running" | "pending" | "degraded" | "unknown";

const CANONICAL_TO_RUN_STATUS_BADGE: Record<CanonicalStatus, UiRunStatusBadge> = {
  pending: "pending",
  queued: "pending",
  waiting: "pending",
  paused: "degraded",
  running: "running",
  succeeded: "success",
  failed: "failed",
  cancelled: "unknown",
  timeout: "failed",
  unknown: "unknown",
};

function runStatusToUiBadge(status: string): UiRunStatusBadge {
  return CANONICAL_TO_RUN_STATUS_BADGE[normalizeExecutionStatus(status)] ?? "unknown";
}

/** Dot-only status for dense step rows (label in tooltip). */
function StatusCue({ status }: { status?: string }) {
  const label = status ? normalizeExecutionStatus(status) : "missing";
  const canonical = status ? normalizeExecutionStatus(status) : "";
  const color =
    canonical === "succeeded"
      ? "bg-green-500"
      : canonical === "failed" || canonical === "timeout"
        ? "bg-destructive"
        : canonical === "running"
          ? "bg-primary"
          : "bg-muted-foreground/50";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-block size-2 shrink-0 rounded-full",
            !status && "bg-border",
            status && color,
          )}
          aria-label={label}
        />
      </TooltipTrigger>
      <TooltipContent side="top" className="text-xs capitalize">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

/** Compact comparison cue: icons only, meaning in tooltip. */
function RowCompareCue({
  diverged,
  extra,
  missingSide,
}: {
  diverged: boolean;
  extra: boolean;
  missingSide: "a" | "b" | null;
}) {
  const tip = diverged
    ? "Status differs between runs"
    : extra
      ? missingSide === "a"
        ? "No step in run A at this index"
        : missingSide === "b"
          ? "No step in run B at this index"
          : "No step in either run at this index"
      : "Same status";

  const node = diverged ? (
    <AlertTriangle className="size-3.5 shrink-0 text-amber-500" aria-hidden />
  ) : extra ? (
    <Minus className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
  ) : (
    <Equal className="size-3.5 shrink-0 text-muted-foreground/25" aria-hidden />
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="flex size-8 items-center justify-center rounded-md text-muted-foreground outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={tip}
          onClick={(e) => e.stopPropagation()}
        >
          {node}
        </button>
      </TooltipTrigger>
      <TooltipContent side="left" className="max-w-xs text-xs">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
}

/** Compact run id for dense headers (full id in title/tooltip). */
function formatCompareRunId(runId: string): string {
  return runId.length <= 14 ? runId : `${runId.slice(0, 10)}…${runId.slice(-4)}`;
}

function durationDeltaLabel(msA?: number, msB?: number): string | null {
  if (msA == null || msB == null) return null;
  if (msA === 0) return null;
  const delta = ((msB - msA) / msA) * 100;
  if (Math.abs(delta) < 1) return null;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(0)}%`;
}

// ─── Step I/O diff section ────────────────────────────────────────────────────

const compareJsonBlockClass =
  "max-h-[min(42vh,22rem)] overflow-auto rounded-md border border-border/60 bg-muted/50 p-3 text-micro-plus leading-relaxed";

function formatBytesCompact(n?: number): string | null {
  if (n == null || n <= 0) return null;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10_240 ? 1 : 0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatStepWhen(iso?: string): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function CompareStepPanel({
  label,
  step,
  detail,
  runId,
  workflowName,
}: {
  label: "A" | "B";
  step: WorkflowDAGLightweightNode | undefined;
  detail: UseQueryResult<WorkflowExecution>;
  runId: string;
  workflowName: string | undefined;
}) {
  const ex = detail.data;
  const startedAt = ex?.started_at ?? step?.started_at;
  const completedAt = ex?.completed_at ?? step?.completed_at;
  const durationMs = ex?.duration_ms ?? step?.duration_ms;
  const statusRaw = ex?.status ?? (step?.status ? normalizeExecutionStatus(step.status) : undefined);
  const statusStr = statusRaw ?? "unknown";

  const layers = extractReasonerInputLayers(ex?.input_data);
  const usageHint = formatOutputUsageHint(ex?.output_data);

  const hasEx = Boolean(ex);
  const hasInputPayload = ex?.input_data != null;
  const hasOutputPayload = ex?.output_data != null;
  const hasErr = Boolean(ex?.error_message);
  const notes = ex?.notes ?? [];
  const inBytes = formatBytesCompact(ex?.input_size);
  const outBytes = formatBytesCompact(ex?.output_size);

  const defaultTab = hasErr
    ? "error"
    : hasInputPayload
      ? "input"
      : hasOutputPayload
        ? "output"
        : notes.length > 0
          ? "notes"
          : "input";

  return (
    <div
      className={cn(
        "flex min-w-0 flex-col gap-3 rounded-lg border border-border bg-card p-3 sm:p-4",
        label === "B" ? "border-l-[3px] border-l-secondary" : "border-l-[3px] border-l-primary",
      )}
    >
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <Badge variant="outline" className="shrink-0 font-mono text-micro">
              Run {label}
            </Badge>
            {workflowName?.trim() ? (
              <span
                className="truncate text-xs font-medium text-foreground"
                title={workflowName}
              >
                {workflowName}
              </span>
            ) : null}
          </div>
          <p
            className="truncate font-mono text-micro-plus text-muted-foreground"
            title={step?.reasoner_id}
          >
            {step?.reasoner_id ?? "—"}
          </p>
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="max-w-[min(100%,11rem)] shrink-0 truncate text-right font-mono text-micro text-muted-foreground">
              {formatCompareRunId(runId)}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-md">
            <p className="break-all font-mono text-xs">{runId}</p>
          </TooltipContent>
        </Tooltip>
      </div>

      {!step && !ex ? (
        <p className="text-xs text-muted-foreground">No step on this side.</p>
      ) : null}

      {(step || ex) && (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
            <StatusBadge
              status={runStatusToUiBadge(statusStr)}
              size="sm"
              showIcon={false}
              className="h-5 text-micro"
            >
              {statusStr}
            </StatusBadge>
            {durationMs != null && durationMs > 0 ? (
              <Badge variant="secondary" className="h-5 font-mono text-micro font-normal">
                {formatDuration(durationMs)}
              </Badge>
            ) : null}
            {formatStepWhen(startedAt) ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="h-5 max-w-[9rem] truncate text-micro font-normal">
                    {formatStepWhen(startedAt)}
                    {completedAt ? ` → ${formatStepWhen(completedAt)}` : ""}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  <div>Started: {startedAt ?? "—"}</div>
                  <div>Completed: {completedAt ?? "—"}</div>
                </TooltipContent>
              </Tooltip>
            ) : null}
            {ex?.agent_node_id ? (
              <Badge variant="outline" className="h-5 max-w-[8rem] truncate text-micro font-normal">
                {ex.agent_node_id}
              </Badge>
            ) : null}
            {ex?.workflow_depth != null && ex.workflow_depth > 0 ? (
              <Badge variant="outline" className="h-5 text-micro font-normal">
                depth {ex.workflow_depth}
              </Badge>
            ) : null}
            {ex != null && ex.retry_count > 0 ? (
              <Badge variant="outline" className="h-5 text-micro font-normal">
                retries {ex.retry_count}
              </Badge>
            ) : null}
            {inBytes ? (
              <Badge variant="outline" className="h-5 text-micro font-normal">
                in {inBytes}
              </Badge>
            ) : null}
            {outBytes ? (
              <Badge variant="outline" className="h-5 text-micro font-normal">
                out {outBytes}
              </Badge>
            ) : null}
            {usageHint ? (
              <Badge variant="secondary" className="h-5 max-w-[14rem] truncate text-micro font-normal">
                {usageHint}
              </Badge>
            ) : null}
          </div>

          {step && !hasEx && detail.isFetched && !detail.isLoading ? (
            <p className="rounded-md border border-dashed border-border/80 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              Execution details are not available for this step. Open the run to inspect I/O.
            </p>
          ) : null}

          {hasEx && (layers.prose.length > 0 || layers.meta.length > 0) ? (
            <div className="rounded-lg border border-border/80 bg-muted/25 p-3">
              <p className="mb-2 text-micro font-medium uppercase tracking-wide text-muted-foreground">
                Reasoner context
              </p>
              {layers.meta.length > 0 ? (
                <div className="mb-2 flex flex-wrap gap-1">
                  {layers.meta.map((m) => (
                    <Tooltip key={m.key}>
                      <TooltipTrigger asChild>
                        <Badge variant="outline" className="h-5 max-w-[11rem] truncate text-micro font-normal">
                          <span className="text-muted-foreground">{m.label}:</span>{" "}
                          <span className="font-mono">{m.value}</span>
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-md break-all text-xs">{m.value}</TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              ) : null}
              <div className="flex flex-col gap-2">
                {layers.prose.map((p) => (
                  <div key={p.key} className="min-w-0">
                    <p className="text-micro font-medium text-muted-foreground">{p.label}</p>
                    <div
                      className="mt-1 max-h-28 overflow-y-auto rounded-md border border-border/50 bg-background/80 px-2.5 py-2 text-xs leading-snug text-foreground"
                      title={p.text}
                    >
                      {p.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {hasEx ? (
          <Tabs defaultValue={defaultTab} className="min-w-0">
            <TabsList variant="soft" className="h-8 w-full justify-start gap-0.5 p-1 sm:w-auto">
              <TabsTrigger variant="soft" size="sm" value="input" disabled={!hasEx} className="text-micro-plus">
                Input JSON
              </TabsTrigger>
              <TabsTrigger variant="soft" size="sm" value="output" disabled={!hasEx} className="text-micro-plus">
                Output
              </TabsTrigger>
              {notes.length > 0 ? (
                <TabsTrigger variant="soft" size="sm" value="notes" className="text-micro-plus">
                  Notes ({notes.length})
                </TabsTrigger>
              ) : null}
              {hasErr ? (
                <TabsTrigger variant="soft" size="sm" value="error" className="text-micro-plus text-destructive">
                  Error
                </TabsTrigger>
              ) : null}
            </TabsList>
            <TabsContent value="input" className="mt-2">
              <JsonHighlightedPre
                data={ex?.input_data}
                className={compareJsonBlockClass}
              />
            </TabsContent>
            <TabsContent value="output" className="mt-2">
              {hasErr ? (
                <p className="mb-2 text-micro-plus text-muted-foreground">
                  This step failed; open the Error tab for the message. Raw output (if any) below.
                </p>
              ) : null}
              <JsonHighlightedPre
                data={ex?.output_data}
                className={compareJsonBlockClass}
              />
            </TabsContent>
            <TabsContent value="notes" className="mt-2 space-y-2">
              {notes.map((note, i) => (
                <div key={i} className="rounded-md border border-border/60 bg-muted/40 p-2 text-xs">
                  <span className="text-muted-foreground">
                    {new Date(note.timestamp).toLocaleString()}
                  </span>
                  <p className="mt-1 whitespace-pre-wrap break-words">{note.message}</p>
                  {note.tags?.length ? (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {note.tags.map((tag) => (
                        <Badge key={tag} variant="outline" className="h-4 text-nano">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </TabsContent>
            <TabsContent value="error" className="mt-2">
              <pre
                className={cn(
                  compareJsonBlockClass,
                  "bg-destructive/10 font-mono text-destructive",
                )}
              >
                {ex?.error_message ?? "—"}
              </pre>
            </TabsContent>
          </Tabs>
          ) : null}

          {ex?.execution_id ? (
            <p className="text-micro text-muted-foreground">
              <span className="font-medium text-foreground/80">Execution</span>{" "}
              <span className="font-mono">{ex.execution_id}</span>
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}

function StepDiff({
  stepA,
  stepB,
  runIdA,
  runIdB,
  workflowNameA,
  workflowNameB,
}: {
  stepA?: WorkflowDAGLightweightNode;
  stepB?: WorkflowDAGLightweightNode;
  runIdA: string;
  runIdB: string;
  workflowNameA?: string;
  workflowNameB?: string;
}) {
  const detailA = useStepDetail(stepA?.execution_id);
  const detailB = useStepDetail(stepB?.execution_id);

  return (
    <div className="space-y-3">
      {detailA.isLoading || detailB.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="min-h-[20rem] w-full rounded-lg" />
          <Skeleton className="min-h-[20rem] w-full rounded-lg" />
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 sm:gap-4">
          <CompareStepPanel
            label="A"
            step={stepA}
            detail={detailA}
            runId={runIdA}
            workflowName={workflowNameA}
          />
          <CompareStepPanel
            label="B"
            step={stepB}
            detail={detailB}
            runId={runIdB}
            workflowName={workflowNameB}
          />
        </div>
      )}
    </div>
  );
}

// ─── Run summary card ──────────────────────────────────────────────────────────

function RunSummaryCard({
  runId,
  label,
  workflowName,
  status,
  stepCount,
  failureCount,
  durationMs,
  deltaLabel,
  isB,
}: {
  runId: string;
  label: "A" | "B";
  workflowName?: string;
  status: string;
  stepCount: number;
  failureCount: number;
  durationMs?: number;
  deltaLabel?: string | null;
  isB?: boolean;
}) {
  const navigate = useNavigate();
  const shortId = formatCompareRunId(runId);

  return (
    <Card
      className={cn(
        "flex-1 overflow-hidden border-l-4",
        isB ? "border-l-secondary" : "border-l-primary",
      )}
    >
      <CardHeader className="gap-3 space-y-0 pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Badge variant="outline" className="shrink-0 font-mono">
              {label}
            </Badge>
            <CardTitle className="truncate leading-none">
              {workflowName ? workflowName : `Run ${label}`}
            </CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 shrink-0 gap-1 px-2 text-xs text-muted-foreground"
            onClick={() => navigate(`/runs/${runId}`)}
          >
            <ExternalLink className="size-3.5" />
            Detail
          </Button>
        </div>
        <div className="space-y-1">
          <p
            className="truncate font-mono text-xs text-muted-foreground"
            title={runId}
          >
            {shortId}
          </p>
          {workflowName ? (
            <p className="text-xs text-muted-foreground">Run {label}</p>
          ) : null}
        </div>
        <StatusBadge
          status={runStatusToUiBadge(status)}
          size="sm"
          showIcon={false}
          className="w-fit"
        >
          {normalizeExecutionStatus(status)}
        </StatusBadge>
      </CardHeader>
      <CardContent className="pb-4 pt-0">
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Steps
            </p>
            <p className="text-sm font-semibold tabular-nums">{stepCount}</p>
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Failures
            </p>
            <p
              className={cn(
                "text-sm font-semibold tabular-nums",
                failureCount > 0 && "text-destructive",
              )}
            >
              {failureCount > 0 ? failureCount : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Duration
            </p>
            <div className="flex flex-wrap items-center justify-center gap-1">
              <p className="text-sm font-semibold tabular-nums">
                {durationMs != null ? formatDuration(durationMs) : "—"}
              </p>
              {deltaLabel ? (
                <span className="text-xs text-muted-foreground">({deltaLabel})</span>
              ) : null}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function ComparisonPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const runIdA = searchParams.get("a") ?? undefined;
  const runIdB = searchParams.get("b") ?? undefined;

  const dagA = useRunDAG(runIdA);
  const dagB = useRunDAG(runIdB);

  const [expandedStepIndex, setExpandedStepIndex] = useState<number | null>(null);

  // ─── Loading ──────────────────────────────────────────────────────────────

  if (dagA.isLoading || dagB.isLoading) {
    return (
      <div className="flex flex-col gap-4 pb-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-8 w-20" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Skeleton className="h-28 w-full rounded-xl" />
          <Skeleton className="h-28 w-full rounded-xl" />
        </div>
        <Skeleton className="min-h-[24rem] w-full rounded-xl" />
      </div>
    );
  }

  // ─── Missing IDs ──────────────────────────────────────────────────────────

  if (!runIdA || !runIdB) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Compare Runs</h1>
        <p className="text-sm text-muted-foreground">
          Select two runs from the Runs page to compare them.
        </p>
        <Button variant="outline" size="sm" className="w-fit" onClick={() => navigate("/runs")}>
          <ArrowLeft className="size-3 mr-1.5" />
          Back to Runs
        </Button>
      </div>
    );
  }

  // ─── Error ────────────────────────────────────────────────────────────────

  if (dagA.isError || dagB.isError) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Compare Runs</h1>
        <div className="rounded-md bg-destructive/10 border border-destructive/20 p-4 text-sm text-destructive">
          {dagA.isError
            ? `Failed to load Run A: ${dagA.error instanceof Error ? dagA.error.message : "Unknown error"}`
            : `Failed to load Run B: ${dagB.error instanceof Error ? dagB.error.message : "Unknown error"}`}
        </div>
        <Button variant="outline" size="sm" className="w-fit" onClick={() => navigate("/runs")}>
          <ArrowLeft className="size-3 mr-1.5" />
          Back to Runs
        </Button>
      </div>
    );
  }

  const dataA = dagA.data!;
  const dataB = dagB.data!;

  const stepsA = dataA.timeline ?? [];
  const stepsB = dataB.timeline ?? [];
  const maxLen = Math.max(stepsA.length, stepsB.length);

  const rootA = stepsA.find((n) => n.workflow_depth === 0) ?? stepsA[0];
  const rootB = stepsB.find((n) => n.workflow_depth === 0) ?? stepsB[0];

  const durationMsA = rootA?.duration_ms;
  const durationMsB = rootB?.duration_ms;

  const failureCountA = stepsA.filter((s) => normalizeExecutionStatus(s.status) === "failed" || normalizeExecutionStatus(s.status) === "timeout").length;
  const failureCountB = stepsB.filter((s) => normalizeExecutionStatus(s.status) === "failed" || normalizeExecutionStatus(s.status) === "timeout").length;

  const deltaLabel = durationDeltaLabel(durationMsA, durationMsB);

  return (
    <TooltipProvider delayDuration={250}>
      <div className="flex flex-col gap-4 pb-8">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">Compare Runs</h1>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            onClick={() => navigate("/runs")}
          >
            <ArrowLeft className="mr-1.5 size-3" />
            Back
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <RunSummaryCard
            runId={runIdA}
            label="A"
            workflowName={dataA.workflow_name}
            status={dataA.workflow_status}
            stepCount={stepsA.length}
            failureCount={failureCountA}
            durationMs={durationMsA}
          />
          <RunSummaryCard
            runId={runIdB}
            label="B"
            workflowName={dataB.workflow_name}
            status={dataB.workflow_status}
            stepCount={stepsB.length}
            failureCount={failureCountB}
            durationMs={durationMsB}
            deltaLabel={deltaLabel}
            isB
          />
        </div>

        <Separator />

        <Card className="flex min-h-0 flex-col">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Steps</CardTitle>
            <p className="text-xs font-normal text-muted-foreground">
              Expand a row for side-by-side metadata, reasoner context (goal / prompts / start tip when
              present in input), and tabbed full JSON. Dots = status; last column = alignment (hover).
            </p>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col p-0">
            <div className="min-h-[12rem] flex-1 overflow-y-auto border-t">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="w-14 px-2 py-2 text-xs font-medium text-muted-foreground">
                      #
                    </TableHead>
                    <TableHead
                      className="min-w-0 px-2 py-2 text-xs font-medium"
                      title={runIdA}
                    >
                      <span className="text-muted-foreground">A</span>
                      <span className="ml-1 font-mono text-micro font-normal text-muted-foreground/80">
                        {formatCompareRunId(runIdA)}
                      </span>
                    </TableHead>
                    <TableHead
                      className="min-w-0 px-2 py-2 text-xs font-medium"
                      title={runIdB}
                    >
                      <span className="text-muted-foreground">B</span>
                      <span className="ml-1 font-mono text-micro font-normal text-muted-foreground/80">
                        {formatCompareRunId(runIdB)}
                      </span>
                    </TableHead>
                    <TableHead className="w-12 px-0 py-2 text-center">
                      <span className="sr-only">Compare</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {maxLen === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={4}
                        className="py-10 text-center text-xs text-muted-foreground"
                      >
                        No steps available for comparison
                      </TableCell>
                    </TableRow>
                  ) : (
                    Array.from({ length: maxLen }).map((_, i) => {
                      const a = stepsA[i];
                      const b = stepsB[i];

                      const statusA = normalizeExecutionStatus(a?.status);
                      const statusB = normalizeExecutionStatus(b?.status);

                      const bothPresent = !!a && !!b;
                      const diverged = bothPresent && statusA !== statusB;
                      const extra = !a || !b;
                      const missingSide =
                        !a && !b ? null : !a ? "a" : !b ? "b" : null;

                      const isExpanded = expandedStepIndex === i;
                      const reasonerLabel =
                        a?.reasoner_id ?? b?.reasoner_id ?? "—";

                      return (
                        <Fragment key={i}>
                          <TableRow
                            data-state={isExpanded ? "selected" : undefined}
                            role="button"
                            tabIndex={0}
                            aria-expanded={isExpanded}
                            aria-label={`Step ${i + 1}, ${reasonerLabel}. ${isExpanded ? "Expanded" : "Collapsed"}. Press to toggle.`}
                            className={cn(
                              "cursor-pointer border-b border-border/60",
                              isExpanded && "bg-muted/60",
                              !isExpanded && "hover:bg-muted/40",
                              diverged && !isExpanded && "bg-amber-500/[0.04]",
                            )}
                            onClick={() => {
                              setExpandedStepIndex(isExpanded ? null : i);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                setExpandedStepIndex(isExpanded ? null : i);
                              }
                            }}
                          >
                            <TableCell className="px-2 py-2 align-middle">
                              <div className="flex items-center gap-1.5 text-xs tabular-nums text-muted-foreground">
                                <ChevronDown
                                  className={cn(
                                    "size-4 shrink-0 text-muted-foreground transition-transform duration-200",
                                    isExpanded && "rotate-180",
                                  )}
                                  aria-hidden
                                />
                                <span>{i + 1}</span>
                              </div>
                            </TableCell>
                            <TableCell className="max-w-0 px-2 py-2 align-middle">
                              <div className="flex min-w-0 items-center gap-2">
                                <span
                                  className="min-w-0 flex-1 truncate font-mono text-xs"
                                  title={a?.reasoner_id}
                                >
                                  {a?.reasoner_id ?? (
                                    <span className="text-muted-foreground/50">—</span>
                                  )}
                                </span>
                                <StatusCue status={a?.status} />
                              </div>
                            </TableCell>
                            <TableCell className="max-w-0 px-2 py-2 align-middle">
                              <div className="flex min-w-0 items-center gap-2">
                                <span
                                  className="min-w-0 flex-1 truncate font-mono text-xs"
                                  title={b?.reasoner_id}
                                >
                                  {b?.reasoner_id ?? (
                                    <span className="text-muted-foreground/50">—</span>
                                  )}
                                </span>
                                <StatusCue status={b?.status} />
                              </div>
                            </TableCell>
                            <TableCell className="w-12 px-0 py-0 align-middle">
                              <RowCompareCue
                                diverged={diverged}
                                extra={extra}
                                missingSide={missingSide}
                              />
                            </TableCell>
                          </TableRow>
                          <TableRow className="border-b border-border/60 hover:bg-transparent">
                            <TableCell colSpan={4} className="p-0">
                              <Collapsible open={isExpanded}>
                                <CollapsibleContent className="overflow-hidden">
                                  <div
                                    className="border-t border-border bg-muted/30 px-3 py-3 sm:px-4 sm:py-4"
                                    role="region"
                                    aria-label={`Step ${i + 1} comparison details for runs A and B`}
                                  >
                                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium text-foreground">
                                          Step comparison
                                        </span>
                                        <span className="text-muted-foreground/70"> · </span>
                                        <span className="font-mono text-micro-plus">
                                          Step {i + 1} · {reasonerLabel}
                                        </span>
                                      </p>
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        className="h-8 shrink-0 text-xs"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setExpandedStepIndex(null);
                                        }}
                                      >
                                        Collapse
                                      </Button>
                                    </div>
                                    <div className="max-h-[min(70vh,48rem)] overflow-y-auto">
                                      <StepDiff
                                        stepA={a}
                                        stepB={b}
                                        runIdA={runIdA}
                                        runIdB={runIdB}
                                        workflowNameA={dataA.workflow_name}
                                        workflowNameB={dataB.workflow_name}
                                      />
                                    </div>
                                  </div>
                                </CollapsibleContent>
                              </Collapsible>
                            </TableCell>
                          </TableRow>
                        </Fragment>
                      );
                    })
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
}
