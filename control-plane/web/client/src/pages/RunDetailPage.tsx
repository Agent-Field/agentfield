import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  useRunDAG,
  useCancelExecution,
  usePauseExecution,
  useResumeExecution,
} from "@/hooks/queries";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Activity,
  BadgeCheck,
  ChevronDown,
  FileJson,
  FileCheck2,
  Info,
  Link2,
  PauseCircle,
  Play,
  RefreshCw,
  RotateCcw,
  XCircle,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useRunNotification } from "@/components/ui/notification";
import { CANCEL_RUN_COPY } from "@/components/runs/RunLifecycleMenu";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { CopyIdentifierChip } from "@/components/ui/copy-identifier-chip";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { RunTrace, buildTraceTree, formatDuration } from "@/components/RunTrace";
import { StepDetail } from "@/components/StepDetail";
import { WorkflowDAGViewer } from "@/components/WorkflowDAG";
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ExecutionObservabilityPanel } from "@/components/execution";
import { normalizeExecutionStatus, isTerminalStatus } from "@/utils/status";
import { StatusPill } from "@/components/ui/status-pill";
import type {
  WebhookFailurePreview,
  WebhookRunSummary,
  WorkflowDAGLightweightNode,
  WorkflowDAGLightweightResponse,
} from "@/types/workflows";
import type { WorkflowExecution } from "@/types/executions";
import { retryExecutionWebhook, getExecutionDetails } from "@/services/executionsApi";
import {
  downloadWorkflowVCAuditFile,
  getWorkflowVCChain,
} from "@/services/vcApi";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function computeMaxDuration(
  timeline: WorkflowDAGLightweightNode[],
): number {
  if (!timeline || timeline.length === 0) return 1;
  const max = Math.max(...timeline.map((n) => n.duration_ms ?? 0));
  return Math.max(max, 1);
}

/** Compact display for long session/actor strings in the meta row. */
function truncateEnd(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, Math.max(0, max - 1))}…`;
}

const RUN_DETAIL_TITLE_MAX_CHARS = 42;

const ZERO_WEBHOOK_SUMMARY: WebhookRunSummary = {
  steps_with_webhook: 0,
  total_deliveries: 0,
  failed_deliveries: 0,
};

function RunContextHint({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-5 shrink-0 items-center justify-center rounded-sm text-muted-foreground/45 transition-colors hover:bg-muted hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          aria-label={label}
        >
          <Info className="size-3" strokeWidth={2.25} />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-[min(18rem,calc(100vw-1.5rem))] border border-border bg-popover px-2.5 py-2 text-left text-micro-plus leading-snug text-popover-foreground shadow-md"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}

type RunParticipantsSource = "api_agent" | "timeline_agent" | "reasoner";

/** Distinct participant ids for the run: API rollup agent ids, else timeline agent_node_id, else reasoner_id. */
function deriveRunParticipants(dag: WorkflowDAGLightweightResponse): {
  ids: string[];
  source: RunParticipantsSource;
} {
  const api = (dag.unique_agent_node_ids ?? []).map((id) => id.trim()).filter(Boolean);
  if (api.length > 0) {
    return { ids: [...new Set(api)].sort(), source: "api_agent" };
  }
  const fromTimeline = new Set<string>();
  for (const n of dag.timeline ?? []) {
    const id = n.agent_node_id?.trim();
    if (id) fromTimeline.add(id);
  }
  if (fromTimeline.size > 0) {
    return { ids: [...fromTimeline].sort(), source: "timeline_agent" };
  }
  const reasoners = new Set<string>();
  for (const n of dag.timeline ?? []) {
    const id = n.reasoner_id?.trim();
    if (id) reasoners.add(id);
  }
  return { ids: [...reasoners].sort(), source: "reasoner" };
}

function RunContextNodesCard({
  participantIds,
  source,
}: {
  participantIds: string[];
  source: "api_agent" | "timeline_agent" | "reasoner";
}) {
  const hasIds = participantIds.length > 0;
  const heading = source === "reasoner" ? "Reasoners" : "Nodes";
  const hint =
    source === "reasoner"
      ? "These are distinct reasoner IDs from the run timeline. Stored executions had no agent_node_id, so the graph labels steps by reasoner — same data as the graph."
      : source === "timeline_agent"
        ? "Distinct agent node IDs taken from the run timeline (execution records had no agent_node_id in the roll-up field)."
        : "Distinct agent node IDs for this run from the server. Select a step for that step's payload and detail.";
  return (
    <Card
      className={cn(
        "min-w-0 border-border/80 shadow-none",
        !hasIds && "border-dashed border-border/50 bg-muted/15",
      )}
    >
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-0.5">
          <p className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
            {heading}
          </p>
          <RunContextHint label={`About ${heading.toLowerCase()} on this run`}>
            {hint}
          </RunContextHint>
        </div>
        {hasIds ? (
          <div className="flex flex-wrap gap-1.5">
            {participantIds.map((id) => (
              <Badge
                key={id}
                variant="secondary"
                className="max-w-full truncate font-mono text-micro font-normal"
                title={id}
              >
                {id}
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            No agent or reasoner identifiers on this run.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/** Run-level webhook roll-up + failed rows with retry (like legacy workflow webhooks tab). */
function RunContextWebhooksCard({
  summary,
  failures,
  onSelectStep,
  onRefetchDag,
}: {
  summary: WebhookRunSummary;
  failures: WebhookFailurePreview[];
  onSelectStep: (executionId: string) => void;
  onRefetchDag: () => void;
}) {
  const [retrying, setRetrying] = useState<Record<string, boolean>>({});
  const [retryAllBusy, setRetryAllBusy] = useState(false);
  const [retryErr, setRetryErr] = useState<string | null>(null);

  const steps = summary.steps_with_webhook;
  const total = summary.total_deliveries;
  const failed = summary.failed_deliveries;
  const succeeded = Math.max(0, total - failed);
  const empty = steps === 0 && total === 0;
  const pendingRegistrations = steps > 0 && total === 0;

  const runRetry = async (executionId: string) => {
    setRetryErr(null);
    setRetrying((r) => ({ ...r, [executionId]: true }));
    try {
      await retryExecutionWebhook(executionId);
      onRefetchDag();
    } catch (e) {
      setRetryErr(e instanceof Error ? e.message : "Retry failed");
    } finally {
      setRetrying((r) => {
        const n = { ...r };
        delete n[executionId];
        return n;
      });
    }
  };

  const runRetryAll = async () => {
    if (failures.length === 0) return;
    setRetryErr(null);
    setRetryAllBusy(true);
    try {
      for (const f of failures) {
        await retryExecutionWebhook(f.execution_id);
      }
      onRefetchDag();
    } catch (e) {
      setRetryErr(e instanceof Error ? e.message : "Retry failed");
    } finally {
      setRetryAllBusy(false);
    }
  };

  return (
    <Card
      className={cn(
        "min-w-0 border-border/80 shadow-none",
        empty && "border-dashed border-border/50 bg-muted/15",
      )}
    >
      <CardContent className={cn("p-3", empty && "py-2.5")}>
        <div className={cn("flex items-center gap-0.5", empty ? "mb-0.5" : "mb-1")}>
          <p className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
            Webhooks
          </p>
          <RunContextHint label="About run-level webhook summary">
            Counts outbound HTTP callbacks registered on steps in this run and delivery
            attempts recorded by the control plane. Failed deliveries listed below can be
            retried here; full attempt history stays in the selected step panel.
          </RunContextHint>
        </div>

        {empty ? (
          <p className="text-micro-plus leading-tight text-muted-foreground">
            No outbound webhooks—register a webhook URL on the reasoner to receive callbacks.
          </p>
        ) : pendingRegistrations ? (
          <p className="text-xs text-foreground">
            {steps} step{steps === 1 ? "" : "s"} registered for callbacks — no delivery
            attempts recorded yet.
          </p>
        ) : (
          <p className="text-xs text-foreground">
            {steps} step{steps === 1 ? "" : "s"} with callbacks · {total} delivery
            {total === 1 ? "" : "ies"}
            {succeeded > 0 ? ` · ${succeeded} succeeded` : ""}
            {failed > 0 ? ` · ${failed} failed` : ""}
          </p>
        )}

        {failures.length > 0 ? (
          <div className="mt-2 space-y-1.5 border-t border-border/60 pt-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
                Failed deliveries
              </p>
              {failures.length > 1 ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 gap-1 px-2 text-micro"
                  disabled={retryAllBusy}
                  onClick={() => void runRetryAll()}
                >
                  {retryAllBusy ? (
                    <RefreshCw className="size-3 animate-spin" />
                  ) : (
                    <RefreshCw className="size-3" />
                  )}
                  Retry all
                </Button>
              ) : null}
            </div>
            <ul className="max-h-40 space-y-1.5 overflow-y-auto pr-0.5">
              {failures.map((f) => {
                const label =
                  f.reasoner_id?.trim() ||
                  f.agent_node_id?.trim() ||
                  f.execution_id.slice(0, 12);
                const busy = Boolean(retrying[f.execution_id]);
                return (
                  <li
                    key={f.execution_id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-muted/40 px-2 py-1.5 text-micro-plus"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium text-foreground" title={label}>
                        {label}
                      </p>
                      <p className="truncate font-mono text-micro text-muted-foreground">
                        {f.event_type}
                        {f.http_status != null ? ` · HTTP ${f.http_status}` : ""}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-micro"
                        onClick={() => onSelectStep(f.execution_id)}
                      >
                        Step
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="h-6 gap-1 px-2 text-micro"
                        disabled={busy}
                        onClick={() => void runRetry(f.execution_id)}
                      >
                        {busy ? (
                          <RefreshCw className="size-3 animate-spin" />
                        ) : (
                          <RefreshCw className="size-3" />
                        )}
                        Retry
                      </Button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}

        {retryErr ? (
          <p className="mt-1.5 text-micro text-destructive">{retryErr}</p>
        ) : null}

        {!empty ? (
          <p
            className={cn(
              "mt-1.5 text-micro leading-snug text-muted-foreground",
              failures.length === 0 && "opacity-80",
            )}
          >
            {failures.length === 0
              ? "Select a step to see each delivery attempt, HTTP status, and retry failed sends."
              : "Use Step to open the execution in the detail panel, or Retry to resend from the control plane."}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: dag, isLoading, isError, error } = useRunDAG(runId);
  const cancelMutation = useCancelExecution();
  const pauseMutation = usePauseExecution();
  const resumeMutation = useResumeExecution();
  const showRunNotification = useRunNotification();
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [lifecycleBusy, setLifecycleBusy] = useState<
    null | "pause" | "resume" | "cancel"
  >(null);

  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"trace" | "graph">("trace");
  const [surfaceTab, setSurfaceTab] = useState<"execution" | "logs">("execution");

  const participants = useMemo(() => {
    if (!dag) {
      return { ids: [] as string[], source: "api_agent" as const };
    }
    return deriveRunParticipants(dag);
  }, [dag]);

  const workflowIdForVc = dag?.root_workflow_id || runId || "";
  const { data: vcChain } = useQuery({
    queryKey: ["workflow-vc-chain", workflowIdForVc],
    queryFn: () => getWorkflowVCChain(workflowIdForVc),
    enabled: Boolean(workflowIdForVc),
    retry: false,
    staleTime: 60_000,
  });

  // Auto-select root step (first in timeline)
  useEffect(() => {
    if (dag?.timeline && dag.timeline.length > 0 && !selectedStepId) {
      const root =
        dag.timeline.find((n) => n.workflow_depth === 0) ?? dag.timeline[0];
      setSelectedStepId(root.execution_id);
    }
  }, [dag, selectedStepId]);

  const traceTree = useMemo(() => {
    if (!dag?.timeline) return null;
    return buildTraceTree(dag.timeline);
  }, [dag]);

  const maxDuration = useMemo(
    () => computeMaxDuration(dag?.timeline ?? []),
    [dag],
  );

  const isSingleStep = (dag?.total_nodes ?? 0) <= 1;
  const shortId = runId ? runId.substring(0, 12) : "—";

  // ─── Loading state ──────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex min-w-0 flex-col gap-4 h-[calc(100vh-8rem)]">
        <div className="flex flex-shrink-0 flex-col gap-2 border-b border-border/50 pb-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2.5">
              <Skeleton className="h-8 w-36 sm:w-48" />
              <Skeleton className="h-9 w-[6rem] rounded-lg" />
              <Skeleton className="h-9 w-[7.25rem] rounded-lg" />
              <Skeleton className="h-8 w-24 rounded-md" />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Skeleton className="h-4 w-64 max-w-full" />
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            <Skeleton className="h-8 w-[4.5rem]" />
            <Skeleton className="h-8 w-24" />
          </div>
        </div>
        <Skeleton className="flex-1 w-full" />
      </div>
    );
  }

  // ─── Error state ────────────────────────────────────────────────────────────
  if (isError) {
    return (
      <div className="flex min-w-0 flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Run {shortId}
        </h1>
        <div className="rounded-md bg-destructive/10 border border-destructive/20 p-4 text-sm text-destructive">
          {error instanceof Error ? error.message : "Failed to load run"}
        </div>
      </div>
    );
  }

  // ─── Empty state ────────────────────────────────────────────────────────────
  if (!dag) {
    return (
      <div className="flex min-w-0 flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Run {shortId}
        </h1>
        <p className="text-sm text-muted-foreground">No data available for this run.</p>
      </div>
    );
  }

  const rootNode = dag.timeline.find((n) => n.workflow_depth === 0) ?? dag.timeline[0];
  const rootExecution: WorkflowExecution = {
    id: 0,
    workflow_id: workflowIdForVc,
    execution_id: rootNode?.execution_id ?? runId ?? "",
    agentfield_request_id: "",
    session_id: dag.session_id ?? undefined,
    actor_id: dag.actor_id ?? undefined,
    agent_node_id: rootNode?.agent_node_id ?? participants.ids[0] ?? "",
    parent_workflow_id: undefined,
    root_workflow_id: dag.root_workflow_id ?? runId ?? undefined,
    workflow_depth: rootNode?.workflow_depth ?? 0,
    reasoner_id: rootNode?.reasoner_id ?? "",
    input_data: null,
    output_data: null,
    input_size: 0,
    output_size: 0,
    workflow_name: dag.workflow_name ?? undefined,
    workflow_tags: [],
    status: normalizeExecutionStatus(dag.workflow_status),
    started_at: rootNode?.started_at ?? dag.timeline[0]?.started_at ?? "",
    completed_at: rootNode?.completed_at ?? undefined,
    duration_ms: rootNode?.duration_ms ?? undefined,
    error_message: undefined,
    retry_count: 0,
    created_at: rootNode?.started_at ?? dag.timeline[0]?.started_at ?? "",
    updated_at: rootNode?.completed_at ?? rootNode?.started_at ?? dag.timeline[0]?.started_at ?? "",
    notes: [],
    webhook_registered: false,
    webhook_events: [],
  };
  const selectedNode =
    dag.timeline.find((node) => node.execution_id === selectedStepId) ?? rootNode;
  const selectedExecution: WorkflowExecution = {
    ...rootExecution,
    execution_id: selectedNode?.execution_id ?? rootExecution.execution_id,
    agent_node_id: selectedNode?.agent_node_id ?? rootExecution.agent_node_id,
    workflow_depth: selectedNode?.workflow_depth ?? rootExecution.workflow_depth,
    reasoner_id: selectedNode?.reasoner_id ?? rootExecution.reasoner_id,
    status: normalizeExecutionStatus(selectedNode?.status ?? dag.workflow_status),
    started_at: selectedNode?.started_at ?? rootExecution.started_at,
    completed_at: selectedNode?.completed_at ?? rootExecution.completed_at,
    duration_ms: selectedNode?.duration_ms ?? rootExecution.duration_ms,
    created_at: selectedNode?.started_at ?? rootExecution.created_at,
    updated_at:
      selectedNode?.completed_at ?? selectedNode?.started_at ?? rootExecution.updated_at,
  };

  const workflowId = dag.root_workflow_id || runId || "";

  const serverWorkflowIssuerDid =
    dag.workflow_issuer_did?.trim() ||
    vcChain?.workflow_vc?.issuer_did?.trim() ||
    "";

  const runTitle =
    dag.workflow_name?.trim() ||
    rootNode?.reasoner_id ||
    "Run";
  const runTitleDisplay = truncateEnd(runTitle, RUN_DETAIL_TITLE_MAX_CHARS);

  const metaParts: string[] = [];
  if (dag.workflow_name?.trim() && rootNode?.reasoner_id) {
    metaParts.push(rootNode.reasoner_id);
  }
  metaParts.push(
    `${dag.total_nodes} ${dag.total_nodes === 1 ? "step" : "steps"}`,
  );
  if (rootNode?.duration_ms != null) {
    metaParts.push(formatDuration(rootNode.duration_ms));
  }
  if (dag.max_depth > 0) {
    metaParts.push(`Depth ${dag.max_depth}`);
  }

  const sessionTrim = dag.session_id?.trim();
  const actorTrim = dag.actor_id?.trim();

  return (
    <div className="flex min-w-0 flex-col h-[calc(100vh-8rem)] max-w-full">
      {/* ─── Header ─────────────────────────────────────────────────────── */}
      <div className="mb-3 flex min-w-0 flex-shrink-0 flex-col gap-2 border-b border-border/50 pb-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-2">
            <h1
              className="min-w-0 text-lg font-semibold leading-snug tracking-tight text-foreground sm:text-xl"
              title={runTitle !== runTitleDisplay ? runTitle : undefined}
            >
              {runTitleDisplay}
            </h1>
            {runId ? (
              <CopyIdentifierChip
                label="Run"
                value={runId}
                tooltip="Copy run ID"
                idTailVisible={6}
              />
            ) : null}
            <CopyIdentifierChip
              label="Identity"
              value={serverWorkflowIssuerDid}
              tooltip="Copy workflow issuer DID"
              noValueMessage="No issuer DID"
              noValueTitle="Verifiable credentials disabled or issuer DID not yet issued"
              idTailVisible={8}
            />
            {(() => {
              const rootNodeForBadge =
                dag.timeline.find((n) => n.workflow_depth === 0) ??
                dag.timeline[0];
              const effective = normalizeExecutionStatus(
                rootNodeForBadge?.status ?? dag.workflow_status,
              );
              return (
                <StatusPill status={effective} size="md" className="shrink-0 shadow-xs" />
              );
            })()}
          </div>

          <div className="flex min-w-0 flex-col gap-1.5 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-3 sm:gap-y-1">
            <p className="m-0 min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
              <span>{metaParts.join(" · ")}</span>
              {sessionTrim ? (
                <>
                  {" · "}
                  <span
                    className="font-mono text-micro-plus text-muted-foreground/90"
                    title={sessionTrim}
                  >
                    Session {truncateEnd(sessionTrim, 28)}
                  </span>
                </>
              ) : null}
              {actorTrim ? (
                <>
                  {" · "}
                  <span
                    className="font-mono text-micro-plus text-muted-foreground/90"
                    title={actorTrim}
                  >
                    Actor {truncateEnd(actorTrim, 24)}
                  </span>
                </>
              ) : null}
            </p>

            {workflowId && workflowId !== runId ? (
              <div className="flex min-w-0 flex-wrap items-center gap-1.5 sm:shrink-0">
                <CopyIdentifierChip
                  label="Flow"
                  value={workflowId}
                  tooltip="Copy workflow ID"
                />
              </div>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 shrink-0 sm:pt-0.5 sm:justify-end">
          {/* Replay */}
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={async () => {
              const agentNodeId = rootNode?.agent_node_id;
              const reasonerId = rootNode?.reasoner_id;
              const execId = rootNode?.execution_id ?? selectedExecution.execution_id;
              const target =
                agentNodeId && reasonerId
                  ? `${agentNodeId}.${reasonerId}`
                  : agentNodeId ?? reasonerId ?? "";
              let replayInput: unknown = null;
              if (execId) {
                try {
                  const details = await getExecutionDetails(execId);
                  replayInput = details.input_data;
                } catch { /* best effort */ }
              }
              navigate(`/playground${target ? `/${target}` : ""}`, {
                state: { replayInput },
              });
            }}
          >
            <RotateCcw className="size-3.5 mr-1" />
            Replay
          </Button>

          {/* Export run provenance (VC chain + audit bundle) */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 px-3 shadow-sm"
                aria-label="Export provenance: verifiable credential chain or audit JSON for this run"
              >
                <BadgeCheck
                  className="size-3.5 shrink-0 text-muted-foreground"
                  aria-hidden
                />
                <span className="text-xs font-medium">Export provenance</span>
                <ChevronDown
                  className="size-3.5 shrink-0 opacity-60"
                  aria-hidden
                />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">
                Provenance for this run
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="flex cursor-pointer flex-col items-start gap-0.5 py-2"
                onClick={() => {
                  void (async () => {
                    try {
                      const data = await getWorkflowVCChain(workflowId);
                      const blob = new Blob([JSON.stringify(data, null, 2)], {
                        type: "application/json",
                      });
                      const url = URL.createObjectURL(blob);
                      window.open(url, "_blank", "noopener,noreferrer");
                      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
                    } catch (e) {
                      console.error(e);
                    }
                  })();
                }}
              >
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Link2 className="size-4 shrink-0" />
                  Preview VC chain
                </span>
                <span className="pl-6 text-xs text-muted-foreground">
                  Authenticated fetch — JSON in a new tab
                </span>
              </DropdownMenuItem>
              <DropdownMenuItem
                className="flex cursor-pointer flex-col items-start gap-0.5 py-2"
                onClick={() => {
                  void downloadWorkflowVCAuditFile(workflowId).catch((e) =>
                    console.error(e),
                  );
                }}
              >
                <span className="flex items-center gap-2 text-sm font-medium">
                  <FileJson className="size-4 shrink-0" />
                  Download VC audit JSON
                </span>
                <span className="pl-6 text-xs text-muted-foreground">
                  Same shape as GET /workflows/…/vc-chain — use with{" "}
                  <code className="text-micro">af verify</code>
                </span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link
                  to="/verify"
                  className="flex cursor-pointer flex-col items-start gap-0.5 py-2"
                >
                  <span className="flex items-center gap-2 text-sm font-medium">
                    <FileCheck2 className="size-4 shrink-0" />
                    Open Audit tool
                  </span>
                  <span className="pl-6 text-xs text-muted-foreground">
                    Upload the file you downloaded for cryptographic checks
                  </span>
                </Link>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Lifecycle cluster — Pause / Resume / Cancel. Uses the ROOT
              execution's own status (not the aggregated workflow status)
              because that's the row the user controls. A run can be
              aggregate-'running' while the root is already 'paused' if
              in-flight children are still finishing. */}
          {(() => {
            const rootNodeForStatus =
              dag.timeline.find((n) => n.workflow_depth === 0) ??
              dag.timeline[0];
            const normalized = normalizeExecutionStatus(
              rootNodeForStatus?.status ?? dag.workflow_status,
            );
            const isRunning = normalized === "running";
            const isPaused = normalized === "paused";
            if (isTerminalStatus(normalized)) return null;

            const rootExecId = rootNodeForStatus?.execution_id;
            if (!rootExecId) return null;

            const busy = lifecycleBusy !== null;

            const runLabelForNotif =
              dag.workflow_name?.trim() ||
              (rootNodeForStatus?.agent_node_id && rootNodeForStatus?.reasoner_id
                ? `${rootNodeForStatus.agent_node_id}.${rootNodeForStatus.reasoner_id}`
                : rootNodeForStatus?.reasoner_id ?? "run");
            const runIdForNotif = runId ?? "";

            const handlePause = async () => {
              setLifecycleBusy("pause");
              try {
                await pauseMutation.mutateAsync(rootExecId);
                showRunNotification({
                  type: "success",
                  eventKind: "pause",
                  title: "Paused",
                  message: `${runLabelForNotif} is now paused. In-flight steps will finish; no new steps will start until you resume.`,
                  runId: runIdForNotif,
                  runLabel: runLabelForNotif,
                });
              } catch (err) {
                showRunNotification({
                  type: "error",
                  eventKind: "error",
                  title: "Pause failed",
                  message:
                    err instanceof Error ? err.message : "Unable to pause run.",
                  runId: runIdForNotif,
                  runLabel: runLabelForNotif,
                });
              } finally {
                setLifecycleBusy(null);
              }
            };

            const handleResume = async () => {
              setLifecycleBusy("resume");
              try {
                await resumeMutation.mutateAsync(rootExecId);
                showRunNotification({
                  type: "success",
                  eventKind: "resume",
                  title: "Resumed",
                  message: `${runLabelForNotif} is running again.`,
                  runId: runIdForNotif,
                  runLabel: runLabelForNotif,
                });
              } catch (err) {
                showRunNotification({
                  type: "error",
                  eventKind: "error",
                  title: "Resume failed",
                  message:
                    err instanceof Error
                      ? err.message
                      : "Unable to resume run.",
                  runId: runIdForNotif,
                  runLabel: runLabelForNotif,
                });
              } finally {
                setLifecycleBusy(null);
              }
            };

            return (
              <>
                {isRunning ? (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 text-xs"
                    disabled={busy}
                    onClick={handlePause}
                  >
                    {lifecycleBusy === "pause" ? (
                      <Activity
                        className="size-3.5 animate-spin"
                        aria-hidden
                      />
                    ) : (
                      <PauseCircle className="size-3.5" aria-hidden />
                    )}
                    Pause
                  </Button>
                ) : null}
                {isPaused ? (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 text-xs"
                    disabled={busy}
                    onClick={handleResume}
                  >
                    {lifecycleBusy === "resume" ? (
                      <Activity
                        className="size-3.5 animate-spin"
                        aria-hidden
                      />
                    ) : (
                      <Play className="size-3.5" aria-hidden />
                    )}
                    Resume
                  </Button>
                ) : null}
                <Button
                  variant="destructive"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                  disabled={busy}
                  onClick={() => setCancelDialogOpen(true)}
                >
                  {lifecycleBusy === "cancel" ? (
                    <Activity className="size-3.5 animate-spin" aria-hidden />
                  ) : (
                    <XCircle className="size-3.5" aria-hidden />
                  )}
                  Cancel
                </Button>

                <AlertDialog
                  open={cancelDialogOpen}
                  onOpenChange={setCancelDialogOpen}
                >
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>
                        {CANCEL_RUN_COPY.title(1)}
                      </AlertDialogTitle>
                      <AlertDialogDescription>
                        {CANCEL_RUN_COPY.description}
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel disabled={busy}>
                        {CANCEL_RUN_COPY.keepLabel}
                      </AlertDialogCancel>
                      <AlertDialogAction
                        disabled={busy}
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        onClick={async () => {
                          setCancelDialogOpen(false);
                          setLifecycleBusy("cancel");
                          try {
                            await cancelMutation.mutateAsync(rootExecId);
                            showRunNotification({
                              type: "success",
                              eventKind: "cancel",
                              title: "Cancelled",
                              message: `${runLabelForNotif} will stop after its current step finishes. In-flight work will be discarded.`,
                              runId: runIdForNotif,
                              runLabel: runLabelForNotif,
                            });
                          } catch (err) {
                            showRunNotification({
                              type: "error",
                              eventKind: "error",
                              title: "Cancel failed",
                              message:
                                err instanceof Error
                                  ? err.message
                                  : "Unable to cancel run.",
                              runId: runIdForNotif,
                              runLabel: runLabelForNotif,
                            });
                          } finally {
                            setLifecycleBusy(null);
                          }
                        }}
                      >
                        {CANCEL_RUN_COPY.confirmLabel(1)}
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </>
            );
          })()}

        </div>
      </div>

      {/* Cancellation / pause registered strip — shown when the root
          execution is cancelled or paused by the user but at least one
          child is still reporting 'running'. This is the honest depiction
          of the backend semantics: the control plane flipped the root's
          status immediately but in-flight HTTP calls to agent workers
          cannot be killed mid-dispatch and will finish naturally. */}
      {(() => {
        const rootNodeForStrip =
          dag.timeline.find((n) => n.workflow_depth === 0) ??
          dag.timeline[0];
        const rootStatus = normalizeExecutionStatus(rootNodeForStrip?.status);
        if (rootStatus !== "cancelled" && rootStatus !== "paused") return null;
        const stillRunning = dag.timeline.filter(
          (n) => normalizeExecutionStatus(n.status) === "running",
        ).length;
        if (stillRunning === 0) return null;
        const verb = rootStatus === "cancelled" ? "Cancellation" : "Pause";
        return (
          <div
            role="status"
            className="mb-3 flex items-start gap-2.5 rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
          >
            <Info
              className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
              aria-hidden
            />
            <p className="leading-snug">
              <span className="font-medium text-foreground">
                {verb} registered
              </span>{" "}
              — {stillRunning} node{stillRunning === 1 ? "" : "s"} still
              finishing the current step. No new nodes will start
              {rootStatus === "cancelled"
                ? "; their output will be discarded."
                : " until you resume."}
            </p>
          </div>
        );
      })()}

      {/* Nodes + webhooks — always show run-level strip (empty states explicit) */}
      <TooltipProvider delayDuration={280}>
        <div className="mb-3 grid min-w-0 gap-3 sm:grid-cols-2">
          <RunContextNodesCard
            participantIds={participants.ids}
            source={participants.source}
          />
          <RunContextWebhooksCard
            summary={dag.webhook_summary ?? ZERO_WEBHOOK_SUMMARY}
            failures={dag.webhook_failures ?? []}
            onSelectStep={setSelectedStepId}
            onRefetchDag={() => {
              void queryClient.invalidateQueries({ queryKey: ["run-dag", runId] });
              void queryClient.invalidateQueries({ queryKey: ["step-detail"] });
            }}
          />
        </div>
      </TooltipProvider>

      <Tabs
        value={surfaceTab}
        onValueChange={(value) => setSurfaceTab(value as "execution" | "logs")}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="mb-3 flex min-w-0 items-center justify-between gap-3 border-b border-border/50 pb-3">
          <TabsList className="h-9" aria-label="Run detail surface">
            <TabsTrigger value="execution" className="px-4 text-sm">
              Execution
            </TabsTrigger>
            <TabsTrigger value="logs" className="px-4 text-sm">
              Logs
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="logs" className="mt-0 min-h-0 flex-1 data-[state=inactive]:hidden">
          {selectedExecution.execution_id ? (
            <ExecutionObservabilityPanel
              execution={selectedExecution}
              relatedNodeIds={participants.ids}
            />
          ) : (
            <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              <CardContent className="flex min-h-0 min-w-0 flex-1 items-center justify-center p-6 text-sm text-muted-foreground">
                Execution logs are unavailable for this run.
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent
          value="execution"
          className="mt-0 min-h-0 flex-1 data-[state=inactive]:hidden"
        >
          {isSingleStep ? (
            <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col p-0">
                {selectedStepId ? (
                  <StepDetail executionId={selectedStepId} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No step selected
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row lg:items-stretch">
              <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:min-w-0 lg:basis-0">
                <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
                  <span className="text-xs font-medium text-muted-foreground">
                    Steps
                  </span>
                  <Tabs
                    value={viewMode}
                    onValueChange={(v) => setViewMode(v as "trace" | "graph")}
                  >
                    <TabsList className="h-8" aria-label="Trace or graph view">
                      <TabsTrigger value="trace" className="h-7 px-3 text-xs">
                        Trace
                      </TabsTrigger>
                      <TabsTrigger value="graph" className="h-7 px-3 text-xs">
                        Graph
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
                <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col p-0">
                  {viewMode === "graph" ? (
                    <div
                      className="flex h-full min-h-[min(45vh,22rem)] min-w-0 flex-1 flex-col"
                      style={{
                        minHeight: "max(280px, min(45vh, 22rem))",
                        width: "100%",
                        flex: "1 1 0%",
                      }}
                    >
                      <ErrorBoundary>
                        <WorkflowDAGViewer
                          key={runId}
                          className="h-full min-h-0 flex-1"
                          workflowId={dag.root_workflow_id || runId || ""}
                          dagData={dag}
                          selectedNodeIds={selectedStepId ? [selectedStepId] : undefined}
                          onExecutionClick={(execution) =>
                            setSelectedStepId(execution.execution_id)
                          }
                        />
                      </ErrorBoundary>
                    </div>
                  ) : (
                    <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
                      {traceTree ? (
                        <RunTrace
                          node={traceTree}
                          maxDuration={maxDuration}
                          selectedId={selectedStepId}
                          onSelect={setSelectedStepId}
                          rootStatus={
                            dag.timeline.find((n) => n.workflow_depth === 0)
                              ?.status ?? dag.workflow_status
                          }
                          runStartedAt={
                            dag.timeline.find((n) => n.workflow_depth === 0)
                              ?.started_at ??
                            dag.timeline[0]?.started_at ??
                            ""
                          }
                        />
                      ) : (
                        <p className="p-4 text-xs text-muted-foreground">
                          No steps to display
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:min-w-0 lg:basis-0">
                <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col p-0">
                  {selectedStepId ? (
                    <StepDetail executionId={selectedStepId} />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                      Select a step to view details
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
