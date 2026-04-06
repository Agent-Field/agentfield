import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useStepDetail } from "@/hooks/queries";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ChevronDown } from "@/components/ui/icon-bridge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Check, ShieldAlert, RefreshCw, Terminal, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { retryExecutionWebhook } from "@/services/executionsApi";
import { formatDuration } from "./RunTrace";
import { JsonHighlightedPre } from "@/components/ui/json-syntax-highlight";
import { StepProvenanceCard } from "@/components/StepProvenanceCard";

// ─── cURL snippet: copy + minimal info (hover) ────────────────────────────────

function CopyCurlSnippet({
  label,
  hint,
  getText,
}: {
  label: string;
  hint: string;
  getText: () => string;
}) {
  const [copied, setCopied] = useState(false);

  const handleClick = () => {
    const text = getText();
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="inline-flex items-center gap-0">
      <Button
        variant="ghost"
        size="sm"
        className="h-6 gap-1 px-1.5 pr-1 text-micro text-muted-foreground"
        onClick={handleClick}
        title={`Copy cURL: ${label}`}
      >
        {copied ? (
          <Check className="size-2.5 shrink-0" />
        ) : (
          <Copy className="size-2.5 shrink-0" />
        )}
        <span className="max-w-[9rem] truncate sm:max-w-none">{label}</span>
      </Button>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="inline-flex size-5 shrink-0 items-center justify-center rounded-sm text-muted-foreground/50 transition-colors hover:bg-muted hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-label={`What this copies: ${label}`}
          >
            <Info className="size-3" strokeWidth={2.25} />
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          className="max-w-[min(20rem,calc(100vw-2rem))] border border-border bg-popover px-2.5 py-2 text-left text-micro-plus leading-snug text-popover-foreground shadow-md"
        >
          {hint}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

/** Workflow ID label + clickable value (copies raw id). */
function CopyableWorkflowId({ workflowId }: { workflowId: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <p className="mt-1 flex flex-wrap items-center gap-1.5 text-micro text-muted-foreground/80">
      <span className="shrink-0 font-sans">Workflow ID:</span>
      <button
        type="button"
        className={cn(
          "max-w-full break-all rounded-sm px-1 py-0.5 text-left font-mono",
          "text-foreground/90 transition-colors",
          "hover:bg-muted hover:text-foreground",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        )}
        title="Copy workflow ID"
        onClick={() => {
          void navigator.clipboard.writeText(workflowId).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {workflowId}
      </button>
      {copied ? (
        <Check className="size-3 shrink-0 text-green-600 dark:text-green-500" aria-hidden />
      ) : (
        <Copy
          className="size-3 shrink-0 opacity-40 pointer-events-none"
          aria-hidden
        />
      )}
    </p>
  );
}

/** Icon-only copy for JSON blocks (sits on Input / Output headers). */
function CopyJsonHeaderButton({
  data,
  ariaLabel,
  disabled,
}: {
  data: unknown;
  ariaLabel: string;
  disabled?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
      disabled={disabled}
      aria-label={ariaLabel}
      title={ariaLabel}
      onClick={(e) => {
        e.preventDefault();
        const text =
          data == null ? "" : JSON.stringify(data, null, 2);
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? (
        <Check className="size-3.5" />
      ) : (
        <Copy className="size-3.5" />
      )}
    </Button>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function StepDetail({ executionId }: { executionId: string }) {
  const { data: execution, isLoading } = useStepDetail(executionId);
  const queryClient = useQueryClient();
  const [approvalBusy, setApprovalBusy] = useState(false);

  const handleApproval = useCallback(
    async (decision: "approved" | "rejected") => {
      if (!execution?.approval_request_id || approvalBusy) return;
      setApprovalBusy(true);
      try {
        const res = await fetch("/api/v1/webhooks/approval-response", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": localStorage.getItem("agentfield_api_key") ?? "",
          },
          body: JSON.stringify({
            requestId: execution.approval_request_id,
            decision,
          }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          console.error("Approval failed:", body);
        }
        void queryClient.invalidateQueries({ queryKey: ["step-detail", executionId] });
        void queryClient.invalidateQueries({ queryKey: ["run-dag"] });
        void queryClient.invalidateQueries({ queryKey: ["executions"] });
      } finally {
        setApprovalBusy(false);
      }
    },
    [execution?.approval_request_id, executionId, queryClient, approvalBusy],
  );

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3 p-4">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-3 w-60" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground p-8">
        Step not found
      </div>
    );
  }

  const hasError = Boolean(execution.error_message);
  const hasOutput = execution.output_data != null;
  const hasInput = execution.input_data != null;
  const notes = execution.notes ?? [];

  const apiUiBase =
    (import.meta.env.VITE_API_BASE_URL as string | undefined) || "/api/ui/v1";
  const curlApiRoot =
    apiUiBase.startsWith("http") ? apiUiBase : `${window.location.origin}${apiUiBase}`;
  const workflowId = execution.workflow_id;

  const buildCurl = () => {
    const origin = window.location.origin;
    return (
      `curl -X POST '${origin}/api/v1/execute/${execution.agent_node_id}.${execution.reasoner_id}' \\\n` +
      `  -H 'Content-Type: application/json' \\\n` +
      `  -H 'X-API-Key: YOUR_API_KEY' \\\n` +
      `  -d '${JSON.stringify({ input: execution.input_data })}'`
    );
  };

  const buildCurlWorkflowDag = () =>
    [
      `curl -sS '${curlApiRoot}/workflows/${workflowId}/dag?mode=lightweight' \\`,
      `  -H 'X-API-Key: YOUR_API_KEY'`,
    ].join("\n");

  const buildCurlWorkflowVCAudit = () =>
    [
      `# JSON matches GET vc-chain — use as: af verify audit.json`,
      `curl -sS '${curlApiRoot}/workflows/${workflowId}/vc-chain' \\`,
      `  -H 'X-API-Key: YOUR_API_KEY'`,
    ].join("\n");

  const buildCurlStepDetails = () =>
    [
      `curl -sS '${curlApiRoot}/executions/${execution.execution_id}/details' \\`,
      `  -H 'X-API-Key: YOUR_API_KEY'`,
    ].join("\n");

  const curlSnippets: {
    id: string;
    label: string;
    hint: string;
    getText: () => string;
  }[] = [
    {
      id: "execute",
      label: "Execute",
      hint:
        "POST /api/v1/execute/{agent}.{reasoner} with the Input JSON from this step. Calls the public execute API and starts a new run (not a replay of this exact execution id).",
      getText: buildCurl,
    },
    {
      id: "dag",
      label: "Workflow DAG",
      hint:
        "GET /api/ui/v1/workflows/{workflowId}/dag?mode=lightweight — graph + timeline metadata for the whole workflow (agents, steps order, lightweight structure).",
      getText: buildCurlWorkflowDag,
    },
    {
      id: "vc",
      label: "VC audit",
      hint:
        "GET /api/ui/v1/workflows/{workflowId}/vc-chain — verifiable credential chain for the workflow. Response matches what af verify expects if you save it as a JSON file.",
      getText: buildCurlWorkflowVCAudit,
    },
    {
      id: "details",
      label: "Execution record",
      hint:
        "GET /api/ui/v1/executions/{this execution id}/details — full JSON for this single step: status, input/output, errors, notes, webhooks; same payload the sidebar loads.",
      getText: buildCurlStepDetails,
    },
  ];

  return (
    <ScrollArea className="h-full min-w-0 max-w-full">
      <div className="flex min-w-0 max-w-full flex-col gap-4 p-4">
        {/* Step header */}
        <div>
          <h3 className="text-sm font-semibold font-mono">
            {execution.reasoner_id}
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Agent: {execution.agent_node_id}
            {" · "}
            Duration: {formatDuration(execution.duration_ms)}
            {execution.workflow_depth != null && (
              <> · Depth: {execution.workflow_depth}</>
            )}
          </p>
          <CopyableWorkflowId workflowId={workflowId} />

          {/* cURL: compact dropdown on small screens; inline from md+ */}
          <TooltipProvider delayDuration={280}>
            <div className="mt-2 flex flex-wrap items-center gap-x-0.5 gap-y-1">
              <div className="hidden md:flex md:flex-wrap md:items-center md:gap-x-0 md:gap-y-1">
                {curlSnippets.map((s) => (
                  <CopyCurlSnippet
                    key={s.id}
                    label={s.label}
                    hint={s.hint}
                    getText={s.getText}
                  />
                ))}
              </div>
              <div className="md:hidden">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 gap-1.5 px-2 text-micro font-medium"
                    >
                      <Terminal className="size-3 shrink-0" />
                      Copy cURL
                      <ChevronDown className="size-3 shrink-0 opacity-60" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="start"
                    className="w-[min(100vw-1.5rem,18rem)]"
                  >
                    <DropdownMenuLabel className="text-micro font-normal leading-snug text-muted-foreground">
                      Each item copies a ready-to-run curl. Descriptions below.
                    </DropdownMenuLabel>
                    {curlSnippets.map((s) => (
                      <DropdownMenuItem
                        key={s.id}
                        className="flex cursor-pointer flex-col items-stretch gap-1 py-2.5"
                        onSelect={() =>
                          void navigator.clipboard.writeText(s.getText())
                        }
                      >
                        <span className="flex items-center gap-2 text-xs font-medium">
                          <Copy className="size-3.5 shrink-0 opacity-70" />
                          {s.label}
                        </span>
                        <span className="pl-[1.375rem] text-micro leading-snug text-muted-foreground">
                          {s.hint}
                        </span>
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </TooltipProvider>
        </div>

        <StepProvenanceCard
          callerDid={execution.caller_did}
          targetDid={execution.target_did}
          inputHash={execution.input_hash}
          outputHash={execution.output_hash}
        />

        {/* Error above input so failures are visible before scrolling past payload */}
        {hasError ? (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3">
            <p className="text-xs font-medium text-destructive">Error</p>
            <p className="text-xs mt-1 font-mono whitespace-pre-wrap break-all">
              {execution.error_message}
            </p>
          </div>
        ) : null}

        {/* Input section */}
        {hasInput && (
          <Collapsible defaultOpen>
            <div className="flex min-w-0 w-full items-center gap-0.5">
              <CollapsibleTrigger className="flex min-w-0 flex-1 items-center gap-1 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
                <ChevronDown className="size-3 shrink-0 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
                Input
              </CollapsibleTrigger>
              <CopyJsonHeaderButton
                data={execution.input_data}
                ariaLabel="Copy input JSON"
              />
            </div>
            <CollapsibleContent>
              <div className="mt-2 min-w-0 max-w-full rounded-md bg-muted p-3 overflow-x-auto overflow-y-auto max-h-64">
                <JsonHighlightedPre
                  data={execution.input_data}
                  className="text-xs font-mono leading-relaxed"
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Output (omit when error present — same as before) */}
        {!hasError && hasOutput ? (
          <Collapsible defaultOpen>
            <div className="flex min-w-0 w-full items-center gap-0.5">
              <CollapsibleTrigger className="flex min-w-0 flex-1 items-center gap-1 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
                <ChevronDown className="size-3 shrink-0 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
                Output
              </CollapsibleTrigger>
              <CopyJsonHeaderButton
                data={execution.output_data}
                ariaLabel="Copy output JSON"
              />
            </div>
            <CollapsibleContent>
              <div className="mt-2 min-w-0 max-w-full rounded-md bg-muted p-3 overflow-x-auto overflow-y-auto max-h-64">
                <JsonHighlightedPre
                  data={execution.output_data}
                  className="text-xs font-mono leading-relaxed"
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        ) : !hasError ? (
          <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
            No output
          </div>
        ) : null}

        {/* Notes */}
        {notes.length > 0 && (
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors w-full text-left">
              <ChevronDown className="size-3 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
              Notes ({notes.length})
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-2 flex flex-col gap-2">
                {notes.map((note, i) => (
                  <div
                    key={i}
                    className="rounded-md bg-muted p-2 text-xs"
                  >
                    <span className="text-muted-foreground">
                      {new Date(note.timestamp).toLocaleTimeString()}
                    </span>{" "}
                    {note.message}
                    {note.tags?.map((tag) => (
                      <Badge
                        key={tag}
                        variant="outline"
                        className="ml-1 text-micro py-0 h-4"
                      >
                        {tag}
                      </Badge>
                    ))}
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Webhook Delivery */}
        {(execution.webhook_registered || (execution.webhook_events && execution.webhook_events.length > 0)) && (
          <Collapsible defaultOpen={false}>
            <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors w-full text-left">
              <ChevronDown className="size-3 transition-transform [[data-state=open]_&]:rotate-0 [[data-state=closed]_&]:-rotate-90" />
              Webhooks ({execution.webhook_events?.length ?? 0})
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-2 flex flex-col gap-1">
                {execution.webhook_events && execution.webhook_events.length > 0 ? (
                  execution.webhook_events.map((event, i) => (
                    <div
                      key={event.id ?? i}
                      className="flex items-center justify-between rounded-md bg-muted px-2 py-1.5 text-micro-plus"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={cn(
                            "size-1.5 rounded-full shrink-0",
                            event.status === "delivered"
                              ? "bg-green-500"
                              : event.status === "failed"
                                ? "bg-red-500"
                                : "bg-amber-500 animate-pulse",
                          )}
                        />
                        <span className="font-mono truncate max-w-[120px]">
                          {event.event_type}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-muted-foreground shrink-0">
                        {event.http_status != null && (
                          <span
                            className={cn(
                              event.http_status >= 200 && event.http_status < 300
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-500",
                            )}
                          >
                            HTTP {event.http_status}
                          </span>
                        )}
                        {!event.http_status && (
                          <span className="capitalize">{event.status}</span>
                        )}
                        <span>{new Date(event.created_at).toLocaleTimeString()}</span>
                        {event.status === "failed" && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-5 px-1.5 text-micro gap-1"
                            onClick={() =>
                              retryExecutionWebhook(execution.execution_id).catch(
                                console.error,
                              )
                            }
                          >
                            <RefreshCw className="size-2.5" />
                            Retry
                          </Button>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-micro-plus text-muted-foreground px-1">
                    Webhook registered but no delivery attempts yet.
                  </p>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* HITL Approval Section */}
        {(execution.status === "waiting" || execution.approval_request_id) && (
          <Card className="border-amber-500/30 bg-amber-500/5">
            <CardHeader className="py-2 px-3">
              <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                <ShieldAlert className="size-3.5 text-amber-500" />
                Human Approval Required
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3 flex flex-col gap-2">
              {execution.approval_status && (
                <p className="text-micro-plus text-muted-foreground">
                  Status:{" "}
                  <Badge variant="outline" className="text-micro ml-1">
                    {execution.approval_status}
                  </Badge>
                </p>
              )}
              {execution.approval_requested_at && (
                <p className="text-micro-plus text-muted-foreground">
                  Requested:{" "}
                  {new Date(execution.approval_requested_at).toLocaleString()}
                </p>
              )}
              {execution.approval_request_id &&
                execution.approval_status === "pending" && (
                  <div className="flex gap-2 mt-1">
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      disabled={approvalBusy}
                      onClick={() => handleApproval("approved")}
                    >
                      {approvalBusy ? "Sending…" : "Approve"}
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="h-7 text-xs"
                      disabled={approvalBusy}
                      onClick={() => handleApproval("rejected")}
                    >
                      {approvalBusy ? "Sending…" : "Reject"}
                    </Button>
                  </div>
                )}
            </CardContent>
          </Card>
        )}
      </div>
    </ScrollArea>
  );
}
