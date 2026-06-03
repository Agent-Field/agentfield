import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { ReasonerNodeCombobox } from "../components/ui/reasoner-node-combobox";
import { Separator } from "../components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  Play,
  InProgress,
  ArrowRight,
  Upload,
  Copy,
  Check,
  ChevronRight,
  ChevronDown,
} from "../components/ui/icon-bridge";
import { reasonersApi } from "../services/reasonersApi";
import type { ReasonerWithNode, ReasonersResponse } from "../types/reasoners";
import { normalizeExecutionStatus } from "../utils/status";
import { JsonHighlightedPre } from "../components/ui/json-syntax-highlight";

interface RecentRun {
  id: string;
  duration_ms?: number;
  status: string;
  input_preview: string;
  output_preview: string;
  created_at: string;
  input_data?: unknown;
}

function formatDuration(ms?: number): string {
  if (ms === undefined || ms === null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function extractPreview(data: unknown, maxLen = 40): string {
  if (data === null || data === undefined) return "—";
  try {
    const str =
      typeof data === "string" ? data : JSON.stringify(data);
    return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
  } catch {
    return "—";
  }
}

function statusVariant(
  status: string
): "default" | "secondary" | "destructive" | "outline" {
  const s = normalizeExecutionStatus(status);
  if (s === "succeeded") return "default";
  if (s === "failed") return "destructive";
  return "secondary";
}

function buildCurlCommand(
  target: string,
  input: string,
  baseUrl: string = window.location.origin
): string {
  const escapedInput = input.replace(/'/g, "'\\''");
  return `curl -X POST '${baseUrl}/api/v1/execute/${target}' \\
  -H 'Content-Type: application/json' \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -d '{"input": ${escapedInput}}'`;
}

function buildAsyncCurlCommand(
  target: string,
  input: string,
  baseUrl: string = window.location.origin
): string {
  const escapedInput = input.replace(/'/g, "'\\''");
  return `curl -X POST '${baseUrl}/api/v1/execute/async/${target}' \\
  -H 'Content-Type: application/json' \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -d '{"input": ${escapedInput}}'`;
}

export function PlaygroundPage() {
  const { reasonerId: paramReasonerId } = useParams<{ reasonerId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const replayInput = (location.state as { replayInput?: unknown } | null)?.replayInput;
  const hasReplayInput = replayInput != null && replayInput !== "" && JSON.stringify(replayInput) !== "{}" && JSON.stringify(replayInput) !== "null";

  // ── reasoner list ─────────────────────────────────────────────────────────
  const [reasonersData, setReasoners] = useState<ReasonersResponse | null>(null);
  const [loadingReasoners, setLoadingReasoners] = useState(true);

  // ── selected reasoner ─────────────────────────────────────────────────────
  const [selectedId, setSelectedId] = useState<string | null>(
    paramReasonerId ?? null
  );
  const [selectedReasoner, setSelectedReasoner] =
    useState<ReasonerWithNode | null>(null);
  const [loadingReasonerDetails, setLoadingReasonerDetails] = useState(false);

  // ── playground state ──────────────────────────────────────────────────────
  const [input, setInput] = useState(() => {
    if (replayInput != null) {
      try {
        return typeof replayInput === "string"
          ? replayInput
          : JSON.stringify(replayInput, null, 2);
      } catch {
        return "{}";
      }
    }
    return "{}";
  });
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<unknown>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const [resultStatus, setResultStatus] = useState<string | null>(null);
  const [resultDuration, setResultDuration] = useState<number | undefined>(undefined);
  const [executing, setExecuting] = useState(false);
  const [lastRunId, setLastRunId] = useState<string | null>(null);

  // ── copy feedback ─────────────────────────────────────────────────────────
  const [copiedSync, setCopiedSync] = useState(false);
  const [copiedAsync, setCopiedAsync] = useState(false);

  // ── schema collapsible ────────────────────────────────────────────────────
  const [schemaOpen, setSchemaOpen] = useState(false);

  // ── recent runs ───────────────────────────────────────────────────────────
  const [recentRuns, setRecentRuns] = useState<RecentRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);

  // ── load reasoners ────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoadingReasoners(true);
    reasonersApi
      .getAllReasoners({ status: "all", limit: 200 })
      .then((data) => {
        if (!cancelled) setReasoners(data);
      })
      .catch(console.error)
      .finally(() => {
        if (!cancelled) setLoadingReasoners(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ── load selected reasoner details ───────────────────────────────────────
  useEffect(() => {
    if (!selectedId) {
      setSelectedReasoner(null);
      return;
    }
    let cancelled = false;
    setLoadingReasonerDetails(true);
    reasonersApi
      .getReasonerDetails(selectedId)
      .then((data) => {
        if (!cancelled) {
          setSelectedReasoner(data);
          // Seed input textarea with schema example, unless replay data was provided
          if (!hasReplayInput) {
            if (data.input_schema?.properties) {
              const example: Record<string, string> = {};
              for (const key of Object.keys(data.input_schema.properties)) {
                example[key] = "";
              }
              setInput(JSON.stringify(example, null, 2));
            } else {
              setInput("{}");
            }
          }
          setResult(null);
          setResultError(null);
          setResultStatus(null);
          setResultDuration(undefined);
          setLastRunId(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("Failed to load reasoner details:", err);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingReasonerDetails(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // ── load recent runs ──────────────────────────────────────────────────────
  const loadRecentRuns = useCallback(async (reasonerId: string) => {
    setLoadingRuns(true);
    try {
      const history = await reasonersApi.getExecutionHistory(reasonerId, 1, 5);
      const runs: RecentRun[] = (history.executions ?? []).map((ex: any) => ({
        id: ex.execution_id ?? ex.id ?? "—",
        duration_ms: ex.duration_ms,
        status: ex.status ?? "unknown",
        input_preview: extractPreview(ex.input_data ?? ex.input),
        output_preview: extractPreview(ex.output_data ?? ex.result ?? ex.output),
        created_at: ex.started_at ?? ex.created_at ?? "",
        input_data: ex.input_data ?? ex.input,
      }));
      setRecentRuns(runs);
    } catch (err) {
      console.error("Failed to load recent runs:", err);
      setRecentRuns([]);
    } finally {
      setLoadingRuns(false);
    }
  }, []);

  useEffect(() => {
    if (selectedId) {
      loadRecentRuns(selectedId);
    } else {
      setRecentRuns([]);
    }
  }, [selectedId, loadRecentRuns]);

  // ── execute ───────────────────────────────────────────────────────────────
  async function handleExecute() {
    if (!selectedId) return;

    // Validate JSON
    let parsed: unknown;
    try {
      parsed = JSON.parse(input);
    } catch {
      setInputError("Invalid JSON — please fix before executing.");
      return;
    }
    setInputError(null);
    setExecuting(true);
    setResult(null);
    setResultError(null);
    setResultStatus(null);
    setResultDuration(undefined);
    setLastRunId(null);

    try {
      const data = await reasonersApi.executeReasoner(selectedId, {
        input: parsed as Record<string, unknown>,
      });
      setResult(data.result ?? data);
      const runId =
        (data as any).execution_id ??
        (data as any).run_id ??
        null;
      setLastRunId(runId);
      setResultStatus((data as any).status ?? "succeeded");
      setResultDuration((data as any).duration_ms);
      // Refresh recent runs after successful execution
      loadRecentRuns(selectedId);
    } catch (err) {
      setResultError(
        err instanceof Error ? err.message : "Execution failed."
      );
      setResultStatus("failed");
    } finally {
      setExecuting(false);
    }
  }

  // ── cURL copy helpers ─────────────────────────────────────────────────────
  function handleCopySync() {
    if (!selectedId) return;
    const cmd = buildCurlCommand(selectedId, input);
    navigator.clipboard.writeText(cmd);
    setCopiedSync(true);
    setTimeout(() => setCopiedSync(false), 2000);
  }

  function handleCopyAsync() {
    if (!selectedId) return;
    const cmd = buildAsyncCurlCommand(selectedId, input);
    navigator.clipboard.writeText(cmd);
    setCopiedAsync(true);
    setTimeout(() => setCopiedAsync(false), 2000);
  }

  // ── route sync ────────────────────────────────────────────────────────────
  function handleReasonerChange(value: string) {
    setSelectedId(value);
    navigate(`/playground/${encodeURIComponent(value)}`, { replace: true });
  }

  // ── load input from recent run ────────────────────────────────────────────
  function handleLoadInput(run: RecentRun) {
    if (run.input_data !== undefined && run.input_data !== null) {
      try {
        setInput(JSON.stringify(run.input_data, null, 2));
        setInputError(null);
      } catch {
        // ignore
      }
    }
  }

  const inputPlaceholder = useMemo(() => {
    if (!selectedReasoner?.input_schema?.properties) return '{\n  \n}';
    const keys = Object.keys(selectedReasoner.input_schema.properties);
    const example: Record<string, string> = {};
    keys.forEach((k) => (example[k] = ""));
    return JSON.stringify(example, null, 2);
  }, [selectedReasoner]);

  const normalizedResultStatus = resultStatus
    ? normalizeExecutionStatus(resultStatus)
    : null;

  return (
    <div className="space-y-6">
      {/* ── Page heading ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Playground</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Pick an agent node (service), then a reasoner (thinking endpoint) or
            skill (deterministic API), provide input, execute, and inspect the result.
          </p>
        </div>
      </div>

      {/* ── Agent node · skill selector ─────────────────────────────────── */}
      <div className="flex min-w-0 flex-wrap items-center gap-3">
        <span className="text-sm font-medium text-muted-foreground whitespace-nowrap">
          Reasoner / skill:
        </span>
        <div className="min-w-0 w-full max-w-md sm:w-[min(100%,24rem)] sm:flex-1">
          <ReasonerNodeCombobox
            reasoners={reasonersData?.reasoners ?? []}
            value={selectedId}
            onValueChange={handleReasonerChange}
            disabled={
              loadingReasoners ||
              (reasonersData?.reasoners?.length ?? 0) === 0
            }
            loading={loadingReasoners}
            className="w-full"
            placeholder={
              !loadingReasoners && (reasonersData?.reasoners?.length ?? 0) === 0
                ? "No skills available"
                : "Select agent node · skill"
            }
          />
        </div>

        {selectedReasoner && (
          <Badge variant="secondary" className="text-xs font-mono">
            {selectedId}
          </Badge>
        )}

        {loadingReasonerDetails && (
          <InProgress className="h-4 w-4 animate-spin text-muted-foreground" />
        )}
      </div>

      {/* ── Input / Result split ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* INPUT */}
        <Card className="flex flex-col">
          <CardHeader className="pb-3">
            <label
              htmlFor="playground-input"
              className="text-sm font-semibold uppercase tracking-wider text-muted-foreground leading-none"
            >
              Input
            </label>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 flex-1">
            <textarea
              id="playground-input"
              autoFocus
              className="flex-1 min-h-[200px] w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-sm resize-vertical focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 placeholder:text-muted-foreground"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (inputError) setInputError(null);
              }}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  if (!executing && selectedId) handleExecute();
                }
              }}
              placeholder={inputPlaceholder}
              spellCheck={false}
              aria-describedby={inputError ? "playground-input-error" : undefined}
              aria-invalid={inputError ? true : undefined}
            />
            {inputError && (
              <p id="playground-input-error" className="text-xs text-destructive">
                {inputError}
              </p>
            )}

            {/* ── Schema collapsible ──────────────────────────────────── */}
            {selectedReasoner && (
              <div className="space-y-2">
                <Collapsible open={schemaOpen} onOpenChange={setSchemaOpen}>
                  <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
                    {schemaOpen ? (
                      <ChevronDown className="size-3" />
                    ) : (
                      <ChevronRight className="size-3" />
                    )}
                    Schema
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="mt-2 space-y-2">
                      {selectedReasoner.input_schema && (
                        <div>
                          <p className="text-micro font-medium text-muted-foreground uppercase tracking-wider mb-1">
                            Input Schema
                          </p>
                          <JsonHighlightedPre
                            data={selectedReasoner.input_schema}
                            className="rounded-md bg-muted p-2 text-micro max-h-32 overflow-auto"
                          />
                        </div>
                      )}
                      {selectedReasoner.output_schema && (
                        <div>
                          <p className="text-micro font-medium text-muted-foreground uppercase tracking-wider mb-1">
                            Output Schema
                          </p>
                          <JsonHighlightedPre
                            data={selectedReasoner.output_schema}
                            className="rounded-md bg-muted p-2 text-micro max-h-32 overflow-auto"
                          />
                        </div>
                      )}
                      {!selectedReasoner.input_schema && !selectedReasoner.output_schema && (
                        <p className="text-xs text-muted-foreground">No schema defined.</p>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}

            {/* ── Execute + cURL buttons ──────────────────────────────── */}
            <div className="flex gap-2">
              <Button
                onClick={handleExecute}
                disabled={executing || !selectedId}
                className="flex-1"
              >
                {executing ? (
                  <>
                    <InProgress className="h-4 w-4 mr-2 animate-spin" />
                    Executing…
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute
                  </>
                )}
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="default"
                    disabled={!selectedId}
                    className="gap-1.5"
                  >
                    {copiedSync || copiedAsync ? (
                      <Check className="size-3.5" />
                    ) : (
                      <Copy className="size-3.5" />
                    )}
                    {copiedSync || copiedAsync ? "Copied!" : "cURL"}
                    <ChevronDown className="size-3 ml-0.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={handleCopySync}>
                    {copiedSync ? (
                      <Check className="size-3.5 mr-2" />
                    ) : (
                      <Copy className="size-3.5 mr-2" />
                    )}
                    {copiedSync ? "Copied!" : "Copy cURL (sync)"}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={handleCopyAsync}>
                    {copiedAsync ? (
                      <Check className="size-3.5 mr-2" />
                    ) : (
                      <Copy className="size-3.5 mr-2" />
                    )}
                    {copiedAsync ? "Copied!" : "Copy cURL (async)"}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </CardContent>
        </Card>

        {/* RESULT */}
        <Card className="flex flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                Result
              </CardTitle>
              {normalizedResultStatus && !executing && (
                <div className="flex items-center gap-2">
                  {resultDuration !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(resultDuration)}
                    </span>
                  )}
                  <Badge
                    variant={statusVariant(normalizedResultStatus)}
                    className="text-xs"
                  >
                    {normalizedResultStatus}
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 flex-1">
            <div className="flex-1 min-h-[200px] rounded-md border border-border bg-muted/30 px-3 py-2 text-sm overflow-auto">
              {executing ? (
                <span className="text-muted-foreground flex items-center gap-2 mt-1">
                  <InProgress className="h-4 w-4 animate-spin" />
                  Running…
                </span>
              ) : resultError ? (
                <span className="text-destructive">{resultError}</span>
              ) : result !== null && result !== undefined ? (
                <JsonHighlightedPre data={result} className="text-sm" />
              ) : (
                <span className="text-muted-foreground">
                  (waiting for execution…)
                </span>
              )}
            </div>

            {/* ── Post-execution action buttons ───────────────────────── */}
            {(lastRunId || (result !== null && !resultError)) && !executing && (
              <div className="flex gap-2">
                {lastRunId && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 gap-2"
                    onClick={() =>
                      navigate(`/runs/${encodeURIComponent(lastRunId)}`)
                    }
                  >
                    View Run
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 gap-2"
                  onClick={handleExecute}
                  disabled={executing || !selectedId}
                >
                  <Play className="h-3.5 w-3.5" />
                  Replay
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Recent Runs ──────────────────────────────────────────────────── */}
      {selectedId && (
        <>
          <Separator />
          <div className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Recent Runs of this Reasoner
            </h2>

            {loadingRuns ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                <InProgress className="h-4 w-4 animate-spin" />
                Loading runs…
              </div>
            ) : recentRuns.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4">
                No runs recorded yet.
              </p>
            ) : (
              <div className="rounded-md border border-border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Run</TableHead>
                      <TableHead className="text-xs">Duration</TableHead>
                      <TableHead className="text-xs">Status</TableHead>
                      <TableHead className="text-xs">Input preview</TableHead>
                      <TableHead className="text-xs">Output preview</TableHead>
                      <TableHead className="text-xs w-24" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentRuns.map((run) => (
                      <TableRow key={run.id} className="text-sm">
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {run.id.slice(0, 8)}…
                        </TableCell>
                        <TableCell className="text-xs">
                          {formatDuration(run.duration_ms)}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={statusVariant(run.status)}
                            className="text-xs"
                          >
                            {normalizeExecutionStatus(run.status)}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs max-w-[160px] truncate">
                          {run.input_preview}
                        </TableCell>
                        <TableCell className="font-mono text-xs max-w-[160px] truncate">
                          {run.output_preview}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-xs gap-1"
                            onClick={() => handleLoadInput(run)}
                            title="Load this run's input into the editor"
                          >
                            <Upload className="h-3 w-3" />
                            Load Input
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
