import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { FilterCombobox } from "@/components/ui/filter-combobox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SearchBar } from "@/components/ui/SearchBar";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Loader2,
  PauseCircle,
  Play,
  RefreshCw,
  Settings,
  Terminal,
} from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";
import { NodeProcessLogsPanel } from "@/components/nodes";
import { getExecutionLogs, streamExecutionLogs } from "@/services/executionsApi";
import type { ExecutionLogEntry, WorkflowExecution } from "@/types/executions";
import { observabilityStyles } from "./observabilityStyles";

const DEFAULT_TAIL = 250;
const MAX_BUFFER = 1000;

function formatTimestamp(ts: string): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function levelVariant(level: string): "default" | "secondary" | "destructive" | "outline" {
  const normalized = level.toLowerCase();
  if (normalized === "error" || normalized === "fatal" || normalized === "critical") {
    return "destructive";
  }
  if (normalized === "warn" || normalized === "warning") {
    return "outline";
  }
  if (normalized === "debug" || normalized === "trace") {
    return "secondary";
  }
  return "default";
}

function hasAttributes(attributes: ExecutionLogEntry["attributes"]): boolean {
  if (attributes == null) return false;
  if (typeof attributes === "string") {
    return attributes.trim() !== "" && attributes.trim() !== "{}";
  }
  if (Array.isArray(attributes)) {
    return attributes.length > 0;
  }
  if (typeof attributes === "object") {
    return Object.keys(attributes).length > 0;
  }
  return true;
}

function appendUniqueSorted(values: string[], nextValue?: string): string[] {
  const set = new Set(values.filter(Boolean));
  if (nextValue?.trim()) {
    set.add(nextValue.trim());
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}

interface ExecutionObservabilityPanelProps {
  execution: WorkflowExecution;
  relatedNodeIds?: string[];
  className?: string;
}

export function ExecutionObservabilityPanel({
  execution,
  relatedNodeIds = [],
  className,
}: ExecutionObservabilityPanelProps) {
  const [entries, setEntries] = useState<ExecutionLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState("all");
  const [nodeFilter, setNodeFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [rawLogsOpen, setRawLogsOpen] = useState(false);
  const [selectedRawNodeId, setSelectedRawNodeId] = useState(execution.agent_node_id);
  const latestSeqRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const loadLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getExecutionLogs(execution.execution_id, { tail: DEFAULT_TAIL });
      const trimmed = response.entries.slice(-MAX_BUFFER);
      setEntries(trimmed);
      latestSeqRef.current = trimmed.reduce((max, entry) => Math.max(max, entry.seq ?? 0), 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load execution logs");
    } finally {
      setLoading(false);
    }
  }, [execution.execution_id]);

  useEffect(() => {
    void loadLogs();
    setSelectedRawNodeId(execution.agent_node_id);
    setLive(false);
  }, [execution.execution_id, execution.agent_node_id, loadLogs]);

  useEffect(() => {
    if (!live) return;

    const source = streamExecutionLogs(execution.execution_id, {
      afterSeq: latestSeqRef.current,
    });

    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as ExecutionLogEntry & { type?: string };
        if (payload.type === "connected" || payload.type === "heartbeat") {
          return;
        }
        setError(null);
        setEntries((prev) => {
          const next = [...prev, payload].slice(-MAX_BUFFER);
          latestSeqRef.current = next.reduce((max, entry) => Math.max(max, entry.seq ?? 0), 0);
          return next;
        });
      } catch {
        // Ignore malformed heartbeat or non-log events.
      }
    };

    source.onerror = () => {
      setError("Execution log stream interrupted");
      setLive(false);
      source.close();
    };

    return () => {
      source.close();
    };
  }, [execution.execution_id, live]);

  const availableLevels = useMemo(() => {
    return entries.reduce<string[]>((acc, entry) => appendUniqueSorted(acc, entry.level), []);
  }, [entries]);

  const availableNodeIds = useMemo(() => {
    return entries.reduce<string[]>(
      (acc, entry) => appendUniqueSorted(acc, entry.agent_node_id),
      [execution.agent_node_id, ...relatedNodeIds].filter(Boolean),
    );
  }, [entries, execution.agent_node_id, relatedNodeIds]);

  const availableSources = useMemo(() => {
    return entries.reduce<string[]>((acc, entry) => appendUniqueSorted(acc, entry.source), []);
  }, [entries]);

  const filteredEntries = useMemo(() => {
    const query = search.trim().toLowerCase();
    return entries.filter((entry) => {
      if (levelFilter !== "all" && entry.level !== levelFilter) return false;
      if (nodeFilter !== "all" && entry.agent_node_id !== nodeFilter) return false;
      if (sourceFilter !== "all" && entry.source !== sourceFilter) return false;
      if (!query) return true;

      return [
        entry.message,
        entry.source,
        entry.level,
        entry.agent_node_id,
        entry.reasoner_id,
        entry.event_type,
        JSON.stringify(entry.attributes ?? {}),
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query));
    });
  }, [entries, levelFilter, nodeFilter, search, sourceFilter]);

  useEffect(() => {
    if (!live || !scrollRef.current) return;
    const viewport = scrollRef.current.querySelector("[data-radix-scroll-area-viewport]");
    if (viewport instanceof HTMLElement) {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, [filteredEntries.length, live]);

  const systemCount = useMemo(
    () => entries.filter((entry) => entry.system_generated).length,
    [entries],
  );

  const logSummary = useMemo(() => {
    return {
      total: entries.length,
      visible: filteredEntries.length,
      nodes: availableNodeIds.length,
      live,
      system: systemCount,
    };
  }, [availableNodeIds.length, entries.length, filteredEntries.length, live, systemCount]);

  const rawNodeOptions = useMemo(
    () => availableNodeIds.map((value) => ({ value, label: value })),
    [availableNodeIds],
  );

  return (
    <Card className={cn(observabilityStyles.card, className)}>
      <CardContent className={observabilityStyles.cardContent}>
        <div className={observabilityStyles.toolbarGrid}>
          <SearchBar
            value={search}
            onChange={setSearch}
            placeholder="Search message, source, reasoner, event, or attributes…"
            size="md"
          />

          <FilterCombobox
            label="Filter by level"
            value={levelFilter}
            onValueChange={setLevelFilter}
            options={[
              { value: "all", label: "All levels" },
              ...availableLevels.map((value) => ({ value, label: value })),
            ]}
            placeholder="All levels"
            className="w-full xl:w-auto"
          />

          <FilterCombobox
            label="Filter by node"
            value={nodeFilter}
            onValueChange={setNodeFilter}
            options={[
              { value: "all", label: "All nodes" },
              ...availableNodeIds.map((value) => ({ value, label: value })),
            ]}
            placeholder="All nodes"
            className="w-full xl:w-auto"
          />

          <FilterCombobox
            label="Filter by source"
            value={sourceFilter}
            onValueChange={setSourceFilter}
            options={[
              { value: "all", label: "All sources" },
              ...availableSources.map((value) => ({ value, label: value })),
            ]}
            placeholder="All sources"
            className="w-full xl:w-auto"
          />

          <div className={observabilityStyles.toolbarActions}>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void loadLogs()}
              disabled={loading}
              className="gap-2"
            >
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
              Refresh
            </Button>
            <Button
              variant={live ? "secondary" : "default"}
              size="sm"
              onClick={() => setLive((prev) => !prev)}
              className="gap-2"
            >
              {live ? <PauseCircle className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {live ? "Pause live" : "Go live"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRawLogsOpen(true)}
              className="gap-2"
            >
              <Terminal className="h-4 w-4" />
              Raw node logs
            </Button>
          </div>
        </div>

        {error ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Observability stream unavailable</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        <div className={observabilityStyles.surface}>
          <div className={observabilityStyles.summaryBar}>
            <div className={observabilityStyles.summaryFilters}>
              <span className="font-medium">
                Showing {logSummary.visible} of {logSummary.total} events
              </span>
              {search ? (
                <Badge variant="outline" className="font-mono">
                  Query: {search}
                </Badge>
              ) : null}
              {levelFilter !== "all" ? (
                <Badge variant="outline" className="font-mono">
                  Level: {levelFilter}
                </Badge>
              ) : null}
              {nodeFilter !== "all" ? (
                <Badge variant="outline" className="font-mono">
                  Node: {nodeFilter}
                </Badge>
              ) : null}
              {sourceFilter !== "all" ? (
                <Badge variant="outline" className="font-mono">
                  Source: {sourceFilter}
                </Badge>
              ) : null}
            </div>

            <div className={observabilityStyles.summaryStats}>
              <Badge variant="secondary" className="font-mono">
                {logSummary.total} events
              </Badge>
              <Badge variant="outline" className="font-mono">
                {logSummary.nodes} node{logSummary.nodes === 1 ? "" : "s"}
              </Badge>
              <Badge variant="outline" className="font-mono">
                {logSummary.system} system
              </Badge>
              {live ? (
                <Badge variant="default" className="font-mono">
                  Live
                </Badge>
              ) : (
                <Badge variant="secondary" className="font-mono">
                  Snapshot
                </Badge>
              )}
            </div>
          </div>

          <ScrollArea ref={scrollRef} className={cn("h-[30rem]", observabilityStyles.scrollArea)}>
            {loading ? (
              <div className={observabilityStyles.loadingState}>
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading execution logs…
                </div>
              </div>
            ) : filteredEntries.length === 0 ? (
              <div className={observabilityStyles.emptyState}>
                No structured execution logs match the current filters yet.
              </div>
            ) : (
              <div className={observabilityStyles.structuredList}>
                {filteredEntries.map((entry) => (
                  <Collapsible
                    key={`${entry.execution_id}-${entry.seq}-${entry.ts}`}
                    className={observabilityStyles.structuredRow}
                  >
                    <div className={observabilityStyles.structuredRowInner}>
                      <div className="flex items-start gap-3">
                        <span className={observabilityStyles.structuredTimestamp}>
                          {formatTimestamp(entry.ts)}
                        </span>

                        <div className="min-w-0 flex-1">
                          <div className={observabilityStyles.structuredMessageRow}>
                            <Badge
                              variant={levelVariant(entry.level)}
                              className="h-5 font-mono uppercase"
                            >
                              {entry.level}
                            </Badge>
                            <p className={observabilityStyles.structuredMessage}>
                              {entry.message}
                            </p>
                          </div>

                          <div className={observabilityStyles.structuredMeta}>
                            <span className="font-mono">seq:{entry.seq}</span>
                            <span className="font-mono">node:{entry.agent_node_id}</span>
                            <span className="font-mono">source:{entry.source}</span>
                            {entry.reasoner_id ? (
                              <span className="font-mono">reasoner:{entry.reasoner_id}</span>
                            ) : null}
                            {entry.event_type ? (
                              <span className="font-mono">event:{entry.event_type}</span>
                            ) : null}
                          </div>
                        </div>

                        {hasAttributes(entry.attributes) ? (
                          <CollapsibleTrigger className={observabilityStyles.detailTrigger}>
                            <ChevronRight className="h-3.5 w-3.5 group-data-[state=open]:hidden" />
                            <ChevronDown className="hidden h-3.5 w-3.5 group-data-[state=open]:block" />
                            attrs
                          </CollapsibleTrigger>
                        ) : null}
                      </div>

                      {hasAttributes(entry.attributes) ? (
                        <CollapsibleContent className="mt-2 pl-[calc(5.5rem+0.75rem)]">
                          <div className={observabilityStyles.detailPanel}>
                            <UnifiedJsonViewer
                              data={entry.attributes}
                              searchable={false}
                              showHeader={false}
                              className="border-0"
                              maxHeight="14rem"
                            />
                          </div>
                        </CollapsibleContent>
                      ) : null}
                    </div>
                  </Collapsible>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </CardContent>

      <Drawer open={rawLogsOpen} onOpenChange={setRawLogsOpen}>
        <DrawerContent className="max-h-[92vh]">
          <DrawerHeader className="space-y-3">
            <DrawerTitle className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Advanced raw node logs
            </DrawerTitle>
            <DrawerDescription>
              Process-level stdout and stderr for the selected node. These logs are useful for
              deep debugging and may include activity outside this execution.
            </DrawerDescription>
            {rawNodeOptions.length > 1 ? (
              <div className="max-w-xs">
                <FilterCombobox
                  label="Select node"
                  value={selectedRawNodeId}
                  onValueChange={setSelectedRawNodeId}
                  options={rawNodeOptions}
                  placeholder="Select a node"
                />
              </div>
            ) : null}
          </DrawerHeader>

          <div className="px-4 pb-4 sm:px-6 sm:pb-6">
            <NodeProcessLogsPanel nodeId={selectedRawNodeId} />
          </div>
        </DrawerContent>
      </Drawer>
    </Card>
  );
}
