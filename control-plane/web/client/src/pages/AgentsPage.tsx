import { useEffect, useMemo, useState } from "react";
import { formatCompactRelativeTime } from "@/utils/dateFormat";
import { Link, useNavigate } from "react-router-dom";
import { useAgents, useAgentTagSummaries } from "@/hooks/queries";
import { getNodeDetails } from "@/services/api";
import { startAgent } from "@/services/configurationApi";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { EndpointKindIconBox } from "@/components/ui/endpoint-kind-icon-box";
import { EntityTag } from "@/components/ui/entity-tag";
import { LifecycleDot } from "@/components/ui/status-pill";
import { isLifecycleOnline } from "@/utils/lifecycle-status";
import { NodeProcessLogsPanel } from "@/components/nodes";
import {
  AgentNodeIcon,
  ChevronRight,
  Play,
  ReasonerIcon,
  RefreshCw,
  Search,
  SkillIcon,
  Terminal,
} from "@/components/ui/icon-bridge";
import type { AgentNodeSummary, ReasonerDefinition, SkillDefinition } from "@/types/agentfield";
import type { AgentTagSummary } from "@/services/tagApprovalApi";
import { useQuery } from "@tanstack/react-query";

// ─── Helpers ────────────────────────────────────────────────────────────────

const formatRelativeTime = formatCompactRelativeTime;

type AgentListStatusFilter = "all" | "online" | "offline";

function matchesAgentNodeSearch(node: AgentNodeSummary, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (node.id.toLowerCase().includes(q)) return true;
  if (node.version?.toLowerCase().includes(q)) return true;
  if (node.lifecycle_status.toLowerCase().includes(q)) return true;
  if (node.base_url?.toLowerCase().includes(q)) return true;
  return false;
}

function matchesAgentStatusFilter(
  node: AgentNodeSummary,
  filter: AgentListStatusFilter
): boolean {
  if (filter === "all") return true;
  const online = isLifecycleOnline(node.lifecycle_status);
  return filter === "online" ? online : !online;
}

// ─── NodeReasonerList ────────────────────────────────────────────────────────

/** When more than this many rows load, cap list height and scroll inside (native overflow — Radix ScrollArea needs fixed height). */
const SCROLL_AFTER = 8;

type NodeEndpointRow = {
  id: string;
  name: string;
  description?: string;
  kind: "reasoner" | "skill";
};

interface NodeReasonerListProps {
  nodeId: string;
  reasonerCount: number;
  skillCount: number;
}

function matchesFilter(q: string, row: NodeEndpointRow): boolean {
  if (!q.trim()) return true;
  const n = q.trim().toLowerCase();
  return (
    row.id.toLowerCase().includes(n) ||
    row.name.toLowerCase().includes(n) ||
    (row.description?.toLowerCase().includes(n) ?? false)
  );
}

function NodeReasonerList({ nodeId, reasonerCount, skillCount }: NodeReasonerListProps) {
  const navigate = useNavigate();
  const [filter, setFilter] = useState("");

  const { data: nodeDetails, isLoading, isError, error } = useQuery({
    queryKey: ["node-details", nodeId],
    queryFn: () => getNodeDetails(nodeId),
    staleTime: 30_000,
  });

  const reasonerRows: NodeEndpointRow[] = useMemo(() => {
    const list = nodeDetails?.reasoners ?? [];
    return list.map((r: ReasonerDefinition) => ({
      id: r.id,
      name: r.name || r.id,
      description: r.description,
      kind: "reasoner" as const,
    }));
  }, [nodeDetails?.reasoners]);

  const skillRows: NodeEndpointRow[] = useMemo(() => {
    const list = nodeDetails?.skills ?? [];
    return list.map((s: SkillDefinition) => ({
      id: s.id,
      name: s.name || s.id,
      description: s.description,
      kind: "skill" as const,
    }));
  }, [nodeDetails?.skills]);

  const filteredReasoners = useMemo(
    () => reasonerRows.filter((r) => matchesFilter(filter, r)),
    [reasonerRows, filter]
  );
  const filteredSkills = useMemo(
    () => skillRows.filter((s) => matchesFilter(filter, s)),
    [skillRows, filter]
  );

  const totalLoaded = reasonerRows.length + skillRows.length;
  const totalExpected = reasonerCount + skillCount;
  const showSearch = totalLoaded >= 10;
  const useScroll = totalLoaded > SCROLL_AFTER;
  const showSectionLabels = reasonerRows.length > 0 && skillRows.length > 0;

  if (isLoading) {
    return (
      <div className="border-t border-border bg-muted/15">
        <div className="pl-10 pr-4 py-2 space-y-2">
          {Array.from({ length: Math.min(Math.max(totalExpected, 1), 4) }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="size-8 shrink-0 rounded-md bg-muted/50 animate-pulse" />
              <div className="h-4 flex-1 max-w-[200px] rounded bg-muted/40 animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="border-t border-border bg-muted/15 px-4 py-2 pl-10">
        <p className="text-xs text-destructive">
          Could not load endpoints
          {error instanceof Error ? `: ${error.message}` : ""}. Try expanding again or check the node is reachable.
        </p>
      </div>
    );
  }

  if (totalLoaded === 0) {
    return (
      <div className="border-t border-border bg-muted/15 pl-10 pr-4 py-2.5">
        <p className="text-xs text-muted-foreground">No reasoners or skills registered on this node.</p>
      </div>
    );
  }

  const listBody = (
    <div className="divide-y divide-border/70">
      {filteredReasoners.length === 0 && filteredSkills.length === 0 ? (
        <div className="px-3 py-3 text-center text-xs text-muted-foreground">
          No matches for &quot;{filter.trim()}&quot;
        </div>
      ) : (
        <>
          {filteredReasoners.length > 0 && (
            <>
              {showSectionLabels && (
                <div
                  className="sticky top-0 z-[1] flex items-center gap-2 bg-muted/30 px-3 py-1.5 text-micro-plus font-medium uppercase tracking-wide text-muted-foreground backdrop-blur-sm"
                  role="presentation"
                >
                  <ReasonerIcon className="size-3.5 opacity-80" aria-hidden />
                  Reasoners
                  <span className="font-mono text-micro normal-case tracking-normal text-muted-foreground/80">
                    ({filteredReasoners.length})
                  </span>
                </div>
              )}
              {filteredReasoners.map((row) => (
                <EndpointRow key={`r-${row.id}`} nodeId={nodeId} row={row} onOpen={navigate} />
              ))}
            </>
          )}
          {filteredSkills.length > 0 && (
            <>
              {showSectionLabels && (
                <div
                  className="sticky top-0 z-[1] flex items-center gap-2 bg-muted/30 px-3 py-1.5 text-micro-plus font-medium uppercase tracking-wide text-muted-foreground backdrop-blur-sm"
                  role="presentation"
                >
                  <SkillIcon className="size-3.5 opacity-80" aria-hidden />
                  Skills
                  <span className="font-mono text-micro normal-case tracking-normal text-muted-foreground/80">
                    ({filteredSkills.length})
                  </span>
                </div>
              )}
              {filteredSkills.map((row) => (
                <EndpointRow key={`s-${row.id}`} nodeId={nodeId} row={row} onOpen={navigate} />
              ))}
            </>
          )}
        </>
      )}
    </div>
  );

  return (
    <div className="border-t border-border bg-muted/15">
      {showSearch && (
        <div className="border-b border-border/60 px-3 py-2 pl-10">
          <div className="relative max-w-md">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by name or id…"
              className="h-8 border-border/80 bg-background/80 pl-8 text-xs shadow-none"
              aria-label="Filter reasoners and skills"
            />
          </div>
        </div>
      )}
      <div
        className={cn(
          "min-h-0 pl-6",
          useScroll &&
            "max-h-[min(45vh,320px)] overflow-y-auto overflow-x-hidden overscroll-y-contain pr-2 [scrollbar-gutter:stable]"
        )}
      >
        {listBody}
      </div>
    </div>
  );
}

interface EndpointRowProps {
  nodeId: string;
  row: NodeEndpointRow;
  onOpen: (path: string) => void;
}

function EndpointRow({ nodeId, row, onOpen }: EndpointRowProps) {
  const isSkill = row.kind === "skill";
  const label = isSkill ? "skill" : "reasoner";

  return (
    <button
      type="button"
      className={cn(
        "flex w-full items-start gap-3 px-3 py-2 pl-4 text-left transition-colors",
        "hover:bg-accent/40",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      )}
      onClick={() => onOpen(`/playground/${nodeId}.${row.id}`)}
      aria-label={`Open ${label} ${row.name} in playground`}
    >
      <EndpointKindIconBox
        kind={isSkill ? "skill" : "reasoner"}
        className="mt-0.5"
      />
      <span className="min-w-0 flex-1 pt-0.5">
        <span className="block font-mono text-xs font-medium leading-snug text-foreground">
          {row.name}
        </span>
        <span className="mt-0.5 flex flex-wrap items-baseline gap-x-1 gap-y-0.5">
          <EntityTag tone={isSkill ? "neutral" : "accent"}>
            {isSkill ? "Skill" : "Reasoner"}
          </EntityTag>
          <span
            className="select-none text-[0.625rem] leading-none text-muted-foreground/30"
            aria-hidden
          >
            ·
          </span>
          {row.description ? (
            <span className="min-w-0 max-w-full text-micro-plus leading-snug text-muted-foreground line-clamp-2">
              {row.description}
            </span>
          ) : (
            <span className="min-w-0 font-mono text-micro leading-snug text-muted-foreground/80">
              {row.id}
            </span>
          )}
        </span>
      </span>
      <span className="flex shrink-0 items-center gap-1.5 self-center text-muted-foreground">
        <span className="hidden text-micro-plus sm:inline">Playground</span>
        <Play className="size-3.5 opacity-70" aria-hidden />
      </span>
    </button>
  );
}

// ─── AgentRow ────────────────────────────────────────────────────────────────

function AgentAuthTagStrip({ summary }: { summary: AgentTagSummary }) {
  const granted = summary.approved_tags ?? [];
  const requested = summary.proposed_tags ?? [];
  const hasTags = granted.length > 0 || requested.length > 0;

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border/50 bg-muted/15 px-4 py-2 pl-14 text-left">
      <span className="w-full text-micro font-medium uppercase tracking-wide text-muted-foreground sm:w-auto">
        Auth tags
      </span>
      {!hasTags && (
        <span className="text-micro text-muted-foreground">No tags recorded</span>
      )}
      {granted.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-micro text-muted-foreground">Granted</span>
          {granted.slice(0, 8).map((t) => (
            <Badge
              key={`g-${t}`}
              variant="secondary"
              size="sm"
              showIcon={false}
              className="max-w-[140px] truncate text-micro"
            >
              {t}
            </Badge>
          ))}
          {granted.length > 8 && (
            <span className="text-micro text-muted-foreground">+{granted.length - 8}</span>
          )}
        </div>
      )}
      {requested.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-micro text-muted-foreground">Requested</span>
          {requested.slice(0, 8).map((t) => (
            <Badge
              key={`p-${t}`}
              variant="outline"
              size="sm"
              showIcon={false}
              className="max-w-[140px] truncate text-micro"
            >
              {t}
            </Badge>
          ))}
          {requested.length > 8 && (
            <span className="text-micro text-muted-foreground">+{requested.length - 8}</span>
          )}
        </div>
      )}
      <Link
        to="/access"
        className="ml-auto text-micro font-medium text-primary hover:underline"
      >
        Access management
      </Link>
    </div>
  );
}

interface AgentRowProps {
  node: AgentNodeSummary;
  tagSummary?: AgentTagSummary;
}

function AgentRow({ node, tagSummary }: AgentRowProps) {
  const [open, setOpen] = useState(false);
  const [detailTab, setDetailTab] = useState<"endpoints" | "logs">(
    "endpoints"
  );
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    if (!open) setDetailTab("endpoints");
  }, [open]);

  const openProcessLogs = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDetailTab("logs");
    setOpen(true);
  };

  const totalItems = node.reasoner_count + node.skill_count;

  const handleRestart = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setRestarting(true);
    try {
      await startAgent(node.id);
    } catch (err) {
      console.error("Failed to restart agent:", node.id, err);
    } finally {
      setRestarting(false);
    }
  };

  return (
    <>
      {/* Main row */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-3 w-full px-4 py-2.5 text-left hover:bg-accent/40 transition-colors"
      >
        <ChevronRight
          className={cn(
            "size-3 text-muted-foreground transition-transform flex-shrink-0",
            open && "rotate-90"
          )}
        />

        <span
          className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-background text-muted-foreground"
          aria-hidden
        >
          <AgentNodeIcon className="size-4 shrink-0" />
        </span>

        <span className="font-mono text-sm font-medium truncate min-w-0 flex-1">
          {node.id || '(unknown)'}
        </span>

        {/* Status dot + label */}
        <div className="flex items-center flex-shrink-0">
          <LifecycleDot status={node.lifecycle_status} size="sm" />
        </div>

        {/* Reasoner / skill counts */}
        {totalItems > 0 && (
          <span className="text-xs text-muted-foreground flex-shrink-0 text-right tabular-nums max-sm:max-w-[5.5rem] max-sm:truncate sm:w-36">
            {node.reasoner_count > 0 && (
              <>
                {node.reasoner_count} reasoner{node.reasoner_count !== 1 ? "s" : ""}
              </>
            )}
            {node.reasoner_count > 0 && node.skill_count > 0 && " · "}
            {node.skill_count > 0 && (
              <>
                {node.skill_count} skill{node.skill_count !== 1 ? "s" : ""}
              </>
            )}
          </span>
        )}

        {/* Version */}
        {node.version && (
          <span className="text-xs text-muted-foreground font-mono flex-shrink-0 w-16 text-right hidden sm:inline">
            v{node.version}
          </span>
        )}

        {/* Heartbeat */}
        <span className="text-xs text-muted-foreground flex-shrink-0 w-20 text-right">
          {formatRelativeTime(node.last_heartbeat)}
        </span>

        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-6 text-muted-foreground hover:text-foreground"
            onClick={openProcessLogs}
            aria-label={`Open process logs for ${node.id}`}
            title="Process logs"
          >
            <Terminal className="size-3" aria-hidden />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-6 text-muted-foreground hover:text-foreground"
            onClick={handleRestart}
            disabled={restarting}
            aria-label="Restart agent"
          >
            <RefreshCw className={cn("size-3", restarting && "animate-spin")} />
          </Button>
        </div>
      </button>

      {tagSummary ? <AgentAuthTagStrip summary={tagSummary} /> : null}

      {/* Expanded: endpoints + process logs (match Node detail tab chrome: rounded-lg muted track, full width) */}
      {open && (
        <div className="border-t border-border bg-muted/10">
          <Tabs
            value={detailTab}
            onValueChange={(v) => setDetailTab(v as "endpoints" | "logs")}
            className="w-full"
          >
            <div
              className="border-b border-border/60 bg-muted/20 px-4 py-3"
              role="presentation"
            >
              <TabsList
                className="grid h-10 w-full grid-cols-2 gap-1 rounded-lg bg-muted/40 p-1 text-muted-foreground shadow-inner"
                aria-label="Agent detail sections"
              >
                <TabsTrigger
                  value="endpoints"
                  className="group gap-2 rounded-md px-3 text-sm font-medium data-[state=active]:text-foreground"
                >
                  <ReasonerIcon
                    className="size-4 shrink-0 opacity-60 group-data-[state=active]:opacity-100"
                    aria-hidden
                  />
                  <span className="truncate">Endpoints</span>
                </TabsTrigger>
                <TabsTrigger
                  value="logs"
                  className="group gap-2 rounded-md px-3 text-sm font-medium data-[state=active]:text-foreground"
                >
                  <Terminal
                    className="size-4 shrink-0 opacity-60 group-data-[state=active]:opacity-100"
                    aria-hidden
                  />
                  <span className="truncate">Process logs</span>
                </TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="endpoints" className="mt-0 focus-visible:outline-none">
              <NodeReasonerList
                nodeId={node.id}
                reasonerCount={node.reasoner_count}
                skillCount={node.skill_count}
              />
            </TabsContent>
            <TabsContent value="logs" className="mt-0 border-t border-border/40 bg-card/30 px-4 pb-4 pt-3 focus-visible:outline-none">
              <NodeProcessLogsPanel nodeId={node.id} />
            </TabsContent>
          </Tabs>
        </div>
      )}
    </>
  );
}

// ─── AgentsPage ──────────────────────────────────────────────────────────────

export function AgentsPage() {
  const { data, isLoading, isError, error } = useAgents();
  const { data: tagAgents } = useAgentTagSummaries();
  const tagsByAgentId = useMemo(() => {
    const m = new Map<string, AgentTagSummary>();
    for (const a of tagAgents ?? []) {
      m.set(a.agent_id, a);
    }
    return m;
  }, [tagAgents]);

  const agentsFromApi = data?.nodes;
  const nodes = agentsFromApi ?? [];
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<AgentListStatusFilter>("all");

  const filteredNodes = useMemo(() => {
    const list = agentsFromApi ?? [];
    return list.filter(
      (n) =>
        matchesAgentNodeSearch(n, searchQuery) &&
        matchesAgentStatusFilter(n, statusFilter)
    );
  }, [agentsFromApi, searchQuery, statusFilter]);

  const { totalAgents, onlineAgents, offlineAgents } = useMemo(() => {
    const list = agentsFromApi ?? [];
    const online = list.filter((n) => isLifecycleOnline(n.lifecycle_status)).length;
    return {
      totalAgents: list.length,
      onlineAgents: online,
      offlineAgents: list.length - online,
    };
  }, [agentsFromApi]);

  const hasActiveFilters =
    searchQuery.trim().length > 0 || statusFilter !== "all";

  return (
    <div className="flex flex-col gap-4">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Agent nodes &amp; logs
        </h1>
        <div className="space-y-1 text-muted-foreground">
          <p className="text-sm">
            {isLoading ? (
              "Loading agents…"
            ) : nodes.length === 0 ? (
              "No agents registered"
            ) : (
              <>
                {nodes.length} agent node{nodes.length !== 1 ? "s" : ""}{" "}
                registered
              </>
            )}
          </p>
          {!isLoading && nodes.length > 0 ? (
            <p className="text-xs leading-relaxed text-muted-foreground/95">
              Expand a row for{" "}
              <span className="font-medium text-foreground/80">Endpoints</span>{" "}
              and{" "}
              <span className="font-medium text-foreground/80">Process logs</span>
              , or click the{" "}
              <Terminal
                className="inline-block size-3.5 align-[-0.125em] opacity-90"
                aria-hidden
              />{" "}
              icon beside restart to jump straight to logs.
            </p>
          ) : null}
        </div>
      </header>

      {/* Search + segmented connection filter (shadcn Tabs — Figma-style control) */}
      {!isLoading && !isError && nodes.length > 0 && (
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between lg:gap-4">
          <div className="relative min-w-0 flex-1 lg:max-w-md">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by node id, version, status…"
              className="h-9 border-border/80 bg-background pl-9 text-sm shadow-sm"
              aria-label="Search agent nodes"
            />
          </div>
          <Tabs
            value={statusFilter}
            onValueChange={(v) => setStatusFilter(v as AgentListStatusFilter)}
            className="w-full shrink-0 lg:w-auto"
            aria-label="Filter agents by connection"
          >
            <TabsList
              variant="segmented"
              density="cosy"
              className="grid h-9 w-full grid-cols-3 gap-0.5 p-1 lg:inline-flex lg:w-auto"
            >
              <TabsTrigger
                variant="segmented"
                size="sm"
                value="all"
                className="gap-1.5 px-2 sm:px-3"
              >
                <span>All</span>
                <span
                  className="min-w-[1.25rem] rounded-md bg-background/60 px-1 py-px text-center text-micro font-medium tabular-nums text-muted-foreground shadow-sm"
                  aria-hidden
                >
                  {totalAgents}
                </span>
              </TabsTrigger>
              <TabsTrigger
                variant="segmented"
                size="sm"
                value="online"
                className="gap-1.5 px-2 sm:px-3"
              >
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="size-1.5 shrink-0 rounded-full bg-green-400"
                    aria-hidden
                  />
                  Online
                </span>
                <span
                  className="min-w-[1.25rem] rounded-md bg-background/60 px-1 py-px text-center text-micro font-medium tabular-nums text-muted-foreground shadow-sm"
                  aria-hidden
                >
                  {onlineAgents}
                </span>
              </TabsTrigger>
              <TabsTrigger
                variant="segmented"
                size="sm"
                value="offline"
                className="gap-1.5 px-2 sm:px-3"
              >
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="size-1.5 shrink-0 rounded-full bg-red-400"
                    aria-hidden
                  />
                  Offline
                </span>
                <span
                  className="min-w-[1.25rem] rounded-md bg-background/60 px-1 py-px text-center text-micro font-medium tabular-nums text-muted-foreground shadow-sm"
                  aria-hidden
                >
                  {offlineAgents}
                </span>
              </TabsTrigger>
            </TabsList>
            {/* Radix associates triggers with panels; list below is the real content — keep panels inert */}
            <TabsContent value="all" className="hidden" tabIndex={-1} />
            <TabsContent value="online" className="hidden" tabIndex={-1} />
            <TabsContent value="offline" className="hidden" tabIndex={-1} />
          </Tabs>
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Failed to load agents:{" "}
          {error instanceof Error ? error.message : "Unknown error"}
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && (
        <Card>
          <div className="divide-y divide-border">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-10 px-4 py-2.5 animate-pulse"
              >
                <div className="h-4 bg-muted/40 rounded w-48" />
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Empty state */}
      {!isLoading && !isError && nodes.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-sm font-medium text-muted-foreground">
            No agent nodes found
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Start an agent to see it here. Run{" "}
            <code className="font-mono bg-muted px-1 rounded">af run</code> in
            your agent directory.
          </p>
        </div>
      )}

      {/* Agent list */}
      {!isLoading && nodes.length > 0 && filteredNodes.length > 0 && (
        <Card className="overflow-hidden p-0">
          <div className="divide-y divide-border">
            {filteredNodes.map((node) => (
              <AgentRow
                key={node.id}
                node={node}
                tagSummary={tagsByAgentId.get(node.id)}
              />
            ))}
          </div>
        </Card>
      )}

      {/* Filtered empty */}
      {!isLoading && !isError && nodes.length > 0 && filteredNodes.length === 0 && (
        <Card className="border-dashed p-8 text-center">
          <p className="text-sm font-medium text-foreground">No matching agents</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Try a different search or connection filter.
          </p>
          {hasActiveFilters && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => {
                setSearchQuery("");
                setStatusFilter("all");
              }}
            >
              Clear filters
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}
