// Node column uses AgentNodeIcon (agent-as-service). Endpoint list uses ReasonerIcon for registry rows from reasonersApi.
import * as React from "react";

import { cn } from "@/lib/utils";
import {
  AgentNodeIcon,
  Check,
  ChevronDown,
  ReasonerIcon,
  Search,
} from "@/components/ui/icon-bridge";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ReasonerWithNode } from "@/types/reasoners";

function reasonerMatchesQuery(r: ReasonerWithNode, q: string): boolean {
  if (!q) return true;
  const n = q.toLowerCase();
  return (
    r.node_id.toLowerCase().includes(n) ||
    r.name.toLowerCase().includes(n) ||
    r.reasoner_id.toLowerCase().includes(n) ||
    !!(r.description && r.description.toLowerCase().includes(n))
  );
}

function nodeMatchesQuery(
  nodeId: string,
  nodeReasoners: ReasonerWithNode[],
  q: string
): boolean {
  if (!q) return true;
  if (nodeId.toLowerCase().includes(q.toLowerCase())) return true;
  return nodeReasoners.some((r) => reasonerMatchesQuery(r, q));
}

function visibleReasonersForNode(
  nodeId: string,
  list: ReasonerWithNode[],
  q: string
): ReasonerWithNode[] {
  if (!q) return list;
  const matched = list.filter((r) => reasonerMatchesQuery(r, q));
  if (matched.length > 0) return matched;
  if (nodeId.toLowerCase().includes(q.toLowerCase())) return list;
  return [];
}

export interface ReasonerNodeComboboxProps {
  reasoners: ReasonerWithNode[];
  value: string | null;
  onValueChange: (reasonerId: string) => void;
  disabled?: boolean;
  loading?: boolean;
  /** Trigger width */
  className?: string;
  placeholder?: string;
}

export function ReasonerNodeCombobox({
  reasoners,
  value,
  onValueChange,
  disabled,
  loading,
  className,
  placeholder = "Select agent node · skill",
}: ReasonerNodeComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [activeNodeId, setActiveNodeId] = React.useState<string | null>(null);

  const grouped = React.useMemo(() => {
    const m = new Map<string, ReasonerWithNode[]>();
    for (const r of reasoners) {
      const list = m.get(r.node_id) ?? [];
      list.push(r);
      m.set(r.node_id, list);
    }
    for (const list of m.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return m;
  }, [reasoners]);

  const sortedNodeIds = React.useMemo(
    () => [...grouped.keys()].sort((a, b) => a.localeCompare(b)),
    [grouped]
  );

  const selected = React.useMemo(
    () => reasoners.find((r) => r.reasoner_id === value) ?? null,
    [reasoners, value]
  );

  const q = search.trim();

  const visibleNodeIds = React.useMemo(() => {
    return sortedNodeIds.filter((id) =>
      nodeMatchesQuery(id, grouped.get(id) ?? [], q)
    );
  }, [sortedNodeIds, grouped, q]);

  const reasonersInActiveNode = React.useMemo(() => {
    if (!activeNodeId) return [];
    const list = grouped.get(activeNodeId) ?? [];
    return visibleReasonersForNode(activeNodeId, list, q);
  }, [grouped, activeNodeId, q]);

  // When opening: focus the node for the current value, or the first visible node.
  React.useEffect(() => {
    if (!open) {
      setSearch("");
      return;
    }
    if (selected) {
      setActiveNodeId(selected.node_id);
      return;
    }
    if (sortedNodeIds.length > 0) {
      setActiveNodeId(sortedNodeIds[0]);
    }
  }, [open, selected, sortedNodeIds]);

  // Smart node focus while typing: single global match, or all matches under one node.
  React.useEffect(() => {
    if (!open || !q) return;
    const matches = reasoners.filter((r) => reasonerMatchesQuery(r, q));
    if (matches.length === 1) {
      setActiveNodeId(matches[0].node_id);
      return;
    }
    const nodes = new Set(matches.map((r) => r.node_id));
    if (nodes.size === 1) {
      setActiveNodeId([...nodes][0]);
    }
  }, [open, q, reasoners]);

  // Keep the active node in sync with the filtered node list (e.g. while searching).
  React.useEffect(() => {
    if (!open) return;
    if (visibleNodeIds.length === 0) {
      setActiveNodeId(null);
      return;
    }
    if (!activeNodeId || !visibleNodeIds.includes(activeNodeId)) {
      setActiveNodeId(visibleNodeIds[0]);
    }
  }, [open, activeNodeId, visibleNodeIds]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-label="Select agent node and skill"
          disabled={disabled || loading}
          className={cn(
            "flex h-10 w-full min-w-0 items-center justify-between gap-2 rounded-md border border-input bg-background px-3 text-sm ring-offset-background",
            "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
            "disabled:cursor-not-allowed disabled:opacity-50",
            !selected && "text-muted-foreground",
            className
          )}
        >
          <span className="flex min-w-0 flex-1 flex-nowrap items-center gap-2 overflow-hidden text-left">
            {selected ? (
              <>
                <span
                  className="flex size-6 shrink-0 items-center justify-center rounded-md border border-border bg-muted/30 text-muted-foreground"
                  aria-hidden
                >
                  <AgentNodeIcon className="size-3.5 shrink-0" />
                </span>
                <span className="flex min-w-0 flex-1 flex-nowrap items-center gap-1.5 overflow-hidden">
                  <span className="min-w-0 shrink truncate font-mono text-xs text-muted-foreground">
                    {selected.node_id}
                  </span>
                  <span className="shrink-0 text-muted-foreground" aria-hidden>
                    ·
                  </span>
                  <span className="flex min-w-0 flex-1 flex-nowrap items-center gap-1.5 overflow-hidden">
                    <span
                      className="flex size-6 shrink-0 items-center justify-center rounded border border-border/80 bg-background text-muted-foreground"
                      aria-hidden
                    >
                      <ReasonerIcon className="size-3 shrink-0" />
                    </span>
                    <span className="min-w-0 flex-1 truncate font-medium text-foreground">
                      {selected.name}
                    </span>
                  </span>
                </span>
              </>
            ) : loading ? (
              "Loading skills…"
            ) : (
              placeholder
            )}
          </span>
          <ChevronDown className="h-4 w-4 shrink-0 opacity-50" aria-hidden />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[min(100vw-1.5rem,32rem)] p-0"
        align="start"
        sideOffset={4}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <div className="border-b border-border px-3 py-2">
          <div className="relative">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search nodes, skills, or full id…"
              className="h-9 pl-9 text-sm"
              autoFocus
            />
          </div>
        </div>

        <div className="grid max-h-[min(50vh,20rem)] grid-cols-[minmax(0,11.5rem)_1fr] divide-x divide-border">
          <ScrollArea className="h-[min(50vh,20rem)]">
            <div className="p-1">
              {visibleNodeIds.length === 0 ? (
                <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                  No matching nodes
                </p>
              ) : (
                visibleNodeIds.map((nodeId) => {
                  const list = grouped.get(nodeId) ?? [];
                  const count = visibleReasonersForNode(nodeId, list, q).length;
                  const isActive = activeNodeId === nodeId;
                  return (
                    <button
                      key={nodeId}
                      type="button"
                      onMouseEnter={() => setActiveNodeId(nodeId)}
                      onFocus={() => setActiveNodeId(nodeId)}
                      onClick={() => setActiveNodeId(nodeId)}
                      className={cn(
                        "flex w-full flex-row items-start gap-2 rounded-sm px-2 py-2 text-left text-sm outline-none transition-colors",
                        "hover:bg-accent hover:text-accent-foreground",
                        "focus-visible:bg-accent focus-visible:text-accent-foreground",
                        isActive && "bg-accent text-accent-foreground"
                      )}
                    >
                      <span
                        className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md border border-border/80 bg-background text-muted-foreground"
                        aria-hidden
                      >
                        <AgentNodeIcon className="size-3.5 shrink-0" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block w-full truncate font-mono text-xs">
                          {nodeId}
                        </span>
                        <span className="text-micro text-muted-foreground">
                          {count} skill{count === 1 ? "" : "s"}
                        </span>
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </ScrollArea>

          <ScrollArea className="h-[min(50vh,20rem)]">
            <div className="p-1">
              {!activeNodeId ? (
                <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                  Select a node
                </p>
              ) : reasonersInActiveNode.length === 0 ? (
                <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                  No skills match your search for this node.
                </p>
              ) : (
                reasonersInActiveNode.map((r) => {
                  const isSelected = value === r.reasoner_id;
                  return (
                    <button
                      key={r.reasoner_id}
                      type="button"
                      onClick={() => {
                        onValueChange(r.reasoner_id);
                        setOpen(false);
                      }}
                      className={cn(
                        "flex w-full flex-col items-start gap-0.5 rounded-sm px-3 py-2 text-left text-sm outline-none transition-colors",
                        "hover:bg-accent hover:text-accent-foreground",
                        "focus-visible:bg-accent focus-visible:text-accent-foreground",
                        isSelected && "bg-accent/80"
                      )}
                    >
                      <span className="flex w-full items-center gap-2">
                        <Check
                          className={cn(
                            "h-4 w-4 shrink-0",
                            isSelected ? "opacity-100" : "opacity-0"
                          )}
                          aria-hidden
                        />
                        <span
                          className="flex size-7 shrink-0 items-center justify-center rounded-md border border-border/80 bg-background text-muted-foreground"
                          aria-hidden
                        >
                          <ReasonerIcon className="size-3.5 shrink-0" />
                        </span>
                        <span className="min-w-0 flex-1 truncate font-medium">
                          {r.name}
                        </span>
                      </span>
                      <span className="w-full pl-15 font-mono text-micro-plus text-muted-foreground">
                        {r.reasoner_id}
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </ScrollArea>
        </div>
      </PopoverContent>
    </Popover>
  );
}
