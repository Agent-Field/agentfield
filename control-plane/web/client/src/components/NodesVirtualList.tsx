import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronUp, WarningFilled, ErrorFilled, CloudOffline } from '@/components/ui/icon-bridge';
import { Skeleton } from "@/components/ui/skeleton";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import type { AgentNodeSummary } from '../types/haxen';
import { NodeCard } from './NodeCard';
import type { DensityMode } from './DensityToggle';
import { getStatusPriority } from './ui/status-indicator';
import { getNodeStatusPresentation } from "@/utils/node-status";
import { cn } from "@/lib/utils";
import { cardVariants } from "@/components/ui/card";

interface NodesVirtualListProps {
  nodes: AgentNodeSummary[];
  searchQuery: string;
  isLoading: boolean;
  density?: DensityMode;
}


export function NodesVirtualList({ nodes, searchQuery, isLoading, density = 'comfortable' }: NodesVirtualListProps) {
  const [showInactive, setShowInactive] = useState(false);
  const [visibleFreshCount, setVisibleFreshCount] = useState(20);
  const [visibleInactiveCount, setVisibleInactiveCount] = useState(20);

  const readyPresentation = getNodeStatusPresentation("ready", "active");
  const stalePresentation = getNodeStatusPresentation("starting");
  const veryStalePresentation = getNodeStatusPresentation("degraded");
  const degradedPresentation = getNodeStatusPresentation("degraded");
  const startingPresentation = getNodeStatusPresentation("starting");
  const offlinePresentation = getNodeStatusPresentation("offline", "inactive");
  const sectionContainerClasses = cn(
    cardVariants({ variant: "surface", interactive: false }),
    "px-4 py-3 shadow-sm"
  );

  // Two-tier categorization: Primary (Running/Offline) + Secondary (Health Details)
  const categorizedNodes = useMemo(() => {
    const now = new Date();
    
    const categorized = nodes.reduce((acc, node) => {
      // Safe date handling - check for null/undefined heartbeat
      const lastHeartbeat = node.last_heartbeat ? new Date(node.last_heartbeat) : null;
      const isValidHeartbeat = lastHeartbeat && !isNaN(lastHeartbeat.getTime());
      const minutesSinceHeartbeat = isValidHeartbeat 
        ? (now.getTime() - lastHeartbeat.getTime()) / (1000 * 60)
        : Infinity;

      // Primary categorization based on lifecycle status (what users care about)
      const isRunning = node.lifecycle_status === 'ready' || node.lifecycle_status === 'degraded';
      
      if (isRunning) {
        // Node is running - categorize by health details
        if (node.lifecycle_status === 'degraded') {
          acc.running.degraded.push(node);
        } else if (minutesSinceHeartbeat <= 2) {
          acc.running.fresh.push(node);
        } else if (minutesSinceHeartbeat <= 5) {
          acc.running.stale.push(node);
        } else {
          acc.running.veryStale.push(node);
        }
      } else {
        // Node is offline - categorize by reason
        if (node.lifecycle_status === 'starting') {
          acc.offline.starting.push(node);
        } else {
          acc.offline.down.push(node);
        }
      }
      
      return acc;
    }, { 
      running: {
        fresh: [] as AgentNodeSummary[],
        stale: [] as AgentNodeSummary[],
        veryStale: [] as AgentNodeSummary[],
        degraded: [] as AgentNodeSummary[],
      },
      offline: {
        starting: [] as AgentNodeSummary[],
        down: [] as AgentNodeSummary[],
      }
    });

    // Sort each category by importance (capability count + recency)
    const sortByImportance = (a: AgentNodeSummary, b: AgentNodeSummary) => {
      // First by status priority (most critical first)
      const aPriority = getStatusPriority(a.lifecycle_status, a.health_status);
      const bPriority = getStatusPriority(b.lifecycle_status, b.health_status);
      
      if (aPriority !== bPriority) {
        return aPriority - bPriority;
      }
      
      // Then by capability count (descending)
      const aImportance = a.reasoner_count + a.skill_count;
      const bImportance = b.reasoner_count + b.skill_count;
      
      if (aImportance !== bImportance) {
        return bImportance - aImportance;
      }
      
      // Finally by recency (most recent first) - with safe date handling
      const aTime = a.last_heartbeat ? new Date(a.last_heartbeat).getTime() : 0;
      const bTime = b.last_heartbeat ? new Date(b.last_heartbeat).getTime() : 0;
      
      // Handle invalid dates
      const aValidTime = isNaN(aTime) ? 0 : aTime;
      const bValidTime = isNaN(bTime) ? 0 : bTime;
      
      return bValidTime - aValidTime;
    };

    // Sort all subcategories
    categorized.running.fresh.sort(sortByImportance);
    categorized.running.stale.sort(sortByImportance);
    categorized.running.veryStale.sort(sortByImportance);
    categorized.running.degraded.sort(sortByImportance);
    categorized.offline.starting.sort(sortByImportance);
    categorized.offline.down.sort(sortByImportance);

    return categorized;
  }, [nodes]);

  // Filter nodes based on search query
  const filteredNodes = useMemo(() => {
    if (!searchQuery) return categorizedNodes;

    const filterBySearch = (nodeList: AgentNodeSummary[]) => {
      return nodeList.filter(node => {
        const query = searchQuery.toLowerCase();
        return (
          node.id.toLowerCase().includes(query) ||
          node.team_id.toLowerCase().includes(query) ||
          node.version.toLowerCase().includes(query)
        );
      });
    };

    return {
      running: {
        fresh: filterBySearch(categorizedNodes.running.fresh),
        stale: filterBySearch(categorizedNodes.running.stale),
        veryStale: filterBySearch(categorizedNodes.running.veryStale),
        degraded: filterBySearch(categorizedNodes.running.degraded),
      },
      offline: {
        starting: filterBySearch(categorizedNodes.offline.starting),
        down: filterBySearch(categorizedNodes.offline.down),
      }
    };
  }, [categorizedNodes, searchQuery]);

  // Enhanced defensive checks for Object.values() calls to prevent Object.entries() errors
  const runningNodes = (filteredNodes?.running && typeof filteredNodes.running === 'object' && filteredNodes.running !== null)
    ? Object.values(filteredNodes.running).flat()
    : [];
  const offlineNodes = (filteredNodes?.offline && typeof filteredNodes.offline === 'object' && filteredNodes.offline !== null)
    ? Object.values(filteredNodes.offline).flat()
    : [];
  const { fresh = [], stale = [], veryStale = [], degraded = [] } = filteredNodes?.running || {};
  const { starting = [], down = [] } = filteredNodes?.offline || {};

  // Loading skeleton
  if (isLoading && nodes.length === 0) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className={sectionContainerClasses}>
            <Skeleton className="h-16 w-full rounded-lg" />
          </div>
        ))}
      </div>
    );
  }

  // Empty state
  if (nodes.length === 0) {
    return (
      <Empty className="min-h-[240px]">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <CloudOffline className="h-8 w-8" />
          </EmptyMedia>
          <EmptyTitle>No agent nodes</EmptyTitle>
          <EmptyDescription>
            {searchQuery
              ? `No nodes found matching "${searchQuery}". Adjust the query or reset your filters.`
              : "Register an agent node to begin monitoring its status and workloads."
            }
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  // Search results view (flat list)
  if (searchQuery) {
    const allResults = [...runningNodes, ...offlineNodes];
    return (
      <div className="space-y-3">
        {allResults.map((node) => (
          <NodeCard 
            key={node.id} 
            nodeSummary={node} 
            searchQuery={searchQuery}
            density={density}
          />
        ))}
      </div>
    );
  }

  // Grouped view (default)
  return (
    <div className="space-y-6">
      {/* Running Nodes Section */}
      {runningNodes.length > 0 && (
        <div className="space-y-6">
          {/* Fresh Running Nodes */}
          {fresh.length > 0 && (
            <div className={sectionContainerClasses}>
              <div className="flex items-center gap-3 border-b border-border/60 pb-2">
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full",
                    readyPresentation.theme.indicatorClass
                  )}
                />
                <h3 className="text-sm font-semibold text-foreground">
                  Ready &amp; responsive
                </h3>
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-body-small font-medium",
                    readyPresentation.theme.pillClass
                  )}
                >
                  {fresh.length} fresh
                </span>
              </div>

              <div className="space-y-3 pt-3">
                {fresh.slice(0, visibleFreshCount).map((node) => (
                  <NodeCard key={node.id} nodeSummary={node} density={density} />
                ))}
              </div>

              {fresh.length > visibleFreshCount && (
                <div className="pt-2 text-center">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setVisibleFreshCount((prev) => prev + 20)}
                  >
                    Show {Math.min(20, fresh.length - visibleFreshCount)} more running nodes
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Stale Running Nodes */}
          {stale.length > 0 && (
            <div className={sectionContainerClasses}>
              <div className="flex items-center gap-3 border-b border-border/60 pb-2">
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full",
                    stalePresentation.theme.indicatorClass
                  )}
                />
                <h3 className="text-sm font-semibold text-foreground">
                  Running (Stale)
                </h3>
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-body-small font-medium",
                    stalePresentation.theme.pillClass
                  )}
                >
                  {stale.length} stale
                </span>
              </div>

              <div className="space-y-3 pt-3">
                {stale.map((node) => (
                  <NodeCard key={node.id} nodeSummary={node} density={density} />
                ))}
              </div>
            </div>
          )}

          {/* Very Stale Running Nodes */}
          {veryStale.length > 0 && (
            <div className={sectionContainerClasses}>
              <div className="flex items-center gap-3 border-b border-border/60 pb-2">
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full",
                    veryStalePresentation.theme.indicatorClass
                  )}
                />
                <h3 className="text-sm font-semibold text-foreground">
                  Running (Very Stale)
                </h3>
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-body-small font-medium",
                    veryStalePresentation.theme.pillClass
                  )}
                >
                  {veryStale.length} very stale
                </span>
              </div>

              <div className="space-y-3 pt-3">
                {veryStale.map((node) => (
                  <NodeCard key={node.id} nodeSummary={node} density={density} />
                ))}
              </div>
            </div>
          )}

          {/* Degraded Running Nodes */}
          {degraded.length > 0 && (
            <div className={sectionContainerClasses}>
              <div className="flex items-center gap-3 border-b border-border/60 pb-2">
                <WarningFilled
                  className={cn("h-4 w-4", degradedPresentation.theme.textClass)}
                />
                <h3 className="text-sm font-semibold text-foreground">
                  Running (Degraded)
                </h3>
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-body-small font-medium",
                    degradedPresentation.theme.pillClass
                  )}
                >
                  {degraded.length} degraded
                </span>
              </div>

              <div className="space-y-3 pt-3">
                {degraded.map((node) => (
                  <NodeCard key={node.id} nodeSummary={node} density={density} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Offline Nodes Section */}
      {offlineNodes.length > 0 && (
        <div className="space-y-6">
          {/* Starting Nodes */}
          {starting.length > 0 && (
            <div className={sectionContainerClasses}>
              <div className="flex items-center gap-3 border-b border-border/60 pb-2">
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full",
                    startingPresentation.theme.indicatorClass
                  )}
                />
                <h3 className="text-sm font-semibold text-foreground">
                  Starting
                </h3>
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-body-small font-medium",
                    startingPresentation.theme.pillClass
                  )}
                >
                  {starting.length} initializing
                </span>
              </div>

              <div className="space-y-3 pt-3">
                {starting.map((node) => (
                  <NodeCard key={node.id} nodeSummary={node} density={density} />
                ))}
              </div>
            </div>
          )}

          {/* Down Nodes - Collapsed by default, more faded */}
          {down.length > 0 && (
            <div className={sectionContainerClasses}>
              <Collapsible open={showInactive} onOpenChange={setShowInactive}>
                <CollapsibleTrigger asChild>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-3 text-left"
                  >
                    <div className="flex items-center gap-3">
                      <ErrorFilled
                        className={cn("h-4 w-4", offlinePresentation.theme.textClass)}
                      />
                      <span className="text-sm font-semibold text-foreground">
                        Offline
                      </span>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-body-small font-medium",
                          offlinePresentation.theme.pillClass
                        )}
                      >
                        {down.length} offline
                      </span>
                    </div>
                    {showInactive ? (
                      <ChevronUp className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                </CollapsibleTrigger>

                <CollapsibleContent className="mt-3 space-y-3">
                  {down.slice(0, visibleInactiveCount).map((node) => (
                    <NodeCard key={node.id} nodeSummary={node} density={density} />
                  ))}

                  {down.length > visibleInactiveCount && (
                    <div className="pt-2 text-center">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setVisibleInactiveCount((prev) => prev + 20)}
                      >
                        Show {Math.min(20, down.length - visibleInactiveCount)} more offline nodes
                      </Button>
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>
            </div>
          )}
        </div>
      )}

      {/* All sections empty */}
      {runningNodes.length === 0 && offlineNodes.length === 0 && (
        <div className="text-center py-12">
          <div className="mx-auto w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
            <div className="w-6 h-6 rounded-full bg-muted-foreground/20" />
          </div>
          <h3 className="text-heading-3 mb-2">No Nodes Found</h3>
          <p className="text-muted-foreground">
            Try adjusting your search query or check if nodes are properly registered.
          </p>
        </div>
      )}
    </div>
  );
}
