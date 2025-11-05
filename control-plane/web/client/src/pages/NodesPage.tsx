import { useState, useEffect, useCallback, useMemo } from "react";
import type { ReactNode } from "react";
import type {
  AgentNodeSummary,
  AgentStatus,
  HealthStatus,
  LifecycleStatus,
} from "../types/agentfield";
import { getNodesSummary } from "../services/api";
import { useNodeEventsSSE, useUnifiedStatusSSE } from "../hooks/useSSE";
import { StatusRefreshButton } from "../components/status";
import { SearchBar } from "@/components/ui/SearchBar";
import { NodesStatusSummary } from "../components/NodesStatusSummary";
import { NodesVirtualList } from "../components/NodesVirtualList";
import { DensityToggle, type DensityMode } from "../components/DensityToggle";
import { ServerlessRegistrationModal } from "../components/ServerlessRegistrationModal";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { ButtonProps } from "@/components/ui/button";
import { PageHeader } from "../components/PageHeader";
import { summarizeNodeStatuses } from "@/utils/node-status";
import {
  ArrowClockwise,
  Plus,
  Terminal,
  WifiHigh,
  WifiSlash,
} from "@/components/ui/icon-bridge";

type HeaderAction = {
  label: string;
  onClick: () => void;
  variant?: ButtonProps["variant"];
  size?: ButtonProps["size"];
  className?: string;
  icon?: ReactNode;
  disabled?: boolean;
};

const formatRelativeTime = (date: Date) => {
  const diffMs = Date.now() - date.getTime();
  if (diffMs < 0) return "just now";

  const diffSeconds = Math.floor(diffMs / 1000);
  if (diffSeconds < 60) return `${diffSeconds}s ago`;

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
};

export function NodesPage() {
  const [nodes, setNodes] = useState<AgentNodeSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [density, setDensity] = useState<DensityMode>("comfortable");
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [showServerlessModal, setShowServerlessModal] = useState(false);

  // Console log to verify we're using the updated build
  console.log(
    "🚀 NodesPage: Component loaded with SSE fixes - Build timestamp:",
    new Date().toISOString()
  );

  // Use the new SSE hook for real-time updates
  const sseHook = useNodeEventsSSE();
  const { connected, reconnecting, latestEvent, reconnect } = sseHook;

  // Use unified status SSE for enhanced status updates
  const unifiedStatusSSE = useUnifiedStatusSSE();
  const { latestEvent: unifiedStatusEvent } = unifiedStatusSSE;

  // Note: Optimistic status updates are handled by the StatusRefreshButton component

  const normalizeHealthStatus = useCallback(
    (value: string | undefined, fallback: HealthStatus): HealthStatus => {
      const allowed: HealthStatus[] = [
        "starting",
        "ready",
        "degraded",
        "offline",
        "active",
        "inactive",
        "unknown",
      ];
      if (value && allowed.includes(value as HealthStatus)) {
        return value as HealthStatus;
      }
      return fallback;
    },
    []
  );

  const normalizeLifecycleStatus = useCallback(
    (value: string | undefined, fallback: LifecycleStatus): LifecycleStatus => {
      const allowed: LifecycleStatus[] = [
        "starting",
        "ready",
        "degraded",
        "offline",
        "running",
        "stopped",
        "error",
        "unknown",
      ];
      if (value && allowed.includes(value as LifecycleStatus)) {
        return value as LifecycleStatus;
      }
      return fallback;
    },
    []
  );

  const fetchNodes = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getNodesSummary();
      setNodes(data.nodes);
      setLastRefresh(new Date());
    } catch (err) {
      console.error("Failed to load nodes summary:", err);
      setError(
        "Failed to load agent nodes. Please ensure the AgentField server is running and accessible."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle real-time events
  useEffect(() => {
    if (!latestEvent) return;

    console.log(
      "🔄 Frontend: Received node SSE event:",
      latestEvent.type,
      latestEvent
    );

    // The new dedicated node event system sends events directly with node data in the 'data' field
    // latestEvent.data contains the NodeEvent structure: { type, node_id, status, timestamp, data }
    const eventData = latestEvent.data;

    // Enhanced defensive checks to prevent Object.entries() errors
    if (!eventData || typeof eventData !== "object" || eventData === null) {
      console.warn("🚨 Frontend: Invalid event data received:", eventData);
      return;
    }

    const nodeData = eventData?.data || eventData; // Extract node data from the nested structure

    console.log("🔍 Frontend: Event data:", eventData);
    console.log("🔍 Frontend: Event type:", latestEvent.type);
    console.log("🔍 Frontend: Node data:", nodeData);
    console.log("🔍 Frontend: Event structure check:", {
      hasEventData: !!eventData,
      hasNestedData: !!eventData?.data,
      eventDataType: eventData?.type,
      nodeId: eventData?.node_id,
    });

    switch (latestEvent.type) {
      case "node_registered":
        console.log("🆕 Frontend: Processing node_registered event");
        if (nodeData) {
          setNodes((prevNodes) => {
            const newNode = nodeData as AgentNodeSummary;
            console.log("🆕 Frontend: New node data:", newNode);
            const existingIndex = prevNodes.findIndex(
              (node) => node.id === newNode.id
            );
            if (existingIndex > -1) {
              // Update existing node
              const updatedNodes = [...prevNodes];
              updatedNodes[existingIndex] = newNode;
              console.log("🔄 Frontend: Updated existing node:", newNode.id);
              return updatedNodes;
            }
            // Add new node
            console.log("➕ Frontend: Added new node:", newNode.id);
            return [...prevNodes, newNode];
          });
        }
        break;

      case "node_online":
      case "node_offline":
      case "node_status_updated":
      case "node_status_changed":
      case "node_health_changed":
        console.log(`🔄 Frontend: Processing ${latestEvent.type} event`);
        if (nodeData) {
          setNodes((prevNodes) => {
            const updatedNode = nodeData as AgentNodeSummary;
            console.log("🔄 Frontend: Updated node data:", {
              id: updatedNode.id,
              health_status: updatedNode.health_status,
              lifecycle_status: updatedNode.lifecycle_status,
              last_heartbeat: updatedNode.last_heartbeat,
            });
            const newNodes = prevNodes.map((node) =>
              node.id === updatedNode.id ? updatedNode : node
            );
            console.log("🔄 Frontend: Node state updated for:", updatedNode.id);
            return newNodes;
          });
        }
        break;

      case "node_removed":
        if (eventData && typeof eventData === "object" && "id" in eventData) {
          setNodes((prevNodes) => {
            const nodeId = (eventData as { id: string }).id;
            return prevNodes.filter((node) => node.id !== nodeId);
          });
        }
        break;

      case "mcp_health_changed":
        // Handle MCP health changes
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const mcpData = eventData as { node_id: string; mcp_summary: any };
          setNodes((prevNodes) =>
            prevNodes.map((node) =>
              node.id === mcpData.node_id
                ? { ...node, mcp_summary: mcpData.mcp_summary }
                : node
            )
          );
        }
        break;

      // New unified status events
      case "node_unified_status_changed":
        console.log(
          "🔄 Frontend: Processing node_unified_status_changed event"
        );
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const statusData = eventData as {
            node_id: string;
            new_status: AgentStatus;
          };
          setNodes((prevNodes) =>
            prevNodes.map((node) => {
              if (node.id === statusData.node_id) {
                // Update node with new unified status
                return {
                  ...node,
                  health_status: normalizeHealthStatus(
                    statusData.new_status.health_status,
                    node.health_status
                  ),
                  lifecycle_status: normalizeLifecycleStatus(
                    statusData.new_status.lifecycle_status,
                    node.lifecycle_status
                  ),
                  last_heartbeat:
                    statusData.new_status.last_seen ?? node.last_heartbeat,
                };
              }
              return node;
            })
          );
        }
        break;

      case "node_state_transition":
        console.log("🔄 Frontend: Processing node_state_transition event");
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const transitionData = eventData as {
            node_id: string;
            to_state: string;
          };
          // Show optimistic update during transition
          setNodes((prevNodes) =>
            prevNodes.map((node) => {
              if (node.id === transitionData.node_id) {
                return {
                  ...node,
                  lifecycle_status: normalizeLifecycleStatus(
                    transitionData.to_state,
                    node.lifecycle_status
                  ),
                };
              }
              return node;
            })
          );
        }
        break;

      case "node_status_refreshed":
        console.log("🔄 Frontend: Processing node_status_refreshed event");
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const refreshData = eventData as {
            node_id: string;
            status: AgentStatus;
          };
          setNodes((prevNodes) =>
            prevNodes.map((node) => {
              if (node.id === refreshData.node_id) {
                return {
                  ...node,
                  health_status: normalizeHealthStatus(
                    refreshData.status.health_status,
                    node.health_status
                  ),
                  lifecycle_status: normalizeLifecycleStatus(
                    refreshData.status.lifecycle_status,
                    node.lifecycle_status
                  ),
                  last_heartbeat:
                    refreshData.status.last_seen ?? node.last_heartbeat,
                };
              }
              return node;
            })
          );
        }
        break;

      case "bulk_status_update":
        console.log("🔄 Frontend: Processing bulk_status_update event");
        // Trigger a full refresh after bulk updates
        fetchNodes();
        break;

      case "connected":
      case "heartbeat":
      case "node_heartbeat":
        // Handle connection and heartbeat events - no action needed
        console.log(
          "🔗 Frontend: Connection event received:",
          latestEvent.type
        );
        break;

      default:
        console.log("Unhandled event type:", latestEvent.type);
    }
  }, [latestEvent, normalizeHealthStatus, normalizeLifecycleStatus]);

  // Handle unified status events
  useEffect(() => {
    if (!unifiedStatusEvent) return;

    console.log(
      "🔄 Frontend: Received unified status event:",
      unifiedStatusEvent.type,
      unifiedStatusEvent
    );

    const eventData = unifiedStatusEvent.data;

    switch (unifiedStatusEvent.type) {
      case "node_unified_status_changed":
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const statusData = eventData as {
            node_id: string;
            new_status: AgentStatus;
          };
          setNodes((prevNodes) =>
            prevNodes.map((node) => {
              if (node.id === statusData.node_id) {
                return {
                  ...node,
                  health_status: normalizeHealthStatus(
                    statusData.new_status.health_status,
                    node.health_status
                  ),
                  lifecycle_status: normalizeLifecycleStatus(
                    statusData.new_status.lifecycle_status,
                    node.lifecycle_status
                  ),
                  last_heartbeat:
                    statusData.new_status.last_seen ?? node.last_heartbeat,
                };
              }
              return node;
            })
          );
        }
        break;

      case "node_state_transition":
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const transitionData = eventData as {
            node_id: string;
            to_state: string;
          };
          setNodes((prevNodes) =>
            prevNodes.map((node) => {
              if (node.id === transitionData.node_id) {
                return {
                  ...node,
                  lifecycle_status: normalizeLifecycleStatus(
                    transitionData.to_state,
                    node.lifecycle_status
                  ),
                };
              }
              return node;
            })
          );
        }
        break;

      case "bulk_status_update":
        // Trigger a full refresh after bulk updates
        fetchNodes();
        break;

      default:
        console.log(
          "Unhandled unified status event type:",
          unifiedStatusEvent.type
        );
    }
  }, [unifiedStatusEvent, normalizeHealthStatus, normalizeLifecycleStatus]);

  // Initial load - trigger status refresh on page load
  useEffect(() => {
    fetchNodes();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  // Handle bulk status refresh
  const handleBulkRefresh = (
    status: AgentStatus | Record<string, AgentStatus>
  ) => {
    if ("state" in status) {
      // Single status - shouldn't happen in bulk refresh but handle it
      console.warn("Received single status in bulk refresh handler");
      return;
    }

    // Multiple statuses
    const statuses = status as Record<string, AgentStatus>;
    setNodes((prevNodes) =>
      prevNodes.map((node) => {
        const newStatus = statuses[node.id];
        if (newStatus) {
          return {
            ...node,
            health_status: normalizeHealthStatus(
              newStatus.health_status,
              node.health_status
            ),
            lifecycle_status: normalizeLifecycleStatus(
              newStatus.lifecycle_status,
              node.lifecycle_status
            ),
            last_heartbeat: newStatus.last_seen ?? node.last_heartbeat,
          };
        }
        return node;
      })
    );
    setLastRefresh(new Date());
  };

  const handleRefreshError = (error: string) => {
    console.error("Status refresh failed:", error);
    setError(error);
  };

  // Filter nodes based on search query
  const filteredNodes = searchQuery
    ? nodes.filter((node) => {
        const query = searchQuery.toLowerCase();
        return (
          node.id.toLowerCase().includes(query) ||
          node.team_id.toLowerCase().includes(query) ||
          node.version.toLowerCase().includes(query)
        );
      })
    : nodes;

  const summary = useMemo(
    () => summarizeNodeStatuses(searchQuery ? filteredNodes : nodes),
    [filteredNodes, nodes, searchQuery]
  );

  const headerSubtitle = searchQuery
    ? `Showing ${filteredNodes.length} result${
        filteredNodes.length === 1 ? "" : "s"
      } for "${searchQuery}"`
    : "Monitor and manage your AI agent nodes in the AgentField orchestration platform.";

  const connectionBadgeVariant = connected
    ? "success"
    : reconnecting
      ? "pending"
      : "failed";
  const ConnectionIcon = connected ? WifiHigh : WifiSlash;
  const connectionLabel = connected
    ? "Live updates"
    : reconnecting
      ? "Reconnecting…"
      : "Disconnected";

  const pageHeaderActions: HeaderAction[] = [];

  if (!connected) {
    pageHeaderActions.push({
      label: reconnecting ? "Reconnecting…" : "Reconnect",
      onClick: reconnect,
      icon: <ArrowClockwise className="h-4 w-4" />,
      variant: "outline",
      disabled: reconnecting,
    });
  }

  pageHeaderActions.push({
    label: "Add Serverless Agent",
    onClick: () => setShowServerlessModal(true),
    icon: <Plus className="h-4 w-4" />,
    variant: "default",
  });

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isRefreshCombo =
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === "r";

      if (isRefreshCombo) {
        event.preventDefault();
        if (connected) {
          fetchNodes();
        } else {
          reconnect();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [connected, reconnect]);

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
        <PageHeader
          title="Agent Nodes"
          description={headerSubtitle}
          actions={pageHeaderActions}
          aside={
            <div className="flex flex-wrap items-center gap-3 sm:justify-end">
              <Badge variant="count" size="sm">
                {summary.total} total
              </Badge>
              <Badge
                variant={connectionBadgeVariant}
                size="sm"
                showIcon={false}
                className="flex items-center gap-1 rounded-full px-3"
              >
                <ConnectionIcon className="h-3.5 w-3.5" />
                {connectionLabel}
              </Badge>
              {connected && lastRefresh && (
                <Badge variant="pill" className="hidden md:flex">
                  Updated {formatRelativeTime(lastRefresh)}
                </Badge>
              )}
              {connected && (
                <StatusRefreshButton
                  nodeIds={nodes.map((node) => node.id)}
                  onRefresh={handleBulkRefresh}
                  onError={handleRefreshError}
                  disabled={isLoading}
                  size="sm"
                  variant="ghost"
                  showLabel
                  showLastVerified
                  lastVerified={lastRefresh.toISOString()}
                  className="hidden md:flex"
                />
              )}
            </div>
          }
        />

        <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search nodes, teams, reasoners, or skills..."
              wrapperClassName="w-full lg:max-w-md"
              inputClassName="border-border bg-background focus-visible:ring-0 focus-visible:outline-none"
            />
            <div className="flex items-center gap-4">
              <DensityToggle density={density} onChange={setDensity} />
              {connected && (
                <StatusRefreshButton
                  nodeIds={nodes.map((node) => node.id)}
                  onRefresh={handleBulkRefresh}
                  onError={handleRefreshError}
                  disabled={isLoading}
                  size="sm"
                  variant="ghost"
                  showLabel={false}
                  className="md:hidden"
                />
              )}
            </div>
          </div>

          <NodesStatusSummary nodes={filteredNodes} searchQuery={searchQuery} />

          {error && (
            <Alert variant="destructive">
              <Terminal className="h-4 w-4" />
              <AlertTitle>Connection Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <NodesVirtualList
              nodes={filteredNodes}
              searchQuery={searchQuery}
              isLoading={isLoading}
              density={density}
            />
          </div>

          {!isLoading && !error && nodes.length > 0 && (
            <div className="py-2 text-center text-body-small">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      <ServerlessRegistrationModal
        isOpen={showServerlessModal}
        onClose={() => setShowServerlessModal(false)}
        onSuccess={(nodeId) => {
          console.log("✅ Serverless agent registered:", nodeId);
          fetchNodes();
        }}
      />
    </>
  );
}
