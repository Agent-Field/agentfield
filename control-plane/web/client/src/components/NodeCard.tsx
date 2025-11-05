import { Badge } from "@/components/ui/badge";
import {
  ChevronRight,
  Code,
  Tools,
  Clock,
  Identification,
  Layers,
  Flash,
} from "@/components/ui/icon-bridge";
import { memo, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { startAgent, stopAgent, reconcileAgent } from "../services/configurationApi";
import { AgentControlButton, type AgentState } from "@/components/ui/AgentControlButton";
import { useDIDStatus } from "../hooks/useDIDInfo";
import { cn } from "../lib/utils";
import type { AgentNodeSummary } from "../types/haxen";
import type { DensityMode } from "./DensityToggle";
import { CompositeDIDStatus } from "./did/DIDStatusBadge";
import { DIDIdentityBadge } from "./did/DIDDisplay";
import type { MCPHealthStatus } from "./mcp/MCPHealthIndicator";
import { MCPHealthDot, MCPHealthIndicator } from "./mcp/MCPHealthIndicator";
import { getNodeStatusPresentation } from "@/utils/node-status";

interface NodeCardProps {
  nodeSummary: AgentNodeSummary;
  searchQuery?: string;
  density?: DensityMode;
}

/**
 * Enhanced NodeCard component for direct navigation to NodeDetailPage with MCP integration.
 * Displays agent node summary with health status, capabilities, and MCP server information.
 * Optimized for navigation-focused interaction with comprehensive accessibility support.
 *
 * @param {NodeCardProps} props - The props for the NodeCard component.
 * @param {AgentNodeSummary} props.nodeSummary - Summary data for the agent node.
 * @param {string} [props.searchQuery] - Optional search query to highlight text.
 * @param {DensityMode} [props.density='comfortable'] - Visual density mode for the card.
 */
const NodeCard = memo(
  ({ nodeSummary, searchQuery, density = "comfortable" }: NodeCardProps) => {
    const navigate = useNavigate();
    const [currentTime, setCurrentTime] = useState(new Date());
    const [isHovered, setIsHovered] = useState(false);
    const [actionLoading, setActionLoading] = useState<string | null>(null);

    // Console log to verify we're using the updated build and check node data
    console.log('🎯 NodeCard: Rendering node with defensive checks - Node ID:', nodeSummary?.id, 'MCP Summary:', nodeSummary?.mcp_summary);

    // Get DID status for this node
    const { status: didStatus } = useDIDStatus(nodeSummary.id);

    // Auto-refresh timer to update timestamps every second
    useEffect(() => {
      const interval = setInterval(() => {
        setCurrentTime(new Date());
      }, 1000);

      return () => clearInterval(interval);
    }, []);

    const now = currentTime;
    const lastHeartbeat = nodeSummary.last_heartbeat
      ? new Date(nodeSummary.last_heartbeat)
      : null;
    const minutesSinceHeartbeat = lastHeartbeat
      ? (now.getTime() - lastHeartbeat.getTime()) / (1000 * 60)
      : Infinity;

    // Primary categorization: Running vs Offline (matches NodesStatusSummary)
    // Use optional chaining and fallbacks for safety
    const lifecycleStatus = nodeSummary.lifecycle_status ?? 'unknown';
    const healthStatus = nodeSummary.health_status ?? 'unknown';
    const statusPresentation = getNodeStatusPresentation(
      lifecycleStatus,
      healthStatus
    );
    const isRunning =
      lifecycleStatus === "ready" ||
      lifecycleStatus === "degraded";
    const isOffline = statusPresentation.kind === "offline" || statusPresentation.kind === "error" || !isRunning;

    // Secondary health indicators for running nodes
    const isDegraded = lifecycleStatus === "degraded";
    const isStarting = lifecycleStatus === "starting";
    const isStale =
      isRunning && minutesSinceHeartbeat > 2 && minutesSinceHeartbeat <= 5;
    const isVeryStale = isRunning && minutesSinceHeartbeat > 5;

    // Calculate importance score for visual weight
    const reasonerCount = nodeSummary.reasoner_count ?? 0;
    const skillCount = nodeSummary.skill_count ?? 0;
    const mcpSummary = nodeSummary.mcp_summary;
    const importanceScore = reasonerCount + skillCount;
    const isHighImportance = importanceScore >= 8;

    // Convert MCP service status to MCPHealthStatus with defensive checks
    const getMCPHealthStatus = (): MCPHealthStatus => {
      // Comprehensive null/undefined checks to prevent Object.entries() errors
      if (!nodeSummary?.mcp_summary ||
          typeof nodeSummary.mcp_summary !== 'object' ||
          nodeSummary.mcp_summary === null) {
        return "unknown";
      }

      // Additional safety check for service_status property
      const serviceStatus = nodeSummary.mcp_summary.service_status;
      if (!serviceStatus || typeof serviceStatus !== 'string') {
        return "unknown";
      }

      switch (serviceStatus) {
        case "ready":
          return "running";
        case "degraded":
          return "error";
        case "unavailable":
          return "stopped";
        default:
          return "unknown";
      }
    };

    // Format time ago with enhanced precision
    const formatTimeAgo = (date: Date | null) => {
      if (!date) return "Never";
      const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
      if (seconds < 60) return `${seconds}s ago`;
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return `${minutes}m ago`;
      const hours = Math.floor(minutes / 60);
      if (hours < 24) return `${hours}h ago`;
      const days = Math.floor(hours / 24);
      return `${days}d ago`;
    };

    // Highlight search matches
    const highlightText = (text: string) => {
      if (!searchQuery) return text;
      const regex = new RegExp(`(${searchQuery})`, "gi");
      const parts = text.split(regex);
      return parts.map((part, index) =>
        regex.test(part) ? (
          <mark
            key={index}
            className="bg-yellow-200 dark:bg-yellow-800 px-1 rounded"
          >
            {part}
          </mark>
        ) : (
          part
        )
      );
    };

    // Status text for accessibility
    const getStatusText = () => statusPresentation.label;

    // Handle navigation
    const handleNavigation = () => {
      navigate(`/nodes/${nodeSummary.id}`);
    };

    // Handle keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleNavigation();
      }
    };

    // Handle start/stop actions
    const handleStartAgent = async () => {
      setActionLoading('start');
      try {
        await startAgent(nodeSummary.id);
        console.log(`Agent ${nodeSummary.id} started successfully`);
      } catch (error) {
        console.error(`Failed to start agent ${nodeSummary.id}:`, error);
      } finally {
        setActionLoading(null);
      }
    };

    const handleStopAgent = async () => {
      setActionLoading('stop');
      try {
        await stopAgent(nodeSummary.id);
        console.log(`Agent ${nodeSummary.id} stopped successfully`);
      } catch (error) {
        console.error(`Failed to stop agent ${nodeSummary.id}:`, error);
      } finally {
        setActionLoading(null);
      }
    };

    const handleReconcileAgent = async () => {
      setActionLoading('reconcile');
      try {
        await reconcileAgent(nodeSummary.id);
        console.log(`Agent ${nodeSummary.id} reconciled successfully`);
      } catch (error) {
        console.error(`Failed to reconcile agent ${nodeSummary.id}:`, error);
      } finally {
        setActionLoading(null);
      }
    };

    // Determine current agent state for the control button
    const getAgentState = (): AgentState => {
      if (actionLoading === 'start') return 'starting';
      if (actionLoading === 'stop') return 'stopping';
      if (actionLoading === 'reconcile') return 'reconciling';

      // Check multiple sources for running state (more robust detection)
      const isRunning =
        lifecycleStatus === 'ready' ||
        lifecycleStatus === 'degraded' ||
        mcpSummary?.service_status === 'ready' ||
        (mcpSummary?.total_servers ?? 0) > 0;

      if (isRunning) {
        // Check for error/degraded states
        if (lifecycleStatus === 'degraded' || mcpSummary?.service_status === 'degraded') {
          return 'error';
        }
        return 'running';
      }

      return 'stopped';
    };

    // Unified handler for agent control actions
    const handleAgentAction = async (action: 'start' | 'stop' | 'reconcile') => {
      try {
        switch (action) {
          case 'start':
            await handleStartAgent();
            break;
          case 'stop':
            await handleStopAgent();
            break;
          case 'reconcile':
            await handleReconcileAgent();
            break;
        }

        // Force a re-render by updating current time (triggers useEffect)
        setTimeout(() => {
          setCurrentTime(new Date());
        }, 1000);

      } catch (error) {
        // Error handling is already done in individual handlers
        console.error(`Failed to ${action} agent:`, error);
      }
    };

    const teamId = nodeSummary.team_id || "unknown";
    const deploymentType = nodeSummary.deployment_type || null;
    const totalMcpServers = mcpSummary?.total_servers ?? 0;
    const runningMcpServers = mcpSummary?.running_servers ?? 0;
    const totalMcpTools = mcpSummary?.total_tools ?? 0;
    const hasMcpIssues = Boolean(mcpSummary?.has_issues);
    const capabilitiesAvailable = Boolean(mcpSummary?.capabilities_available);

    const containerPadding =
      density === "compact"
        ? "px-3 py-3"
        : density === "spacious"
          ? "px-5 py-5"
          : "px-4 py-4";

    const containerClasses = cn(
      "group relative flex cursor-pointer flex-col rounded-xl border border-border/60 bg-background/95 shadow-sm transition-all duration-200",
      containerPadding,
      !isOffline && "hover:border-primary/50 hover:shadow-lg",
      isOffline && "opacity-70 hover:opacity-90",
      isHighImportance && "border-primary/40",
      isHovered && !isOffline && "shadow-lg"
    );

    const statusDotClass = cn(
      "h-2.5 w-2.5 rounded-full",
      statusPresentation.theme.indicatorClass,
      statusPresentation.shouldPulse && "animate-pulse"
    );

    const highlightHeartbeat =
      isStale || isVeryStale || isDegraded || isStarting || isOffline;
    const heartbeatClass = cn(
      "text-xs flex items-center gap-1",
      highlightHeartbeat
        ? statusPresentation.theme.textClass
        : "text-muted-foreground"
    );

    return (
      <div
        className={containerClasses}
        onClick={handleNavigation}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        role="button"
        tabIndex={0}
        aria-label={`Navigate to details for node ${
          nodeSummary.id
        }. Status: ${getStatusText()}. ${
          nodeSummary.reasoner_count
        } reasoners, ${nodeSummary.skill_count} skills.`}
        onKeyDown={handleKeyDown}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className={statusDotClass} />
              <span
                className={cn(
                  "text-xs font-medium uppercase tracking-wide",
                  statusPresentation.theme.textClass
                )}
              >
                {statusPresentation.label}
              </span>
              <h3
                className={cn(
                  "truncate text-sm font-semibold text-foreground",
                  "max-w-xs md:max-w-md",
                  isOffline && "text-muted-foreground"
                )}
              >
                {highlightText(nodeSummary.id)}
              </h3>
              <Badge variant="outline" className="h-6 rounded-full px-2 text-body-small">
                v{nodeSummary.version}
              </Badge>
              {deploymentType === "serverless" && (
                <Badge
                  variant="outline"
                  className="h-6 rounded-full px-2 text-body-small flex items-center gap-1"
                >
                  <Flash className="h-3.5 w-3.5" />
                  Serverless
                </Badge>
              )}
              {isHighImportance && (
                <Badge variant="outline" className="h-6 rounded-full px-2 text-body-small">
                  High capability
                </Badge>
              )}
              {hasMcpIssues && (
                <Badge variant="destructive" className="h-6 rounded-full px-2 text-body-small">
                  Issues detected
                </Badge>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <span className={heartbeatClass}>
                <Clock className="h-3.5 w-3.5" />
                {lastHeartbeat ? formatTimeAgo(lastHeartbeat) : "No heartbeat"}
              </span>
              {didStatus && didStatus.has_did && (
                <div className="flex items-center gap-1 text-body-small">
                  <Identification className="h-3.5 w-3.5" />
                  <CompositeDIDStatus
                    status={didStatus.did_status}
                    reasonerCount={didStatus.reasoner_count}
                    skillCount={didStatus.skill_count}
                    compact={true}
                    className="text-xs"
                  />
                </div>
              )}
              {mcpSummary && (
                <div className="flex items-center gap-2 text-body-small">
                  <MCPHealthDot
                    status={getMCPHealthStatus()}
                    size="sm"
                    className="flex-shrink-0"
                  />
                  <span>
                    {runningMcpServers}/{totalMcpServers} MCP servers
                  </span>
                  {totalMcpTools > 0 && (
                    <span className="text-muted-foreground/80">
                      ({totalMcpTools} tools)
                    </span>
                  )}
                  {capabilitiesAvailable && (
                    <span className="rounded-full bg-status-success-bg px-2 py-0.5 text-[10px] font-medium text-status-success">
                      Capabilities ready
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 pl-2">
            <div onClick={(e) => e.stopPropagation()}>
              <AgentControlButton
                agentId={nodeSummary.id}
                currentState={getAgentState()}
                onToggle={handleAgentAction}
                size="sm"
                variant="minimal"
                className="shadow-none"
              />
            </div>
            <ChevronRight
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform duration-200",
                isHovered && "translate-x-1 text-foreground"
              )}
            />
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-body-small">
          <div className="flex items-center gap-1.5">
            <Code className="h-4 w-4" />
            <span>
              {reasonerCount} reasoner{reasonerCount === 1 ? "" : "s"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Tools className="h-4 w-4" />
            <span>
              {skillCount} skill{skillCount === 1 ? "" : "s"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Layers className="h-4 w-4" />
            <span>Team {highlightText(teamId)}</span>
          </div>
          {mcpSummary && (
            <MCPHealthIndicator
              status={getMCPHealthStatus()}
              size="sm"
              showText={true}
              className="flex-shrink-0"
            />
          )}
          {didStatus && didStatus.has_did && (
            <DIDIdentityBadge
              nodeId={nodeSummary.id}
              showDID={true}
              className={cn(
                "text-xs opacity-70 transition-opacity",
                isHovered ? "opacity-100" : "opacity-70"
              )}
            />
          )}
        </div>
      </div>
    );
  }
);

NodeCard.displayName = "NodeCard";

export { NodeCard };
