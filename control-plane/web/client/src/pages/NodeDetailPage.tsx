import {
  ErrorAnnouncer,
  StatusAnnouncer,
  useAccessibility,
} from "@/components/AccessibilityEnhancements";
import { DIDInfoModal } from "@/components/did/DIDInfoModal";
import { EnvironmentVariableForm } from "@/components/forms/EnvironmentVariableForm";
import { ReasonersSkillsTable } from "@/components/ReasonersSkillsTable";
import { StatusRefreshButton } from "@/components/status";
import {
  AgentControlButton,
  type AgentState,
} from "@/components/ui/AgentControlButton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { NotificationProvider } from "@/components/ui/notification";
import { RestartRequiredBanner } from "@/components/ui/RestartRequiredBanner";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AnimatedTabs,
  AnimatedTabsContent,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "@/components/ui/animated-tabs";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { useMode } from "@/contexts/ModeContext";
import { useDIDInfo } from "@/hooks/useDIDInfo";
import { useNodeUnifiedStatusSSE } from "@/hooks/useSSE";
import {
  getNodeDetailsWithPackageInfo,
  getNodeStatus,
} from "@/services/api";
import {
  reconcileAgent,
  startAgent,
  stopAgent,
} from "@/services/configurationApi";
import { AlertCircle, Flash } from "@/components/ui/icon-bridge";

import {
  useErrorNotification,
  useInfoNotification,
  useSuccessNotification,
} from "@/components/ui/notification";
import { cn } from "@/lib/utils";
import type {
  AgentNodeDetailsForUIWithPackage,
  AgentStatus,
} from "@/types/agentfield";
import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { EnhancedNodeDetailHeader, NodeProcessLogsPanel } from "@/components/nodes";
import { getNodeStatusPresentation } from "@/utils/node-status";

/**
 * Comprehensive NodeDetailPage component with agent management interface.
 * Features tabbed navigation, real-time updates, and mode-aware rendering.
 */
function NodeDetailPageContent() {
  const { nodeId } = useParams<{ nodeId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { mode } = useMode();
  const { announceStatus: _announceStatus } = useAccessibility();

  // Notification hooks
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();
  const showInfo = useInfoNotification();

  // State management
  const [node, setNode] = useState<AgentNodeDetailsForUIWithPackage | null>(
    null
  );
  const [liveStatus, setLiveStatus] = useState<AgentStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showRestartBanner, setShowRestartBanner] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // DID-related state
  const { didInfo } = useDIDInfo(nodeId || "");
  const [showDIDModal, setShowDIDModal] = useState(false);

  // Real-time updates using optimized SSE hook
  const { latestEvent: unifiedStatusEvent } = useNodeUnifiedStatusSSE(
    nodeId || null
  );
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Handle unified status events for real-time status updates
  useEffect(() => {
    if (!unifiedStatusEvent) return;

    const eventData = unifiedStatusEvent.data;

    switch (unifiedStatusEvent.type) {
      case "node_unified_status_changed":
        if (
          eventData &&
          typeof eventData === "object" &&
          "new_status" in eventData
        ) {
          const statusData = eventData as {
            node_id: string;
            new_status: AgentStatus;
          };
          if (statusData.node_id === nodeId) {
            // Update live status with the new status from SSE
            setLiveStatus(statusData.new_status);
            setLastUpdate(new Date());
          }
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
            from_state: string;
          };
          if (transitionData.node_id === nodeId) {
            // State transition detected
            setLastUpdate(new Date());
          }
        }
        break;

      case "node_status_refreshed":
        if (
          eventData &&
          typeof eventData === "object" &&
          "node_id" in eventData
        ) {
          const refreshData = eventData as {
            node_id: string;
            status: AgentStatus;
          };
          if (refreshData.node_id === nodeId) {
            // Update live status with the refreshed status from SSE
            setLiveStatus(refreshData.status);
            setLastUpdate(new Date());
          }
        }
        break;

      default:
    }
  }, [unifiedStatusEvent, nodeId]);

  // Extract tab from URL hash
  useEffect(() => {
    const hash = location.hash.replace("#", "");
    if (
      hash &&
      [
        "overview",
        "configuration",
        "logs",
      ].includes(hash)
    ) {
      setActiveTab(hash);
    }
  }, [location.hash]);

  // Update URL hash when tab changes
  const handleTabChange = useCallback(
    (value: string) => {
      setActiveTab(value);
      navigate(`${location.pathname}#${value}`, { replace: true });
    },
    [navigate, location.pathname]
  );

  // Fetch node details with progressive loading
  const fetchData = useCallback(
    async (showSpinner = true) => {
      if (!nodeId) {
        setError("Node ID is missing.");
        setLoading(false);
        return;
      }

      if (showSpinner) {
        setLoading(true);
      } else {
        setRefreshing(true);
      }
      setError(null);

      try {
        // Phase 1: Load critical node data first with shorter timeout
        const nodeData = await getNodeDetailsWithPackageInfo(nodeId, mode);
        setNode(nodeData);
        setLastUpdate(new Date());

        // If we're showing spinner, hide it now that we have basic data
        if (showSpinner) {
          setLoading(false);
        }

        // Phase 2: Load unified status in background (non-blocking)
        getNodeStatus(nodeId)
          .then((statusData) => {
            setLiveStatus(statusData);
            setLastUpdate(new Date());
          })
          .catch((err) => {
            console.warn("Failed to fetch unified status:", err);
          });
      } catch (err: any) {
        const errorMessage = err.message || "Failed to load node details.";
        setError(errorMessage);
        console.error("Failed to fetch node data:", err);
      } finally {
        setRefreshing(false);
        // Only set loading to false if we haven't already done so
        if (showSpinner) {
          setLoading(false);
        }
      }
    },
    [nodeId, mode]
  );

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleBack = () => {
    navigate(-1);
  };

  // Handle start/stop actions
  const handleStartAgent = async () => {
    if (!nodeId) return;
    setActionLoading("start");
    showInfo(`Initiating start sequence for ${nodeId}...`);

    try {
      await startAgent(nodeId);
      showSuccess(`🚀 Agent ${nodeId} launch sequence completed!`);
      // Refresh data to get updated status
      fetchData(false);
    } catch (error: any) {
      const errorMessage = `Failed to start agent ${nodeId}`;

      // Handle specific error cases with clever messaging
      if (error.message?.includes("already running")) {
        showInfo(`⚡ Agent ${nodeId} is already active and ready!`);
      } else if (error.message?.includes("not installed")) {
        showError(`📦 Agent ${nodeId} needs to be installed first`);
      } else if (error.message?.includes("port")) {
        showError(`🔌 Port conflict detected - please try again`);
      } else {
        showError(errorMessage, error.message);
      }

      console.error("Failed to start agent", {
        nodeId,
        error,
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleStopAgent = async () => {
    if (!nodeId) return;
    setActionLoading("stop");
    showInfo(`Initiating shutdown sequence for ${nodeId}...`);

    try {
      await stopAgent(nodeId);
      showSuccess(`🛑 Agent ${nodeId} shutdown completed successfully!`);
      // Refresh data to get updated status
      fetchData(false);
    } catch (error: any) {
      const errorMessage = `Failed to stop agent ${nodeId}`;

      // Handle specific error cases with clever messaging
      if (error.message?.includes("not running")) {
        showInfo(`💤 Agent ${nodeId} is already in standby mode`);
      } else if (error.message?.includes("not installed")) {
        showError(`📦 Agent ${nodeId} is not installed`);
      } else {
        showError(errorMessage, error.message);
      }

      console.error("Failed to stop agent", {
        nodeId,
        error,
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleReconcileAgent = async () => {
    if (!nodeId) return;
    setActionLoading("reconcile");
    showInfo(`🔄 Reconciling agent ${nodeId} state...`);

    try {
      await reconcileAgent(nodeId);
      showSuccess(`✅ Agent ${nodeId} state reconciled successfully!`);
      // Refresh data to get updated status
      fetchData(false);
    } catch (error: any) {
      const errorMessage = `Failed to reconcile agent ${nodeId}`;

      if (error.message?.includes("not installed")) {
        showError(`📦 Agent ${nodeId} is not installed`);
      } else {
        showError(errorMessage, error.message);
      }

      console.error("Failed to reconcile agent", {
        nodeId,
        error,
      });
    } finally {
      setActionLoading(null);
    }
  };

  useEffect(() => {
    if (isFullscreen) {
      document.body.classList.add("overflow-hidden");
    } else {
      document.body.classList.remove("overflow-hidden");
    }

    return () => {
      document.body.classList.remove("overflow-hidden");
    };
  }, [isFullscreen]);

  useEffect(() => {
    if (!isFullscreen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsFullscreen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);


  // Determine current agent state for the control button
  const getAgentState = (): AgentState => {
    if (actionLoading === "start") return "starting";
    if (actionLoading === "stop") return "stopping";
    if (actionLoading === "reconcile") return "reconciling";

    // PRIORITY 1: Use live status data from the unified status system (live health checks)
    if (liveStatus) {
      switch (liveStatus.lifecycle_status) {
        case "ready":
          return "running";
        case "degraded":
          return "error";
        case "starting":
          return "starting";
        case "offline":
          return "stopped";
        default:
          break;
      }
    }

    // FALLBACK: Legacy logic using cached data (for backward compatibility)
    const isRunning =
      node?.lifecycle_status === "ready" ||
      node?.lifecycle_status === "degraded";

    if (isRunning) {
      if (node?.lifecycle_status === "degraded") {
        return "error";
      }
      return "running";
    }

    return "stopped";
  };

  // Unified handler for agent control actions
  const handleAgentAction = async (action: "start" | "stop" | "reconcile") => {
    try {
      switch (action) {
        case "start":
          await handleStartAgent();
          break;
        case "stop":
          await handleStopAgent();
          break;
        case "reconcile":
          await handleReconcileAgent();
          break;
      }

      // Give the backend a moment to update, then refresh data
      setTimeout(() => {
        fetchData(false);
      }, 1000);
    } catch (error) {
      // Error handling is already done in individual handlers
      console.error(`Failed to ${action} agent:`, error);
    }
  };

  // Loading state with enhanced skeleton
  if (loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        <StatusAnnouncer status="Loading node details" />

        {/* Header skeleton */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Skeleton className="h-10 w-20" />
            <div className="space-y-2">
              <Skeleton className="h-8 w-48" />
              <div className="flex space-x-2">
                <Skeleton className="h-6 w-16" />
                <Skeleton className="h-6 w-20" />
                <Skeleton className="h-6 w-24" />
              </div>
            </div>
          </div>
          <div className="flex space-x-2">
            <Skeleton className="h-10 w-20" />
            <Skeleton className="h-4 w-32" />
          </div>
        </div>

        {/* Tabs skeleton */}
        <div className="space-y-4">
          <div className="flex space-x-2">
            {["Overview", "Configuration", "Logs"].map(
              (_, i) => (
                <Skeleton key={i} className="h-10 w-24" />
              )
            )}
          </div>
          <div className="space-y-4">
            <Skeleton className="h-32 w-full" />
            <ResponsiveGrid columns={{ base: 1, md: 3 }} gap="sm">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </ResponsiveGrid>
          </div>
        </div>
      </div>
    );
  }

  // Error state with accessibility
  if (error) {
    return (
      <div className="p-4">
        <ErrorAnnouncer error={error} />
        <Alert variant="destructive" title="Error" role="alert">
          {error}
        </Alert>
        <div className="mt-4 flex space-x-2">
          <Button
            onClick={() => fetchData()}
            aria-label="Retry loading node details"
          >
            Retry
          </Button>
          <Button
            variant="secondary"
            onClick={handleBack}
            aria-label="Go back to previous page"
          >
            Back
          </Button>
        </div>
      </div>
    );
  }

  // No data state
  if (!node) {
    return (
      <div className="p-4">
        <Alert variant="default" title="No Data">
          Node details not found.
        </Alert>
        <Button onClick={handleBack} className="mt-4">
          Back
        </Button>
      </div>
    );
  }

  const isDeveloperMode = mode === "developer";

  const reasonerCount = node.reasoners?.length ?? 0;
  const skillCount = node.skills?.length ?? 0;

  const agentStatusForTable = liveStatus
    ? {
        health_status: liveStatus.health_status ?? 'unknown',
        lifecycle_status: liveStatus.lifecycle_status ?? 'unknown'
      }
    : {
        health_status: node.health_status ?? 'unknown',
        lifecycle_status: node.lifecycle_status ?? 'unknown'
      };

  const effectiveLifecycleStatus = liveStatus?.lifecycle_status ?? node.lifecycle_status ?? null;
  const effectiveHealthStatus = liveStatus?.health_status ?? node.health_status ?? null;
  const liveStatusPresentation = getNodeStatusPresentation(
    effectiveLifecycleStatus,
    effectiveHealthStatus
  );

  const headerMetadata: Array<{ label: string; value: string }> = [
    { label: "Reasoners", value: String(reasonerCount) },
    { label: "Skills", value: String(skillCount) },
    { label: "Mode", value: isDeveloperMode ? "Developer" : "User" },
  ];

  const liveStatusBadge = (
    <span
      className={cn(
        "flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium",
        liveStatusPresentation.theme.bgClass,
        liveStatusPresentation.theme.textClass,
        liveStatusPresentation.theme.borderClass
      )}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          liveStatusPresentation.theme.indicatorClass,
          liveStatusPresentation.shouldPulse && "animate-pulse"
        )}
      />
      {liveStatusPresentation.label}
    </span>
  );

  const statusRefreshButton = (
    <StatusRefreshButton
      nodeId={nodeId}
      onRefresh={(status) => {
        if (
          status &&
          typeof status === "object" &&
          "lifecycle_status" in status
        ) {
          setLiveStatus(status as AgentStatus);
        }
        setLastUpdate(new Date());
        fetchData(false);
      }}
      onError={(error) => {
        console.error("Status refresh failed:", error);
        showError(`Failed to refresh status: ${error}`);
      }}
      disabled={refreshing}
      size="sm"
      variant="ghost"
      showLabel={false}
      showLastVerified={true}
      lastVerified={lastUpdate.toISOString()}
      className="hidden md:flex"
    />
  );

  const headerActions = (
    <div className="flex items-center gap-2">
      <div onClick={(event) => event.stopPropagation()}>
        <AgentControlButton
          agentId={nodeId || ""}
          currentState={getAgentState()}
          onToggle={handleAgentAction}
          size="sm"
          showLabel={false}
          className="shadow-none"
        />
      </div>
      {statusRefreshButton}
    </div>
  );

  const mobileStatusRefreshButton = (
    <StatusRefreshButton
      nodeId={nodeId}
      onRefresh={(status) => {
        if (
          status &&
          typeof status === "object" &&
          "lifecycle_status" in status
        ) {
          setLiveStatus(status as AgentStatus);
        }
        setLastUpdate(new Date());
        fetchData(false);
      }}
      onError={(error) => {
        console.error("Status refresh failed:", error);
        showError(`Failed to refresh status: ${error}`);
      }}
      disabled={refreshing}
      size="sm"
      variant="ghost"
      showLabel={false}
      showLastVerified={false}
      className="md:hidden"
    />
  );

  const contentWrapperClass = cn(
    "flex min-h-0 flex-1 flex-col overflow-hidden"
  );

  const pageWrapperClass = cn(
    "flex min-h-0 flex-1 flex-col overflow-hidden",
    isFullscreen && "fixed inset-0 z-50 bg-background"
  );

  const formatRelative = (date: Date) => {
    const diffSeconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    const diffMinutes = Math.floor(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  const statusBadges = (
    <span className="text-body-small">
      Verified {formatRelative(lastUpdate)}
    </span>
  );

  return (
    <div className={cn(pageWrapperClass, "h-screen")}>
      <EnhancedNodeDetailHeader
        nodeId={node.id}
        lifecycleStatus={effectiveLifecycleStatus}
        healthStatus={effectiveHealthStatus}
        lastHeartbeat={liveStatus?.last_seen ?? node.last_heartbeat ?? null}
        onBack={handleBack}
        isFullscreen={isFullscreen}
        onFullscreenChange={setIsFullscreen}
        rightActions={headerActions}
        statusBadges={statusBadges}
        liveStatusBadge={liveStatusBadge}
        teamId={node.team_id}
        version={node.version}
        deploymentType={node.deployment_type}
        metadata={headerMetadata}
        focusMode={false}
        onFocusModeChange={() => {}}
        viewMode="standard"
        onViewModeChange={() => {}}
      />

      <div className={contentWrapperClass}>
        <AnimatedTabs
          value={activeTab}
          onValueChange={handleTabChange}
          className="flex min-h-0 flex-1 flex-col overflow-hidden"
        >
          <div className="flex items-center justify-between px-6 pt-4 pb-2">
            <AnimatedTabsList className="h-11 gap-1 rounded-lg bg-muted/40 p-1 flex-1">
              <AnimatedTabsTrigger value="overview" className="gap-2 px-4">
              Overview
            </AnimatedTabsTrigger>
            <AnimatedTabsTrigger value="configuration" className="gap-2 px-4">
              Configuration
            </AnimatedTabsTrigger>
            <AnimatedTabsTrigger value="logs" className="gap-2 px-4">
              Logs
            </AnimatedTabsTrigger>
          </AnimatedTabsList>
            {mobileStatusRefreshButton}
          </div>

          <AnimatedTabsContent
            value="overview"
            className="flex-1 overflow-y-auto"
          >
            <div className="flex flex-col gap-6 px-6 pb-6">
              <Card>
                <CardHeader>
                  <CardTitle>Node Information</CardTitle>
                  <CardDescription>
                    Comprehensive details about this agent node
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-5">
                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Node ID
                      </dt>
                      <dd className="text-sm font-mono break-all">{node.id}</dd>
                    </div>

                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Team ID
                      </dt>
                      <dd className="text-sm">{node.team_id}</dd>
                    </div>

                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Version
                      </dt>
                      <dd className="text-sm font-mono">{node.version}</dd>
                    </div>

                    <div className="space-y-1 md:col-span-2">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Base URL
                      </dt>
                      <dd className="text-sm font-mono break-all">{node.base_url}</dd>
                    </div>

                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Deployment Type
                      </dt>
                      <dd className="text-sm">
                        {node.deployment_type === "serverless" ? (
                          <Badge variant="outline" className="inline-flex items-center gap-1">
                            <Flash className="h-3.5 w-3.5" />
                            Serverless
                          </Badge>
                        ) : (
                          <Badge variant="outline">Long Running</Badge>
                        )}
                      </dd>
                    </div>

                    {node.deployment_type === "serverless" && node.invocation_url && (
                      <div className="space-y-1 md:col-span-2 lg:col-span-3">
                        <dt className="text-sm font-medium text-muted-foreground">
                          Invocation URL
                        </dt>
                        <dd className="text-sm font-mono break-all bg-muted/50 rounded-md px-3 py-2">
                          {node.invocation_url}
                        </dd>
                      </div>
                    )}

                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Last Heartbeat
                      </dt>
                      <dd className="text-sm">
                        {node.last_heartbeat
                          ? new Date(node.last_heartbeat).toLocaleString()
                          : "N/A"}
                      </dd>
                    </div>

                    <div className="space-y-1">
                      <dt className="text-sm font-medium text-muted-foreground">
                        Registered At
                      </dt>
                      <dd className="text-sm">
                        {node.registered_at
                          ? new Date(node.registered_at).toLocaleString()
                          : "N/A"}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              <ReasonersSkillsTable
                reasoners={node.reasoners ?? []}
                skills={node.skills ?? []}
                reasonerDIDs={didInfo?.reasoners}
                skillDIDs={didInfo?.skills}
                agentDID={didInfo?.did}
                agentStatus={agentStatusForTable}
                nodeId={nodeId}
                className="w-full"
              />
            </div>
          </AnimatedTabsContent>

          <AnimatedTabsContent
            value="configuration"
            className="flex-1 overflow-y-auto"
          >
            <div className="flex flex-col gap-6 px-6 pb-6">
              {showRestartBanner && (
                <RestartRequiredBanner
                  agentId={nodeId || ""}
                  onRestart={async () => {
                    try {
                      await handleStopAgent();
                      setTimeout(async () => {
                        await handleStartAgent();
                        setShowRestartBanner(false);
                      }, 2000);
                    } catch (error) {
                      showError("Failed to restart agent");
                    }
                  }}
                  onDismiss={() => setShowRestartBanner(false)}
                  className="mb-4"
                />
              )}

              {node?.package_info ? (
                <EnvironmentVariableForm
                  agentId={nodeId || ""}
                  packageId={node.package_info.package_id}
                  onConfigurationChange={() => {
                    setShowRestartBanner(true);
                    fetchData(false);
                  }}
                />
              ) : (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    No package information available for this agent. Configuration
                    cannot be managed.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </AnimatedTabsContent>

          <AnimatedTabsContent
            value="logs"
            className="flex-1 overflow-y-auto"
          >
            <div className="flex flex-col gap-6 px-6 pb-6">
              {nodeId ? (
                <NodeProcessLogsPanel nodeId={nodeId} />
              ) : null}
            </div>
          </AnimatedTabsContent>
        </AnimatedTabs>
      </div>

      <DIDInfoModal
        isOpen={showDIDModal}
        onClose={() => setShowDIDModal(false)}
        nodeId={node.id}
      />
    </div>
  );
}

export function NodeDetailPage() {
  return (
    <NotificationProvider>
      <NodeDetailPageContent />
    </NotificationProvider>
  );
}
