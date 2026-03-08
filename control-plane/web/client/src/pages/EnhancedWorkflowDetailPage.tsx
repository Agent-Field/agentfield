import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { EnhancedWorkflowHeader, type WorkflowNavigationTab } from "../components/workflow/EnhancedWorkflowHeader";
import { EnhancedWorkflowFlow } from "../components/workflow/EnhancedWorkflowFlow";
import { EnhancedWorkflowData } from "../components/workflow/EnhancedWorkflowData";
import { EnhancedWorkflowEvents } from "../components/workflow/EnhancedWorkflowEvents";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { Skeleton } from "@/components/ui/skeleton";
import { EnhancedWorkflowPerformance } from "../components/workflow/EnhancedWorkflowPerformance";
import { EnhancedWorkflowOverview } from "../components/workflow/EnhancedWorkflowOverview";
import { EnhancedWorkflowIdentity } from "../components/workflow/EnhancedWorkflowIdentity";
import { EnhancedWorkflowWebhooks } from "../components/workflow/EnhancedWorkflowWebhooks";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { NotificationProvider } from "../components/ui/notification";
import {
  GitBranch,
  Database,
  BarChart3,
  ShieldCheck,
  RadioTower,
  FileText,
} from "@/components/ui/icon-bridge";
import { getWorkflowRunSummary } from "../services/workflowsApi";
import { getWorkflowVCChain } from "../services/vcApi";
import { useWorkflowDAGSmart } from "../hooks/useWorkflowDAG";
import { summarizeWorkflowWebhook } from "../utils/webhook";
import type { WorkflowSummary } from "../types/workflows";
import type { WorkflowVCChainResponse } from "../types/did";
import { normalizeExecutionStatus } from "../utils/status";

type TabType = 'graph' | 'io' | 'webhooks' | 'notes' | 'identity' | 'insights';

const WORKFLOW_TAB_VALUES = ['graph', 'io', 'webhooks', 'notes', 'identity', 'insights'] as const;
const DEFAULT_WORKFLOW_TAB: TabType = 'graph';

function isWorkflowTab(value: string | null): value is TabType {
  return value !== null && WORKFLOW_TAB_VALUES.includes(value as TabType);
}

function resolveWorkflowTab(value: string | null): TabType {
  return isWorkflowTab(value) ? value : DEFAULT_WORKFLOW_TAB;
}

function EnhancedWorkflowDetailPageContent() {
  const { workflowId: runId } = useParams<{ workflowId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  // Core data state
  const [workflow, setWorkflow] = useState<WorkflowSummary | null>(null);
  const [vcChain, setVcChain] = useState<WorkflowVCChainResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI state derived from URL
  const activeTab = resolveWorkflowTab(searchParams.get('tab'));
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Enhanced workflow state
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [focusMode, setFocusMode] = useState(false);
  const [viewMode, setViewMode] = useState<'standard' | 'performance' | 'debug'>('standard');

  // Use smart polling hook for DAG data
  const {
    data: dagData,
    loading: dagLoading,
    error: dagError,
    isRefreshing: dagRefreshing,
    refresh: refreshDAG,
  } = useWorkflowDAGSmart(runId || null);

  const timelineForStatus = dagData?.timeline ?? [];
  const aggregatedStatus = normalizeExecutionStatus(
    dagData?.workflow_status ?? workflow?.status ?? 'unknown'
  );

  const timelineStatusCounts = useMemo(() => {
    if (!timelineForStatus.length) {
      return null;
    }
    return timelineForStatus.reduce<Record<string, number>>((acc, node) => {
      const normalized = normalizeExecutionStatus(node.status);
      acc[normalized] = (acc[normalized] ?? 0) + 1;
      return acc;
    }, {});
  }, [timelineForStatus]);

  const timelineActiveExecutions = useMemo(() => {
    if (!timelineForStatus.length) {
      return null;
    }
    return timelineForStatus.reduce((acc, node) => {
      const normalized = normalizeExecutionStatus(node.status);
      if (
        normalized === 'running' ||
        normalized === 'queued' ||
        normalized === 'pending'
      ) {
        return acc + 1;
      }
      return acc;
    }, 0);
  }, [timelineForStatus]);

  const timelineTerminal = useMemo(() => {
    if (!timelineForStatus.length) {
      return null;
    }
    return timelineForStatus.every((node) => {
      const normalized = normalizeExecutionStatus(node.status);
      return (
        normalized === 'succeeded' ||
        normalized === 'failed' ||
        normalized === 'timeout' ||
        normalized === 'cancelled'
      );
    });
  }, [timelineForStatus]);

  const displayWorkflow = useMemo<WorkflowSummary | null>(() => {
    if (!workflow) {
      return null;
    }

    const statusCounts = timelineStatusCounts ?? workflow.status_counts;
    const activeExecutions =
      timelineActiveExecutions ?? workflow.active_executions;
    const terminal = timelineTerminal ?? workflow.terminal;

    return {
      ...workflow,
      status: aggregatedStatus,
      total_executions:
        timelineForStatus.length || workflow.total_executions || 0,
      max_depth: dagData?.max_depth ?? workflow.max_depth,
      duration_ms: dagData?.dag?.duration_ms ?? workflow.duration_ms,
      status_counts: statusCounts,
      active_executions: activeExecutions,
      terminal,
    };
  }, [
    workflow,
    aggregatedStatus,
    timelineForStatus.length,
    dagData?.max_depth,
    dagData?.dag?.duration_ms,
    timelineStatusCounts,
    timelineActiveExecutions,
    timelineTerminal,
  ]);

  useEffect(() => {
    if (!runId) {
      navigate("/workflows");
      return;
    }

    const controller = new AbortController();

    const fetchWorkflow = async () => {
      try {
        setLoading(true);
        setError(null);

        const workflowSummary = await getWorkflowRunSummary(runId, controller.signal);

        if (!workflowSummary) {
          setError("Workflow run not found");
          return;
        }

        setWorkflow(workflowSummary);

        try {
          const vcData = await getWorkflowVCChain(workflowSummary.workflow_id);
          setVcChain(vcData);
        } catch (vcError) {
          console.error('Failed to fetch VC chain:', vcError);
          setVcChain(null);
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load workflow");
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflow();

    return () => {
      controller.abort();
    };
  }, [runId, navigate]);

  const handleTabChange = useCallback((tab: TabType) => {
    if (tab === 'graph' && selectedNodeIds.length === 0 && dagData?.timeline?.[0]) {
      setSelectedNodeIds([dagData.timeline[0].execution_id]);
    }

    if (tab === activeTab) {
      return;
    }

    const params = new URLSearchParams(searchParams);
    params.set('tab', tab);
    setSearchParams(params, { replace: false });
  }, [activeTab, dagData?.timeline, searchParams, selectedNodeIds.length, setSearchParams]);

  // Ensure URL always has a valid tab parameter
  useEffect(() => {
    const currentValue = searchParams.get('tab');
    if (!isWorkflowTab(currentValue)) {
      const params = new URLSearchParams(searchParams);
      params.set('tab', activeTab);
      setSearchParams(params, { replace: true });
    }
  }, [activeTab, searchParams, setSearchParams]);

  const previousTabRef = useRef<TabType | null>(null);
  useEffect(() => {
    if (
      previousTabRef.current !== activeTab &&
      activeTab === 'graph' &&
      selectedNodeIds.length === 0 &&
      dagData?.timeline?.[0]
    ) {
      setSelectedNodeIds([dagData.timeline[0].execution_id]);
    }
    previousTabRef.current = activeTab;
  }, [activeTab, dagData?.timeline, selectedNodeIds.length]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleTabChange('graph');
            break;
          case '2':
            event.preventDefault();
            handleTabChange('io');
            break;
          case '3':
            event.preventDefault();
            handleTabChange('webhooks');
            break;
          case '4':
            event.preventDefault();
            handleTabChange('notes');
            break;
          case '5':
            event.preventDefault();
            handleTabChange('identity');
            break;
          case '6':
            event.preventDefault();
            handleTabChange('insights');
            break;
          case 'f':
            event.preventDefault();
            setFocusMode(!focusMode);
            break;
          case 'r':
            event.preventDefault();
            refreshDAG();
            break;
        }
      }

      if (event.key === "Escape") {
        if (isFullscreen) {
          setIsFullscreen(false);
        } else if (selectedNodeIds.length > 0) {
          setSelectedNodeIds([]);
        } else {
          navigate("/workflows");
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [focusMode, handleTabChange, isFullscreen, selectedNodeIds, navigate, refreshDAG]);

  const handleNodeSelection = (nodeIds: string[], replace: boolean = true) => {
    if (replace) {
      setSelectedNodeIds(nodeIds);
    } else {
      setSelectedNodeIds(prev => [...new Set([...prev, ...nodeIds])]);
    }
  };

  if (loading) {
    return <EnhancedWorkflowSkeleton />;
  }

  if (error || !displayWorkflow) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <h2 className="text-heading-2">
            {error || "Workflow not found"}
          </h2>
          <button
            onClick={() => navigate("/workflows")}
            className="text-body-small text-muted-foreground hover:text-foreground underline"
          >
            ← Back to workflows
          </button>
        </div>
      </div>
    );
  }

  const containerClasses = isFullscreen
    ? "fixed inset-0 z-50 bg-background flex flex-col overflow-hidden"
    : "bg-background flex flex-col min-h-0 flex-1 overflow-hidden";

  const contentAreaClasses = "flex flex-1 min-h-0 flex-col overflow-hidden relative z-0";
  const showGraphLoading = dagLoading && !dagData;

  const timeline = dagData?.timeline ?? [];
  const webhookSummary = summarizeWorkflowWebhook(timeline);

  const getTabCount = (tabType: string): number | undefined => {
    switch (tabType) {
      case 'graph':
        return timeline.length || displayWorkflow.total_executions;
      case 'io':
        return timeline.filter((node: any) => node.input_data || node.output_data)?.length || 0;
      case 'webhooks':
        return webhookSummary.nodesWithWebhook;
      case 'notes':
        return timeline.reduce((count: number, node: any) => count + (node.notes?.length || 0), 0) || 0;
      case 'identity':
        return vcChain?.component_vcs?.length || 0;
      case 'insights':
        return timeline.filter((node: any) => node.duration_ms)?.length || 0;
      default:
        return undefined;
    }
  };

  const navigationTabs: WorkflowNavigationTab[] = [
    { id: 'graph', label: 'Graph', icon: GitBranch, description: 'Live workflow topology', shortcut: '1', count: getTabCount('graph') },
    { id: 'io', label: 'Inputs & Outputs', icon: Database, description: 'Inspect node inputs and outputs', shortcut: '2', count: getTabCount('io') },
    { id: 'webhooks', label: 'Webhooks', icon: RadioTower, description: 'Callback deliveries and status', shortcut: '3', count: getTabCount('webhooks') },
    { id: 'notes', label: 'Notes', icon: FileText, description: 'Operator notes, annotations, and context', shortcut: '4', count: getTabCount('notes') },
    { id: 'identity', label: 'Identity', icon: ShieldCheck, description: 'Trust, credentials, and verification chain', shortcut: '5', count: getTabCount('identity') },
    { id: 'insights', label: 'Insights', icon: BarChart3, description: 'Performance and health analytics', shortcut: '6', count: getTabCount('insights') },
  ];

  return (
    <ErrorBoundary>
      <div className={containerClasses}>
        <EnhancedWorkflowHeader
          workflow={displayWorkflow}
          dagData={dagData}
          isRefreshing={dagRefreshing}
          onRefresh={refreshDAG}
          onClose={() => navigate("/workflows")}
          isFullscreen={isFullscreen}
          onFullscreenChange={setIsFullscreen}
          selectedNodeCount={selectedNodeIds.length}
          activeTab={activeTab}
          onTabChange={(tab) => handleTabChange(tab as TabType)}
          navigationTabs={navigationTabs}
        />

        {/* Dynamic Content Area */}
        <div className={contentAreaClasses}>
          {activeTab === 'graph' && (
            <EnhancedWorkflowFlow
              workflow={displayWorkflow}
              dagData={dagData}
              loading={showGraphLoading}
              isRefreshing={dagRefreshing}
              error={dagError?.message || null}
              selectedNodeIds={selectedNodeIds}
              onNodeSelection={handleNodeSelection}
              viewMode={viewMode}
              onViewModeChange={setViewMode}
              focusMode={focusMode}
              onFocusModeChange={setFocusMode}
            />
          )}

          {activeTab === 'io' && (
            <EnhancedWorkflowData
              workflow={displayWorkflow}
              dagData={dagData}
              selectedNodeIds={selectedNodeIds}
              onNodeSelection={handleNodeSelection}
            />
          )}

          {activeTab === 'webhooks' && (
            <EnhancedWorkflowWebhooks
              workflow={displayWorkflow}
              dagData={dagData}
              onNodeSelection={handleNodeSelection}
              onRefresh={refreshDAG}
            />
          )}

          {activeTab === 'notes' && (
            <EnhancedWorkflowEvents
              workflow={displayWorkflow}
              dagData={dagData}
              selectedNodeIds={selectedNodeIds}
              onNodeSelection={handleNodeSelection}
            />
          )}

          {activeTab === 'identity' && (
            <EnhancedWorkflowIdentity
              workflow={displayWorkflow}
              vcChain={vcChain}
            />
          )}

          {activeTab === 'insights' && (
            <div className="h-full overflow-hidden">
              <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                <EnhancedWorkflowOverview
                  workflow={displayWorkflow}
                  dagData={dagData}
                  vcChain={vcChain}
                  selectedNodeIds={selectedNodeIds}
                  onNodeSelection={handleNodeSelection}
                />
                <div className="px-6 pb-6">
                  <EnhancedWorkflowPerformance
                    workflow={displayWorkflow}
                    dagData={dagData}
                    selectedNodeIds={selectedNodeIds}
                    onNodeSelection={handleNodeSelection}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
}

export function EnhancedWorkflowDetailPage() {
  return (
    <NotificationProvider>
      <EnhancedWorkflowDetailPageContent />
    </NotificationProvider>
  );
}

function EnhancedWorkflowSkeleton() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header Skeleton */}
      <div className="h-16 border-b border-border bg-card/50 flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <Skeleton className="h-4 w-4 rounded-full" />
          <Skeleton className="h-5 w-48 rounded" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="flex items-center gap-3">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
        </div>
      </div>

      {/* Tabs Skeleton */}
      <div className="h-12 border-b border-border bg-background flex items-center px-6">
        <div className="flex gap-6">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-6 w-20 rounded-full" />
          ))}
        </div>
      </div>

      {/* Content Skeleton */}
      <div className="p-6 space-y-8">
        <ResponsiveGrid preset="quarters" gap="md">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </ResponsiveGrid>
        <Skeleton className="h-96 rounded-xl" />
      </div>
    </div>
  );
}
