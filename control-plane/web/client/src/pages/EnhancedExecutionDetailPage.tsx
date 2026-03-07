import { useCallback, useEffect, useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { CompactExecutionHeader, type NavigationTab } from "../components/execution/CompactExecutionHeader";
import { RedesignedErrorPanel } from "../components/execution/RedesignedErrorPanel";
import { EnhancedNotesSection } from "../components/execution/EnhancedNotesSection";
import { ExecutionRetryPanel } from "../components/execution/ExecutionRetryPanel";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { Skeleton } from "@/components/ui/skeleton";
import { ExecutionIdentityPanel } from "../components/execution/ExecutionIdentityPanel";
import { ExecutionDataColumns } from "../components/execution/ExecutionDataColumns";
import { CollapsibleSection } from "../components/execution/CollapsibleSection";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { NotificationProvider } from "../components/ui/notification";
import { getExecutionDetails, retryExecutionWebhook } from "../services/executionsApi";
import { getExecutionVCStatus } from "../services/vcApi";
import type { WorkflowExecution } from "../types/executions";
import { Database, Bug, Shield, Wrench, FileText, RadioTower, Cog, PauseCircle } from "../components/ui/icon-bridge";
import { ExecutionWebhookActivity } from "../components/execution/ExecutionWebhookActivity";
import { ExecutionApprovalPanel } from "../components/execution/ExecutionApprovalPanel";

type TabType = 'io' | 'webhook' | 'approval' | 'debug' | 'identity' | 'meta' | 'notes';

const EXECUTION_TAB_VALUES = ['io', 'webhook', 'approval', 'debug', 'identity', 'meta', 'notes'] as const;
const DEFAULT_EXECUTION_TAB: TabType = 'io';

function isExecutionTab(value: string | null): value is TabType {
  return value !== null && EXECUTION_TAB_VALUES.includes(value as TabType);
}

function resolveExecutionTab(value: string | null): TabType {
  return isExecutionTab(value) ? value : DEFAULT_EXECUTION_TAB;
}

export function EnhancedExecutionDetailPage() {
  const { executionId } = useParams<{ executionId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  // Core data state
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [vcStatus, setVcStatus] = useState<{
    has_vc: boolean;
    vc_id?: string;
    status: string;
    created_at?: string;
    vc_document?: any;
    storage_uri?: string;
    document_size_bytes?: number;
    original_status?: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [vcLoading, setVcLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [retryingWebhook, setRetryingWebhook] = useState(false);
  const [retryWebhookError, setRetryWebhookError] = useState<string | null>(null);

  // UI state derived from URL
  const activeTab = resolveExecutionTab(searchParams.get('tab'));

  const refreshExecution = async () => {
    if (!executionId) return;

    try {
      setIsRefreshing(true);
      const data = await getExecutionDetails(executionId);
      setExecution(data);
    } catch (err) {
      console.error("Failed to refresh execution details:", err);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    if (!executionId) {
      setError("Execution ID is required");
      setLoading(false);
      return;
    }

    const fetchExecution = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getExecutionDetails(executionId);
        setExecution(data);
      } catch (err) {
        console.error("Failed to fetch execution details:", err);
        setError(err instanceof Error ? err.message : "Failed to load execution details");
      } finally {
        setLoading(false);
      }
    };

    fetchExecution();
  }, [executionId]);

  useEffect(() => {
    if (!executionId) return;

    const fetchVCStatus = async () => {
      try {
        setVcLoading(true);
        const vcData = await getExecutionVCStatus(executionId);
        setVcStatus(vcData);
      } catch (err) {
        console.error("Failed to fetch VC status:", err);
        setVcStatus({ has_vc: false, status: "error" });
      } finally {
        setVcLoading(false);
      }
    };

    fetchVCStatus();
  }, [executionId]);

  // Ensure URL always has a valid tab parameter
  useEffect(() => {
    const currentValue = searchParams.get('tab');
    if (!isExecutionTab(currentValue)) {
      const params = new URLSearchParams(searchParams);
      params.set('tab', activeTab);
      setSearchParams(params, { replace: true });
    }
  }, [activeTab, searchParams, setSearchParams]);

  const handleTabChange = useCallback((tab: TabType) => {
    if (tab === activeTab) {
      return;
    }
    const params = new URLSearchParams(searchParams);
    params.set('tab', tab);
    setSearchParams(params, { replace: false });
  }, [activeTab, searchParams, setSearchParams]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleTabChange('io');
            break;
          case '2':
            event.preventDefault();
            handleTabChange('webhook');
            break;
          case '3':
            event.preventDefault();
            handleTabChange('approval');
            break;
          case '4':
            event.preventDefault();
            handleTabChange('debug');
            break;
          case '5':
            event.preventDefault();
            handleTabChange('identity');
            break;
          case '6':
            event.preventDefault();
            handleTabChange('meta');
            break;
          case '7':
            event.preventDefault();
            handleTabChange('notes');
            break;
        }
      }

      if (event.key === "Escape") {
        navigate("/executions");
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleTabChange, navigate]);

  if (loading) {
    return <EnhancedExecutionSkeleton />;
  }

  if (error || !execution) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <h2 className="text-heading-2">
            {error || "Execution not found"}
          </h2>
          <button
            onClick={() => navigate("/executions")}
            className="text-body-small text-muted-foreground hover:text-foreground underline"
          >
            ← Back to executions
          </button>
        </div>
      </div>
    );
  }

  const navigationTabs: (NavigationTab & { id: TabType })[] = [
    {
      id: 'io',
      label: 'Inputs & Outputs',
      icon: Database,
      description: 'Inspect execution inputs and outputs',
      shortcut: '1',
      count: (execution.input_data ? 1 : 0) + (execution.output_data ? 1 : 0),
    },
    {
      id: 'webhook',
      label: 'Webhooks',
      icon: RadioTower,
      description: 'Delivery history and retry controls',
      shortcut: '2',
      count: (execution.webhook_events?.length || 0) + (execution.webhook_registered ? 1 : 0),
    },
    ...(execution.approval_request_id ? [{
      id: 'approval' as TabType,
      label: 'Approval',
      icon: PauseCircle,
      description: 'Human approval workflow status',
      shortcut: '3',
      count: execution.approval_status === 'pending' ? 1 : 0,
    }] : []),
    {
      id: 'debug',
      label: 'Debug',
      icon: Bug,
      description: 'Retry tools and error analysis',
      shortcut: execution.approval_request_id ? '4' : '3',
      count: execution.error_message ? 1 : 0,
    },
    {
      id: 'identity',
      label: 'Identity',
      icon: Shield,
      description: 'Trust, credentials, and verification',
      shortcut: execution.approval_request_id ? '5' : '4',
      count: vcStatus?.has_vc ? 1 : 0,
    },
    {
      id: 'meta',
      label: 'Technical',
      icon: Wrench,
      description: 'Metadata and system details',
      shortcut: execution.approval_request_id ? '6' : '5',
    },
    {
      id: 'notes',
      label: 'Notes',
      icon: FileText,
      description: 'Live execution notes and workflow context',
      shortcut: execution.approval_request_id ? '7' : '6',
      count: execution.notes?.length || 0,
    },
  ];

  return (
    <ErrorBoundary>
      <NotificationProvider>
      <div className="bg-background flex flex-col min-h-0 min-w-0 flex-1 overflow-hidden">
        <CompactExecutionHeader
          execution={execution}
          onClose={() => navigate("/executions")}
          onRefresh={refreshExecution}
          isRefreshing={isRefreshing}
          activeTab={activeTab}
          onTabChange={(tab) => handleTabChange(tab as TabType)}
          navigationTabs={navigationTabs}
        />

        {/* Dynamic Content Area */}
        <div className="flex-1 min-h-0 min-w-0 overflow-hidden">
          {activeTab === 'io' && (
            <div className="h-full overflow-hidden">
              <ExecutionDataColumns execution={execution} />
            </div>
          )}

          {activeTab === 'webhook' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border p-6">
              <ExecutionWebhookActivity
                execution={execution}
                onRetry={async () => {
                  if (!executionId) return;
                  setRetryWebhookError(null);
                  setRetryingWebhook(true);
                  try {
                    await retryExecutionWebhook(executionId);
                    await refreshExecution();
                  } catch (err) {
                    console.error("Failed to retry webhook:", err);
                    setRetryWebhookError(err instanceof Error ? err.message : "Retry failed");
                  } finally {
                    setRetryingWebhook(false);
                  }
                }}
                isRetrying={retryingWebhook}
                retryError={retryWebhookError}
              />
            </div>
          )}

          {activeTab === 'approval' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border p-6">
              <ExecutionApprovalPanel execution={execution} />
            </div>
          )}

          {activeTab === 'debug' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border p-6">
              <div className="space-y-6">
                {execution.error_message && (
                  <RedesignedErrorPanel execution={execution} />
                )}
                <ExecutionRetryPanel execution={execution} />
              </div>
            </div>
          )}

          {activeTab === 'identity' && (
            <ExecutionIdentityPanel
              execution={execution}
              vcStatus={vcStatus}
              vcLoading={vcLoading}
            />
          )}

          {activeTab === 'meta' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border p-6">
              <CollapsibleSection
                  title="Technical Details"
                  icon={Cog}
                  defaultOpen={true}
                  badge={
                    <span className="text-body-small bg-muted/50 px-2 py-0.5 rounded">
                      Metadata
                    </span>
                  }
                >
                  <div className="p-4 space-y-4">
                    <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="md" align="start" className="text-sm">
                      <div className="space-y-3">
                        <div>
                          <label className="text-muted-foreground">Created At</label>
                          <p className="font-mono text-xs text-foreground">
                            {new Date(execution.created_at).toLocaleString()}
                          </p>
                        </div>

                        <div>
                          <label className="text-muted-foreground">Started At</label>
                          <p className="font-mono text-xs text-foreground">
                            {new Date(execution.started_at).toLocaleString()}
                          </p>
                        </div>

                        {execution.completed_at && (
                          <div>
                            <label className="text-muted-foreground">Completed At</label>
                            <p className="font-mono text-xs text-foreground">
                              {new Date(execution.completed_at).toLocaleString()}
                            </p>
                          </div>
                        )}

                        <div>
                          <label className="text-muted-foreground">Workflow Depth</label>
                          <p className="font-mono text-xs text-foreground">
                            {execution.workflow_depth}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        {execution.parent_workflow_id && (
                          <div>
                            <label className="text-muted-foreground">Parent Workflow</label>
                            <p className="font-mono text-xs text-foreground">
                              {execution.parent_workflow_id}
                            </p>
                          </div>
                        )}

                        {execution.root_workflow_id && (
                          <div>
                            <label className="text-muted-foreground">Root Workflow</label>
                            <p className="font-mono text-xs text-foreground">
                              {execution.root_workflow_id}
                            </p>
                          </div>
                        )}

                        {execution.actor_id && (
                          <div>
                            <label className="text-muted-foreground">Actor ID</label>
                            <p className="font-mono text-xs text-foreground">
                              {execution.actor_id}
                            </p>
                          </div>
                        )}

                      <div>
                          <label className="text-muted-foreground">Updated At</label>
                          <p className="font-mono text-xs text-foreground">
                            {new Date(execution.updated_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    </ResponsiveGrid>

                    {/* URI Information */}
                    {(execution.input_uri || execution.result_uri) && (
                      <div className="border-t border-border pt-4">
                        <h4 className="text-sm font-medium text-foreground mb-3">Storage URIs</h4>
                        <div className="space-y-2">
                          {execution.input_uri && (
                            <div>
                              <label className="text-body-small">Input URI</label>
                              <p className="font-mono text-xs text-foreground break-all">
                                {execution.input_uri}
                              </p>
                            </div>
                          )}
                          {execution.result_uri && (
                            <div>
                              <label className="text-body-small">Result URI</label>
                              <p className="font-mono text-xs text-foreground break-all">
                                {execution.result_uri}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
            </div>
          )}

          {activeTab === 'notes' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border p-6">
              <EnhancedNotesSection execution={execution} onRefresh={refreshExecution} />
            </div>
          )}
        </div>
      </div>
      </NotificationProvider>
    </ErrorBoundary>
  );
}

function EnhancedExecutionSkeleton() {
  return (
    <div className="h-full bg-background">
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

      {/* Performance Strip Skeleton */}
      <div className="h-12 border-b border-border bg-muted/20 flex items-center px-6">
        <div className="flex items-center gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-24" />
          ))}
        </div>
      </div>

      {/* Tabs Skeleton */}
      <div className="h-12 border-b border-border bg-background flex items-center px-6">
        <div className="flex gap-6">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-6 w-20 rounded-full" />
          ))}
        </div>
      </div>

      {/* Content Skeleton */}
      <div className="p-6 space-y-8">
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    </div>
  );
}
