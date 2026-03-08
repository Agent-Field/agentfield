import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { ExecutionDetailsLayout } from "../components/execution/ExecutionDetailsLayout";
import { ExecutionHeader } from "../components/execution/ExecutionHeader";
import { EnhancedDataPanel } from "../components/execution/EnhancedDataPanel";
import { RedesignedErrorPanel } from "../components/execution/RedesignedErrorPanel";
import { EnhancedNotesSection } from "../components/execution/EnhancedNotesSection";
import { CollapsibleSection } from "../components/execution/CollapsibleSection";
import { getExecutionDetails } from "../services/executionsApi";
import { getExecutionVCStatus } from "../services/vcApi";
import type { WorkflowExecution } from "../types/executions";
import type { VCStatusData } from "../types/vc";
import { Settings } from "@/components/ui/icon-bridge";

export function RedesignedExecutionDetailPage() {
  const { executionId } = useParams<{ executionId: string }>();
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [vcStatus, setVcStatus] = useState<VCStatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [vcLoading, setVcLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshExecution = async () => {
    if (!executionId) return;

    try {
      const data = await getExecutionDetails(executionId);
      setExecution(data);
    } catch (err) {
      console.error("Failed to refresh execution details:", err);
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

  if (loading) {
    return (
      <ExecutionDetailsLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading execution details...</p>
          </div>
        </div>
      </ExecutionDetailsLayout>
    );
  }

  if (error || !execution) {
    return (
      <ExecutionDetailsLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <h2 className="text-heading-2 mb-2">
              Execution Not Found
            </h2>
            <p className="text-body">
              {error || "The requested execution could not be found."}
            </p>
          </div>
        </div>
      </ExecutionDetailsLayout>
    );
  }

  return (
    <ExecutionDetailsLayout>
      <div className="space-y-6">
        {/* Enhanced Header with Critical Info */}
        <ExecutionHeader
          execution={execution}
          vcStatus={vcStatus}
          vcLoading={vcLoading}
        />

        {/* Main Content Sections */}
        <div className="space-y-4">
          {/* Enhanced Input Data Panel with Modal Support */}
          <EnhancedDataPanel execution={execution} type="input" />

          {/* Enhanced Output Data Panel with Modal Support */}
          <EnhancedDataPanel execution={execution} type="output" />

          {/* Error Section - Only show if there's an error */}
          {execution.error_message && (
            <RedesignedErrorPanel execution={execution} />
          )}

          {/* Enhanced Notes Section - Auto-collapses when empty */}
          <EnhancedNotesSection execution={execution} onRefresh={refreshExecution} />

          {/* Technical Details Section */}
          <CollapsibleSection
            title="Technical Details"
            icon={Settings}
            defaultOpen={false}
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
      </div>
    </ExecutionDetailsLayout>
  );
}
