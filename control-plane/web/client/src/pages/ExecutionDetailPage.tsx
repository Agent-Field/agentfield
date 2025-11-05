import { Info, Loader2, MessageSquare, Copy, Check } from "@/components/ui/icon-bridge";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { ExecutionDetailsLayout } from "../components/execution/ExecutionDetailsLayout";
import { ExecutionHeader } from "../components/execution/ExecutionHeader";
import { ExecutionTimeline } from "../components/execution/ExecutionTimeline";
import { RedesignedInputDataPanel } from "../components/execution/RedesignedInputDataPanel";
import { RedesignedOutputDataPanel } from "../components/execution/RedesignedOutputDataPanel";
import { RedesignedErrorPanel } from "../components/execution/RedesignedErrorPanel";
import { WorkflowBreadcrumb } from "../components/execution/WorkflowBreadcrumb";
import { CollapsibleSection } from "../components/execution/CollapsibleSection";
import { NotesPanel } from "../components/notes";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { getExecutionDetails } from "../services/executionsApi";
import { getExecutionVCStatus } from "../services/vcApi";
import type { WorkflowExecution } from "../types/executions";

function InlineCopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error("Failed to copy value", err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex h-5 w-5 items-center justify-center rounded hover:bg-muted/80"
      title="Copy value"
    >
      {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3 text-muted-foreground" />}
    </button>
  );
}

interface MetadataRowProps {
  label: string;
  value?: string | React.ReactNode;
  copyValue?: string;
  onClick?: () => void;
}

function MetadataRow({ label, value, copyValue, onClick }: MetadataRowProps) {
  const displayValue = value ?? "—";
  const isInteractive = Boolean(onClick);

  return (
    <div className="space-y-1">
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <div className="flex items-center gap-2 text-sm text-foreground">
        <span
          className={isInteractive ? "cursor-pointer text-primary hover:underline" : "break-all"}
          onClick={onClick}
        >
          {displayValue}
        </span>
        {copyValue && <InlineCopyButton value={copyValue} />}
      </div>
    </div>
  );
}

export function ExecutionDetailPage() {
  const { executionId } = useParams<{ executionId: string }>();
  const navigate = useNavigate();
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [vcStatus, setVcStatus] = useState<{
    has_vc: boolean;
    vc_id?: string;
    status: string;
    created_at?: string;
    vc_document?: any;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [vcLoading, setVcLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  const handleNavigateBack = () => navigate("/executions");

  if (loading) {
    return (
      <ExecutionDetailsLayout>
        <div className="flex min-h-[400px] items-center justify-center">
          <div className="flex items-center gap-3 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <span className="text-heading-3">Loading execution details…</span>
          </div>
        </div>
      </ExecutionDetailsLayout>
    );
  }

  if (error) {
    return (
      <ExecutionDetailsLayout>
        <Card className="border-destructive bg-destructive/5">
          <CardContent className="p-6">
            <div className="text-center">
              <h2 className="mb-2 text-heading-2 text-destructive">Failed to Load Execution</h2>
              <p className="mb-4 text-body">{error}</p>
              <Button onClick={() => window.location.reload()}>Try Again</Button>
            </div>
          </CardContent>
        </Card>
      </ExecutionDetailsLayout>
    );
  }

  if (!execution) {
    return (
      <ExecutionDetailsLayout>
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <h2 className="mb-2 text-heading-2 text-muted-foreground">Execution Not Found</h2>
              <p className="text-body">
                The execution with ID “{executionId}” could not be located.
              </p>
            </div>
          </CardContent>
        </Card>
      </ExecutionDetailsLayout>
    );
  }

  const navigateToWorkflow = () => navigate(`/workflows/${execution.workflow_id}`);
  const navigateToSession = () => execution.session_id && navigate(`/executions?session_id=${execution.session_id}`);

  return (
    <ExecutionDetailsLayout>
      <ExecutionHeader execution={execution} vcStatus={vcStatus} vcLoading={vcLoading} onNavigateBack={handleNavigateBack} />

      <WorkflowBreadcrumb execution={execution} onNavigateBack={handleNavigateBack} />

      <div className="rounded-lg border border-border bg-background p-4">
        <ExecutionTimeline execution={execution} />
      </div>

      <div className="space-y-4">
        <RedesignedInputDataPanel execution={execution} />
        <RedesignedOutputDataPanel execution={execution} />
        {execution.error_message && <RedesignedErrorPanel execution={execution} />}

        <CollapsibleSection
          title="Notes & Comments"
          icon={MessageSquare}
          defaultOpen
          contentClassName="p-0"
        >
          <NotesPanel executionId={execution.execution_id} className="border-0 shadow-none" />
        </CollapsibleSection>

        <CollapsibleSection
          title="Execution Metadata"
          icon={Info}
          defaultOpen={false}
          contentClassName="p-4"
        >
          <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="md" align="start">
            <MetadataRow
              label="Workflow"
              value={execution.workflow_name ?? execution.workflow_id}
              copyValue={execution.workflow_id}
              onClick={navigateToWorkflow}
            />
            <MetadataRow
              label="Session"
              value={execution.session_id ?? '—'}
              copyValue={execution.session_id}
              onClick={execution.session_id ? navigateToSession : undefined}
            />
            <MetadataRow
              label="Haxen Request ID"
              value={execution.haxen_request_id || '—'}
              copyValue={execution.haxen_request_id || undefined}
            />
            <MetadataRow
              label="Actor ID"
              value={execution.actor_id ?? '—'}
              copyValue={execution.actor_id}
            />
            <MetadataRow
              label="Parent Workflow"
              value={execution.parent_workflow_id ?? '—'}
              copyValue={execution.parent_workflow_id ?? undefined}
            />
            <MetadataRow
              label="Root Workflow"
              value={execution.root_workflow_id ?? '—'}
              copyValue={execution.root_workflow_id ?? undefined}
            />
            <MetadataRow
              label="Workflow Depth"
              value={execution.workflow_depth?.toString() ?? '0'}
            />
            <MetadataRow
              label="Retry Count"
              value={execution.retry_count.toString()}
            />
            <MetadataRow
              label="Created"
              value={new Date(execution.created_at).toLocaleString()}
            />
            <MetadataRow
              label="Updated"
              value={new Date(execution.updated_at).toLocaleString()}
            />
            {execution.completed_at && (
              <MetadataRow
                label="Completed"
                value={new Date(execution.completed_at).toLocaleString()}
              />
            )}
            {execution.notes && execution.notes.length > 0 && (
              <MetadataRow
                label="Notes Count"
                value={`${execution.notes.length}`}
              />
            )}
          </ResponsiveGrid>

          {execution.workflow_tags && execution.workflow_tags.length > 0 && (
            <div className="mt-4">
              <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Tags
              </span>
              <div className="mt-2 flex flex-wrap gap-2">
                {execution.workflow_tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CollapsibleSection>
      </div>
    </ExecutionDetailsLayout>
  );
}
