import { useEffect, useMemo, useState } from 'react';
import { Trash, WarningOctagon, SpinnerGap } from '@/components/ui/icon-bridge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge, type BadgeProps } from '../ui/badge';
import type { WorkflowSummary } from '../../types/workflows';
import type { WorkflowCleanupResult } from '../../services/workflowsApi';
import { normalizeExecutionStatus } from '../../utils/status';
import { statusTone } from '@/lib/theme';
import { cn } from '@/lib/utils';

interface WorkflowDeleteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  workflows: WorkflowSummary[];
  onConfirm: (workflowIds: string[]) => Promise<WorkflowCleanupResult[]>;
}

const statusVariantMap: Record<string, NonNullable<BadgeProps["variant"]>> = {
  running: "running",
  queued: "pending",
  pending: "pending",
  succeeded: "success",
  failed: "failed",
  timeout: "failed",
  cancelled: "unknown",
  inactive: "unknown",
};

const getBadgeVariantForStatus = (status: string): NonNullable<BadgeProps["variant"]> => {
  return statusVariantMap[status] ?? "unknown";
};

export function WorkflowDeleteDialog({
  isOpen,
  onClose,
  workflows,
  onConfirm,
}: WorkflowDeleteDialogProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const workflowIds = useMemo(() => {
    return workflows
      .map((workflow) => workflow.workflow_id || workflow.run_id)
      .filter((id): id is string => Boolean(id && id.trim()));
  }, [workflows]);
  const isSingleWorkflow = workflowIds.length === 1;

  const lifecycleTotals = useMemo(() => {
    return workflows.reduce(
      (acc, workflow) => {
        const counts = workflow.status_counts ?? {};
        acc.active += (counts.running ?? 0) + (counts.queued ?? 0) + (counts.pending ?? 0);
        acc.failed += (counts.failed ?? 0) + (counts.timeout ?? 0);
        acc.succeeded += counts.succeeded ?? 0;
        if (normalizeExecutionStatus(workflow.status) === 'running') {
          acc.runningWorkflows += 1;
        }
        return acc;
      },
      { active: 0, failed: 0, succeeded: 0, runningWorkflows: 0 }
    );
  }, [workflows]);

  useEffect(() => {
    if (!isOpen) {
      setError(null);
      setIsDeleting(false);
    }
  }, [isOpen]);

  const handleClose = () => {
    if (isDeleting) {
      return;
    }
    setError(null);
    onClose();
  };

  const handleConfirm = async () => {
    if (workflowIds.length === 0) {
      return;
    }

    setIsDeleting(true);
    setError(null);

    try {
      await onConfirm(workflowIds);
      setIsDeleting(false);
      setError(null);
      onClose();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete workflows. Please try again.';
      setError(message);
      setIsDeleting(false);
    }
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          handleClose();
        }
      }}
    >
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trash className={cn("w-5 h-5", statusTone.error.accent)} />
            Delete {isSingleWorkflow ? 'workflow' : 'workflows'}
          </DialogTitle>
          <DialogDescription>
            Permanently remove {isSingleWorkflow ? 'this workflow' : `these ${workflowIds.length} workflows`} and all
            associated executions, verifiable credentials, and stored artifacts. This action cannot be undone.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="border border-border rounded-lg p-4 max-h-60 overflow-y-auto">
            <h4 className="font-medium mb-3 text-sm">
              {isSingleWorkflow ? 'Workflow to delete' : `${workflowIds.length} workflows to delete`}
            </h4>
            {workflowIds.length === 0 ? (
              <p className="text-sm text-muted-foreground">Select at least one workflow to delete.</p>
            ) : (
              <div className="space-y-2">
                {workflows.map((workflow) => {
                  const counts = workflow.status_counts ?? {};
                  const active = (counts.running ?? 0) + (counts.queued ?? 0) + (counts.pending ?? 0);
                  const failed = (counts.failed ?? 0) + (counts.timeout ?? 0);
                  const normalizedStatus = normalizeExecutionStatus(workflow.status);
                  const workflowIdentifier = workflow.workflow_id || workflow.run_id;

                  return (
                    <div
                      key={workflowIdentifier ?? workflow.display_name ?? workflow.run_id ?? workflow.workflow_id}
                      className="flex items-center justify-between gap-3 rounded bg-muted/50 p-2"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">
                          {workflow.display_name || 'Unnamed Workflow'}
                        </div>
                        <div className="text-sm text-muted-foreground truncate">
                          {workflowIdentifier ?? "unknown"} • {workflow.agent_name}
                        </div>
                        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground text-muted-foreground mt-1">
                          <span>{workflow.total_executions} nodes</span>
                          <span>•</span>
                          <span>{counts.succeeded ?? 0} succeeded</span>
                          {active > 0 && (
                            <>
                              <span>•</span>
                              <span className="text-foreground">{active} active</span>
                            </>
                          )}
                          {failed > 0 && (
                            <>
                              <span>•</span>
                              <span className={statusTone.error.accent}>{failed} issues</span>
                            </>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 ml-2">
                        <Badge
                          variant={getBadgeVariantForStatus(normalizedStatus)}
                          className="capitalize"
                        >
                          {normalizedStatus}
                        </Badge>
                        {active > 0 && (
                          <Badge variant="running" className="text-sm text-muted-foreground h-5 px-2">
                            {active} active
                          </Badge>
                        )}
                        {failed > 0 && (
                          <Badge variant="failed" className="text-sm text-muted-foreground h-5 px-2">
                            {failed} issues
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className={cn("space-y-2 rounded-lg p-4 text-sm", statusTone.error.bg, statusTone.error.border)}>
            <div className={cn("flex items-center gap-2 font-medium", statusTone.error.accent)}>
              <WarningOctagon className="w-4 h-4" />
              Everything linked to these workflows will be deleted
            </div>
            <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
              <li>Execution history and intermediate results</li>
              <li>Uploaded inputs, generated artifacts, and outputs</li>
              <li>Issued verifiable credentials</li>
              {lifecycleTotals.active > 0 && (
                <li>
                  {lifecycleTotals.active} in-flight execution
                  {lifecycleTotals.active === 1 ? '' : 's'} will be force-cancelled.
                </li>
              )}
              {lifecycleTotals.failed > 0 && (
                <li>
                  Includes {lifecycleTotals.failed} failed/timeout run
                  {lifecycleTotals.failed === 1 ? '' : 's'} for forensic review.
                </li>
              )}
            </ul>
          </div>

          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            disabled={isDeleting || workflowIds.length === 0}
            className="min-w-[160px]"
          >
            {isDeleting ? (
              <span className="flex items-center gap-2">
                <SpinnerGap className="h-4 w-4 animate-spin" />
                Deleting...
              </span>
            ) : (
              `Delete ${isSingleWorkflow ? 'workflow' : `${workflowIds.length} workflows`}`
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
