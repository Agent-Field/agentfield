import { ChevronLeft } from "@/components/ui/icon-bridge";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { CompactWorkflowSummary } from "../components/CompactWorkflowSummary";
import { CompactWorkflowInputOutput } from "../components/workflow/CompactWorkflowInputOutput";
import { TwoColumnLayout } from "../components/layout/TwoColumnLayout";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { WorkflowDAGViewer } from "../components/WorkflowDAGViewer";
import { WorkflowTimeline } from "../components/workflow/WorkflowTimeline";
import { SimpleWorkflowVC } from "../components/vc";
import { getWorkflowRunSummary } from "../services/workflowsApi";
import { getWorkflowVCChain } from "../services/vcApi";
import { useWorkflowDAGSmart } from "../hooks/useWorkflowDAG";
import type { WorkflowSummary } from "../types/workflows";
import type { WorkflowVCChainResponse } from "../types/did";

export function WorkflowDetailPage() {
  const { workflowId: runId } = useParams<{ workflowId: string }>();
  const navigate = useNavigate();
  const [workflow, setWorkflow] = useState<WorkflowSummary | null>(null);
  const [vcChain, setVcChain] = useState<WorkflowVCChainResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [vcLoading, setVcLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Timeline UI state - lifted up to prevent reset on data refresh
  const [timelineSortOrder, setTimelineSortOrder] = useState<'asc' | 'desc'>('asc');
  const [timelineSelectedTags, setTimelineSelectedTags] = useState<string[]>([]);
  const [timelineExpandAll, setTimelineExpandAll] = useState(false);
  const [timelineCardExpansions, setTimelineCardExpansions] = useState<Record<string, boolean>>({});
  const [timelineNoteExpansions, setTimelineNoteExpansions] = useState<Record<string, Set<number>>>({});

  // Use smart polling hook for DAG data
  const {
    data: dagData,
    loading: dagLoading,
    error: dagError,
    hasRunningWorkflows,
    currentPollingInterval,
    refresh: refreshDAG
  } = useWorkflowDAGSmart(runId || null);

  useEffect(() => {
    if (!runId) {
      navigate("/executions");
      return;
    }

    const fetchWorkflow = async () => {
      try {
        setLoading(true);
        setError(null);

        const workflowSummary = await getWorkflowRunSummary(runId);

        if (!workflowSummary) {
          setError("Workflow not found");
          return;
        }

        setWorkflow(workflowSummary);
      } catch (err: any) {
        setError(
          err instanceof Error ? err.message : "Failed to load workflow"
        );
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflow();
  }, [runId, navigate]);

  // Fetch VC chain
  useEffect(() => {
    if (!workflow?.workflow_id) return;

    const fetchVCChain = async () => {
      try {
        setVcLoading(true);
        const vcData = await getWorkflowVCChain(workflow.workflow_id);
        setVcChain(vcData);
      } catch (err) {
        console.error('Failed to fetch VC chain:', err);
        setVcChain(null);
      } finally {
        setVcLoading(false);
      }
    };

    fetchVCChain();
  }, [workflow?.workflow_id]);


  // Keyboard support for Escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        handleClose();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleTagFilter = (tags: string[]) => {
    setTimelineSelectedTags(tags);
  };

  // Timeline state handlers
  const handleTimelineSortChange = (sortOrder: 'asc' | 'desc') => {
    setTimelineSortOrder(sortOrder);
  };

  const handleTimelineExpandAllChange = (expandAll: boolean) => {
    setTimelineExpandAll(expandAll);
  };

  const handleTimelineCardExpansionChange = (executionId: string, expanded: boolean) => {
    setTimelineCardExpansions(prev => ({
      ...prev,
      [executionId]: expanded
    }));
  };

  const handleTimelineNoteExpansionChange = (executionId: string, noteIndex: number, expanded: boolean) => {
    setTimelineNoteExpansions(prev => {
      const currentSet = prev[executionId] || new Set();
      const newSet = new Set(currentSet);

      if (expanded) {
        newSet.add(noteIndex);
      } else {
        newSet.delete(noteIndex);
      }

      return {
        ...prev,
        [executionId]: newSet
      };
    });
  };

  const handleClose = () => {
    navigate("/workflows");
  };

  if (loading) {
    return <WorkflowDetailSkeleton />;
  }

  if (error || !workflow) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <div className="text-red-600 mb-2">
                {error || "Workflow not found"}
              </div>
              <Button variant="outline" onClick={handleClose} className="mt-4">
                <ChevronLeft className="h-4 w-4 mr-2" />
                Back to Workflows
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      {/* Ultra-Compact Workflow Summary with Close and Live Updates */}
      <CompactWorkflowSummary
        workflow={workflow}
        onClose={handleClose}
        isLiveUpdating={!!dagData}
        hasRunningWorkflows={hasRunningWorkflows}
        pollingInterval={currentPollingInterval}
        isRefreshing={dagLoading}
        onRefresh={refreshDAG}
      />

      {/* DAG Error Display */}
      {dagError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded-md text-sm">
          Failed to load workflow data: {dagError.message}
          <button
            onClick={refreshDAG}
            className="ml-2 text-red-600 hover:text-red-800 underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Workflow VC Chain */}
      <SimpleWorkflowVC
        workflowId={workflow.workflow_id}
        vcChain={vcChain || undefined}
        loading={vcLoading}
      />

      {/* Compact Input/Output Display */}
      <CompactWorkflowInputOutput dagData={dagData} />

      {/* Two-Column Layout: DAG (2/3) + Timeline (1/3) */}
      <TwoColumnLayout
        leftColumn={
          <WorkflowDAGViewer
            workflowId={workflow.workflow_id}
            dagData={dagData}
            loading={dagLoading}
            error={dagError?.message || null}
            className="min-h-[800px]"
          />
        }
        rightColumn={
          <WorkflowTimeline
            nodes={dagData?.timeline || []}
            onTagFilter={handleTagFilter}
            className="min-h-[800px]"
            // Lifted state props
            sortOrder={timelineSortOrder}
            onSortOrderChange={handleTimelineSortChange}
            selectedTags={timelineSelectedTags}
            expandAll={timelineExpandAll}
            onExpandAllChange={handleTimelineExpandAllChange}
            cardExpansions={timelineCardExpansions}
            onCardExpansionChange={handleTimelineCardExpansionChange}
            noteExpansions={timelineNoteExpansions}
            onNoteExpansionChange={handleTimelineNoteExpansionChange}
          />
        }
        leftWidth="2/3"
        rightWidth="1/3"
        gap="md"
      />
    </div>
  );
}

function WorkflowDetailSkeleton() {
  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      {/* Ultra-Compact Summary Skeleton */}
      <div className="bg-card border border-border rounded-lg px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          {/* Left: Status + Name + ID */}
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <Skeleton className="h-4 w-4 rounded-full" />
            <Skeleton className="h-4 w-32" />
            <span className="text-muted-foreground/60">•</span>
            <Skeleton className="h-4 w-8" />
            <Skeleton className="h-5 w-48 rounded" />
            <Skeleton className="h-5 w-5 rounded" />
          </div>

          {/* Center: Metrics */}
          <div className="hidden md:flex items-center gap-3">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-12" />
          </div>

          {/* Right: Timestamps + Close */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-5 w-5 rounded" />
          </div>
        </div>

        {/* Mobile: Metrics Row */}
        <div className="md:hidden mt-2 pt-2 border-t border-border/50">
          <div className="flex items-center gap-3">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-12" />
            <Skeleton className="h-3 w-20" />
          </div>
        </div>
      </div>

      {/* VC Chain Skeleton */}
      <Card>
        <CardContent className="p-4">
          <Skeleton className="h-4 w-32 mb-2" />
          <Skeleton className="h-3 w-full" />
        </CardContent>
      </Card>

      {/* Two-Column Layout Skeleton */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* DAG Column Skeleton (2/3) */}
        <div className="w-full lg:w-2/3 min-w-0">
          <Card>
            <CardContent className="p-0">
              <div className="h-[800px] bg-muted/20 flex items-center justify-center">
                <div className="text-center space-y-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                  <div className="text-body-small">
                    Loading workflow DAG...
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Timeline Column Skeleton (1/3) */}
        <div className="w-full lg:w-1/3 min-w-0">
          <Card>
            <CardContent className="p-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Skeleton className="h-6 w-32" />
                  <div className="flex gap-2">
                    <Skeleton className="h-8 w-8" />
                    <Skeleton className="h-8 w-8" />
                  </div>
                </div>

                {/* Tag filter skeleton */}
                <div className="flex gap-2">
                  <Skeleton className="h-6 w-16" />
                  <Skeleton className="h-6 w-20" />
                  <Skeleton className="h-6 w-14" />
                </div>

                {/* Timeline cards skeleton */}
                <div className="space-y-4">
                  {Array.from({ length: 4 }).map((_, index) => (
                    <Card key={index} className="border-l-4 border-l-bg-tertiary">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between gap-3 mb-3">
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <Skeleton className="h-4 w-4" />
                            <div className="min-w-0 flex-1">
                              <Skeleton className="h-5 w-24 mb-1" />
                              <Skeleton className="h-4 w-32" />
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <Skeleton className="h-4 w-16 mb-1" />
                            <Skeleton className="h-4 w-12" />
                          </div>
                        </div>
                        <div className="flex items-center justify-between gap-3 mb-3">
                          <Skeleton className="h-4 w-20" />
                          <Skeleton className="h-6 w-16" />
                        </div>
                        <div className="flex gap-1">
                          <Skeleton className="h-5 w-12" />
                          <Skeleton className="h-5 w-16" />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
