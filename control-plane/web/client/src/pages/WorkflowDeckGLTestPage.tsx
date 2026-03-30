import { useCallback, useEffect, useMemo, useState } from "react";
import { getWorkflowDAGLightweight } from "../services/workflowsApi";
import type { WorkflowDAGLightweightResponse } from "../types/workflows";
import { WorkflowDeckGLView } from "../components/WorkflowDAG/DeckGLView";
import {
  buildDeckGraph,
  type DeckGraphData,
  type WorkflowDAGNode,
} from "../components/WorkflowDAG/DeckGLGraph";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardHeader } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";

const DEFAULT_WORKFLOW_ID = "run_0844e417efec";

function normalizeTimeline(
  response: WorkflowDAGLightweightResponse
): WorkflowDAGNode[] {
  return response.timeline.map((node) => ({
    ...node,
    workflow_id: response.root_workflow_id,
  }));
}

export function WorkflowDeckGLTestPage() {
  const [workflowId, setWorkflowId] = useState(DEFAULT_WORKFLOW_ID);
  const [inputValue, setInputValue] = useState(DEFAULT_WORKFLOW_ID);
  const [deckData, setDeckData] = useState<DeckGraphData | null>(null);
  const [summary, setSummary] = useState<{
    totalNodes: number;
    maxDepth: number;
    workflowName: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<WorkflowDAGNode | null>(null);

  const fetchData = useCallback(
    async (id: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await getWorkflowDAGLightweight(id);
        if (import.meta.env.DEV) {
          console.debug("[DeckGLTest] sample timeline", response.timeline.slice(0, 5));
        }
        const timeline = normalizeTimeline(response);
        const graph = buildDeckGraph(timeline);
        setDeckData(graph);
        setSummary({
          totalNodes: response.total_nodes,
          maxDepth: response.max_depth,
          workflowName: response.workflow_name,
        });
      } catch (err) {
        console.error("Failed to fetch lightweight DAG:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load workflow data"
        );
        setDeckData(null);
        setSummary(null);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    fetchData(workflowId);
  }, [workflowId, fetchData]);

  const handleSubmit = useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!inputValue.trim()) return;
      setWorkflowId(inputValue.trim());
    },
    [inputValue]
  );

  const stats = useMemo(() => {
    if (!deckData || !summary) return null;
    const nodeCount = deckData.nodes.length;
    const edgeCount = deckData.edges.length;
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-muted-foreground">
        <div>
          <div className="font-semibold text-lg text-foreground">
            {summary.workflowName || "Unnamed Workflow"}
          </div>
          <div>Workflow</div>
        </div>
        <div>
          <div className="font-semibold text-lg text-foreground">
            {nodeCount.toLocaleString()}
          </div>
          <div>Nodes rendered</div>
        </div>
        <div>
          <div className="font-semibold text-lg text-foreground">
            {edgeCount.toLocaleString()}
          </div>
          <div>Edges rendered</div>
        </div>
      </div>
    );
  }, [deckData, summary]);

  return (
    <div className="flex flex-col gap-6 h-full px-6 pb-6">
      <Card>
        <CardHeader>
          <h2 className="text-heading-3 text-foreground">
            Deck.GL Workflow Visualization Test
          </h2>
          <p className="text-body text-muted-foreground">
            High-density workflow visualization with smooth curves and agent-aware colors.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <form
            className="flex flex-col lg:flex-row gap-3 items-start lg:items-end"
            onSubmit={handleSubmit}
          >
            <div className="flex-1 w-full">
              <label
                className="block text-sm font-medium text-muted-foreground mb-1"
                htmlFor="workflow-id-input"
              >
                Workflow ID
              </label>
              <Input
                id="workflow-id-input"
                placeholder="run_xxx"
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
              />
            </div>
            <Button type="submit" disabled={loading}>
              {loading ? "Loading..." : "Render"}
            </Button>
          </form>
          {summary && (
            <div className="space-y-3">
              {stats}
              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                Max depth:&nbsp;
                <span className="font-semibold text-foreground">
                  {summary.maxDepth}
                </span>
              </div>
            </div>
          )}
          {deckData?.agentPalette.length ? (
            <div className="flex flex-wrap gap-2 pt-1">
              {deckData.agentPalette.map((agent) => (
                <div
                  key={agent.agentId}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs border border-white/10 shadow-sm"
                  style={{
                    background: agent.background,
                    color: agent.text,
                  }}
                >
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full shadow"
                    style={{ background: agent.color }}
                  />
                  <span className="font-medium">{agent.label}</span>
                </div>
              ))}
            </div>
          ) : null}
          {error && (
            <div className="mt-2 text-sm text-red-500">{error}</div>
          )}
        </CardContent>
      </Card>

      <div className="relative flex-1 min-h-[540px] rounded-3xl overflow-hidden border border-white/5 bg-[#0b1220]">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.18),transparent_55%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(147,51,234,0.14),transparent_55%)]" />
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(15,23,42,0.6),rgba(2,6,23,0.8))]" />
        </div>
        {loading && (
          <div className="relative z-10 w-full h-full flex items-center justify-center">
            <div className="space-y-4 text-center">
              <Skeleton className="h-10 w-10 rounded-full mx-auto bg-white/10" />
              <div className="text-sm text-slate-300">
                Rendering deck.gl scene…
              </div>
            </div>
          </div>
        )}
        {!loading && deckData && (
          <div className="relative z-10 h-full">
            <WorkflowDeckGLView
              nodes={deckData.nodes}
              edges={deckData.edges}
              onNodeClick={(node) =>
                console.log("Node selected:", node.execution_id)
              }
              onNodeHover={(node) => setHoveredNode(node)}
            />
            {hoveredNode && (
              <div className="absolute bottom-4 left-4 max-w-sm rounded-2xl bg-black/60 border border-white/10 backdrop-blur px-4 py-3 text-xs text-slate-100 shadow-lg">
                <div className="text-sm font-semibold text-white">
                  {hoveredNode.reasoner_id || hoveredNode.execution_id}
                </div>
                <div className="grid grid-cols-2 gap-y-1 mt-2 text-[11px] text-slate-300">
                  <span className="uppercase tracking-wide text-slate-400">
                    Agent
                  </span>
                  <span>{hoveredNode.agent_node_id ?? "—"}</span>
                  <span className="uppercase tracking-wide text-slate-400">
                    Status
                  </span>
                  <span>{hoveredNode.status}</span>
                  <span className="uppercase tracking-wide text-slate-400">
                    Depth
                  </span>
                  <span>{hoveredNode.workflow_depth ?? "—"}</span>
                  <span className="uppercase tracking-wide text-slate-400">
                    Execution
                  </span>
                  <span className="truncate">{hoveredNode.execution_id}</span>
                </div>
              </div>
            )}
          </div>
        )}
        {!loading && !deckData && !error && (
          <div className="relative z-10 w-full h-full flex items-center justify-center">
            <div className="text-sm text-slate-300">
              Enter a workflow ID to render the visualization.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
