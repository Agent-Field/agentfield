import { ChevronDown, ChevronUp, Filter, Search } from "@/components/ui/icon-bridge";
import type { Node } from "@xyflow/react";
import { useMemo, useState } from "react";
import { cn } from "../../lib/utils";
import { agentColorManager } from "../../utils/agentColorManager";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { AgentBadge, AgentColorDot } from "./AgentBadge";

interface WorkflowDAGNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  parent_execution_id?: string;
  workflow_depth: number;
  children: WorkflowDAGNode[];
  agent_name?: string;
  task_name?: string;
}

interface AgentLegendProps {
  className?: string;
  onAgentFilter?: (agentName: string | null) => void;
  selectedAgent?: string | null;
  compact?: boolean;
  nodes?: Node[]; // Add nodes prop to filter agents by current workflow
}

export function AgentLegend({
  className,
  onAgentFilter,
  selectedAgent,
  compact = false,
  nodes = [],
}: AgentLegendProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");

  // Extract unique agents from current workflow nodes
  const workflowAgents = useMemo(() => {
    const agentSet = new Set<string>();

    nodes.forEach((node) => {
      const nodeData = node.data as unknown as WorkflowDAGNode;
      const agentName = nodeData.agent_name || nodeData.agent_node_id;
      if (agentName) {
        agentSet.add(agentName);
      }
    });

    const agents = Array.from(agentSet);
    return agents;
  }, [nodes]);

  // Get agent colors only for agents in current workflow
  const agentColors = useMemo(() => {
    // Clean up unused agents from the color manager
    if (workflowAgents.length > 0) {
      agentColorManager.cleanupUnusedAgents(workflowAgents);
    }

    return workflowAgents.map((agentName) => {
      // Ensure the agent color is registered
      const agentColor = agentColorManager.getAgentColor(agentName);
      return agentColor;
    });
  }, [workflowAgents]);

  // Filter agents based on search term
  const filteredAgents = agentColors.filter((agent) =>
    agent.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Don't render if no agents
  if (agentColors.length === 0) {
    return null;
  }

  // Compact mode for small number of agents
  if (compact || agentColors.length <= 6) {
    return (
      <div
        className={cn(
          "bg-background/90 backdrop-blur-sm border rounded-lg p-3 shadow-sm",
          "min-w-[200px] max-w-[300px]",
          className
        )}
      >
        <div className="flex items-center gap-2 mb-3">
          <Filter size={14} className="text-muted-foreground" />
          <span className="text-sm font-medium">Agents</span>
          <span className="text-body-small">
            ({agentColors.length})
          </span>
        </div>

        <div className="space-y-2">
          {agentColors.map((agent) => (
            <button
              key={agent.name}
              onClick={() =>
                onAgentFilter?.(
                  selectedAgent === agent.name ? null : agent.name
                )
              }
              className={cn(
                "w-full flex items-center gap-3 p-2 rounded-md text-left",
                "hover:bg-muted/50 transition-colors duration-150",
                "focus:outline-none focus:ring-2 focus:ring-primary/20",
                selectedAgent === agent.name &&
                  "bg-muted ring-2 ring-primary/30"
              )}
            >
              <AgentBadge
                agentName={agent.name}
                size="sm"
                showTooltip={false}
              />
              <span className="text-sm font-medium truncate flex-1">
                {agent.name}
              </span>
              {selectedAgent === agent.name && (
                <div className="w-2 h-2 rounded-full bg-primary" />
              )}
            </button>
          ))}
        </div>

        {selectedAgent && (
          <button
            onClick={() => onAgentFilter?.(null)}
            className="w-full mt-3 text-body-small hover:text-foreground transition-colors"
          >
            Clear filter
          </button>
        )}
      </div>
    );
  }

  // Full legend for many agents
  return (
    <div
      className={cn(
        "bg-background/90 backdrop-blur-sm border rounded-lg shadow-sm",
        "min-w-[250px] max-w-[350px]",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-muted-foreground" />
          <span className="text-sm font-medium">Agents</span>
          <span className="text-body-small">
            ({agentColors.length})
          </span>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="h-6 w-6 p-0"
        >
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </Button>
      </div>

      {isExpanded && (
        <>
          {/* Search */}
          {agentColors.length > 6 && (
            <div className="p-3 border-b">
              <div className="relative">
                <Search
                  size={14}
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                />
                <Input
                  placeholder="Search agents..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-9 h-8 text-sm"
                />
              </div>
            </div>
          )}

          {/* Agent List */}
          <div className="p-2">
            <div
              className={cn(
                "space-y-1",
                filteredAgents.length > 8 && "max-h-64 overflow-y-auto"
              )}
            >
              {filteredAgents.map((agent) => (
                <button
                  key={agent.name}
                  onClick={() =>
                    onAgentFilter?.(
                      selectedAgent === agent.name ? null : agent.name
                    )
                  }
                  className={cn(
                    "w-full flex items-center gap-3 p-2 rounded-md text-left",
                    "hover:bg-muted/50 transition-colors duration-150",
                    "focus:outline-none focus:ring-2 focus:ring-primary/20",
                    selectedAgent === agent.name &&
                      "bg-muted ring-2 ring-primary/30"
                  )}
                >
                  <AgentBadge
                    agentName={agent.name}
                    size="sm"
                    showTooltip={false}
                  />
                  <span className="text-sm font-medium truncate flex-1">
                    {agent.name}
                  </span>
                  {selectedAgent === agent.name && (
                    <div className="w-2 h-2 rounded-full bg-primary" />
                  )}
                </button>
              ))}
            </div>

            {filteredAgents.length === 0 && searchTerm && (
              <div className="text-center py-4 text-body-small">
                No agents found matching "{searchTerm}"
              </div>
            )}
          </div>

          {/* Footer */}
          {selectedAgent && (
            <div className="p-3 border-t">
              <button
                onClick={() => onAgentFilter?.(null)}
                className="w-full text-body-small hover:text-foreground transition-colors"
              >
                Clear filter
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Minimal legend for very constrained spaces
export function AgentLegendMini({
  className,
  onAgentFilter,
  selectedAgent,
  nodes = [],
}: AgentLegendProps) {
  // Extract unique agents from current workflow nodes
  const workflowAgents = useMemo(() => {
    const agentSet = new Set<string>();

    nodes.forEach((node) => {
      const nodeData = node.data as unknown as WorkflowDAGNode;
      const agentName = nodeData.agent_name || nodeData.agent_node_id;
      if (agentName) {
        agentSet.add(agentName);
      }
    });

    return Array.from(agentSet);
  }, [nodes]);

  // Get agent colors only for agents in current workflow
  const agentColors = useMemo(() => {
    // Clean up unused agents from the color manager
    if (workflowAgents.length > 0) {
      agentColorManager.cleanupUnusedAgents(workflowAgents);
    }

    return workflowAgents.map((agentName) => {
      // Ensure the agent color is registered
      const agentColor = agentColorManager.getAgentColor(agentName);
      return agentColor;
    });
  }, [workflowAgents]);

  if (agentColors.length === 0) return null;

  return (
    <div
      className={cn(
        "bg-background/90 backdrop-blur-sm border rounded-lg p-2 shadow-sm",
        "flex items-center gap-2 flex-wrap",
        className
      )}
    >
      <span className="text-body-small mr-1">Agents:</span>
      {agentColors.slice(0, 8).map((agent) => (
        <button
          key={agent.name}
          onClick={() =>
            onAgentFilter?.(selectedAgent === agent.name ? null : agent.name)
          }
          className={cn(
            "relative transition-transform duration-150",
            "hover:scale-110 focus:outline-none focus:scale-110",
            selectedAgent === agent.name &&
              "ring-2 ring-primary/50 rounded-full"
          )}
          title={agent.name}
        >
          <AgentColorDot
            agentName={agent.name}
            size={12}
            className={cn(
              selectedAgent === agent.name && "ring-2 ring-primary/30"
            )}
          />
        </button>
      ))}
      {agentColors.length > 8 && (
        <span className="text-body-small">
          +{agentColors.length - 8}
        </span>
      )}
    </div>
  );
}
