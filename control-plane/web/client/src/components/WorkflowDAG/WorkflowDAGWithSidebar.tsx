import { useState } from "react";
import { NodeDetailSidebar } from "./NodeDetailSidebar";

interface WorkflowNodeData {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  workflow_depth: number;
  task_name?: string;
  agent_name?: string;
}

interface WorkflowDAGWithSidebarProps {
  // Your existing DAG component props
  nodes: WorkflowNodeData[];
  edges: any[];
  // Add any other props your DAG component needs
}

export function WorkflowDAGWithSidebar({
  nodes,
}: WorkflowDAGWithSidebarProps) {
  const [selectedNode, setSelectedNode] = useState<WorkflowNodeData | null>(
    null
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleNodeClick = (node: WorkflowNodeData) => {
    setSelectedNode(node);
    setSidebarOpen(true);
  };

  const handleCloseSidebar = () => {
    setSidebarOpen(false);
    // Optionally clear selected node after animation
    setTimeout(() => setSelectedNode(null), 300);
  };

  return (
    <div className="relative w-full h-full">
      {/* Your existing DAG component */}
      <div className="w-full h-full">
        {/* Example DAG rendering - replace with your actual DAG component */}
        <div className="p-4 space-y-4">
          <h3 className="text-base font-semibold text-foreground">
            Workflow DAG
          </h3>

          {/* Example node rendering - replace with your actual node rendering logic */}
          <div className="grid grid-cols-1 gap-2">
            {nodes.map((node) => (
              <button
                key={node.execution_id}
                onClick={() => handleNodeClick(node)}
                className="p-3 text-left border border-border rounded-lg hover:bg-muted transition-colors"
              >
                <div className="text-sm font-medium text-foreground">
                  {node.task_name || node.reasoner_id}
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  Status: {node.status} | Agent:{" "}
                  {node.agent_name || node.agent_node_id}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Node Detail Sidebar */}
      <NodeDetailSidebar
        node={selectedNode}
        isOpen={sidebarOpen}
        onClose={handleCloseSidebar}
      />
    </div>
  );
}

// Example usage component
export function ExampleWorkflowDAGUsage() {
  // Example data - replace with your actual data
  const exampleNodes: WorkflowNodeData[] = [
    {
      workflow_id: "wf_123456789",
      execution_id: "exec_987654321",
      agent_node_id: "agent_001",
      reasoner_id: "sentiment_analyzer",
      status: "succeeded",
      started_at: "2024-01-15T10:30:00Z",
      completed_at: "2024-01-15T10:30:05Z",
      duration_ms: 5000,
      workflow_depth: 1,
      task_name: "Analyze Customer Sentiment",
      agent_name: "Support Agent",
    },
    {
      workflow_id: "wf_123456789",
      execution_id: "exec_987654322",
      agent_node_id: "agent_002",
      reasoner_id: "response_generator",
      status: "running",
      started_at: "2024-01-15T10:30:05Z",
      duration_ms: 2000,
      workflow_depth: 2,
      task_name: "Generate Response",
      agent_name: "Support Agent",
    },
    {
      workflow_id: "wf_123456789",
      execution_id: "exec_987654323",
      agent_node_id: "agent_003",
      reasoner_id: "quality_checker",
      status: "failed",
      started_at: "2024-01-15T10:30:10Z",
      completed_at: "2024-01-15T10:30:12Z",
      duration_ms: 2000,
      workflow_depth: 3,
      task_name: "Quality Check",
      agent_name: "QA Agent",
    },
  ];

  return (
    <div className="w-full h-screen bg-background">
      <WorkflowDAGWithSidebar nodes={exampleNodes} edges={[]} />
    </div>
  );
}
