export interface AgentNode {
  id: string;
  base_url: string;
  version: string;
  health_status: string;
  deployment_type?: string; // "long_running" or "serverless"
  invocation_url?: string; // For serverless agents
}

export interface Execution {
  execution_id: string;
  workflow_id: string;
  status: string;
  updated_at: string;
}

export interface Workflow {
  workflow_id: string;
  name: string;
  updated_at: string;
}

export interface Reasoner {
  id: string;
  name: string;
  description: string;
}
