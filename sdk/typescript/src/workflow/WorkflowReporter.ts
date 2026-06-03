import type { AgentFieldClient } from '../client/AgentFieldClient.js';

export interface WorkflowMetadata {
  executionId: string;
  runId?: string;
  workflowId?: string;
  agentNodeId?: string;
  reasonerId?: string;
}

export interface WorkflowProgressOptions {
  status?: string;
  result?: Record<string, any>;
  error?: string;
  durationMs?: number;
}

export class WorkflowReporter {
  private readonly client: AgentFieldClient;
  private readonly metadata: WorkflowMetadata;

  constructor(client: AgentFieldClient, metadata: WorkflowMetadata) {
    if (!metadata.executionId) {
      throw new Error('WorkflowReporter requires an executionId');
    }
    this.client = client;
    this.metadata = metadata;
  }

  async progress(progress: number, options?: WorkflowProgressOptions) {
    const normalized = Math.min(100, Math.max(0, Math.round(progress)));
    return this.client.updateExecutionStatus(this.metadata.executionId, {
      status: options?.status ?? 'running',
      progress: normalized,
      result: options?.result,
      error: options?.error,
      durationMs: options?.durationMs
    });
  }
}
