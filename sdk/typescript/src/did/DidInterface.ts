import type { ExecutionMetadata } from '../context/ExecutionContext.js';
import { DidClient, type AuditTrailFilters, type ExecutionCredential, type AuditTrailExport } from './DidClient.js';

export interface GenerateCredentialOptions {
  inputData?: any;
  outputData?: any;
  status?: string;
  errorMessage?: string;
  durationMs?: number;
  timestamp?: string | Date;
  callerDid?: string;
  targetDid?: string;
  agentNodeDid?: string;
  workflowId?: string;
  sessionId?: string;
  executionId?: string;
  headers?: Record<string, string>;
}

export class DidInterface {
  private readonly client: DidClient;
  private readonly metadata: ExecutionMetadata;
  private readonly enabled: boolean;
  private readonly defaultInput: any;

  constructor(params: { client: DidClient; metadata: ExecutionMetadata; enabled: boolean; defaultInput?: any }) {
    this.client = params.client;
    this.metadata = params.metadata;
    this.enabled = params.enabled;
    this.defaultInput = params.defaultInput;
  }

  async generateCredential(options: GenerateCredentialOptions = {}): Promise<ExecutionCredential> {
    if (!this.enabled) {
      throw new Error('DID/VC features are disabled. Enable didEnabled in AgentConfig to use ctx.did.');
    }

    const executionContext = {
      executionId: options.executionId ?? this.metadata.executionId,
      workflowId: options.workflowId ?? this.metadata.workflowId ?? this.metadata.runId,
      sessionId: options.sessionId ?? this.metadata.sessionId,
      callerDid: options.callerDid ?? this.metadata.callerDid,
      targetDid: options.targetDid ?? this.metadata.targetDid,
      agentNodeDid: options.agentNodeDid ?? this.metadata.agentNodeDid,
      timestamp: options.timestamp
    };

    return this.client.generateCredential({
      executionContext,
      inputData: options.inputData ?? this.defaultInput,
      outputData: options.outputData,
      status: options.status,
      errorMessage: options.errorMessage,
      durationMs: options.durationMs,
      headers: options.headers
    });
  }

  exportAuditTrail(filters?: AuditTrailFilters): Promise<AuditTrailExport> {
    if (!this.enabled) {
      throw new Error('DID/VC features are disabled. Enable didEnabled in AgentConfig to use ctx.did.');
    }
    return this.client.exportAuditTrail(filters);
  }
}
