/**
 * Approval workflow helpers for the AgentField TypeScript SDK.
 *
 * Provides methods to request human approval for an execution,
 * poll for approval status, and wait until resolved.
 */

import axios, { type AxiosInstance } from 'axios';
import { httpAgent, httpsAgent } from '../utils/httpAgents.js';

/** Payload sent when requesting approval. */
export interface RequestApprovalPayload {
  title?: string;
  description?: string;
  templateType?: string;
  payload?: Record<string, any>;
  projectId: string;
  expiresInHours?: number;
}

/** Response from the control plane after creating an approval request. */
export interface ApprovalRequestResponse {
  approvalRequestId: string;
  approvalRequestUrl: string;
}

/** Approval status returned by the polling endpoint. */
export interface ApprovalStatusResponse {
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  response?: Record<string, any>;
  requestUrl?: string;
  requestedAt?: string;
  respondedAt?: string;
}

/** Options for the blocking `waitForApproval` helper. */
export interface WaitForApprovalOptions {
  /** Initial polling interval in milliseconds (default: 5000). */
  pollIntervalMs?: number;
  /** Maximum polling interval in milliseconds (default: 60000). */
  maxIntervalMs?: number;
  /** Total timeout in milliseconds (default: unlimited). */
  timeoutMs?: number;
}

export class ApprovalClient {
  private readonly http: AxiosInstance;
  private readonly nodeId: string;
  private readonly headers: Record<string, string>;

  constructor(opts: {
    baseURL: string;
    nodeId: string;
    apiKey?: string;
    headers?: Record<string, string>;
  }) {
    this.http = axios.create({
      baseURL: opts.baseURL.replace(/\/$/, ''),
      timeout: 30_000,
      httpAgent,
      httpsAgent,
    });
    this.nodeId = opts.nodeId;

    const merged: Record<string, string> = { ...(opts.headers ?? {}) };
    if (opts.apiKey) {
      merged['X-API-Key'] = opts.apiKey;
    }
    this.headers = merged;
  }

  /**
   * Request human approval, transitioning the execution to `waiting`.
   *
   * Calls `POST /api/v1/agents/{node}/executions/{id}/request-approval`.
   */
  async requestApproval(
    executionId: string,
    payload: RequestApprovalPayload
  ): Promise<ApprovalRequestResponse> {
    const body = {
      title: payload.title ?? 'Approval Request',
      description: payload.description ?? '',
      template_type: payload.templateType ?? 'plan-review-v1',
      payload: payload.payload ?? {},
      project_id: payload.projectId,
      expires_in_hours: payload.expiresInHours ?? 72,
    };

    const res = await this.http.post(
      `/api/v1/agents/${encodeURIComponent(this.nodeId)}/executions/${encodeURIComponent(executionId)}/request-approval`,
      body,
      { headers: { ...this.headers, 'Content-Type': 'application/json' } }
    );

    return {
      approvalRequestId: res.data.approval_request_id ?? '',
      approvalRequestUrl: res.data.approval_request_url ?? '',
    };
  }

  /**
   * Get the current approval status for an execution.
   *
   * Calls `GET /api/v1/agents/{node}/executions/{id}/approval-status`.
   */
  async getApprovalStatus(executionId: string): Promise<ApprovalStatusResponse> {
    const res = await this.http.get(
      `/api/v1/agents/${encodeURIComponent(this.nodeId)}/executions/${encodeURIComponent(executionId)}/approval-status`,
      { headers: this.headers }
    );

    const data = res.data;
    return {
      status: data.status ?? 'pending',
      response: data.response,
      requestUrl: data.request_url,
      requestedAt: data.requested_at,
      respondedAt: data.responded_at,
    };
  }

  /**
   * Poll approval status with exponential backoff until resolved.
   *
   * Returns once the status is no longer `pending` (i.e. approved, rejected,
   * or expired).
   */
  async waitForApproval(
    executionId: string,
    opts?: WaitForApprovalOptions
  ): Promise<ApprovalStatusResponse> {
    const pollInterval = opts?.pollIntervalMs ?? 5_000;
    const maxInterval = opts?.maxIntervalMs ?? 60_000;
    const timeout = opts?.timeoutMs;
    const backoffFactor = 2;

    const startTime = Date.now();
    let interval = pollInterval;

    while (true) {
      if (timeout != null && Date.now() - startTime >= timeout) {
        throw new Error(
          `Approval for execution ${executionId} timed out after ${timeout}ms`
        );
      }

      await sleep(interval);

      let data: ApprovalStatusResponse;
      try {
        data = await this.getApprovalStatus(executionId);
      } catch {
        // Transient failure — back off and retry
        interval = Math.min(interval * backoffFactor, maxInterval);
        continue;
      }

      if (data.status !== 'pending') {
        return data;
      }

      interval = Math.min(interval * backoffFactor, maxInterval);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
