import type express from 'express';

// Minimal structural logger type — keeps pause.ts independent of whatever
// logger the agent uses today (same approach as cancel.ts).
type PauseLogger = { info(message: string, meta?: Record<string, unknown>): void };

/** Canonical human-decision values carried on an {@link ApprovalResult}. */
export type ApprovalDecision =
  | 'approved'
  | 'rejected'
  | 'request_changes'
  | 'expired'
  | 'error'
  | 'cancelled'
  | (string & {});

/**
 * The outcome of a paused execution once a human (or upstream service)
 * resolves it. Mirrors the Python SDK's `ApprovalResult` dataclass so a
 * reasoner authored against either SDK reads the same fields.
 */
export class ApprovalResult {
  readonly decision: ApprovalDecision;
  readonly feedback: string;
  readonly executionId: string;
  readonly approvalRequestId: string;
  readonly rawResponse?: Record<string, any>;

  constructor(params: {
    decision: ApprovalDecision;
    feedback?: string;
    executionId?: string;
    approvalRequestId?: string;
    rawResponse?: Record<string, any>;
  }) {
    this.decision = params.decision;
    this.feedback = params.feedback ?? '';
    this.executionId = params.executionId ?? '';
    this.approvalRequestId = params.approvalRequestId ?? '';
    this.rawResponse = params.rawResponse;
  }

  /** True when the human approved the request. */
  get approved(): boolean {
    return this.decision === 'approved';
  }

  /** True when the human asked for changes rather than approving/rejecting. */
  get changesRequested(): boolean {
    return this.decision === 'request_changes';
  }
}

/**
 * Tracks how long a single execution has spent inside `ctx.pause()`.
 *
 * The async-execution watchdog (see Agent.runReasonerAsync) must not count
 * time spent waiting for an external approval against the reasoner's active
 * wall-clock budget — otherwise `expiresInHours` on a pause would be silently
 * capped at the reasoner timeout. The watchdog reads `totalPaused()` and
 * subtracts it from elapsed wall-clock. On the awaiter side, the same clock
 * is used so a parent that is blocked waiting on a paused descendant does not
 * burn its own budget.
 *
 * A reasoner runs as a single async chain, so pause intervals cannot overlap
 * on one clock; no locking is needed. All values are milliseconds.
 */
export class PauseClock {
  private totalPausedMs = 0;
  private pauseStartedAt: number | null = null;
  /**
   * Set by the watchdog when it aborts the reasoner for exceeding the active
   * budget. Distinguishes a budget-timeout abort from an external cooperative
   * cancel arriving via the cancel dispatcher.
   */
  timedOut = false;

  startPause(): void {
    if (this.pauseStartedAt === null) {
      this.pauseStartedAt = Date.now();
    }
  }

  endPause(): void {
    if (this.pauseStartedAt !== null) {
      this.totalPausedMs += Date.now() - this.pauseStartedAt;
      this.pauseStartedAt = null;
    }
  }

  /** Cumulative paused milliseconds, including any in-progress pause. */
  totalPaused(): number {
    if (this.pauseStartedAt === null) {
      return this.totalPausedMs;
    }
    return this.totalPausedMs + (Date.now() - this.pauseStartedAt);
  }
}

interface PendingPause {
  resolve: (result: ApprovalResult) => void;
  promise: Promise<ApprovalResult>;
}

/**
 * Registry of pending execution-pause promises resolved via webhook callback.
 *
 * Each `ctx.pause()` call registers a promise keyed by `approvalRequestId`.
 * When the control plane POSTs a resolution to the agent's
 * `/webhooks/approval` route, the matching promise is resolved and the paused
 * reasoner unblocks. Mirrors the Python SDK's `_PauseManager`.
 */
export class PauseManager {
  private readonly pending = new Map<string, PendingPause>();
  /** execution_id -> approval_request_id, for fallback resolution. */
  private readonly execToRequest = new Map<string, string>();

  /**
   * Register a new pending pause and return the promise to await. Idempotent:
   * a second register() for the same `approvalRequestId` returns the existing
   * promise rather than replacing it.
   */
  register(approvalRequestId: string, executionId = ''): Promise<ApprovalResult> {
    const existing = this.pending.get(approvalRequestId);
    if (existing) {
      return existing.promise;
    }
    let resolveFn!: (result: ApprovalResult) => void;
    const promise = new Promise<ApprovalResult>((resolve) => {
      resolveFn = resolve;
    });
    this.pending.set(approvalRequestId, { resolve: resolveFn, promise });
    if (executionId) {
      this.execToRequest.set(executionId, approvalRequestId);
    }
    return promise;
  }

  /**
   * Resolve a pending pause by `approvalRequestId`. Returns true if a waiter
   * was found and resolved, false otherwise.
   */
  resolve(approvalRequestId: string, result: ApprovalResult): boolean {
    const entry = this.pending.get(approvalRequestId);
    if (!entry) {
      return false;
    }
    this.pending.delete(approvalRequestId);
    for (const [eid, rid] of this.execToRequest) {
      if (rid === approvalRequestId) {
        this.execToRequest.delete(eid);
        break;
      }
    }
    entry.resolve(result);
    return true;
  }

  /**
   * Fallback: resolve by `executionId` when the callback omits the
   * `approvalRequestId`. Returns true if a waiter was found.
   */
  resolveByExecutionId(executionId: string, result: ApprovalResult): boolean {
    const requestId = this.execToRequest.get(executionId);
    if (!requestId) {
      return false;
    }
    return this.resolve(requestId, result);
  }

  /**
   * Resolve every pending pause with a `cancelled` result. Used on shutdown so
   * a reasoner blocked in `ctx.pause()` doesn't hang the process forever.
   */
  cancelAll(): void {
    for (const [approvalRequestId, entry] of this.pending) {
      entry.resolve(
        new ApprovalResult({
          decision: 'cancelled',
          feedback: 'agent shutting down',
          approvalRequestId,
        })
      );
    }
    this.pending.clear();
    this.execToRequest.clear();
  }

  /** Number of currently-pending pauses. Useful for tests. */
  pendingCount(): number {
    return this.pending.size;
  }
}

/**
 * Parse the `response` field of an approval webhook callback, which the
 * control plane may deliver as either a JSON object or a JSON-encoded string.
 */
function parseRawResponse(raw: unknown): Record<string, any> | undefined {
  if (raw && typeof raw === 'object') {
    return raw as Record<string, any>;
  }
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        return parsed as Record<string, any>;
      }
    } catch {
      // Non-JSON string response — surface it under a `text` key so the
      // reasoner can still read it rather than silently dropping it.
      return { text: raw };
    }
  }
  return undefined;
}

/**
 * Install the `POST /webhooks/approval` route on the agent's Express app.
 *
 * The control plane POSTs here (via the `callback_url` registered when the
 * execution paused) once a human resolves the approval. The body carries
 * `{ execution_id, decision, feedback, approval_request_id, response }`. We
 * resolve the matching pending pause — first by `approval_request_id`, then by
 * `execution_id` as a fallback — and reply `{ status, resolved }`.
 *
 * Always installed so the control plane reaches the worker regardless of which
 * routes the user registered first (same rationale as installCancelRoute).
 */
export function installApprovalWebhookRoute(
  app: express.Express,
  manager: PauseManager,
  logger?: PauseLogger
): void {
  app.post(
    '/webhooks/approval',
    (req: express.Request, res: express.Response) => {
      const body = (req.body ?? {}) as Record<string, unknown>;
      const executionId = String(body.execution_id ?? '');
      const approvalRequestId = String(body.approval_request_id ?? '');
      const decision = String(body.decision ?? '');
      const feedback = typeof body.feedback === 'string' ? body.feedback : '';
      const rawResponse = parseRawResponse(body.response);

      const result = new ApprovalResult({
        decision,
        feedback,
        executionId,
        approvalRequestId,
        rawResponse,
      });

      let resolved = false;
      if (approvalRequestId) {
        resolved = manager.resolve(approvalRequestId, result);
      }
      if (!resolved && executionId) {
        resolved = manager.resolveByExecutionId(executionId, result);
      }

      logger?.info('approval-webhook received', {
        executionId,
        approvalRequestId,
        decision,
        resolved,
      });

      res.status(200).json({ status: 'received', resolved });
    }
  );
}
