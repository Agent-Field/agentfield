import type express from 'express';

// Minimal structural type — we just need to log info-level messages on
// successful cancel. Using a structural type keeps cancel.ts independent
// of whatever logger the agent uses today.
type CancelLogger = { info(message: string, meta?: Record<string, unknown>): void };

/**
 * Cooperative cancellation registry.
 *
 * The control plane's cancel dispatcher (control-plane/internal/services/
 * cancel_dispatcher.go) POSTs to `/_internal/executions/:execution_id/cancel`
 * whenever an execution flips to cancelled — from per-execution cancel,
 * the bottom-up cancel-tree endpoint, or any future source publishing on
 * the bus. This module holds an `AbortController` per in-flight reasoner
 * invocation; the cancel route looks up the controller and calls
 * `.abort()`, which surfaces as `signal.aborted === true` and rejects any
 * pending `fetch()` / Anthropic-SDK / OpenAI-SDK request bound to that
 * signal.
 *
 * Reasoner-author contract:
 *   - `await fetch(url, { signal: ctx.signal })` — aborts mid-flight.
 *   - The official @anthropic-ai/sdk and openai clients accept
 *     `{ signal }` in their request options. Pass `ctx.signal` through.
 *   - For CPU loops or pure-JS work, periodically check
 *     `ctx.signal.aborted` and throw if true.
 *
 * Backwards compatible: reasoners that ignore `signal` finish naturally
 * and their output is discarded by the control plane (existing per-
 * execution cancel behaviour). Workers running without a dispatcher in
 * front of them simply never see the cancel callback.
 */
export class CancelRegistry {
  private readonly controllers = new Map<string, AbortController>();

  /**
   * Register a fresh AbortController against an execution_id. Returns the
   * controller and a `release()` cleanup function that must be called on
   * completion (success, failure, or cancellation) to drop the entry.
   * Empty `executionId` produces a controller that is NOT registered,
   * so callers always get a usable signal even outside the control-plane
   * dispatch path (e.g. local manual invocations).
   */
  register(executionId: string | undefined): {
    controller: AbortController;
    release: () => void;
  } {
    const controller = new AbortController();
    if (!executionId) {
      return { controller, release: () => {} };
    }
    this.controllers.set(executionId, controller);
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      // Only delete our own entry — a racing register() that replaced
      // us shouldn't get its registration clobbered.
      const current = this.controllers.get(executionId);
      if (current === controller) {
        this.controllers.delete(executionId);
      }
    };
    return { controller, release };
  }

  /**
   * Cancel the AbortController registered for `executionId`. Returns
   * true if a matching controller was found and aborted, false if there
   * was no active execution (already finished, or never dispatched here).
   */
  cancel(executionId: string, reason?: string): boolean {
    if (!executionId) return false;
    const controller = this.controllers.get(executionId);
    if (!controller) return false;
    if (controller.signal.aborted) {
      // Already aborted by something else; treat as a no-op success so
      // double-cancels are idempotent.
      return true;
    }
    controller.abort(reason ?? 'cancelled_by_control_plane');
    this.controllers.delete(executionId);
    return true;
  }

  /** Number of in-flight registrations. Useful for tests. */
  size(): number {
    return this.controllers.size;
  }
}

/**
 * Install the `POST /_internal/executions/:execution_id/cancel` route on
 * the agent's Express app. The route bypasses path-based DID verification
 * by way of its prefix (control-plane→worker notification, not a
 * DID-signed user-initiated call). Bearer-token origin auth, when
 * configured, still applies through the existing middleware path.
 */
export function installCancelRoute(
  app: express.Express,
  registry: CancelRegistry,
  logger?: CancelLogger
): void {
  app.post(
    '/_internal/executions/:executionId/cancel',
    (req: express.Request, res: express.Response) => {
      const executionId = (req.params.executionId ?? '').trim();
      if (!executionId) {
        res.status(404).json({ error: 'invalid execution_id' });
        return;
      }
      const cancelled = registry.cancel(
        executionId,
        typeof req.body?.reason === 'string' ? req.body.reason : undefined
      );
      if (cancelled) {
        logger?.info('cancel-callback fired', {
          executionId,
          source: req.headers['x-agentfield-source'] ?? 'unknown'
        });
        res.status(200).json({ cancelled: true, execution_id: executionId });
      } else {
        // 200 with the marker keeps the dispatcher's best-effort logic
        // happy and avoids spurious warning logs. The execution may
        // have already finished naturally, or never landed here.
        res.status(200).json({
          cancelled: false,
          execution_id: executionId,
          reason: 'execution_not_active'
        });
      }
    }
  );
}
