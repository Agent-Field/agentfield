/**
 * Trigger dispatch envelope detection and TriggerContext construction.
 *
 * When the control plane dispatches a webhook event to a reasoner, it wraps
 * the payload in an envelope: `{ event: <payload>, _meta: <trigger metadata> }`.
 * This module detects that shape, unwraps the event, constructs a TriggerContext,
 * and applies the binding's `transform` if one is declared.
 *
 * Direct calls (no envelope) pass through unchanged — the handler receives
 * its input as before.
 *
 * @module triggers/dispatch
 */

import type { TriggerContext, TriggerBinding, EventTriggerBinding } from './types.js';

/**
 * Shape of the dispatcher's envelope sent by the control plane for
 * webhook-triggered executions.
 */
export interface TriggerEnvelope {
  event: Record<string, unknown>;
  _meta: {
    trigger_id: string;
    source: string;
    event_type: string;
    event_id: string;
    idempotency_key: string;
    received_at: string;
    vc_id?: string;
  };
}

/**
 * Result of envelope detection and unwrap.
 */
export interface UnwrapResult {
  /** The unwrapped input (event payload or original input for direct calls). */
  input: unknown;
  /** TriggerContext if this was a trigger dispatch; undefined for direct calls. */
  triggerContext?: TriggerContext;
}

/**
 * Detect whether a request body is a dispatcher trigger envelope.
 *
 * The envelope shape is `{ event: ..., _meta: { trigger_id, source, ... } }`.
 * Only objects with both `event` and `_meta` keys (where `_meta` contains
 * `trigger_id`) are considered envelopes.
 */
export function isTriggerEnvelope(body: unknown): body is TriggerEnvelope {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return false;
  const obj = body as Record<string, unknown>;
  if (!('event' in obj && '_meta' in obj)) return false;
  const meta = obj._meta;
  if (!meta || typeof meta !== 'object' || Array.isArray(meta)) return false;
  return 'trigger_id' in (meta as Record<string, unknown>);
}

/**
 * Unwrap a trigger envelope (if present) and construct TriggerContext.
 *
 * For trigger dispatches: extracts the event payload and builds a typed
 * TriggerContext from `_meta`. For direct calls: returns the body unchanged
 * with `triggerContext: undefined`.
 */
export function unwrapEnvelope(body: unknown): UnwrapResult {
  if (!isTriggerEnvelope(body)) {
    return { input: body };
  }

  const meta = body._meta;
  let receivedAt: Date;
  try {
    receivedAt = new Date(meta.received_at);
    if (isNaN(receivedAt.getTime())) {
      receivedAt = new Date();
    }
  } catch {
    receivedAt = new Date();
  }

  const triggerContext: TriggerContext = {
    triggerId: meta.trigger_id,
    source: meta.source,
    eventType: meta.event_type,
    eventId: meta.event_id,
    idempotencyKey: meta.idempotency_key,
    receivedAt,
    vcId: meta.vc_id,
  };

  return {
    input: body.event,
    triggerContext,
  };
}

/**
 * Apply the matching binding's `transform` function to the unwrapped event.
 *
 * Matching logic (mirrors Python SDK's `_apply_trigger_transform`):
 * 1. Find bindings where `binding.spec.source === triggerContext.source`
 * 2. For event bindings with non-empty `types`, check prefix match against
 *    `triggerContext.eventType`
 * 3. Most specific match (non-empty types) wins over catch-all (empty types)
 * 4. Apply `transform` if the matched binding declares one
 *
 * Returns the (possibly transformed) input.
 */
export function applyTriggerTransform(
  triggerContext: TriggerContext,
  bindings: TriggerBinding[],
  input: unknown
): unknown {
  if (!bindings || bindings.length === 0) return input;

  let bestMatch: EventTriggerBinding | undefined;
  let bestSpecificity = -1;

  for (const binding of bindings) {
    if (binding.kind !== 'event') continue;
    if (binding.spec.source !== triggerContext.source) continue;

    const types = binding.spec.types ?? [];
    if (types.length > 0) {
      // Binding has specific types — check for prefix match
      const matched = types.some(
        (t) =>
          triggerContext.eventType === t ||
          triggerContext.eventType.startsWith(t + '.')
      );
      if (!matched) continue;
      // Specific match beats catch-all
      if (1 > bestSpecificity) {
        bestMatch = binding;
        bestSpecificity = 1;
      }
    } else {
      // Binding accepts all types (catch-all)
      if (0 > bestSpecificity) {
        bestMatch = binding;
        bestSpecificity = 0;
      }
    }
  }

  if (bestMatch?.spec.transform) {
    try {
      return bestMatch.spec.transform(input as Record<string, unknown>);
    } catch {
      // Transform failed — return raw input (matches Python SDK behaviour)
      return input;
    }
  }

  return input;
}
