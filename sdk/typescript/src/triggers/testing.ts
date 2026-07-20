/**
 * Testing helpers for reasoners that handle webhook triggers.
 *
 * The standard SDK runtime delivers trigger events via the agent's HTTP
 * endpoint — your reasoner sees `ctx.trigger` populated and (when `transform`
 * is set) the unwrapped, transformed input. For unit tests you want the same
 * shape without spinning up a control plane, an HTTP server, or a real
 * provider. `simulateTrigger` gives you that: it crafts the `TriggerContext`
 * the agent runtime would have produced, applies any matching `transform`
 * from the reasoner's declared bindings, and invokes the handler directly
 * with a minimal ReasonerContext-like object. No HTTP, no workflow
 * registration, no VC mint.
 *
 * @module triggers/testing
 */

import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { TriggerContext, TriggerBinding } from './types.js';
import { applyTriggerTransform } from './dispatch.js';

/**
 * Options for `simulateTrigger`.
 */
export interface SimulateTriggerOptions {
  /** Provider source name (e.g. "stripe", "github", "cron"). */
  source: string;
  /** Inbound event body. Defaults to `{}`. */
  body?: Record<string, unknown>;
  /** Provider's event type (e.g. "payment_intent.succeeded"). Defaults to "". */
  eventType?: string;
  /** Override the generated event ID. */
  eventId?: string;
  /** Override the generated idempotency key. */
  idempotencyKey?: string;
  /** Override the generated trigger ID. */
  triggerId?: string;
  /** Override the received-at timestamp. Defaults to `new Date()`. */
  receivedAt?: Date;
  /** Optional VC ID. */
  vcId?: string;
  /**
   * Trigger bindings for the handler. Used to find the matching binding's
   * `transform`. If not provided, the body is passed through as-is.
   */
  bindings?: TriggerBinding[];
}

/**
 * Options for `simulateSchedule`.
 */
export interface SimulateScheduleOptions {
  /** Cron expression (for test introspection). */
  cron?: string;
  /** Override the received-at timestamp. Defaults to `new Date()`. */
  receivedAt?: Date;
  /** Trigger bindings for the handler. */
  bindings?: TriggerBinding[];
}

/**
 * Minimal execution context surface exposed as `ctx` in simulate tests.
 * Only carries the bits a webhook reasoner is likely to read.
 */
export interface SimulatedContext<TInput = unknown> {
  input: TInput;
  trigger: TriggerContext;
  executionId: string;
}

/**
 * Simulate a trigger dispatch and invoke the handler.
 *
 * Builds a synthetic `TriggerContext`, optionally applies the matched
 * binding's `transform`, and calls the handler with a minimal context
 * object that mirrors what the real dispatch path produces.
 *
 * @param handler - Function accepting `(ctx: { input, trigger, executionId })`.
 *   The same shape a `ReasonerHandler` receives, but trimmed to the fields
 *   relevant for trigger-driven logic.
 * @param options - Trigger simulation options.
 * @returns Whatever the handler returns (awaits async handlers transparently).
 */
export async function simulateTrigger<R>(
  handler: (ctx: SimulatedContext) => R | Promise<R>,
  options: SimulateTriggerOptions
): Promise<R> {
  const body = options.body ?? {};
  const triggerContext: TriggerContext = {
    triggerId: options.triggerId ?? `trg_sim_${randomUUID().slice(0, 12)}`,
    source: options.source,
    eventType: options.eventType ?? '',
    eventId: options.eventId ?? `evt_sim_${randomUUID().slice(0, 12)}`,
    idempotencyKey: options.idempotencyKey ?? `idem_sim_${randomUUID().slice(0, 12)}`,
    receivedAt: options.receivedAt ?? new Date(),
    vcId: options.vcId,
  };

  // Apply transform from bindings if provided
  let resolvedInput: unknown = body;
  if (options.bindings && options.bindings.length > 0) {
    resolvedInput = applyTriggerTransform(triggerContext, options.bindings, body);
  }

  const ctx: SimulatedContext = {
    input: resolvedInput,
    trigger: triggerContext,
    executionId: `exec_sim_${randomUUID().slice(0, 12)}`,
  };

  return handler(ctx);
}

/**
 * Simulate a schedule (cron) trigger dispatch.
 *
 * Convenience wrapper around `simulateTrigger` for cron-triggered reasoners.
 * Passes an empty body and `source: 'cron'`, `eventType: 'tick'`.
 */
export async function simulateSchedule<R>(
  handler: (ctx: SimulatedContext) => R | Promise<R>,
  options?: SimulateScheduleOptions
): Promise<R> {
  return simulateTrigger(handler, {
    source: 'cron',
    body: options?.cron ? { cron: options.cron } : {},
    eventType: 'tick',
    receivedAt: options?.receivedAt,
    bindings: options?.bindings,
  });
}

/**
 * Load a captured provider payload from the SDK fixture library.
 *
 * Fixtures live at `src/triggers/fixtures/<source>.json` relative to the
 * package root. Returns a parsed object — each call re-reads from disk so
 * tests can mutate freely.
 */
export function loadFixture(source: string): Record<string, unknown> {
  const dir = dirname(fileURLToPath(import.meta.url));
  // When running from source: dir = .../src/triggers
  // When running from dist bundle: dir = .../dist
  // Try both possible locations.
  const candidates = [
    resolve(dir, 'fixtures', `${source}.json`),
    resolve(dir, '..', 'src', 'triggers', 'fixtures', `${source}.json`),
  ];
  for (const fixturePath of candidates) {
    try {
      const content = readFileSync(fixturePath, 'utf-8');
      return JSON.parse(content);
    } catch {
      // Try next candidate
    }
  }
  throw new Error(
    `No fixture for source="${source}". Looked at ${candidates.join(', ')}`
  );
}
