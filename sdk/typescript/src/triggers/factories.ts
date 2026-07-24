/**
 * Factory functions for creating trigger bindings.
 *
 * Usage:
 * ```ts
 * import { eventTrigger, scheduleTrigger } from "@agentfield/sdk";
 *
 * app.reasoner("handle_payment", handler, {
 *   triggers: [
 *     eventTrigger({
 *       source: "stripe",
 *       types: ["payment_intent.succeeded"],
 *       secretEnv: "STRIPE_WEBHOOK_SECRET",
 *     }),
 *     scheduleTrigger({ cron: "0 * * * *" }),
 *   ],
 * });
 * ```
 */

import type {
  EventTriggerSpec,
  EventTriggerBinding,
  ScheduleTriggerSpec,
  ScheduleTriggerBinding,
  TriggerBinding,
} from './types.js';

/**
 * Create an event trigger binding.
 *
 * Binds a reasoner to events emitted by an HTTP-driven Source plugin
 * (Stripe, GitHub, Slack, generic_hmac, generic_bearer, etc.).
 *
 * @param spec - Event trigger specification
 * @returns A typed TriggerBinding for use in `ReasonerOptions.triggers`
 *
 * @throws TypeError if `transform` is provided but is an async function
 */
export function eventTrigger(spec: EventTriggerSpec): EventTriggerBinding {
  // Validate that transform is not async
  if (spec.transform) {
    const fnStr = spec.transform.toString();
    // Check for async functions — constructor name is most reliable
    if (
      spec.transform.constructor.name === 'AsyncFunction' ||
      fnStr.startsWith('async ')
    ) {
      throw new TypeError(
        `EventTrigger transform must be synchronous, not async. ` +
        `Got: ${spec.transform.name || '(anonymous)'}`
      );
    }
  }

  return {
    kind: 'event',
    spec,
  };
}

/**
 * Create a schedule trigger binding.
 *
 * Binds a reasoner to a cron schedule. The control plane fires the reasoner
 * at the specified cadence; no external webhook needed.
 *
 * @param spec - Schedule trigger specification
 * @returns A typed TriggerBinding for use in `ReasonerOptions.triggers`
 */
export function scheduleTrigger(spec: ScheduleTriggerSpec): ScheduleTriggerBinding {
  return {
    kind: 'schedule',
    spec,
  };
}

/**
 * Convert a typed TriggerBinding into the wire payload sent to the control
 * plane at registration time.
 *
 * The control plane expects `{source, event_types, config, secret_env_var}`;
 * schedule triggers normalize to the "cron" source with their expression
 * embedded in `config`.
 *
 * Note: `transform` is not serialized (it's a runtime JS callable).
 */
export function triggerToPayload(trigger: TriggerBinding): Record<string, unknown> {
  if (trigger.kind === 'event') {
    const payload: Record<string, unknown> = {
      source: trigger.spec.source,
      event_types: trigger.spec.types ?? [],
    };
    if (trigger.spec.config) {
      payload.config = { ...trigger.spec.config };
    }
    if (trigger.spec.secretEnv) {
      payload.secret_env_var = trigger.spec.secretEnv;
    }
    if (trigger.spec.codeOrigin) {
      payload.code_origin = trigger.spec.codeOrigin;
    }
    return payload;
  }

  if (trigger.kind === 'schedule') {
    const payload: Record<string, unknown> = {
      source: 'cron',
      event_types: [],
      config: {
        expression: trigger.spec.cron,
        timezone: trigger.spec.timezone ?? 'UTC',
      },
    };
    if (trigger.spec.codeOrigin) {
      payload.code_origin = trigger.spec.codeOrigin;
    }
    return payload;
  }

  // Exhaustiveness guard
  const _exhaustive: never = trigger;
  throw new TypeError(`Unknown trigger kind: ${(_exhaustive as TriggerBinding)}`);
}
