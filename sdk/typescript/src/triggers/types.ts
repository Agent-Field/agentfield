/**
 * Trigger binding types for AgentField TypeScript SDK.
 *
 * A reasoner declares external event sources via the `triggers` option on
 * `app.reasoner(...)`. The canonical form passes typed `TriggerBinding` instances
 * created by `eventTrigger()` / `scheduleTrigger()` factories.
 *
 * The control plane registers a code-managed Trigger row per binding when the
 * agent registers, so the agent never has to provision webhooks itself.
 *
 * Field-for-field equivalent of `sdk/python/agentfield/triggers.py`.
 */

/**
 * Webhook-trigger metadata exposed to reasoners at runtime.
 *
 * Available as `ctx.trigger` (undefined when the reasoner was invoked directly
 * via app.call(...) instead of by an inbound event).
 *
 * @experimental This type is exported for forward compatibility. Runtime
 * construction and injection into handler context is planned for #510
 * (dispatch envelope unwrap + TriggerContext injection). Do not depend on
 * this being populated until that issue ships.
 */
export interface TriggerContext {
  /** AgentField trigger row ID; stable, equals the public URL slug. */
  triggerId: string;

  /** Provider source ("stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer"). */
  source: string;

  /** Provider's event type (or "" for cron tick). */
  eventType: string;

  /** AgentField inbound_event ID (replay key). */
  eventId: string;

  /** Provider's idempotency key (e.g. evt_xxx). */
  idempotencyKey: string;

  /** When control plane received the inbound event. */
  receivedAt: Date;

  /** Trigger event VC ID, if DID enabled. */
  vcId?: string;
}

/**
 * Specification for binding a reasoner to events from an HTTP-driven Source.
 */
export interface EventTriggerSpec {
  /**
   * Registered Source name (e.g. "stripe", "github", "slack",
   * "generic_hmac", "generic_bearer").
   */
  source: string;

  /**
   * Event types the reasoner cares about. Empty array means "all".
   * Supports prefix-match: "pull_request" matches "pull_request.opened" etc.
   */
  types?: string[];

  /**
   * Name of the env var on the **control plane** that holds
   * the provider's webhook secret. Required for Sources whose
   * `secret_required` is true.
   */
  secretEnv?: string;

  /**
   * Source-specific JSON config (timestamp tolerance, custom header names, etc).
   * The Source's `Validate` runs server-side.
   */
  config?: Record<string, unknown>;

  /**
   * Optional sync transform to convert raw provider event to reasoner input.
   * When set, SDK runs transform(event) before invoking the reasoner.
   * Must be synchronous (no Promises).
   */
  transform?: (event: Record<string, unknown>) => unknown;

  /**
   * Optional source code location (e.g. "path/to/file.ts:42") where this
   * trigger is declared. Used for observability and drift detection.
   */
  codeOrigin?: string;
}

/**
 * Specification for binding a reasoner to a cron schedule.
 */
export interface ScheduleTriggerSpec {
  /** 5-field cron expression (minute hour dom month dow). */
  cron: string;

  /** IANA timezone name. Defaults to "UTC". */
  timezone?: string;

  /**
   * Optional source code location (e.g. "path/to/file.ts:42") where this
   * trigger is declared. Used for observability and drift detection.
   */
  codeOrigin?: string;
}

/**
 * A typed trigger binding — either an event trigger or a schedule trigger.
 * Created via `eventTrigger()` or `scheduleTrigger()` factory functions.
 */
export type TriggerBinding = EventTriggerBinding | ScheduleTriggerBinding;

export interface EventTriggerBinding {
  kind: 'event';
  spec: EventTriggerSpec;
}

export interface ScheduleTriggerBinding {
  kind: 'schedule';
  spec: ScheduleTriggerSpec;
}
