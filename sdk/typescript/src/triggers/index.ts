/**
 * Trigger system — public re-exports.
 *
 * @module triggers
 */

export type {
  TriggerContext,
  EventTriggerSpec,
  ScheduleTriggerSpec,
  TriggerBinding,
  EventTriggerBinding,
  ScheduleTriggerBinding,
} from './types.js';

export {
  eventTrigger,
  scheduleTrigger,
  triggerToPayload,
} from './factories.js';
