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

export {
  isTriggerEnvelope,
  unwrapEnvelope,
  applyTriggerTransform,
} from './dispatch.js';

export type {
  TriggerEnvelope,
  UnwrapResult,
} from './dispatch.js';

export {
  simulateTrigger,
  simulateSchedule,
  loadFixture,
} from './testing.js';

export type {
  SimulateTriggerOptions,
  SimulateScheduleOptions,
  SimulatedContext,
} from './testing.js';
