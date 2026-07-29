import { describe, it, expect } from 'vitest';
import {
  simulateTrigger,
  simulateSchedule,
  loadFixture,
} from '../src/triggers/testing.js';
import { eventTrigger } from '../src/triggers/factories.js';
import type { SimulatedContext } from '../src/triggers/testing.js';

describe('loadFixture()', () => {
  it('loads stripe fixture', () => {
    const fixture = loadFixture('stripe');
    expect(fixture).toBeDefined();
    expect(fixture.type).toBe('payment_intent.succeeded');
    expect((fixture as any).data.object.amount).toBe(5000);
  });

  it('loads github fixture', () => {
    const fixture = loadFixture('github');
    expect(fixture).toBeDefined();
    expect(fixture.action).toBe('opened');
    expect((fixture as any).pull_request.number).toBe(42);
  });

  it('loads slack fixture', () => {
    const fixture = loadFixture('slack');
    expect(fixture).toBeDefined();
    expect(fixture.type).toBe('event_callback');
    expect((fixture as any).event.type).toBe('app_mention');
  });

  it('loads cron fixture', () => {
    const fixture = loadFixture('cron');
    expect(fixture).toBeDefined();
    expect(fixture.cron).toBe('0 9 * * *');
    expect(fixture.fired_at).toBeDefined();
  });

  it('loads generic_hmac fixture', () => {
    const fixture = loadFixture('generic_hmac');
    expect(fixture).toBeDefined();
    expect(fixture.event).toBe('order.created');
    expect((fixture as any).order_id).toBe('ord_demo_42');
  });

  it('loads generic_bearer fixture', () => {
    const fixture = loadFixture('generic_bearer');
    expect(fixture).toBeDefined();
    expect(fixture.kind).toBe('internal.notification');
    expect((fixture as any).notification_id).toBe('notif_demo_77');
  });

  it('throws for non-existent fixture', () => {
    expect(() => loadFixture('nonexistent_provider')).toThrow(/No fixture/);
  });
});

describe('simulateTrigger()', () => {
  it('invokes handler with TriggerContext and input', async () => {
    const handler = async (ctx: SimulatedContext) => ({
      source: ctx.trigger.source,
      input: ctx.input,
    });

    const result = await simulateTrigger(handler, {
      source: 'stripe',
      eventType: 'payment_intent.succeeded',
      body: { amount: 5000 },
    });

    expect(result.source).toBe('stripe');
    expect(result.input).toEqual({ amount: 5000 });
  });

  it('applies transform from bindings', async () => {
    const bindings = [
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: (evt) => (evt as any).data?.object,
      }),
    ];

    const handler = async (ctx: SimulatedContext) => ctx.input;
    const stripeFixture = loadFixture('stripe');

    const result = await simulateTrigger(handler, {
      source: 'stripe',
      eventType: 'payment_intent.succeeded',
      body: stripeFixture,
      bindings,
    });

    expect((result as any).id).toBe('pi_3NYExample1234');
    expect((result as any).amount).toBe(5000);
  });

  it('generates unique IDs per call', async () => {
    const ids: string[] = [];
    const handler = async (ctx: SimulatedContext) => {
      ids.push(ctx.trigger.eventId);
      return null;
    };

    await simulateTrigger(handler, { source: 'github' });
    await simulateTrigger(handler, { source: 'github' });

    expect(ids[0]).not.toBe(ids[1]);
  });

  it('allows overriding trigger IDs', async () => {
    const handler = async (ctx: SimulatedContext) => ctx.trigger;

    const result = await simulateTrigger(handler, {
      source: 'stripe',
      triggerId: 'my_trigger',
      eventId: 'my_event',
      idempotencyKey: 'my_key',
    });

    expect(result.triggerId).toBe('my_trigger');
    expect(result.eventId).toBe('my_event');
    expect(result.idempotencyKey).toBe('my_key');
  });

  it('defaults body to empty object', async () => {
    const handler = async (ctx: SimulatedContext) => ctx.input;
    const result = await simulateTrigger(handler, { source: 'cron' });
    expect(result).toEqual({});
  });

  it('works with all six fixture files', async () => {
    const sources = ['stripe', 'github', 'slack', 'cron', 'generic_hmac', 'generic_bearer'];

    for (const source of sources) {
      const fixture = loadFixture(source);
      const handler = async (ctx: SimulatedContext) => ({
        source: ctx.trigger.source,
        hasInput: ctx.input !== undefined,
      });

      const result = await simulateTrigger(handler, {
        source,
        eventType: source === 'cron' ? 'tick' : `${source}.event`,
        body: fixture,
      });

      expect(result.source).toBe(source);
      expect(result.hasInput).toBe(true);
    }
  });

  it('populates receivedAt with provided value', async () => {
    const date = new Date('2026-01-15T10:00:00Z');
    const handler = async (ctx: SimulatedContext) => ctx.trigger.receivedAt;

    const result = await simulateTrigger(handler, {
      source: 'stripe',
      receivedAt: date,
    });

    expect(result).toEqual(date);
  });

  it('populates vcId when provided', async () => {
    const handler = async (ctx: SimulatedContext) => ctx.trigger.vcId;

    const result = await simulateTrigger(handler, {
      source: 'github',
      vcId: 'vc_test_123',
    });

    expect(result).toBe('vc_test_123');
  });
});

describe('simulateSchedule()', () => {
  it('invokes handler with cron source and tick event type', async () => {
    const handler = async (ctx: SimulatedContext) => ({
      source: ctx.trigger.source,
      eventType: ctx.trigger.eventType,
    });

    const result = await simulateSchedule(handler);
    expect(result.source).toBe('cron');
    expect(result.eventType).toBe('tick');
  });

  it('passes cron expression in body when provided', async () => {
    const handler = async (ctx: SimulatedContext) => ctx.input;

    const result = await simulateSchedule(handler, { cron: '0 9 * * *' });
    expect(result).toEqual({ cron: '0 9 * * *' });
  });

  it('passes empty body when no cron provided', async () => {
    const handler = async (ctx: SimulatedContext) => ctx.input;
    const result = await simulateSchedule(handler);
    expect(result).toEqual({});
  });

  it('respects receivedAt override', async () => {
    const date = new Date('2026-06-01T09:00:00Z');
    const handler = async (ctx: SimulatedContext) => ctx.trigger.receivedAt;

    const result = await simulateSchedule(handler, { receivedAt: date });
    expect(result).toEqual(date);
  });
});
