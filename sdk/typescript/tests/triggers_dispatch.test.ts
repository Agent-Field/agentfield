import { describe, it, expect } from 'vitest';
import {
  isTriggerEnvelope,
  unwrapEnvelope,
  applyTriggerTransform,
} from '../src/triggers/dispatch.js';
import { eventTrigger, scheduleTrigger } from '../src/triggers/factories.js';
import type { TriggerContext } from '../src/triggers/types.js';

describe('isTriggerEnvelope()', () => {
  it('returns true for valid envelope shape', () => {
    const body = {
      event: { id: 'pi_x', amount: 4200 },
      _meta: {
        trigger_id: 'tr_abc',
        source: 'stripe',
        event_type: 'payment_intent.succeeded',
        event_id: 'evt_x',
        idempotency_key: 'idem_1',
        received_at: '2026-04-28T22:29:54Z',
      },
    };
    expect(isTriggerEnvelope(body)).toBe(true);
  });

  it('returns false for plain input (no _meta)', () => {
    expect(isTriggerEnvelope({ id: 'pi_x', amount: 4200 })).toBe(false);
  });

  it('returns false for null/undefined', () => {
    expect(isTriggerEnvelope(null)).toBe(false);
    expect(isTriggerEnvelope(undefined)).toBe(false);
  });

  it('returns false for arrays', () => {
    expect(isTriggerEnvelope([1, 2, 3])).toBe(false);
  });

  it('returns false when _meta is missing trigger_id', () => {
    const body = {
      event: { id: 'pi_x' },
      _meta: { source: 'stripe' },
    };
    expect(isTriggerEnvelope(body)).toBe(false);
  });

  it('returns false when event key is absent', () => {
    const body = {
      _meta: { trigger_id: 'tr_abc', source: 'stripe' },
    };
    expect(isTriggerEnvelope(body)).toBe(false);
  });

  it('returns false for strings', () => {
    expect(isTriggerEnvelope('hello')).toBe(false);
  });

  it('returns false when _meta is a non-object', () => {
    expect(isTriggerEnvelope({ event: {}, _meta: 'not_an_object' })).toBe(false);
    expect(isTriggerEnvelope({ event: {}, _meta: null })).toBe(false);
    expect(isTriggerEnvelope({ event: {}, _meta: [1] })).toBe(false);
  });
});

describe('unwrapEnvelope()', () => {
  it('unwraps a valid envelope and returns TriggerContext', () => {
    const body = {
      event: { id: 'pi_x', amount: 4200 },
      _meta: {
        trigger_id: 'tr_abc',
        source: 'stripe',
        event_type: 'payment_intent.succeeded',
        event_id: 'evt_x',
        idempotency_key: 'idem_1',
        received_at: '2026-04-28T22:29:54Z',
        vc_id: 'vc_123',
      },
    };

    const result = unwrapEnvelope(body);
    expect(result.input).toEqual({ id: 'pi_x', amount: 4200 });
    expect(result.triggerContext).toBeDefined();
    expect(result.triggerContext!.triggerId).toBe('tr_abc');
    expect(result.triggerContext!.source).toBe('stripe');
    expect(result.triggerContext!.eventType).toBe('payment_intent.succeeded');
    expect(result.triggerContext!.eventId).toBe('evt_x');
    expect(result.triggerContext!.idempotencyKey).toBe('idem_1');
    expect(result.triggerContext!.receivedAt).toEqual(new Date('2026-04-28T22:29:54Z'));
    expect(result.triggerContext!.vcId).toBe('vc_123');
  });

  it('returns input unchanged and no triggerContext for direct calls', () => {
    const body = { id: 'pi_x', amount: 4200 };
    const result = unwrapEnvelope(body);
    expect(result.input).toEqual({ id: 'pi_x', amount: 4200 });
    expect(result.triggerContext).toBeUndefined();
  });

  it('handles missing vc_id gracefully', () => {
    const body = {
      event: { data: 'test' },
      _meta: {
        trigger_id: 'tr_1',
        source: 'github',
        event_type: 'push',
        event_id: 'evt_1',
        idempotency_key: 'key_1',
        received_at: '2026-01-01T00:00:00Z',
      },
    };
    const result = unwrapEnvelope(body);
    expect(result.triggerContext!.vcId).toBeUndefined();
  });

  it('handles invalid received_at by defaulting to current time', () => {
    const body = {
      event: {},
      _meta: {
        trigger_id: 'tr_1',
        source: 'cron',
        event_type: 'tick',
        event_id: 'evt_1',
        idempotency_key: '',
        received_at: 'not-a-date',
      },
    };
    const before = Date.now();
    const result = unwrapEnvelope(body);
    const after = Date.now();
    expect(result.triggerContext!.receivedAt.getTime()).toBeGreaterThanOrEqual(before);
    expect(result.triggerContext!.receivedAt.getTime()).toBeLessThanOrEqual(after);
  });

  it('passes through null/undefined bodies unchanged', () => {
    expect(unwrapEnvelope(null)).toEqual({ input: null });
    expect(unwrapEnvelope(undefined)).toEqual({ input: undefined });
  });
});

describe('applyTriggerTransform()', () => {
  const baseTriggerCtx: TriggerContext = {
    triggerId: 'tr_1',
    source: 'stripe',
    eventType: 'payment_intent.succeeded',
    eventId: 'evt_1',
    idempotencyKey: 'idem_1',
    receivedAt: new Date('2026-01-01T00:00:00Z'),
  };

  it('applies matching transform', () => {
    const bindings = [
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: (evt) => (evt as any).data?.object,
      }),
    ];

    const input = { data: { object: { id: 'pi_1', amount: 5000 } } };
    const result = applyTriggerTransform(baseTriggerCtx, bindings, input);
    expect(result).toEqual({ id: 'pi_1', amount: 5000 });
  });

  it('returns raw input when no bindings match', () => {
    const bindings = [
      eventTrigger({
        source: 'github',
        types: ['push'],
        transform: (evt) => evt['commits'],
      }),
    ];
    const input = { id: 'pi_1' };
    const result = applyTriggerTransform(baseTriggerCtx, bindings, input);
    expect(result).toEqual({ id: 'pi_1' });
  });

  it('returns raw input when bindings array is empty', () => {
    const input = { id: 'pi_1' };
    expect(applyTriggerTransform(baseTriggerCtx, [], input)).toEqual(input);
  });

  it('returns raw input when transform throws', () => {
    const bindings = [
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: () => { throw new Error('boom'); },
      }),
    ];
    const input = { id: 'pi_1' };
    expect(applyTriggerTransform(baseTriggerCtx, bindings, input)).toEqual(input);
  });

  it('prefers specific type match over catch-all', () => {
    const bindings = [
      eventTrigger({
        source: 'stripe',
        // catch-all — no types
        transform: (evt) => ({ ...evt, from: 'catch-all' }),
      }),
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: (evt) => ({ ...evt, from: 'specific' }),
      }),
    ];

    const input = { id: 'pi_1' };
    const result = applyTriggerTransform(baseTriggerCtx, bindings, input) as any;
    expect(result.from).toBe('specific');
  });

  it('falls back to catch-all when event type does not match specific', () => {
    const ctx: TriggerContext = { ...baseTriggerCtx, eventType: 'charge.failed' };
    const bindings = [
      eventTrigger({
        source: 'stripe',
        // catch-all
        transform: (evt) => ({ ...evt, from: 'catch-all' }),
      }),
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: (evt) => ({ ...evt, from: 'specific' }),
      }),
    ];

    const input = { id: 'ch_1' };
    const result = applyTriggerTransform(ctx, bindings, input) as any;
    expect(result.from).toBe('catch-all');
  });

  it('supports prefix matching (e.g. "pull_request" matches "pull_request.opened")', () => {
    const ctx: TriggerContext = { ...baseTriggerCtx, source: 'github', eventType: 'pull_request.opened' };
    const bindings = [
      eventTrigger({
        source: 'github',
        types: ['pull_request'],
        transform: (evt) => ({ pr: (evt as any).pull_request }),
      }),
    ];

    const input = { action: 'opened', pull_request: { number: 42 } };
    const result = applyTriggerTransform(ctx, bindings, input) as any;
    expect(result).toEqual({ pr: { number: 42 } });
  });

  it('returns raw input when no transform is declared on the matched binding', () => {
    const bindings = [
      eventTrigger({ source: 'stripe', types: ['payment_intent.succeeded'] }),
    ];
    const input = { id: 'pi_1' };
    expect(applyTriggerTransform(baseTriggerCtx, bindings, input)).toEqual(input);
  });

  it('ignores schedule bindings', () => {
    const bindings = [
      scheduleTrigger({ cron: '* * * * *' }),
      eventTrigger({
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        transform: (evt) => ({ transformed: true }),
      }),
    ];
    const input = { id: 'pi_1' };
    const result = applyTriggerTransform(baseTriggerCtx, bindings, input) as any;
    expect(result.transformed).toBe(true);
  });
});

describe('Agent dispatch integration (envelope → handler)', () => {
  it('direct call (no envelope) passes input unchanged, trigger is undefined', async () => {
    const { Agent } = await import('../src/agent/Agent.js');
    const app = new Agent({ nodeId: 'test-dispatch', devMode: true });

    let receivedInput: any;
    let receivedTrigger: any;
    app.reasoner('handler', async (ctx) => {
      receivedInput = ctx.input;
      receivedTrigger = ctx.trigger;
      return { ok: true };
    }, {
      triggers: [eventTrigger({ source: 'stripe', types: ['payment_intent.succeeded'] })],
    });

    // Simulate a direct call (no envelope)
    const result = await app.call('handler', { id: 'pi_x', amount: 4200 });
    expect(result).toEqual({ ok: true });
    expect(receivedInput).toEqual({ id: 'pi_x', amount: 4200 });
    expect(receivedTrigger).toBeUndefined();
  });

  it('trigger envelope unwraps and injects TriggerContext', async () => {
    const { Agent } = await import('../src/agent/Agent.js');
    const app = new Agent({ nodeId: 'test-dispatch-envelope', devMode: true });

    let receivedInput: any;
    let receivedTrigger: any;
    app.reasoner('handle_payment', async (ctx) => {
      receivedInput = ctx.input;
      receivedTrigger = ctx.trigger;
      return { saved: true };
    }, {
      triggers: [
        eventTrigger({
          source: 'stripe',
          types: ['payment_intent.succeeded'],
          transform: (evt) => (evt as any).data?.object,
        }),
      ],
    });

    const envelope = {
      event: { data: { object: { id: 'pi_1', amount: 5000 } } },
      _meta: {
        trigger_id: 'tr_abc',
        source: 'stripe',
        event_type: 'payment_intent.succeeded',
        event_id: 'evt_1',
        idempotency_key: 'idem_1',
        received_at: '2026-04-28T22:29:54Z',
      },
    };

    const result = await app.call('handle_payment', envelope);
    expect(result).toEqual({ saved: true });
    // Transform was applied — handler receives the nested object
    expect(receivedInput).toEqual({ id: 'pi_1', amount: 5000 });
    // TriggerContext is populated
    expect(receivedTrigger).toBeDefined();
    expect(receivedTrigger.triggerId).toBe('tr_abc');
    expect(receivedTrigger.source).toBe('stripe');
    expect(receivedTrigger.eventType).toBe('payment_intent.succeeded');
    expect(receivedTrigger.eventId).toBe('evt_1');
  });

  it('envelope without matching transform passes raw event', async () => {
    const { Agent } = await import('../src/agent/Agent.js');
    const app = new Agent({ nodeId: 'test-dispatch-no-transform', devMode: true });

    let receivedInput: any;
    app.reasoner('handle_pr', async (ctx) => {
      receivedInput = ctx.input;
      return { ok: true };
    }, {
      triggers: [eventTrigger({ source: 'github', types: ['pull_request'] })],
    });

    const envelope = {
      event: { action: 'opened', number: 42, pull_request: { title: 'test' } },
      _meta: {
        trigger_id: 'tr_gh',
        source: 'github',
        event_type: 'pull_request.opened',
        event_id: 'evt_gh_1',
        idempotency_key: 'delivery_1',
        received_at: '2026-04-28T22:29:54Z',
      },
    };

    await app.call('handle_pr', envelope);
    expect(receivedInput).toEqual({ action: 'opened', number: 42, pull_request: { title: 'test' } });
  });
});
