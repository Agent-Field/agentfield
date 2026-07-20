import { describe, it, expect } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { eventTrigger, scheduleTrigger } from '../src/triggers/factories.js';

describe('app.onEvent() sugar', () => {
  it('registers a reasoner with the specified event trigger', () => {
    const app = new Agent({ nodeId: 'test-sugar', devMode: true });

    app.onEvent(
      { source: 'stripe', types: ['payment_intent.succeeded'], name: 'handle_payment' },
      async (ctx) => ({ ok: true })
    );

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'handle_payment');
    expect(def).toBeDefined();
    expect(def.triggers).toHaveLength(1);
    expect(def.triggers[0].source).toBe('stripe');
    expect(def.triggers[0].event_types).toEqual(['payment_intent.succeeded']);
    expect(def.accepts_webhook).toBe('true');
  });

  it('defaults reasoner name from handler function name', () => {
    const app = new Agent({ nodeId: 'test-sugar-name', devMode: true });

    async function handle_github(ctx: any) { return {}; }
    app.onEvent({ source: 'github', types: ['push'] }, handle_github);

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'handle_github');
    expect(def).toBeDefined();
  });

  it('falls back to on_<source> when handler has no name', () => {
    const app = new Agent({ nodeId: 'test-sugar-fallback', devMode: true });

    // Use Object.defineProperty to force an empty name
    const handler = async () => ({});
    Object.defineProperty(handler, 'name', { value: '' });
    app.onEvent(
      { source: 'slack', types: ['app_mention'] },
      handler
    );

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'on_slack');
    expect(def).toBeDefined();
  });

  it('produces same registration payload as app.reasoner with explicit triggers', () => {
    const app1 = new Agent({ nodeId: 'test-sugar-equiv-1', devMode: true });
    app1.onEvent(
      { source: 'github', types: ['pull_request'], secretEnv: 'GH_SECRET', name: 'handle_pr' },
      async () => ({})
    );

    const app2 = new Agent({ nodeId: 'test-sugar-equiv-2', devMode: true });
    app2.reasoner('handle_pr', async () => ({}), {
      triggers: [eventTrigger({ source: 'github', types: ['pull_request'], secretEnv: 'GH_SECRET' })],
    });

    const defs1 = (app1 as any).reasonerDefinitions();
    const defs2 = (app2 as any).reasonerDefinitions();

    const d1 = defs1.find((d: any) => d.id === 'handle_pr');
    const d2 = defs2.find((d: any) => d.id === 'handle_pr');
    expect(d1.triggers).toEqual(d2.triggers);
    expect(d1.accepts_webhook).toEqual(d2.accepts_webhook);
  });

  it('merges extra options (tags, description) through', () => {
    const app = new Agent({ nodeId: 'test-sugar-opts', devMode: true });

    app.onEvent(
      { source: 'stripe', types: ['charge.failed'], name: 'handle_charge' },
      async () => ({}),
      { tags: ['billing'], description: 'Handles failed charges' }
    );

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'handle_charge');
    expect(def.tags).toEqual(['billing']);
  });

  it('dispatches trigger envelope correctly through onEvent-registered handler', async () => {
    const app = new Agent({ nodeId: 'test-sugar-dispatch', devMode: true });

    let receivedTrigger: any;
    let receivedInput: any;
    app.onEvent(
      {
        source: 'stripe',
        types: ['payment_intent.succeeded'],
        name: 'pay_handler',
        transform: (evt) => (evt as any).data?.object,
      },
      async (ctx) => {
        receivedTrigger = ctx.trigger;
        receivedInput = ctx.input;
        return { saved: true };
      }
    );

    const envelope = {
      event: { data: { object: { id: 'pi_1', amount: 5000 } } },
      _meta: {
        trigger_id: 'tr_1',
        source: 'stripe',
        event_type: 'payment_intent.succeeded',
        event_id: 'evt_1',
        idempotency_key: 'idem_1',
        received_at: '2026-04-28T22:29:54Z',
      },
    };

    const result = await app.call('pay_handler', envelope);
    expect(result).toEqual({ saved: true });
    expect(receivedInput).toEqual({ id: 'pi_1', amount: 5000 });
    expect(receivedTrigger).toBeDefined();
    expect(receivedTrigger.source).toBe('stripe');
  });
});

describe('app.onSchedule() sugar', () => {
  it('registers a reasoner with a cron schedule trigger', () => {
    const app = new Agent({ nodeId: 'test-schedule', devMode: true });

    app.onSchedule('* * * * *', async () => ({ tick: true }), { name: 'handle_tick' });

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'handle_tick');
    expect(def).toBeDefined();
    expect(def.triggers).toHaveLength(1);
    expect(def.triggers[0]).toEqual({
      source: 'cron',
      event_types: [],
      config: { expression: '* * * * *', timezone: 'UTC' },
    });
    expect(def.accepts_webhook).toBe('true');
  });

  it('respects custom timezone', () => {
    const app = new Agent({ nodeId: 'test-schedule-tz', devMode: true });

    app.onSchedule('0 9 * * 1-5', async () => ({}), {
      name: 'daily_standup',
      timezone: 'America/New_York',
    });

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'daily_standup');
    expect(def.triggers[0].config.timezone).toBe('America/New_York');
  });

  it('defaults reasoner name from handler function name', () => {
    const app = new Agent({ nodeId: 'test-schedule-name', devMode: true });

    async function my_cron_job(ctx: any) { return {}; }
    app.onSchedule('*/5 * * * *', my_cron_job);

    const defs = (app as any).reasonerDefinitions();
    const def = defs.find((d: any) => d.id === 'my_cron_job');
    expect(def).toBeDefined();
  });

  it('dispatches cron envelope correctly', async () => {
    const app = new Agent({ nodeId: 'test-schedule-dispatch', devMode: true });

    let receivedTrigger: any;
    app.onSchedule('* * * * *', async (ctx) => {
      receivedTrigger = ctx.trigger;
      return { count: 1 };
    }, { name: 'tick_handler' });

    const envelope = {
      event: { cron: '* * * * *', fired_at: '2026-04-28T09:00:00Z' },
      _meta: {
        trigger_id: 'tr_cron',
        source: 'cron',
        event_type: 'tick',
        event_id: 'evt_cron_1',
        idempotency_key: '',
        received_at: '2026-04-28T09:00:00Z',
      },
    };

    const result = await app.call('tick_handler', envelope);
    expect(result).toEqual({ count: 1 });
    expect(receivedTrigger).toBeDefined();
    expect(receivedTrigger.source).toBe('cron');
    expect(receivedTrigger.eventType).toBe('tick');
  });
});
