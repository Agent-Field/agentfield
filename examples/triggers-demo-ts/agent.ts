/**
 * AgentField triggers demo — sample TypeScript agent.
 *
 * Three deterministic reasoners, each wired to a different Source plugin:
 *
 * - handle_payment   ← Stripe webhook (Stripe-Signature HMAC)
 * - handle_pr        ← GitHub webhook (X-Hub-Signature-256 HMAC)
 * - handle_tick      ← cron schedule (every minute)
 *
 * No LLM calls. Each reasoner just transforms its inbound event into a
 * small, deterministic record and writes it to per-agent memory so the UI's
 * event log + run detail surfaces show real data flowing through.
 *
 * When the agent registers with the control plane, the `onEvent` /
 * `onSchedule` sugar auto-creates code-managed Trigger rows. The CP returns
 * the public URLs for each.
 *
 * Equivalent to `examples/triggers-demo/agent.py` — same memory writes,
 * same shapes, same dispatch behaviour. Uses the TS SDK surface proposed
 * in #507.
 */

import { Agent, eventTrigger, type ReasonerContext } from '@agentfield/sdk';

const AGENT_NODE_ID = process.env.AGENT_NODE_ID ?? 'triggers-demo-agent';
const AGENTFIELD_URL = process.env.AGENTFIELD_URL ?? 'http://localhost:8080';
const PORT = parseInt(process.env.PORT ?? '8001', 10);

const app = new Agent({
  nodeId: AGENT_NODE_ID,
  agentFieldUrl: AGENTFIELD_URL,
  port: PORT,
  host: '0.0.0.0',
  devMode: true,
  didEnabled: true,
});

// ---------------------------------------------------------------------------
// Stripe — payment events
//
// The Stripe source plugin verifies Stripe-Signature: t=<ts>,v1=<hmac> over
// "<ts>.<body>" using the secret read from STRIPE_DEMO_SECRET on the CP host.
// The transform here pulls the bits we actually care about out of Stripe's
// fairly nested envelope so the reasoner body stays clean.
// ---------------------------------------------------------------------------

function stripeToPayment(event: Record<string, unknown>): Record<string, unknown> {
  const data = event.data as Record<string, unknown> | undefined;
  const obj = (data?.object ?? {}) as Record<string, unknown>;
  return {
    id: obj.id,
    amount: obj.amount,
    currency: (obj.currency as string) ?? 'usd',
    customer: obj.customer,
    status: obj.status,
    metadata: obj.metadata ?? {},
  };
}

app.reasoner('handle_payment', async (ctx: ReasonerContext) => {
  const payment = ctx.input as Record<string, unknown>;
  const record = {
    kind: 'payment',
    stripe_id: payment.id,
    amount_cents: payment.amount,
    currency: payment.currency,
    customer: payment.customer,
    received_via: ctx.trigger?.source ?? 'direct_call',
    trigger_event_id: ctx.trigger?.eventId ?? null,
  };
  await ctx.memory.set(`payment:${record.stripe_id}`, record);
  console.log(`[handle_payment] saved`, record);
  return record;
}, {
  triggers: [
    eventTrigger({
      source: 'stripe',
      types: ['payment_intent.succeeded'],
      secretEnv: 'STRIPE_DEMO_SECRET',
      transform: stripeToPayment,
    }),
  ],
});

// ---------------------------------------------------------------------------
// GitHub — pull-request events
//
// The GitHub source verifies X-Hub-Signature-256 = sha256=<hmac of body>
// using the secret from GITHUB_DEMO_SECRET. Reads X-GitHub-Event +
// X-GitHub-Delivery for type and idempotency.
// ---------------------------------------------------------------------------

app.onEvent(
  { source: 'github', types: ['pull_request'], secretEnv: 'GITHUB_DEMO_SECRET', name: 'handle_pr' },
  async (ctx: ReasonerContext) => {
    const event = ctx.input as Record<string, unknown>;
    const pr = (event.pull_request ?? {}) as Record<string, unknown>;
    const user = (pr.user ?? {}) as Record<string, unknown>;
    const repo = (event.repository ?? {}) as Record<string, unknown>;
    const record = {
      kind: 'pull_request',
      action: event.action,
      number: event.number ?? pr.number,
      title: pr.title,
      html_url: pr.html_url,
      user: user.login,
      repo: repo.full_name,
      received_via: ctx.trigger?.source ?? 'direct_call',
      delivery_id: ctx.trigger?.idempotencyKey ?? null,
    };
    if (record.repo && record.number) {
      await ctx.memory.set(`pr:${record.repo}#${record.number}`, record);
    }
    console.log(`[handle_pr] saved`, record);
    return record;
  }
);

// ---------------------------------------------------------------------------
// Cron — periodic tick
//
// The cron source runs as a LoopSource inside the CP, emitting a "tick" event
// every time its schedule fires. The agent sees the same dispatch shape as
// any other webhook delivery.
// ---------------------------------------------------------------------------

app.onSchedule('* * * * *', async (ctx: ReasonerContext) => {
  const counterKey = 'cron:tick:count';
  const current = (await ctx.memory.get(counterKey)) as Record<string, unknown> | null;
  const record = {
    count: ((current?.count as number) ?? 0) + 1,
    last_fired_at: ctx.trigger?.receivedAt?.toISOString() ?? null,
    received_via: ctx.trigger?.source ?? 'direct_call',
  };
  await ctx.memory.set(counterKey, record);
  console.log(`[handle_tick]`, record);
  return record;
}, { name: 'handle_tick' });

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

function heartbeat() {
  let n = 0;
  setInterval(() => {
    console.log(`[${AGENT_NODE_ID}] alive heartbeat #${n}`);
    n++;
  }, 30_000);
}

console.error(
  `AgentField triggers demo (TypeScript) — sample agent starting\n` +
  `  node_id            = ${AGENT_NODE_ID}\n` +
  `  agentfield_server  = ${AGENTFIELD_URL}\n` +
  `  callback url       = ${process.env.AGENT_CALLBACK_URL ?? `http://localhost:${PORT}`}\n` +
  `  reasoners          = handle_payment (stripe), handle_pr (github), handle_tick (cron)`
);

heartbeat();
app.serve().catch((err) => {
  console.error('Failed to start agent:', err);
  process.exit(1);
});
