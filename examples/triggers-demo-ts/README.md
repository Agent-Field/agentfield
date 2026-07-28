# Triggers demo (TypeScript)

TypeScript counterpart to `examples/triggers-demo/`. Same demo, same
`fire-events.sh` script, equivalent memory writes — but using the TS SDK
surface (`app.onEvent()`, `app.onSchedule()`, `eventTrigger()`).

---

## What you get

| Service | Port | What it is |
|---|---|---|
| `control-plane` | 8080 | AgentField server with the embedded UI |
| `triggers-demo-ts-agent` | 8001 | TypeScript agent declaring three triggers |

| Reasoner | Source | Fires on | Notes |
|---|---|---|---|
| `handle_payment` | `stripe` | `payment_intent.succeeded` | applies a Stripe transform before invoking the reasoner |
| `handle_pr` | `github` | `pull_request.*` | registered via `app.onEvent()` sugar |
| `handle_tick` | `cron` | every minute | registered via `app.onSchedule()` sugar |

---

## Quick start

```bash
cd examples/triggers-demo-ts

# 1. Bring up control plane + agent
docker compose up --build -d

# 2. Wait ~30 seconds for both containers to come up and the agent to register
docker compose logs -f triggers-demo-ts-agent

# 3. Open the UI
open http://localhost:8080/ui/triggers

# 4. Fire signed Stripe + GitHub events (cron fires on its own)
./scripts/fire-events.sh
```

---

## How it works

The TS agent declares triggers via two patterns:

### Option-bag on `app.reasoner`

```ts
app.reasoner('handle_payment', handler, {
  triggers: [
    eventTrigger({
      source: 'stripe',
      types: ['payment_intent.succeeded'],
      secretEnv: 'STRIPE_DEMO_SECRET',
      transform: stripeToPayment,
    }),
  ],
});
```

### Sugar helpers

```ts
app.onEvent(
  { source: 'github', types: ['pull_request'], secretEnv: 'GITHUB_DEMO_SECRET', name: 'handle_pr' },
  async (ctx) => { /* handler */ }
);

app.onSchedule('* * * * *', async (ctx) => { /* handler */ }, { name: 'handle_tick' });
```

Both register identically with the control plane. The handler receives:
- `ctx.input` — the (optionally transformed) event payload
- `ctx.trigger` — a `TriggerContext` with source, event ID, etc. (`undefined` for direct calls)

---

## Architecture

```
fire-events.sh
   │   signs body with STRIPE_DEMO_SECRET / GITHUB_DEMO_SECRET
   ▼
POST /sources/<trigger_id>          ← public ingest URL on CP
   │
   ▼
control-plane:
   1. resolves trigger row from <trigger_id>
   2. asks the Source plugin to verify the signature
   3. persists InboundEvent
   4. dispatches {event, _meta} envelope to the agent's reasoner endpoint
   ▼
triggers-demo-ts-agent:
   - SDK detects {event, _meta} envelope shape
   - SDK runs the per-binding `transform` (Stripe only here)
   - SDK injects ctx.trigger (TriggerContext)
   - Reasoner runs deterministically, writes to memory
   ▼
UI:
   - SSE stream pushes the event lifecycle into the open Sheet
   - Run detail page picks up the run + trigger field
```

---

## Comparing with the Python demo

| Aspect | Python demo | TypeScript demo |
|---|---|---|
| Trigger declaration | `@on_event` / `@on_schedule` decorators | `app.onEvent()` / `app.onSchedule()` sugar |
| Trigger context | `trigger: TriggerContext \| None` parameter injection | `ctx.trigger?: TriggerContext` on ReasonerContext |
| Transform | `transform=` kwarg on `EventTrigger(...)` | `transform:` in `eventTrigger({...})` spec |
| Memory writes | `await app.memory.set(...)` | `await ctx.memory.set(...)` |
| Same script | ✅ `fire-events.sh` | ✅ Same `fire-events.sh` |

---

## Tearing down

```bash
docker compose down --volumes
```
