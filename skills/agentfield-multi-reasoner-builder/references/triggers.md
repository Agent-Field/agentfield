# Trigger declarations — reference for scaffold generation

This file is consumed by the `agentfield-multi-reasoner-builder` skill when
generating multi-reasoner agent scaffolds. It contains canonical examples
in both Python and TypeScript so the skill emits correct trigger wiring
regardless of the target language.

---

## Python

### Event trigger (decorator style)

```python
from agentfield import Agent, EventTrigger, TriggerContext, on_event

app = Agent(node_id="my-agent")

# Option A: triggers= kwarg on @app.reasoner
@app.reasoner(
    triggers=[
        EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],
            secret_env="STRIPE_WEBHOOK_SECRET",
            transform=lambda evt: evt["data"]["object"],
        ),
    ],
)
async def handle_payment(payment: dict, trigger: TriggerContext | None = None):
    # payment is the transformed payload (data.object)
    # trigger is populated when invoked by webhook, None for direct calls
    ...

# Option B: @on_event decorator
@app.reasoner()
@on_event(source="github", types=["pull_request"], secret_env="GITHUB_SECRET")
async def handle_pr(event: dict, trigger: TriggerContext | None = None):
    ...
```

### Schedule trigger

```python
from agentfield import on_schedule

@app.reasoner()
@on_schedule("* * * * *")
async def handle_tick(_input, trigger: TriggerContext | None = None):
    # trigger.source == "cron", trigger.event_type == "tick"
    ...
```

### TriggerContext fields (Python)

```python
@dataclass(frozen=True)
class TriggerContext:
    trigger_id: str        # AgentField trigger row ID
    source: str            # "stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer"
    event_type: str        # Provider event type (or "" for cron)
    event_id: str          # AgentField inbound_event ID (replay key)
    idempotency_key: str   # Provider's idempotency key
    received_at: datetime  # When CP received the event
    vc_id: str | None      # VC ID if DID enabled
```

---

## TypeScript

### Event trigger (option-bag style)

```typescript
import { Agent, eventTrigger, type ReasonerContext } from '@agentfield/sdk';

const app = new Agent({ nodeId: 'my-agent' });

// Option A: triggers in ReasonerOptions
app.reasoner('handle_payment', async (ctx: ReasonerContext) => {
  const payment = ctx.input;         // transformed payload (data.object)
  const trigger = ctx.trigger;       // TriggerContext | undefined
  const source = trigger?.source;    // "stripe" when invoked by webhook
  // ...
}, {
  triggers: [
    eventTrigger({
      source: 'stripe',
      types: ['payment_intent.succeeded'],
      secretEnv: 'STRIPE_WEBHOOK_SECRET',
      transform: (evt) => (evt as any).data.object,
    }),
  ],
});

// Option B: app.onEvent() sugar (same registration, terser syntax)
app.onEvent(
  { source: 'github', types: ['pull_request'], secretEnv: 'GITHUB_SECRET', name: 'handle_pr' },
  async (ctx: ReasonerContext) => {
    const event = ctx.input;
    const trigger = ctx.trigger;   // populated by dispatch; undefined for direct calls
    // ...
  }
);
```

### Schedule trigger

```typescript
import { scheduleTrigger } from '@agentfield/sdk';

// Option A: explicit
app.reasoner('handle_tick', async (ctx) => {
  // ctx.trigger?.source === 'cron'
  // ctx.trigger?.eventType === 'tick'
}, {
  triggers: [scheduleTrigger({ cron: '* * * * *' })],
});

// Option B: app.onSchedule() sugar
app.onSchedule('* * * * *', async (ctx) => {
  // same shape as above
}, { name: 'handle_tick', timezone: 'UTC' });
```

### TriggerContext fields (TypeScript)

```typescript
interface TriggerContext {
  triggerId: string;        // AgentField trigger row ID
  source: string;           // "stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer"
  eventType: string;        // Provider event type (or "" for cron)
  eventId: string;          // AgentField inbound_event ID (replay key)
  idempotencyKey: string;   // Provider's idempotency key
  receivedAt: Date;         // When CP received the event
  vcId?: string;            // VC ID if DID enabled
}
```

### Key differences from Python

| Aspect | Python | TypeScript |
|---|---|---|
| Declaration | `@on_event` / `@on_schedule` decorators | `app.onEvent()` / `app.onSchedule()` methods |
| Trigger context access | `trigger` parameter (injected by name) | `ctx.trigger` property on `ReasonerContext` |
| Transform | `transform=callable` on `EventTrigger(...)` | `transform: (evt) => ...` in `eventTrigger({...})` |
| Null-safety | `trigger: TriggerContext \| None` | `ctx.trigger?: TriggerContext` (optional) |
| Naming convention | snake_case (`secret_env`) | camelCase (`secretEnv`) |

---

## Registration wire format

Both SDKs produce the same control-plane registration payload per trigger:

```json
{
  "source": "stripe",
  "event_types": ["payment_intent.succeeded"],
  "secret_env_var": "STRIPE_WEBHOOK_SECRET",
  "config": {},
  "code_origin": "agent.ts:42"
}
```

Schedule triggers normalize to:

```json
{
  "source": "cron",
  "event_types": [],
  "config": {
    "expression": "* * * * *",
    "timezone": "UTC"
  }
}
```

`transform` is never serialized — it's a runtime callable applied agent-side.

---

## Dispatch envelope

The control plane dispatches triggers as:

```json
{
  "event": { /* raw provider payload */ },
  "_meta": {
    "trigger_id": "tr_abc",
    "source": "stripe",
    "event_type": "payment_intent.succeeded",
    "event_id": "evt_123",
    "idempotency_key": "evt_xxx",
    "received_at": "2026-04-28T22:29:54Z",
    "vc_id": "vc_456"
  }
}
```

Both SDKs detect this shape, unwrap `event`, build `TriggerContext` from
`_meta`, apply the matching binding's `transform`, and deliver the result
to the handler. Direct calls (no `_meta`) pass through unchanged.
