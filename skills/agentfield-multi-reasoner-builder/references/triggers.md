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

## Go

Go has no decorators, so registration is explicit. The trigger types live in
their own package, `github.com/Agent-Field/agentfield/sdk/go/triggers`.

### Event trigger (sugar method)

```go
import (
    "context"

    "github.com/Agent-Field/agentfield/sdk/go/agent"
    "github.com/Agent-Field/agentfield/sdk/go/triggers"
)

// Option A: app.OnEvent sugar — one call, registers and binds
app.OnEvent(triggers.EventOpts{
    Source:    "stripe",
    Types:     []string{"payment_intent.succeeded"},
    SecretEnv: "STRIPE_WEBHOOK_SECRET",
    Transform: func(evt map[string]any) any {
        data, _ := evt["data"].(map[string]any)
        return data["object"]
    },
}, "handle_payment", func(ctx context.Context, payment map[string]any) (any, error) {
    // payment is the transformed value (data.object)
    tc := triggers.FromContext(ctx) // nil on direct calls
    if tc != nil {
        _ = tc.Source // "stripe"
    }
    return map[string]any{"ok": true}, nil
})

// Option B: the option-struct form on RegisterReasoner
app.RegisterReasoner("handle_pr", handlePR,
    agent.WithTriggers(triggers.Event(triggers.EventOpts{
        Source:    "github",
        Types:     []string{"pull_request"},
        SecretEnv: "GITHUB_SECRET",
    })),
)
```

### Schedule trigger

```go
// Option A: app.OnSchedule sugar
app.OnSchedule("* * * * *", "handle_tick", func(ctx context.Context, _ map[string]any) (any, error) {
    tc := triggers.FromContext(ctx)
    // tc.Source == "cron", tc.EventType == "tick"
    return nil, nil
}, agent.WithTimezone("America/New_York")) // optional

// Option B: explicit binding
app.RegisterReasoner("handle_tick", handleTick,
    agent.WithTriggers(triggers.Schedule(triggers.ScheduleOpts{
        Cron:     "* * * * *",
        Timezone: "UTC",
    })),
)
```

### Context fields (Go)

```go
type Context struct {
    TriggerID      string    // AgentField trigger row ID
    Source         string    // "stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer"
    EventType      string    // Provider event type (or "" for cron)
    EventID        string    // AgentField inbound_event ID (replay key)
    IdempotencyKey string    // Provider's idempotency key
    ReceivedAt     time.Time // When CP received the event
    VCID           string    // VC ID if DID enabled (may be empty)
}
```

Retrieve it with `triggers.FromContext(ctx)`, which returns `nil` for direct
calls. Handler signatures are unchanged, so a reasoner can serve both the
trigger and direct paths:

```go
func handlePayment(ctx context.Context, input map[string]any) (any, error) {
    if tc := triggers.FromContext(ctx); tc != nil {
        // dispatched via a trigger
    }
    return nil, nil
}
```

### Testing Go trigger reasoners

The `triggers` package ships helpers and a fixture library, so a reasoner can
be tested without a control plane, HTTP server, or real provider:

```go
func TestHandlePayment(t *testing.T) {
    result, err := triggers.SimulateEvent(t, handlePayment, triggers.SimulateEventOpts{
        Source:    "stripe",
        EventType: "payment_intent.succeeded",
        Body:      triggers.LoadFixture(t, "stripe"),
        Bindings:  []triggers.Binding{triggers.Event(opts)}, // applies Transform
    })
    if err != nil {
        t.Fatal(err)
    }
    // assert on result...
}

func TestHandleTick(t *testing.T) {
    result, err := triggers.SimulateSchedule(t, handleTick, triggers.SimulateScheduleOpts{
        Cron: "* * * * *",
    })
    // ...
}
```

`LoadFixture` reads from an embedded library of six captured payloads
(`stripe`, `github`, `slack`, `cron`, `generic_hmac`, `generic_bearer`) that
are byte-identical to the Python SDK's, so behaviour is comparable across
SDKs. `FixtureNames()` lists them for table-driven tests.

### Key differences from Python and TypeScript

| Aspect | Python | TypeScript | Go |
|---|---|---|---|
| Declaration | `@on_event` decorator | `app.onEvent()` | `app.OnEvent()` or `agent.WithTriggers(triggers.Event(...))` |
| Trigger context access | `trigger` parameter | `ctx.trigger` property | `triggers.FromContext(ctx)` |
| Nil / null check | `if trigger:` | `ctx.trigger?` | `if tc != nil` |
| Transform | `transform=callable` | `transform: (evt) => ...` | `Transform: func(map[string]any) any` |
| Cron field name | `cron` | `cron` | `Cron` |
| Naming convention | snake_case | camelCase | PascalCase fields |

Why `FromContext` rather than a third handler parameter: Go's `HandlerFunc` is
`func(context.Context, map[string]any) (any, error)`, and adding a parameter
would break every existing reasoner. Context propagation matches how the SDK
already threads `ExecutionContext` and the cost tracker.

---

## Registration wire format

All three SDKs produce the same control-plane registration payload per trigger:

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

All three SDKs detect this shape, unwrap `event`, build the trigger context from
`_meta`, apply the matching binding's `transform`, and deliver the result
to the handler. Direct calls (no `_meta`) pass through unchanged.
