# Triggers — entry surface for events and schedules

Reasoners are the unit of intelligence. **Triggers are the unit of arrival** — the way the outside world reaches a reasoner without anyone calling it programmatically. Pick triggers when the system runs on inbound events (Stripe webhook, GitHub issue, Slack mention) or recurring time (cron). Skip triggers when the system runs synchronously off a request from your own code.

If the use case is event-driven, the triggered reasoner IS the entry reasoner — there is no `app.run()`-only-then-curl path. Webhook delivery hits the control plane, the CP fans out to the agent over HTTP, the SDK unwraps the envelope, and the decorated function runs.

## The six built-in sources

| Source | Kind | Auth | Use when |
|---|---|---|---|
| `stripe` | webhook | `Stripe-Signature` HMAC over `t.<body>` | Payments, subscriptions, customer events |
| `github` | webhook | `X-Hub-Signature-256` HMAC | PRs, issues, push, workflow events |
| `slack` | webhook | Slack signing-secret HMAC | Workspace events, slash commands |
| `generic_hmac` | webhook | Caller-defined HMAC scheme | Any signed third-party webhook |
| `generic_bearer` | webhook | Static bearer token | Internal-only / token-auth webhooks |
| `cron` | loop | none (server-side) | Recurring jobs |

The CP verifies signatures before dispatch — the reasoner never runs on an unsigned event.

## Three ways to declare a trigger

All three end at the same `Trigger` row in the CP. Pick the form that reads cleanest where the reasoner is defined.

```python
from agentfield import EventTrigger, ScheduleTrigger, TriggerContext, on_event, on_schedule

# Form 1 — explicit kwarg on @app.reasoner
@app.reasoner(
    triggers=[
        EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],
            secret_env="STRIPE_WEBHOOK_SECRET",
            transform=lambda evt: evt["data"]["object"],   # optional pre-shaper
        ),
    ],
)
async def handle_payment(payment: dict, trigger: TriggerContext | None = None):
    ...

# Form 2 — sugar decorator (event)
@app.reasoner()
@on_event(source="github", types=["pull_request"], secret_env="GITHUB_WEBHOOK_SECRET")
async def handle_pr(event: dict, trigger: TriggerContext | None = None):
    ...

# Form 3 — sugar decorator (cron)
@app.reasoner()
@on_schedule("* * * * *")
async def handle_tick(_input, trigger: TriggerContext | None = None):
    ...
```

`transform=` is a synchronous pure function the SDK applies before calling the reasoner. Use it to peel provider-specific envelopes (Stripe's `data.object`, GitHub's nested `pull_request`) so the reasoner body works on a clean shape. It must NOT do I/O.

## TriggerContext — the inbound metadata

When a reasoner is invoked through a trigger, the SDK binds a `TriggerContext` to the optional `trigger` parameter. It's None when the same function is invoked directly via `app.call(...)` or curl.

```python
@dataclass(frozen=True)
class TriggerContext:
    trigger_id: str            # the configured Trigger row
    source: str                # "stripe" | "github" | "cron" | ...
    event_type: str            # "payment_intent.succeeded", "issues.opened", "tick", ...
    event_id: str              # CP-assigned inbound event row id
    idempotency_key: str       # provider-supplied or CP-derived
    received_at: datetime      # when the CP first saw it
    vc_id: Optional[str]       # parent trigger-event VC, if DID is wired
```

Use `trigger.idempotency_key` to dedupe. Use `trigger.event_id` for cross-references. **Do NOT** assume `trigger` is non-None — keep the function callable from tests and direct curls.

## Architectural rules

**Triggered reasoners are entry reasoners. Keep them thin.** A triggered reasoner's job is: validate the event shape, decide if this event is interesting, hand off to specialists via `app.call(...)`. It's a router, not a worker. The actual reasoning lives in downstream `@app.reasoner` units that the trigger fans out to via `asyncio.gather`. If the triggered function is more than ~30 lines of orchestration, decompose it.

**One trigger per (source, event_type) you actually handle differently.** Don't subscribe to `*` and switch internally — that's a static-pipeline anti-pattern wearing a webhook hat. Declare separate `@on_event(types=[...])` decorators if `issues.opened` and `pull_request.merged` need different downstream graphs.

**Cron is for periodic work, not delays.** `@on_schedule("* * * * *")` runs every minute forever. If you need "fire once in five minutes," that's a job for an external scheduler or a self-rearming reasoner — not a cron trigger.

**Idempotency is on the reasoner, not the source.** The CP stores every inbound event before dispatch and dedupes by `(source, idempotency_key)`. A retried delivery from the provider is filtered at ingest. But anything inside your reasoner — memory writes, downstream `app.call`s — must still be safe to repeat in case the reasoner itself retries.

**Verify shape, never trust the payload.** A `payment` dict from Stripe may be missing fields when the upstream object is malformed. Use `dict.get(...)` with defaults; raise early with a known error if a required field is absent. The CP records the failure and exposes it on the trigger sheet.

## When NOT to use triggers

- The system runs synchronously from your own code (`app.call("...")` from a script). Use `tags=["entry"]` and a normal reasoner.
- The "event" is just a periodic poll of a third-party API. Cron + reasoner is fine, but consider whether the third party offers a real webhook first.
- The use case fans out to many parallel "agents". Triggers don't fan out — they call ONE reasoner. Fan-out happens inside the triggered reasoner via `app.call(...)` + `asyncio.gather`.

## Concrete example — GitHub issues → LLM triage

A trigger that summarizes new issues via OpenRouter, hands off to a downstream specialist, never blocks on the LLM call:

```python
from agentfield import on_event, TriggerContext

@app.reasoner()
@on_event(source="github", types=["issues"], secret_env="GITHUB_WEBHOOK_SECRET")
async def on_issue(event: dict, trigger: TriggerContext | None = None):
    """Triage entry. Thin: skip non-content actions, hand off to specialists."""
    if event.get("action") not in {"opened", "edited", "reopened"}:
        return {"skipped": True, "action": event.get("action")}

    issue = event.get("issue") or {}
    repo  = (event.get("repository") or {}).get("full_name", "")
    number = issue.get("number")

    summary, severity = await asyncio.gather(
        app.call(f"{app.node_id}.summarize_issue", title=issue["title"], body=issue["body"]),
        app.call(f"{app.node_id}.classify_severity", title=issue["title"], body=issue["body"]),
    )
    return {
        "repo": repo,
        "number": number,
        "summary": summary["summary"],
        "severity": severity["level"],
        "trigger_event_id": trigger.event_id if trigger else None,
    }


@app.reasoner()
async def summarize_issue(title: str, body: str) -> dict:
    """One LLM call, narrow scope, returns a clean dict."""
    out = await app.ai(
        system="Summarize this GitHub issue in 2-3 sentences. Plain prose.",
        user=f"Title: {title}\n\n{body}",
        model="openrouter/anthropic/claude-haiku-4-5",
    )
    return {"summary": str(out)}


@app.reasoner()
async def classify_severity(title: str, body: str) -> dict:
    """Fast classifier with a tight schema."""
    class Sev(BaseModel):
        level: Literal["low", "medium", "high"]
        confident: bool
    return (await app.ai(
        system="Classify the severity of this GitHub issue.",
        user=f"{title}\n\n{body}",
        schema=Sev,
        model="openrouter/openai/gpt-4o-mini",
    )).model_dump()
```

The trigger reasoner is ~15 lines. The work lives in the two specialists. Both run in parallel.

## Hard rejections specific to triggers

| ❌ | ✅ |
|---|---|
| Long synthesis or multi-step reasoning inside the triggered reasoner | Trigger reasoner is a router; fan out to `@app.reasoner` specialists via `app.call` + `asyncio.gather` |
| `transform=` doing async work or HTTP calls | `transform` is sync, pure, envelope-peeling only. Do I/O in the reasoner |
| Catching the trigger payload's whole nested provider envelope into the reasoner signature | Use `transform=` to peel down to the fields the reasoner actually consumes |
| Hardcoded webhook secret in source | `secret_env="STRIPE_WEBHOOK_SECRET"` — the CP reads the env var at request time, the value never leaves the host |
| Ignoring `trigger.idempotency_key` and writing to memory unconditionally | Read existing key first OR scope writes by `trigger.event_id` so retries are no-ops |
| Subscribing to all event types and branching internally | One `@on_event(types=[...])` per behavior; let the CP filter at ingest |
| Cron used as a one-shot delay (`* * * * *` then `if first_run: ...`) | Cron is periodic. One-shots belong elsewhere |
| Triggered reasoner with no fallback for missing optional fields | `event.get("action") in {...}` early; bail with `{"skipped": True}` for shapes you don't handle |

## Smoke test for a triggered build

After `docker compose up`:

1. Confirm the trigger registered:
   ```bash
   curl -s http://localhost:8080/api/v1/triggers | jq '.triggers[] | {source_name, target_reasoner, id}'
   ```
2. Fire a signed test event (use the provider's CLI or `scripts/fire-events.sh` from `examples/triggers-demo/`).
3. Watch the trigger sheet in the UI (`/ui/triggers` → click the row) — events tab should show the delivery with valid signature, dispatch outcome, and the run that executed.

If the run has `trigger_source: "<your-source>"` on `/runs` AND the run-detail page's Webhooks card shows an Inbound section, the chain is healthy.
