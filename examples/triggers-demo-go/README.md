# Triggers demo (Go)

Go counterpart to `examples/triggers-demo/` (Python) and
`examples/triggers-demo-ts/` (TypeScript). Same demo, same unmodified
`fire-events.sh`, equivalent memory writes, built on the Go SDK's
`OnEvent` / `OnSchedule` surface and the `triggers` package.

---

## What you get

| Service | Port | What it is |
|---|---|---|
| `control-plane` | 8080 | AgentField server with the embedded UI |
| `triggers-demo-go-agent` | 8001 | Go agent declaring three triggers |

The agent's code-managed triggers auto-register on startup:

| Reasoner | Source | Fires on | Notes |
|---|---|---|---|
| `handle_payment` | `stripe` | `payment_intent.succeeded` | applies a `Transform` to flatten Stripe's nested envelope before the handler runs |
| `handle_pr` | `github` | `pull_request.*` | no transform; the handler reads the raw event |
| `handle_tick` | `cron` | every minute | increments a counter in memory |

All three are deterministic, with no LLM calls.

---

## Quick start

```bash
cd examples/triggers-demo-go

# 1. Bring up control plane + agent
docker compose up --build -d

# 2. Wait ~30 seconds for both containers to come up and the agent to register
docker compose logs -f triggers-demo-go-agent

# 3. Open the UI
open http://localhost:8080/ui/triggers

# 4. Fire signed Stripe + GitHub events (cron fires on its own)
./scripts/fire-events.sh
```

Within a few seconds of running `fire-events.sh`, the events appear in the
right-side detail Sheet (click any trigger row) with `signature: valid`, and
within a minute the cron trigger has fired at least once on its own.

---

## What the code looks like

Declaring an event trigger with a transform:

```go
app.OnEvent(triggers.EventOpts{
    Source:    "stripe",
    Types:     []string{"payment_intent.succeeded"},
    SecretEnv: "STRIPE_DEMO_SECRET",
    Transform: stripeToPayment,
}, "handle_payment", func(ctx context.Context, payment map[string]any) (any, error) {
    tc := triggers.FromContext(ctx) // nil on direct calls
    ...
})
```

Declaring a cron schedule:

```go
app.OnSchedule("* * * * *", "handle_tick", func(ctx context.Context, _ map[string]any) (any, error) {
    tc := triggers.FromContext(ctx)
    ...
})
```

The handler reads trigger metadata through `triggers.FromContext(ctx)`, which
returns `nil` when the reasoner was invoked directly rather than dispatched by
an inbound event. That nil check is what lets the same reasoner serve both
paths, and every record in this demo falls back to `"direct_call"` accordingly.

`Transform` runs only on trigger dispatches, never on direct calls, so a direct
`POST` with a flat body reaches the handler untouched.

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
   1. resolves the trigger row from <trigger_id>
   2. asks the Source plugin to verify the signature
   3. persists InboundEvent
   4. dispatches {event, _meta} envelope to the agent's reasoner endpoint
   ▼
triggers-demo-go-agent:
   - SDK detects the {event, _meta} envelope shape
   - SDK runs the per-binding Transform (Stripe only here)
   - SDK attaches *triggers.Context, readable via triggers.FromContext(ctx)
   - reasoner runs deterministically and writes to memory
   ▼
UI:
   - SSE stream pushes the event lifecycle into the open Sheet
   - run detail picks up the run and its trigger
```

---

## Memory keys

Identical to the Python and TypeScript demos, so the UI surfaces read the same
shapes regardless of which demo is running:

| Reasoner | Key | Record |
|---|---|---|
| `handle_payment` | `payment:<stripe_id>` | `kind`, `stripe_id`, `amount_cents`, `currency`, `customer`, `received_via`, `trigger_event_id` |
| `handle_pr` | `pr:<repo>#<number>` | `kind`, `action`, `number`, `title`, `html_url`, `user`, `repo`, `received_via`, `delivery_id` |
| `handle_tick` | `cron:tick:count` | `count`, `last_fired_at`, `received_via` |

`handle_pr` only writes when both the repo and number are present, so a
malformed payload does not create a `pr:<nil>#<nil>` key.

---

## A note on the Slack / HMAC / Bearer sections of the script

`fire-events.sh` is shared verbatim with the Python demo, and its later
sections lazily create UI-managed Slack, `generic_hmac`, and `generic_bearer`
triggers routed at a `handle_inbound` catch-all reasoner. The Python demo
defines that reasoner; this demo and the TypeScript one do not, matching the
three-reasoner scope of the demo.

The practical effect: those three triggers are created and their signatures
verify, but dispatch fails because the target reasoner does not exist on this
node. The Stripe, GitHub, and cron paths, which are what this demo is about,
work end to end. Run the Python demo if you want to exercise all six source
plugins against a live handler.

---

## Running the tests

The demo's reasoners are unit-tested with the `triggers` package test helpers,
without a control plane, HTTP server, or real provider:

```bash
cd examples/triggers-demo-go
go test ./...
```

`main_test.go` doubles as a worked example of `triggers.SimulateEvent`,
`triggers.SimulateSchedule`, and `triggers.LoadFixture` against the shared
fixture library.

---

## Custom secrets

The demo bakes plain-text demo secrets into `docker-compose.yml`. To use your
own, override before `docker compose up`:

```bash
STRIPE_DEMO_SECRET=whsec_xxx \
GITHUB_DEMO_SECRET=ghsecret_xxx \
  docker compose up --build -d
```

Then re-run `fire-events.sh` with the same values exported in your shell so the
script signs with the matching secret.

---

## Tearing down

```bash
docker compose down --volumes
```

`--volumes` deletes the SQLite database so the next run starts clean.
