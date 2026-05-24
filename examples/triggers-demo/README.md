# Triggers demo

Run AgentField with a sample agent that exercises every built-in
source plugin — **Stripe payments**, **GitHub pull requests + issues**, a
**cron schedule**, plus **Slack**, **generic HMAC**, and **generic Bearer**
inbound webhooks — and watch the events flow through the UI live.

Most of the agent's reasoners are fully deterministic (no LLM calls); the
GitHub `issues` reasoner optionally calls OpenRouter to summarise issue
bodies and is skipped if `OPENROUTER_API_KEY` isn't set. This is the
quickest way to see what the trigger plugin system looks like end-to-end.

---

## What you get

| Service | Port | What it is |
|---|---|---|
| `control-plane` | 8080 | AgentField server with the embedded UI |
| `triggers-demo-agent` | 8001 | Python agent declaring three triggers |

The agent's code-managed triggers, declared via `@on_event` /
`@on_schedule`, auto-register on startup:

| Reasoner | Source | Fires on | Notes |
|---|---|---|---|
| `handle_payment` | `stripe` | `payment_intent.succeeded` | applies a Stripe-specific transform before invoking the reasoner |
| `handle_pr` | `github` | `pull_request.*` | |
| `summarize_issue` | `github` | `issues.*` | calls OpenRouter (Claude Haiku) when `OPENROUTER_API_KEY` is set |
| `handle_tick` | `cron` | every minute | |
| `handle_inbound` | `slack` / `generic_hmac` / `generic_bearer` | bound at runtime | catch-all reasoner for the three UI-managed triggers `fire-events.sh` lazily creates |

---

## Quick start

```bash
cd examples/triggers-demo

# 1. Bring up control plane + agent
docker compose up --build -d

# 2. Wait ~30 seconds for both containers to come up and the agent to register
docker compose logs -f triggers-demo-agent

# 3. Open the UI (the embedded SPA mounts under /ui/, so all UI URLs are
#    /ui/<route>; the equivalent React route inside the app is just /<route>)
open http://localhost:8080/ui/triggers

# 4. Fire a signed Stripe + GitHub event (cron fires on its own)
./scripts/fire-events.sh
```

When you load `http://localhost:8080/ui/triggers`, you'll see four
**code** trigger rows — Stripe, GitHub × 2, cron — each with a public URL,
an enabled toggle, and the source code location they were declared at
(`agent.py:NN`). After `fire-events.sh` runs, three more **ui** rows
appear: Slack, generic_hmac, generic_bearer.

Within a few seconds of running `fire-events.sh`, the events appear live in
the right-side detail Sheet (click any trigger row), and within a minute
the cron trigger has fired at least once on its own.

---

## What to look at in the UI

The trigger feature surfaces in **seven places** — open each as you walk
through the demo to see how the pieces fit together.

### 1. `/triggers` — the master list

The single Triggers page is a master-detail layout. You should see:

- A **Sources strip** at the top showing the six built-in plugins
- A filterable **Active triggers table** with the three demo triggers
- A `code` badge on each row (since they were registered via decorators)
- The public ingest URL with a copy button
- A **secret-status pill** showing whether the env var is set on the CP host
- An **enabled** Switch
- 24h event count + last-event timestamp

### 2. Right-side detail Sheet (click a trigger row)

The Sheet has four tabs:

- **Events** — the live event list. Each row collapses by default; click to expand inline. New events appear in real time over Server-Sent Events as the cron fires and as you run `fire-events.sh`.
- **Configuration** — read-only JSON view (because these triggers are code-managed)
- **Secrets** — the env var name + a "set / missing" status pill
- **Dispatch logs** — placeholder for now

The Sheet header carries the source icon, name, public URL with copy, an enabled `Switch`, and a Delete button (disabled with a tooltip for code-managed rows).

For code-managed triggers, the Sheet also shows a **drift card** with the file:line where the decorator was declared (`agent.py:42`) and the timestamp of the most recent agent re-registration.

### 3. Inline event detail (click a row in the Events tab)

Each event expands inline to show:

- A **Verification card** — status badge ("dispatched" / "failed"), received-at + processed-at timestamps
- A **Payload viewer** with three tabs: **Raw** / **Normalized** / **Headers**
- A **VC chain card** (when DID is enabled — this demo runs with DID off, so it shows the "DID not enabled" empty state)
- **Replay** + **Copy as fixture** action buttons

### 4. `/ui/runs` — execution rows tagged with their trigger

When the agent finishes processing an event, the resulting run shows up in
`/ui/runs` (or the existing executions/workflows surface). Rows whose run
was kicked off by an inbound event show a small `↪ Stripe` / `↪ GitHub` /
`↪ Cron` badge next to the run identifier.

### 5. `/ui/runs/:id` — run detail with the webhook input

Click into one of those runs and the run detail page shows a new
**Trigger** card at the top with:

- Source icon + name
- Event type, event ID, received-at
- The webhook payload that fired the run (as the run input, in a
  `UnifiedJsonViewer`)
- A "View this trigger →" deep-link back to the Sheet

### 6. Node detail sidebar — bound triggers per node

Open the demo agent's node in the workflow / nodes UI and the detail
sidebar shows a "Triggers" section listing the three bound triggers with
their public URLs and enabled status.

### 7. Dashboard tile

The main dashboard gains an **"Inbound events (24h)"** `MetricCard` showing
the recent event count, dispatch success rate, and a destructive `Badge`
when DLQ depth goes above zero.

---

## How the pieces fit together

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
   4. dispatches to the agent's reasoner endpoint
   │   (sets X-Trigger-ID, X-Source-Name, X-Event-Type, X-Event-ID)
   ▼
triggers-demo-agent:
   - SDK auto-unwraps the {event, _meta} envelope
   - SDK runs the per-binding `transform` (Stripe-only here)
   - SDK injects ctx.trigger (TriggerContext)
   - Reasoner runs deterministically, writes to memory
   ▼
UI:
   - SSE stream pushes the event lifecycle into the open Sheet
   - Run detail page picks up the run + trigger field
```

The cron trigger doesn't need an inbound HTTP request — the CP runs a
goroutine per cron trigger that fires on schedule and dispatches directly.

---

## Custom secrets

The demo bakes plain-text demo secrets into `docker-compose.yml`. To use
your own, override before `docker compose up`:

```bash
STRIPE_DEMO_SECRET=whsec_xxx \
GITHUB_DEMO_SECRET=ghsecret_xxx \
SLACK_DEMO_SIGNING_SECRET=xxx \
GENERIC_HMAC_DEMO_SECRET=xxx \
GENERIC_BEARER_DEMO_TOKEN=xxx \
  docker compose up --build -d
```

Then re-run `fire-events.sh` with the same values exported in your shell so
the script signs with the matching secret.

## Optional: enable the LLM-powered issue summary

The `summarize_issue` reasoner is wired to OpenRouter and is the only
reasoner in the demo that makes an external network call. Without an
API key it will fail (visibly) on the trigger event row — the rest of the
demo still works. To enable it:

```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
docker compose up --build -d
```

`docker-compose.yml` forwards the variable into the agent container.

---

## Pointing real Stripe / GitHub at the demo

If you want to skip `fire-events.sh` and use the real providers:

1. Expose the CP to the public internet, e.g.

   ```bash
   ngrok http 8080
   ```

2. Copy the Stripe trigger's public URL from `/triggers` (or curl `GET /api/v1/triggers`) and prepend the ngrok host:

   ```
   https://abc-123.ngrok.app/sources/<stripe_trigger_id>
   ```

3. Paste it into the Stripe dashboard's webhook settings. Use the
   `STRIPE_DEMO_SECRET` value (or rotate to a real signing secret) when
   Stripe asks for the endpoint's signing secret.

4. Repeat for GitHub: copy the GitHub trigger's URL into the repository's
   Settings → Webhooks page; set the secret to `GITHUB_DEMO_SECRET`; pick
   the `pull_request` event type.

Real provider events flow through the same pipeline as the script's
synthetic ones.

---

## Tearing down

```bash
docker compose down --volumes
```

`--volumes` deletes the SQLite database so the next run starts clean.
