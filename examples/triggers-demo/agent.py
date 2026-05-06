"""
AgentField triggers demo — sample Python agent.

Three deterministic reasoners, each wired to a different Source plugin:

- handle_payment   ← Stripe webhook (Stripe-Signature HMAC)
- handle_pr        ← GitHub webhook (X-Hub-Signature-256 HMAC)
- handle_tick      ← cron schedule (every minute)

No LLM calls. Each reasoner just transforms its inbound event into a
small, deterministic record and writes it to per-agent memory so the UI's
event log + run detail surfaces show real data flowing through.

When the agent registers with the control plane, the @on_event /
@on_schedule decorators auto-create code-managed Trigger rows. The CP
returns the public URLs for each, which the SDK prints at startup:

    Stripe webhook URL: http://localhost:8080/sources/<id>
    GitHub webhook URL: http://localhost:8080/sources/<id>
    Cron schedule "* * * * *" registered

Paste those URLs into provider dashboards (or use the included
`scripts/fire-events.sh` to fire signed test events locally).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict

from agentfield import (
    Agent,
    EventTrigger,
    ScheduleTrigger,
    TriggerContext,
    on_event,
    on_schedule,
    reasoner,
)


app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "triggers-demo-agent"),
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    dev_mode=True,
)


# ---------------------------------------------------------------------------
# Stripe — payment events
#
# The Stripe source plugin verifies Stripe-Signature: t=<ts>,v1=<hmac> over
# "<ts>.<body>" using the secret read from STRIPE_DEMO_SECRET on the CP host.
# The transform here pulls the bits we actually care about out of Stripe's
# fairly nested envelope so the reasoner body stays clean.
# ---------------------------------------------------------------------------


def _stripe_to_payment(event: dict) -> Dict[str, Any]:
    obj = event.get("data", {}).get("object", {})
    return {
        "id": obj.get("id"),
        "amount": obj.get("amount"),
        "currency": obj.get("currency", "usd"),
        "customer": obj.get("customer"),
        "status": obj.get("status"),
        "metadata": obj.get("metadata", {}),
    }


@app.reasoner(
    triggers=[
        EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],
            secret_env="STRIPE_DEMO_SECRET",
            transform=_stripe_to_payment,
        ),
    ],
)
async def handle_payment(payment: dict, trigger: TriggerContext | None = None):
    """Records a Stripe payment, deterministically."""
    record = {
        "kind": "payment",
        "stripe_id": payment.get("id"),
        "amount_cents": payment.get("amount"),
        "currency": payment.get("currency"),
        "customer": payment.get("customer"),
        "received_via": trigger.source if trigger else "direct_call",
        "trigger_event_id": trigger.event_id if trigger else None,
    }
    await app.memory.set(key=f"payment:{record['stripe_id']}", data=record)
    print(f"[handle_payment] saved {record}", flush=True)
    return record


# ---------------------------------------------------------------------------
# GitHub — pull-request events
#
# The GitHub source verifies X-Hub-Signature-256 = sha256=<hmac of body>
# using the secret from GITHUB_DEMO_SECRET. Reads X-GitHub-Event +
# X-GitHub-Delivery for type and idempotency.
# ---------------------------------------------------------------------------


@app.reasoner()
@on_event(
    source="github",
    types=["pull_request"],
    secret_env="GITHUB_DEMO_SECRET",
)
async def handle_pr(event: dict, trigger: TriggerContext | None = None):
    """Records a GitHub pull_request action."""
    pr = event.get("pull_request", {})
    record = {
        "kind": "pull_request",
        "action": event.get("action"),
        "number": event.get("number") or pr.get("number"),
        "title": pr.get("title"),
        "html_url": pr.get("html_url"),
        "user": (pr.get("user") or {}).get("login"),
        "repo": (event.get("repository") or {}).get("full_name"),
        "received_via": trigger.source if trigger else "direct_call",
        "delivery_id": trigger.idempotency_key if trigger else None,
    }
    if record["repo"] and record["number"]:
        key = f"pr:{record['repo']}#{record['number']}"
        await app.memory.set(key=key, data=record)
    print(f"[handle_pr] saved {record}", flush=True)
    return record


# ---------------------------------------------------------------------------
# GitHub issues — LLM-powered summary
#
# Same source as handle_pr (single signed GitHub webhook can carry multiple
# event types), but matches on the `issues` event. Calls OpenRouter via
# app.ai() to produce a short summary, then writes it to per-agent memory
# so the result shows up in the trigger event row + run detail surfaces.
# ---------------------------------------------------------------------------


@app.reasoner()
@on_event(
    source="github",
    types=["issues"],
    secret_env="GITHUB_DEMO_SECRET",
)
async def summarize_issue(event: dict, trigger: TriggerContext | None = None):
    """Summarises a GitHub issue via OpenRouter on `issues.opened`."""
    issue = event.get("issue") or {}
    action = event.get("action", "")
    repo = (event.get("repository") or {}).get("full_name", "")
    number = issue.get("number")

    # Skip non-content actions (labeled, assigned, etc.). We only summarise
    # when the issue is first opened or its body materially changes.
    if action not in {"opened", "edited", "reopened"}:
        record = {
            "kind": "issue_skipped",
            "repo": repo,
            "number": number,
            "action": action,
        }
        print(f"[summarize_issue] skipped action={action}", flush=True)
        return record

    title = issue.get("title", "(no title)")
    body = issue.get("body") or "(no body provided)"
    author = (issue.get("user") or {}).get("login", "unknown")
    html_url = issue.get("html_url", "")

    summary = await app.ai(
        system=(
            "You are a triage assistant. Given a GitHub issue, write a 2-3 "
            "sentence summary that captures (a) the core problem, (b) any "
            "reproduction steps, and (c) what the reporter expects. Plain "
            "prose, no headers."
        ),
        user=(
            f"Repo: {repo}\n"
            f"Author: {author}\n"
            f"Title: {title}\n\n"
            f"Body:\n{body}"
        ),
        model="openrouter/anthropic/claude-haiku-4-5",
        max_tokens=300,
    )

    record = {
        "kind": "issue_summary",
        "repo": repo,
        "number": number,
        "title": title,
        "url": html_url,
        "author": author,
        "summary": str(summary),
        "received_via": trigger.source if trigger else "direct_call",
        "trigger_event_id": trigger.event_id if trigger else None,
    }
    if repo and number:
        await app.memory.set(key=f"issue:{repo}#{number}", data=record)
    print(f"[summarize_issue] saved {repo}#{number} — {record['summary'][:120]}…", flush=True)
    return record


# ---------------------------------------------------------------------------
# Generic catch-all — exercised by UI-managed Slack / HMAC / Bearer triggers
#
# The demo's three code-managed triggers cover Stripe, GitHub, and cron via
# decorators. To exercise the other three built-in source plugins (Slack
# Events API, generic HMAC, generic Bearer) the fire-events.sh script lazily
# creates UI-managed triggers via POST /api/v1/triggers, all routed at this
# one reasoner. handle_inbound just records the payload + provider context
# into per-agent memory so the UI's trigger event surface shows real data.
# The `accepts_webhook="yes"` opt-in lets the CP create UI triggers for it
# without the soft-warning fallback.
# ---------------------------------------------------------------------------


@app.reasoner(accepts_webhook=True)
async def handle_inbound(payload, trigger: TriggerContext | None = None):
    """Catch-all handler for UI-managed triggers (Slack / HMAC / Bearer)."""
    record = {
        "kind": "inbound",
        "received_via": trigger.source if trigger else "direct_call",
        "event_type": trigger.event_type if trigger else None,
        "event_id": trigger.event_id if trigger else None,
        "idempotency_key": trigger.idempotency_key if trigger else None,
        "payload": payload,
    }
    if trigger and trigger.event_id:
        await app.memory.set(key=f"inbound:{trigger.source}:{trigger.event_id}", data=record)
    print(f"[handle_inbound] {trigger.source if trigger else 'direct'} event_type={record['event_type']}", flush=True)
    return record


# ---------------------------------------------------------------------------
# Cron — periodic tick
#
# The cron source runs as a LoopSource inside the CP, emitting a "tick" event
# every time its schedule fires. The agent sees the same dispatch shape as
# any other webhook delivery — so the reasoner code path is identical.
# ---------------------------------------------------------------------------


@app.reasoner()
@on_schedule("* * * * *")
async def handle_tick(_input, trigger: TriggerContext | None = None):
    """Increments a cron-fire counter and records the wall-clock time."""
    counter_key = "cron:tick:count"
    current = (await app.memory.get(key=counter_key)) or {"count": 0}
    record = {
        "count": (current.get("count") or 0) + 1,
        "last_fired_at": trigger.received_at.isoformat() if trigger else None,
        "received_via": trigger.source if trigger else "direct_call",
    }
    await app.memory.set(key=counter_key, data=record)
    print(f"[handle_tick] {record}", flush=True)
    return record


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def _heartbeat() -> None:
    """Surface in container logs that the agent is alive between events."""
    n = 0
    node = app.node_id
    while True:
        print(f"[{node}] alive heartbeat #{n}", flush=True)
        n += 1
        time.sleep(30)


if __name__ == "__main__":
    threading.Thread(target=_heartbeat, daemon=True).start()
    port = int(os.getenv("PORT", "8001"))
    # Banner so the user sees the agent come up. The SDK separately prints
    # the assigned trigger URLs once it registers with the CP.
    print(
        "AgentField triggers demo — sample agent starting\n"
        f"  node_id            = {app.node_id}\n"
        f"  agentfield_server  = {os.getenv('AGENTFIELD_URL', 'http://localhost:8080')}\n"
        f"  callback url       = {os.getenv('AGENT_CALLBACK_URL', f'http://localhost:{port}')}\n"
        "  reasoners          = handle_payment (stripe), handle_pr (github), summarize_issue (github),\n"
        "                       handle_inbound (slack/hmac/bearer via UI), handle_tick (cron)",
        flush=True,
        file=sys.stderr,
    )
    app.run(host="0.0.0.0", port=port, auto_port=False)
