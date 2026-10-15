# Pydantic AI inside an AgentField reasoner

A Pydantic AI `Agent` is ordinary Python, so it runs inside a reasoner with no
adapter. The split that works well in practice:

- **Pydantic AI** owns one step: the prompt, the tool loop, and the validated
  output type.
- **AgentField** owns everything around it: the run DAG, cross-agent calls,
  async execution, pause/resume, replay and the audit trail.

This page covers the part that is not obvious — getting one Logfire trace per
reasoner execution that contains both the Pydantic AI spans and the `app.ai`
completions, correlated with the AgentField run.

A runnable version of everything below is in
[`examples/python_agent_nodes/pydantic_ai_logfire`](../examples/python_agent_nodes/pydantic_ai_logfire).

## Install

```bash
pip install "agentfield" "pydantic-ai-slim[openai]" "logfire[fastapi,litellm]"
```

`logfire[fastapi]` provides `instrument_fastapi()`, `logfire[litellm]` provides
`instrument_litellm()`. Neither is an AgentField dependency.

## Wiring

```python
import os
from typing import Literal

import logfire
from agentfield import Agent, AIConfig
from agentfield.execution_context import get_current_context
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Two model names for the same model: app.ai takes a LiteLLM-style name, the
# Pydantic AI model takes the provider's own id.
AF_MODEL = os.getenv("AGENTFIELD_AI_MODEL", "openrouter/openai/gpt-4o-mini")
PYDANTIC_AI_MODEL = os.getenv("PYDANTIC_AI_MODEL", "openai/gpt-4o-mini")

# One tracer provider for the whole agent process.
logfire.configure(service_name="ticket-triage")
logfire.instrument_pydantic_ai()  # Pydantic AI agent runs, tool calls, model requests
logfire.instrument_litellm()      # every app.ai(...) completion

app = Agent(
    node_id="ticket-triage",
    version="1.0.0",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(model=AF_MODEL),
)

# Agent subclasses FastAPI, so the inbound reasoner request becomes the root span
# and everything below it lands in the same trace.
logfire.instrument_fastapi(app, excluded_urls="/health,/status,/agentfield/.*")


class Triage(BaseModel):
    severity: Literal["low", "medium", "high"]
    summary: str = Field(description="one sentence, no more")


triager = PydanticAIAgent(
    OpenAIChatModel(PYDANTIC_AI_MODEL, provider=OpenRouterProvider()),
    output_type=Triage,
    system_prompt="Triage the incoming support ticket.",
)


def agentfield_baggage() -> dict:
    """AgentField run correlation, as OpenTelemetry baggage."""
    ctx = get_current_context()
    if ctx is None:
        return {}
    values = {
        "agentfield.run_id": ctx.run_id,
        "agentfield.execution_id": ctx.execution_id,
        "agentfield.node_id": ctx.agent_node_id or app.node_id,
        "agentfield.reasoner": ctx.reasoner_name,
        "agentfield.session_id": ctx.session_id,
    }
    return {k: v for k, v in values.items() if v}


@app.reasoner()
async def triage_ticket(ticket: str) -> dict:
    with logfire.set_baggage(**agentfield_baggage()):
        triaged = await triager.run(ticket)
        reply = await app.ai(
            system="You write one-sentence support replies.",
            user=f"Ticket: {ticket}\nSeverity: {triaged.output.severity}",
        )
        return {
            "severity": triaged.output.severity,
            "summary": triaged.output.summary,
            "reply": str(reply),
        }
```

Three things are doing the work:

1. `logfire.configure()` runs once per process, at import time, before the agent
   serves traffic. Both instrumentors and the FastAPI middleware attach to that
   single tracer provider, so Pydantic AI and LiteLLM spans end up in the same
   trace rather than two disconnected trees.
2. `logfire.instrument_fastapi(app)` works because `agentfield.Agent` is a
   `FastAPI` subclass. The reasoner request itself becomes the root span.
3. `logfire.set_baggage(...)` puts the AgentField identifiers into
   OpenTelemetry baggage. Logfire copies baggage onto every span created inside
   the block (`add_baggage_to_attributes`, on by default), so the correlation
   lands on Pydantic AI's spans and LiteLLM's without threading anything
   through either library's API.

## What you get

One trace per reasoner execution:

```
POST /reasoners/triage_ticket            opentelemetry.instrumentation.fastapi
├── invoke_agent triager                 pydantic-ai
│   └── chat openai/gpt-4o-mini          pydantic-ai
└── acompletion                          openinference.instrumentation.litellm
```

Every span below the root carries `agentfield.run_id`,
`agentfield.execution_id`, `agentfield.node_id` and `agentfield.reasoner`, so a
Logfire trace joins back to an AgentField run by attribute. This holds for
`POST /api/v1/execute/...` and `POST /api/v1/execute/async/...` alike.

## Keeping Pydantic AI usage in AgentField's accounting

`app.ai` records tokens and cost into a per-execution cost tracker, which the
SDK reports back to the control plane. Pydantic AI talks to the provider
itself, so its tokens are invisible to that tracker unless you hand them over:

```python
from agentfield.cost_tracker import get_current_cost_tracker


def record_pydantic_ai_usage(result) -> None:
    tracker = get_current_cost_tracker() or app.cost_tracker
    if tracker is None:
        return
    usage = result.usage
    ctx = get_current_context()
    tracker.record(
        model=f"openrouter/{PYDANTIC_AI_MODEL}",
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_write_tokens,
        cost_usd=float(usage.cost) if usage.cost is not None else None,
        cost_source="genai-prices" if usage.cost is not None else None,
        reasoner_name=ctx.reasoner_name if ctx else None,
    )


triaged = await triager.run(ticket)
record_pydantic_ai_usage(triaged)
```

The execution's usage then carries both calls. On a synchronous dict result the
SDK merges it into the response under the reserved `__agentfield_usage__` key;
on a control-plane-dispatched async execution the same entries go back in the
`usage` field of the status callback. Either way it is recorded against the
execution and kept out of the stored result:

```json
{
  "total_cost_usd": 5.1e-05,
  "total_input_tokens": 112,
  "total_output_tokens": 57,
  "entries": [
    {"provider": "openrouter", "model": "openrouter/openai/gpt-4o-mini", "total_tokens": 117, "cost_usd": 3.42e-05, "cost_source": "genai-prices"},
    {"provider": "openai", "model": "openai/gpt-4o-mini", "total_tokens": 52, "cost_usd": 1.68e-05, "cost_source": "provider"}
  ]
}
```

The first entry is the Pydantic AI run, the second is the `app.ai` completion.

## `instrument_litellm()` or `AGENTFIELD_LITELLM_CALLBACKS`

AgentField can also register LiteLLM's own `logfire` callback — see
[LLM observability](llm-observability.md). The two are different mechanisms and
you normally want only one:

| | `logfire.instrument_litellm()` | `AGENTFIELD_LITELLM_CALLBACKS=logfire` |
|---|---|---|
| Configured in | agent code | environment |
| Exporter | the process's `logfire.configure()` provider | LiteLLM's own OTLP exporter, straight to `LOGFIRE_BASE_URL` |
| Requires `LOGFIRE_TOKEN` | no (console or local OTLP works) | yes — AgentField registers the name either way, but LiteLLM then logs a non-blocking `LOGFIRE_TOKEN not found` init error and the callback does nothing |
| Correlation | baggage attributes (above) | the `agentfield_*` LiteLLM `metadata` keys stamped by AgentField |

If you enable both, LiteLLM notices the active Logfire span, joins its trace and
— under its defaults — emits only a `raw_gen_ai_request` child rather than a
second completion span. It still ships that span through its own exporter, so
it only reaches the same backend when `LOGFIRE_BASE_URL` points there. Pick one.

## What this does not cover

- **Pydantic AI's durable execution adapters.** Their checkpointing and replay
  sit on top of AgentField's own run DAG, `X-AgentField-Replay-*` and
  pause/resume rather than plugging into them. Use AgentField's async
  execution, `app.pause()` and restart/replay for durability.
- **`app.ai` configuration does not reach Pydantic AI.** Model choice, provider
  routing and retries on the Pydantic AI side are configured on its `Model`.
- **The control plane's spans are a separate trace.** AgentField does not
  propagate W3C `traceparent` to agent nodes, so join the control plane's OTLP
  spans and the node's Logfire trace on `agentfield.run_id`, not on trace ID.

## Gotchas

- `instrument_fastapi()` also traces the node's own endpoints, and the control
  plane polls `/status` continuously for liveness. Keep the `excluded_urls`
  above or that poll dominates your traces.
- On the async dispatch path the reasoner returns `202` immediately and the work
  continues in a background task. The spans stay in the same trace, but the root
  HTTP span closes before its children finish, so its duration is not the
  execution's duration — read that from AgentField.
- LiteLLM callback state is process-global: a process hosting several Agents
  applies the union of their callbacks.
- `result.usage` is a property in Pydantic AI 2.x (it was a method in 1.x).
- `Field(description=...)` documents a field, it does not constrain it — use
  `Literal`/`Enum` when you want the model's output actually validated.

Verified against agentfield 0.1.140-rc.1, pydantic-ai-slim 2.46.0, logfire 5.1.0
and litellm 1.83.0.
