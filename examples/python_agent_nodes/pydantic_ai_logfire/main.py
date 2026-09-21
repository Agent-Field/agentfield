"""Pydantic AI + Logfire inside an AgentField reasoner.

Pydantic AI owns one step (prompt, tool loop, validated output); AgentField owns
the run DAG around it. Logfire traces both halves in a single trace tagged with
the AgentField run. See docs/pydantic-ai.md.

    export OPENROUTER_API_KEY=...   # used by app.ai and by the Pydantic AI model
    export LOGFIRE_TOKEN=...        # or LOGFIRE_SEND_TO_LOGFIRE=false for console-only
    python main.py
"""

import os
from typing import Literal

import logfire
from agentfield import Agent, AIConfig
from agentfield.cost_tracker import get_current_cost_tracker
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
logfire.instrument_litellm()  # every app.ai(...) completion

app = Agent(
    node_id="ticket-triage",
    version="1.0.0",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(model=AF_MODEL),
)

# Agent subclasses FastAPI, so the inbound reasoner request becomes the root span
# and everything below it lands in the same trace. The control plane polls
# /status continuously; excluding it keeps that noise out of Logfire.
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


def record_pydantic_ai_usage(result) -> None:
    """Fold a Pydantic AI run's tokens into AgentField's per-execution usage."""
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


@app.reasoner()
async def triage_ticket(ticket: str) -> dict:
    """Triage a ticket with Pydantic AI, then draft a reply with app.ai."""
    with logfire.set_baggage(**agentfield_baggage()):
        triaged = await triager.run(ticket)
        record_pydantic_ai_usage(triaged)

        reply = await app.ai(
            system="You write one-sentence support replies.",
            user=f"Ticket: {ticket}\nSeverity: {triaged.output.severity}",
        )
        return {
            "severity": triaged.output.severity,
            "summary": triaged.output.summary,
            "reply": str(reply),
        }


if __name__ == "__main__":
    app.run(auto_port=True)
