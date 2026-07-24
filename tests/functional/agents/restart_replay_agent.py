"""
Agent definitions for restart/fork replay functional tests.

The graph intentionally fails once after successful upstream app.call edges:

root_investigation
  -> plan_scope                 deterministic planning checkpoint
  -> assess_dimension x 2       deterministic parallel fan-out
  -> synthesize                 fails once, then succeeds
  -> verify_recovery            deterministic post-restart check

The module-level counters let tests prove replay skipped upstream dispatches
after restarting from the failed synthesize node.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, Optional

from agentfield import AIConfig, Agent
from agentfield.async_config import AsyncConfig
from agents import AgentSpec


AGENT_SPEC = AgentSpec(
    key="restart_replay",
    display_name="Restart Replay Functional Agent",
    default_node_id="restart-replay-agent",
    description="Complex graph that validates restart reuse across app.call boundaries.",
    reasoners=(
        "root_investigation",
        "plan_scope",
        "assess_dimension",
        "synthesize",
        "verify_recovery",
    ),
    skills=(),
)

CALL_COUNTS: defaultdict[str, int] = defaultdict(int)
FAILED_SCENARIOS: set[str] = set()


def reset_state() -> None:
    CALL_COUNTS.clear()
    FAILED_SCENARIOS.clear()


def snapshot_counts() -> Dict[str, int]:
    return dict(CALL_COUNTS)


def _bump(name: str) -> None:
    CALL_COUNTS[name] += 1


def create_agent(
    ai_config: AIConfig,
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    resolved_node_id = node_id or AGENT_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )
    agent_kwargs.setdefault(
        "async_config",
        AsyncConfig(
            completed_execution_retention_seconds=300.0,
            result_cache_ttl=300.0,
            cleanup_interval=30.0,
        ),
    )

    agent = Agent(
        node_id=resolved_node_id,
        ai_config=ai_config,
        **agent_kwargs,
    )

    @agent.reasoner(name="root_investigation", tags=["entry"])
    async def root_investigation(
        scenario_id: str,
        question: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        _bump("root_investigation")

        plan = await agent.call(
            f"{agent.node_id}.plan_scope",
            question=question,
            model=model,
        )

        dimensions = ["technical", "product"]
        assessments = await asyncio.gather(
            *[
                agent.call(
                    f"{agent.node_id}.assess_dimension",
                    scenario_id=scenario_id,
                    dimension=dimension,
                    question=question,
                    plan=plan,
                    model=model,
                )
                for dimension in dimensions
            ]
        )

        synthesis = await agent.call(
            f"{agent.node_id}.synthesize",
            scenario_id=scenario_id,
            question=question,
            plan=plan,
            assessments=assessments,
            model=model,
        )

        verdict = await agent.call(
            f"{agent.node_id}.verify_recovery",
            scenario_id=scenario_id,
            question=question,
            synthesis=synthesis,
            model=model,
        )

        return {
            "scenario_id": scenario_id,
            "plan": plan,
            "assessments": assessments,
            "synthesis": synthesis,
            "verdict": verdict,
            "counts": snapshot_counts(),
        }

    @agent.reasoner(name="plan_scope")
    async def plan_scope(question: str, model: Optional[str] = None) -> Dict[str, Any]:
        _bump("plan_scope")
        return {
            "focus": "restart recovery",
            "risk": "medium",
            "confident": True,
            "question": question,
        }

    @agent.reasoner(name="assess_dimension")
    async def assess_dimension(
        scenario_id: str,
        dimension: str,
        question: str,
        plan: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        _bump(f"assess_dimension:{dimension}")
        return {
            "scenario_id": scenario_id,
            "dimension": dimension,
            "finding": (
                f"{dimension} checkpoint can be reused before rerunning synthesis"
            ),
            "severity": 3 if dimension == "technical" else 2,
            "confident": bool(plan.get("confident", True)),
            "question": question,
        }

    @agent.reasoner(name="synthesize")
    async def synthesize(
        scenario_id: str,
        question: str,
        plan: Dict[str, Any],
        assessments: list[Dict[str, Any]],
        model: Optional[str] = None,
        execution_context: Any = None,
    ) -> Dict[str, Any]:
        _bump("synthesize")
        ctx = execution_context or agent.ctx
        is_replay_restart = bool(ctx and ctx.replay_source_run_id)
        if not is_replay_restart and scenario_id not in FAILED_SCENARIOS:
            FAILED_SCENARIOS.add(scenario_id)
            raise RuntimeError(
                "transient failure after upstream checkpoints: "
                f"{scenario_id}; replay_source={getattr(ctx, 'replay_source_run_id', None)} "
                f"mode={getattr(ctx, 'replay_mode', None)}"
            )

        high = [item for item in assessments if int(item.get("severity", 0)) >= 3]
        return {
            "scenario_id": scenario_id,
            "question": question,
            "focus": plan.get("focus"),
            "high_count": len(high),
            "replay_source_run_id": ctx.replay_source_run_id if ctx else None,
            "summary": "restart reused upstream checkpoints and continued synthesis",
        }

    @agent.reasoner(name="verify_recovery")
    async def verify_recovery(
        scenario_id: str,
        question: str,
        synthesis: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        _bump("verify_recovery")
        summary = str(synthesis.get("summary") or "")
        return {
            "scenario_id": scenario_id,
            "question": question,
            "verdict": "restart recovered and continued",
            "confident": "reused upstream checkpoints" in summary,
        }

    return agent


def create_agent_from_env() -> Agent:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-3.1-flash-lite")
    node_id = os.environ.get("AGENT_NODE_ID")

    ai_config = AIConfig(
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=500,
        timeout=120.0,
        retry_attempts=3,
    )
    return create_agent(ai_config, node_id=node_id)


__all__ = [
    "AGENT_SPEC",
    "create_agent",
    "create_agent_from_env",
    "reset_state",
    "snapshot_counts",
]


if __name__ == "__main__":
    create_agent_from_env().run()
