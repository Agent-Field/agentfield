#!/usr/bin/env python3
"""
Seed the local AgentField SQLite store with realistic deep-research workflows.

The script generates configurable DAGs of workflow executions, associated runs,
steps, and timeline events so they show up inside the control-plane UI
(`http://localhost:8080/ui/workflows`).

Example:
    python scripts/seed_workflows.py --nodes-per-workflow 10000 --workflow-count 1
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# --- Data classes ---------------------------------------------------------


@dataclass(frozen=True)
class AgentNodeDefinition:
    node_id: str
    base_url: str
    version: str
    deployment_type: str
    invocation_url: str
    reasoners: Sequence[dict]  # Changed from Sequence[str] to Sequence[dict]
    skills: Sequence[dict]  # Changed from Sequence[str] to Sequence[dict]
    communication: dict
    health_status: str = "healthy"
    lifecycle_status: str = "active"
    features: Optional[dict] = None
    metadata: Optional[dict] = None


@dataclass
class WorkflowScenario:
    workflow_id: str
    workflow_name: str
    workflow_tags: List[str]
    session_id: str
    actor_id: str
    run_id: str
    root_execution_id: str
    root_request_id: str
    started_at: datetime
    completed_at: datetime


@dataclass
class NodeRecord:
    execution_id: str
    request_id: str
    parent_execution_id: Optional[str]
    depth: int
    agent_node_id: str
    reasoner_id: str
    status: str
    status_reason: Optional[str]
    error_message: Optional[str]
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    input_payload: dict
    output_payload: dict
    notes: List[dict]


# --- Constants ------------------------------------------------------------

DEFAULT_DB_PATH = Path("~/.agentfield/data/agentfield.db").expanduser()

AGENT_NODE_POOL: Tuple[AgentNodeDefinition, ...] = (
    AgentNodeDefinition(
        node_id="atlas_scope_orchestrator",
        base_url="https://agents.internal/atlas-scope",
        version="1.14.3",
        deployment_type="long_running",
        invocation_url="https://agents.internal/atlas-scope/invoke",
        reasoners=(
            {
                "id": "Deep Scope Orchestrator",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Context Window Planner",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "graph_navigation", "input_schema": {}, "tags": ["navigation"]},
            {"id": "source_enrichment", "input_schema": {}, "tags": ["enrichment"]},
        ),
        communication={"webhook": "https://callbacks.internal/atlas-scope"},
        features={"supports_multithread": True},
        metadata={"owner": "Atlas Research"},
    ),
    AgentNodeDefinition(
        node_id="source_link_parser",
        base_url="https://agents.internal/source-parser",
        version="2.3.1",
        deployment_type="ephemeral",
        invocation_url="https://agents.internal/source-parser/invoke",
        reasoners=(
            {
                "id": "Signal Enrichment Parser",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Evidence Link Resolver",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "html_extraction", "input_schema": {}, "tags": ["extraction"]},
            {"id": "ocr", "input_schema": {}, "tags": ["extraction"]},
            {"id": "link_resolution", "input_schema": {}, "tags": ["resolution"]},
        ),
        communication={"queue": "source-parser-events"},
        features={"supports_batch": True},
        metadata={"coverage": "public+subscription"},
    ),
    AgentNodeDefinition(
        node_id="signal_prioritizer",
        base_url="https://agents.internal/signal-prioritizer",
        version="0.9.12",
        deployment_type="long_running",
        invocation_url="https://agents.internal/signal-prioritizer/invoke",
        reasoners=(
            {
                "id": "Signal Weighting Pipeline",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Early Warning Ranker",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "ranking", "input_schema": {}, "tags": ["prioritization"]},
            {"id": "anomaly_detection", "input_schema": {}, "tags": ["detection"]},
        ),
        communication={"kafka_topic": "signal-priority"},
        features={"weighted_scores": True},
        metadata={"owner": "SignalOps"},
    ),
    AgentNodeDefinition(
        node_id="narrative_synthesizer",
        base_url="https://agents.internal/narrative-synth",
        version="3.0.2",
        deployment_type="long_running",
        invocation_url="https://agents.internal/narrative-synth/invoke",
        reasoners=(
            {
                "id": "Narrative Multi-Lens Synth",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Cross Lens Summarizer",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "summarization", "input_schema": {}, "tags": ["synthesis"]},
            {"id": "storylining", "input_schema": {}, "tags": ["narrative"]},
        ),
        communication={"webhook": "https://callbacks.internal/narrative"},
        features={"supports_sections": True},
        metadata={"owner": "NarrativeOps"},
    ),
    AgentNodeDefinition(
        node_id="risk_modeler",
        base_url="https://agents.internal/risk-modeler",
        version="1.8.7",
        deployment_type="long_running",
        invocation_url="https://agents.internal/risk-modeler/invoke",
        reasoners=(
            {
                "id": "Risk Quant Scorer",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Scenario Stress Analyzer",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "scenario_planning", "input_schema": {}, "tags": ["planning"]},
            {"id": "stress_testing", "input_schema": {}, "tags": ["testing"]},
        ),
        communication={"queue": "risk-modeler-events"},
        features={"supports_probabilistic": True},
        metadata={"coverage": "macroeconomic"},
    ),
    AgentNodeDefinition(
        node_id="market_lens_vectorizer",
        base_url="https://agents.internal/market-lens",
        version="0.6.5",
        deployment_type="ephemeral",
        invocation_url="https://agents.internal/market-lens/invoke",
        reasoners=(
            {
                "id": "Market Vector Indexer",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Capital Flow Mapper",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "embedding", "input_schema": {}, "tags": ["vectorization"]},
            {"id": "semantic_alignment", "input_schema": {}, "tags": ["alignment"]},
        ),
        communication={"queue": "market-lens-jobs"},
        features={"supports_dimensionality": 1536},
        metadata={"owner": "MarketIntel"},
    ),
    AgentNodeDefinition(
        node_id="insight_validation_bureau",
        base_url="https://agents.internal/insight-validation",
        version="1.2.9",
        deployment_type="long_running",
        invocation_url="https://agents.internal/insight-validation/invoke",
        reasoners=(
            {
                "id": "Insight Validation Hub",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
            {
                "id": "Confidence Calibration Board",
                "input_schema": {},
                "output_schema": {},
                "memory_config": {
                    "auto_inject": [],
                    "memory_retention": "session",
                    "cache_results": False,
                },
            },
        ),
        skills=(
            {"id": "source_corrobortion", "input_schema": {}, "tags": ["validation"]},
            {"id": "confidence_estimation", "input_schema": {}, "tags": ["estimation"]},
        ),
        communication={"webhook": "https://callbacks.internal/insight-validation"},
        features={"supports_panel_reviews": True},
        metadata={"owner": "InsightOps"},
    ),
)

STATUS_WEIGHTS = (("succeeded", 1.0),)

REGIONS = (
    "EU robotics",
    "LATAM fintech",
    "APAC climate tech",
    "MENA mobility",
    "US defense tech",
    "Nordic energy storage",
    "India healthtech",
    "Sub-Saharan agri-tech",
)

OBJECTIVES = (
    "Map competitive pivots",
    "Surface latent demand",
    "Track supplier pressure",
    "Quantify regulatory heat",
    "Trace capital allocations",
    "Forecast leadership churn",
    "Score narrative momentum",
    "Detect stealth partnerships",
)

DELIVERABLES = (
    "executive_brief",
    "risk_dossier",
    "market_signal_digest",
    "threat_model",
    "investment_heatmap",
    "cx_opportunity_brief",
)


# --- Helpers --------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the AgentField SQLite workflow tables with realistic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Tips:
              • Use --nodes-per-workflow 10000 to mirror dense research DAGs.
              • Combine --workflow-count with a smaller node count to simulate breadth.
              • Re-run with the same --seed to get reproducible structures.
            """
        ),
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--workflow-count",
        type=int,
        default=1,
        help="Number of distinct root workflows to generate.",
    )
    parser.add_argument(
        "--nodes-per-workflow",
        type=int,
        default=50,
        help="Number of execution nodes per workflow (excluding the root).",
    )
    parser.add_argument(
        "--workflow-prefix",
        type=str,
        default="wf_deep_context_probe",
        help="Prefix used when minting workflow IDs.",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="session_deep_research",
        help="Prefix for session identifiers.",
    )
    parser.add_argument(
        "--actor-pool",
        type=str,
        nargs="+",
        default=["research_lead_marissa", "intel_director_akeem", "strategy_gm_li"],
        help="Actor IDs to sample from for ownership.",
    )
    parser.add_argument(
        "--team-id",
        type=str,
        default="intel_ops_enterprise",
        help="Team ID used for registering agent nodes.",
    )
    parser.add_argument(
        "--start-hours-ago",
        type=int,
        default=18,
        help="Anchor time (in hours before now) for the first workflow start.",
    )
    parser.add_argument(
        "--stagger-minutes",
        type=int,
        default=42,
        help="Minutes between successive workflow kickoffs.",
    )
    parser.add_argument(
        "--purge-prefix",
        action="store_true",
        help="Delete existing workflows whose IDs start with the workflow prefix before seeding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the workflow plan but skip database writes.",
    )
    return parser.parse_args(argv)


def ensure_agent_nodes(conn: sqlite3.Connection, team_id: str) -> None:
    """Upsert the canned agent nodes so UI views have metadata."""
    now_iso = isoformat(datetime.utcnow())
    for node in AGENT_NODE_POOL:
        payload = (
            node.node_id,
            team_id,
            node.base_url,
            node.version,
            node.deployment_type,
            node.invocation_url,
            json.dumps(node.reasoners).encode(),
            json.dumps(node.skills).encode(),
            json.dumps(node.communication).encode(),
            node.health_status,
            node.lifecycle_status,
            now_iso,
            now_iso,
            json.dumps(node.features or {}).encode(),
            json.dumps(node.metadata or {}).encode(),
        )
        conn.execute("DELETE FROM agent_nodes WHERE id = ?", (node.node_id,))
        conn.execute(
            """
            INSERT INTO agent_nodes (
                id, team_id, base_url, version, deployment_type, invocation_url,
                reasoners, skills, communication_config, health_status, lifecycle_status,
                last_heartbeat, registered_at, features, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def purge_workflows_with_prefix(conn: sqlite3.Connection, prefix: str) -> int:
    """Remove existing data for workflows whose IDs share the prefix."""
    pattern = f"{prefix}%"
    workflow_ids = [
        row[0]
        for row in conn.execute(
            "SELECT workflow_id FROM workflows WHERE workflow_id LIKE ?", (pattern,)
        )
    ]
    removed = 0
    for workflow_id in workflow_ids:
        run_ids = [
            row[0]
            for row in conn.execute(
                "SELECT run_id FROM workflow_runs WHERE root_workflow_id = ?",
                (workflow_id,),
            )
        ]
        for run_id in run_ids:
            conn.execute("DELETE FROM executions WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM workflow_steps WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM workflow_run_events WHERE run_id = ?", (run_id,))
            conn.execute(
                "DELETE FROM workflow_execution_events WHERE run_id = ?", (run_id,)
            )
        conn.execute(
            "DELETE FROM workflow_execution_events WHERE workflow_id = ?",
            (workflow_id,),
        )
        conn.execute(
            "DELETE FROM workflow_executions WHERE workflow_id = ?", (workflow_id,)
        )
        conn.execute(
            "DELETE FROM workflow_runs WHERE root_workflow_id = ?", (workflow_id,)
        )
        conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
        removed += 1
    return removed


def choose_weighted(options: Sequence[Tuple[str, float]]) -> str:
    labels, weights = zip(*options)
    return random.choices(labels, weights=weights, k=1)[0]


def isoformat(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def generate_scenario(
    *,
    prefix: str,
    session_prefix: str,
    actor_pool: Sequence[str],
    idx: int,
    started_at: datetime,
    nodes_per_workflow: int,
) -> WorkflowScenario:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    root_exec = f"exec_root_{uuid.uuid4().hex[:12]}"
    root_req = f"af_req_{uuid.uuid4().hex[:12]}"
    suffix = started_at.strftime("%Y%m%d%H%M%S")
    workflow_id = f"{prefix}_{suffix}_{idx:03d}"

    region = random.choice(REGIONS)
    objective = random.choice(OBJECTIVES)
    workflow_name = f"{objective} – {region}"
    tags = [
        "deep_research",
        "market_intelligence",
        region.replace(" ", "_").lower(),
        objective.replace(" ", "_").lower(),
    ]
    session_id = f"{session_prefix}_{region.split()[0].lower()}_{idx:02d}"
    actor_id = random.choice(actor_pool)
    duration = timedelta(minutes=18 + int(nodes_per_workflow / 40))
    return WorkflowScenario(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        workflow_tags=tags,
        session_id=session_id,
        actor_id=actor_id,
        run_id=run_id,
        root_execution_id=root_exec,
        root_request_id=root_req,
        started_at=started_at,
        completed_at=started_at + duration,
    )


def synthesize_nodes(
    scenario: WorkflowScenario, *, nodes_per_workflow: int
) -> List[NodeRecord]:
    nodes: List[NodeRecord] = []

    root_input = {
        "objective": scenario.workflow_name,
        "initial_sources": random.sample(
            ["eu-parliament", "ifri", "techcrunch-eu", "g7-briefs", "cb-insights"], k=3
        ),
        "run_context": "executive-intelligence-brief",
    }
    root_output = {
        "summary": "Initial orchestration launched with validated scope.",
        "confidence": round(random.uniform(0.82, 0.95), 2),
    }
    root_notes = [
        {
            "author": scenario.actor_id,
            "note": "Scope locked with intelligence leadership.",
            "timestamp": isoformat(scenario.started_at),
        }
    ]

    root_end = scenario.started_at + timedelta(minutes=6)
    root_duration_ms = int((root_end - scenario.started_at).total_seconds() * 1000)
    nodes.append(
        NodeRecord(
            execution_id=scenario.root_execution_id,
            request_id=scenario.root_request_id,
            parent_execution_id=None,
            depth=0,
            agent_node_id="atlas_scope_orchestrator",
            reasoner_id=AGENT_NODE_POOL[0].reasoners[0]["id"],
            status="succeeded",
            status_reason=None,
            error_message=None,
            started_at=scenario.started_at,
            completed_at=root_end,
            duration_ms=root_duration_ms,
            input_payload=root_input,
            output_payload=root_output,
            notes=root_notes,
        )
    )

    for i in range(1, nodes_per_workflow + 1):
        parent_node = random.choice(nodes)
        status = choose_weighted(STATUS_WEIGHTS)
        status_reason = None
        error_message = None

        start_offset = random.randint(90, 480)
        node_start = parent_node.started_at + timedelta(seconds=start_offset)
        node_duration = random.randint(120, 1500)
        node_end = node_start + timedelta(seconds=node_duration)

        agent_choice = random.choice(AGENT_NODE_POOL)
        reasoner_choice = random.choice(agent_choice.reasoners)

        notes: List[dict] = []
        notes.append(
            {
                "author": random.choice(
                    ["analyst_jonah", "analyst_li", "analyst_manuela", "analyst_mira"]
                ),
                "note": "Reviewed output and advanced to synthesis track.",
                "timestamp": isoformat(node_end),
            }
        )

        nodes.append(
            NodeRecord(
                execution_id=f"exec_{i:05d}_{uuid.uuid4().hex[:10]}",
                request_id=f"af_req_{uuid.uuid4().hex[:12]}",
                parent_execution_id=parent_node.execution_id,
                depth=parent_node.depth + 1,
                agent_node_id=agent_choice.node_id,
                reasoner_id=reasoner_choice["id"],
                status=status,
                status_reason=status_reason,
                error_message=error_message,
                started_at=node_start,
                completed_at=node_end,
                duration_ms=node_duration * 1000,
                input_payload={
                    "focus": random.choice(
                        [
                            "supply_chain",
                            "policy_shift",
                            "funding_rounds",
                            "sentiment_trace",
                            "talent_flows",
                        ]
                    ),
                    "signals_processed": random.randint(12, 120),
                    "parent_execution": parent_node.execution_id,
                },
                output_payload={
                    "key_findings": random.randint(3, 8),
                    "priority_score": round(random.uniform(0.32, 0.98), 2),
                    "insight_hash": uuid.uuid4().hex[:16],
                },
                notes=notes,
            )
        )

    return nodes


def insert_workflow(
    conn: sqlite3.Connection, scenario: WorkflowScenario, nodes: Sequence[NodeRecord]
) -> None:
    total_executions = len(nodes)
    success_count = sum(1 for n in nodes if n.status == "succeeded")
    failed_count = sum(1 for n in nodes if n.status != "succeeded")
    max_depth = max(n.depth for n in nodes)
    workflow_status = "succeeded" if failed_count == 0 else "failed"
    total_duration_ms = int(
        (
            max(n.completed_at for n in nodes) - min(n.started_at for n in nodes)
        ).total_seconds()
        * 1000
    )
    tags_json = json.dumps(scenario.workflow_tags)

    conn.execute(
        """
        INSERT INTO workflows (
            workflow_id, workflow_name, workflow_tags, session_id, actor_id,
            parent_workflow_id, parent_execution_id, root_workflow_id,
            workflow_depth, total_executions, successful_executions, failed_executions,
            total_duration_ms, status, started_at, completed_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scenario.workflow_id,
            scenario.workflow_name,
            tags_json,
            scenario.session_id,
            scenario.actor_id,
            None,
            None,
            scenario.workflow_id,
            max_depth,
            total_executions,
            success_count,
            failed_count,
            total_duration_ms,
            workflow_status,
            isoformat(min(n.started_at for n in nodes)),
            isoformat(max(n.completed_at for n in nodes)),
            isoformat(scenario.started_at),
            isoformat(datetime.utcnow()),
        ),
    )

    run_metadata = {
        "objective": scenario.workflow_name,
        "deliverable": random.choice(DELIVERABLES),
        "regions": list({scenario.workflow_tags[2]}),
    }

    conn.execute(
        """
        INSERT INTO workflow_runs (
            run_id, root_workflow_id, root_execution_id, status, total_steps,
            completed_steps, failed_steps, state_version, last_event_sequence,
            metadata, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scenario.run_id,
            scenario.workflow_id,
            scenario.root_execution_id,
            workflow_status,
            total_executions,
            success_count,
            failed_count,
            3,
            3,
            json.dumps(run_metadata).encode(),
            isoformat(scenario.started_at),
            isoformat(datetime.utcnow()),
            isoformat(max(n.completed_at for n in nodes)),
        ),
    )

    execution_rows = []
    event_rows = []
    step_rows = []
    step_lookup = {}
    execution_rows_simple = []

    for node in nodes:
        notes_payload = json.dumps(node.notes).encode()
        execution_rows.append(
            (
                scenario.workflow_id,
                node.execution_id,
                node.request_id,
                scenario.run_id,
                scenario.session_id,
                scenario.actor_id,
                node.agent_node_id,
                scenario.workflow_id,
                node.parent_execution_id,
                scenario.workflow_id,
                node.depth,
                node.reasoner_id,
                json.dumps(node.input_payload).encode(),
                json.dumps(node.output_payload).encode(),
                len(json.dumps(node.input_payload)),
                len(json.dumps(node.output_payload)),
                scenario.workflow_name,
                tags_json,
                node.status,
                isoformat(node.started_at),
                isoformat(node.completed_at),
                node.duration_ms,
                1,
                2,
                0,
                0,
                None,
                node.status_reason,
                None,
                None,
                node.error_message,
                0,
                notes_payload,
                isoformat(node.started_at),
                isoformat(node.completed_at),
            )
        )

        event_rows.extend(
            [
                (
                    node.execution_id,
                    scenario.workflow_id,
                    scenario.run_id,
                    node.parent_execution_id,
                    1,
                    0,
                    "status_changed",
                    "running",
                    None,
                    json.dumps({"detail": "Execution started"}),
                    isoformat(node.started_at),
                    isoformat(node.started_at),
                ),
                (
                    node.execution_id,
                    scenario.workflow_id,
                    scenario.run_id,
                    node.parent_execution_id,
                    2,
                    1,
                    "status_changed",
                    node.status,
                    node.status_reason,
                    json.dumps(
                        {
                            "detail": "Execution completed"
                            if node.status != "failed"
                            else "Execution failed"
                        }
                    ),
                    isoformat(node.completed_at),
                    isoformat(node.completed_at),
                ),
            ]
        )

        step_id = f"step_{uuid.uuid4().hex[:14]}"
        step_lookup[node.execution_id] = step_id
        parent_step_id = (
            step_lookup.get(node.parent_execution_id)
            if node.parent_execution_id
            else None
        )
        step_rows.append(
            (
                step_id,
                scenario.run_id,
                parent_step_id,
                node.execution_id,
                node.agent_node_id,
                f"agent://{node.agent_node_id}",
                "succeeded" if node.status.startswith("succeeded") else node.status,
                1,
                random.randint(0, 4),
                isoformat(node.started_at),
                None,
                None,
                node.error_message,
                b"{}",
                isoformat(node.started_at),
                isoformat(node.completed_at),
                None,
                None,
                isoformat(node.started_at),
                isoformat(node.completed_at),
            )
        )

        execution_rows_simple.append(
            (
                node.execution_id,
                scenario.run_id,
                node.parent_execution_id,
                node.agent_node_id,
                node.reasoner_id,
                node.agent_node_id,
                node.status,
                json.dumps(node.input_payload).encode(),
                json.dumps(node.output_payload).encode(),
                node.error_message,
                None,
                None,
                scenario.session_id,
                scenario.actor_id,
                isoformat(node.started_at),
                isoformat(node.completed_at),
                node.duration_ms,
                isoformat(node.started_at),
                isoformat(node.completed_at),
            )
        )

    placeholders = ",".join("?" for _ in execution_rows[0])
    conn.executemany(
        f"""
        INSERT INTO workflow_executions (
            workflow_id, execution_id, agentfield_request_id, run_id, session_id,
            actor_id, agent_node_id, parent_workflow_id, parent_execution_id,
            root_workflow_id, workflow_depth, reasoner_id, input_data, output_data,
            input_size, output_size, workflow_name, workflow_tags, status,
            started_at, completed_at, duration_ms, state_version, last_event_sequence,
            active_children, pending_children, pending_terminal_status, status_reason,
            lease_owner, lease_expires_at, error_message, retry_count, notes, created_at,
            updated_at
        ) VALUES ({placeholders})
        """,
        execution_rows,
    )

    conn.executemany(
        """
        INSERT INTO workflow_execution_events (
            execution_id, workflow_id, run_id, parent_execution_id, sequence,
            previous_sequence, event_type, status, status_reason, payload,
            emitted_at, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        event_rows,
    )

    conn.executemany(
        """
        INSERT INTO workflow_steps (
            step_id, run_id, parent_step_id, execution_id, agent_node_id, target,
            status, attempt, priority, not_before, input_uri, result_uri, error_message,
            metadata, started_at, completed_at, leased_at, lease_timeout, created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        step_rows,
    )

    run_events = [
        (
            scenario.run_id,
            1,
            0,
            "run_started",
            "running",
            None,
            json.dumps({"kickoff": "approved"}),
            isoformat(scenario.started_at),
            isoformat(scenario.started_at),
        ),
        (
            scenario.run_id,
            2,
            1,
            "run_checkpoint",
            "running",
            None,
            json.dumps({"progress": 0.55}),
            isoformat(scenario.started_at + timedelta(minutes=9)),
            isoformat(scenario.started_at + timedelta(minutes=9)),
        ),
        (
            scenario.run_id,
            3,
            2,
            "run_completed",
            workflow_status,
            None,
            json.dumps({"deliverable": random.choice(DELIVERABLES)}),
            isoformat(max(n.completed_at for n in nodes)),
            isoformat(max(n.completed_at for n in nodes)),
        ),
    ]
    conn.executemany(
        """
        INSERT INTO workflow_run_events (
            run_id, sequence, previous_sequence, event_type, status,
            status_reason, payload, emitted_at, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        run_events,
    )

    conn.executemany(
        """
        INSERT INTO executions (
            execution_id, run_id, parent_execution_id, agent_node_id, reasoner_id,
            node_id, status, input_payload, result_payload, error_message,
            input_uri, result_uri, session_id, actor_id, started_at, completed_at,
            duration_ms, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        execution_rows_simple,
    )


def seed_database(args: argparse.Namespace) -> List[Tuple[str, str, int]]:
    if args.seed is not None:
        random.seed(args.seed)

    db_path = args.db_path.expanduser()
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. Start the control plane once to create it."
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")

        ensure_agent_nodes(conn, args.team_id)
        if args.purge_prefix:
            purge_workflows_with_prefix(conn, args.workflow_prefix)

        inserted: List[Tuple[str, str, int]] = []
        base_start = datetime.utcnow() - timedelta(hours=args.start_hours_ago)

        for wf_index in range(args.workflow_count):
            scenario = generate_scenario(
                prefix=args.workflow_prefix,
                session_prefix=args.session_prefix,
                actor_pool=args.actor_pool,
                idx=wf_index,
                started_at=base_start
                + timedelta(minutes=wf_index * args.stagger_minutes),
                nodes_per_workflow=args.nodes_per_workflow,
            )
            nodes = synthesize_nodes(
                scenario, nodes_per_workflow=args.nodes_per_workflow
            )
            insert_workflow(conn, scenario, nodes)
            inserted.append((scenario.workflow_id, scenario.run_id, len(nodes)))

        conn.commit()

    return inserted


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.nodes_per_workflow <= 0:
        print("nodes-per-workflow must be positive.", file=sys.stderr)
        return 1
    if args.workflow_count <= 0:
        print("workflow-count must be positive.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("Dry-run mode currently not implemented. Remove --dry-run to write data.")
        return 0

    try:
        inserted = seed_database(args)
    except Exception as exc:
        print(f"Failed to seed database: {exc}", file=sys.stderr)
        return 2

    print(f"Seeded {len(inserted)} workflow(s) into {args.db_path.expanduser()}")
    for workflow_id, run_id, total_nodes in inserted[:10]:
        print(f"  • {workflow_id} / {run_id} ({total_nodes} executions incl. root)")
    if len(inserted) > 10:
        print("  • ...")

    if inserted:
        sample_workflow_id, sample_run_id, _ = inserted[0]
        print("\nValidate with:")
        print(
            "  curl 'http://localhost:8080/api/ui/v1/executions/enhanced?sort_by=started_at&sort_order=desc&page=1&limit=5'"
        )
        print(
            f"  curl 'http://localhost:8080/api/ui/v1/workflows/{sample_workflow_id}/dag'"
        )
        print(f"  curl 'http://localhost:8080/api/ui/v2/workflow-runs/{sample_run_id}'")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
