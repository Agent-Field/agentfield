#!/usr/bin/env python3
"""Standalone debug script: test harness with complex JSON schemas.

Usage (from worktree sdk/python/ dir):
    .venv/bin/python tests/debug_complex_json.py --provider claude-code
    .venv/bin/python tests/debug_complex_json.py --provider codex
    .venv/bin/python tests/debug_complex_json.py --provider claude-code --test simple
    .venv/bin/python tests/debug_complex_json.py --provider claude-code --test complex
    .venv/bin/python tests/debug_complex_json.py --provider claude-code --test deeply_nested

No control plane needed. Exercises the harness directly via HarnessRunner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────
# SCHEMAS — escalating complexity
# ────────────────────────────────────────────────────────────────────


# Level 1: Simple (baseline — should always work)
class SimpleResult(BaseModel):
    greeting: str
    number: int


# Level 2: Medium — list fields, optional, constrained types
class Finding(BaseModel):
    title: str
    severity: str = Field(description="One of: low, medium, high, critical")
    description: str
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None


class MediumResult(BaseModel):
    summary: str
    overall_score: int = Field(ge=1, le=10)
    findings: List[Finding]
    tags: List[str]
    reviewed_file: str


# Level 3: Complex — deeply nested objects, enums, dicts, arrays of arrays
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CodeLocation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    column: Optional[int] = None


class TestCase(BaseModel):
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    is_edge_case: bool = False


class SecurityVulnerability(BaseModel):
    cve_id: Optional[str] = None
    title: str
    description: str
    priority: Priority
    location: CodeLocation
    affected_versions: List[str]
    remediation_steps: List[str]
    references: List[str] = Field(default_factory=list)


class PerformanceIssue(BaseModel):
    title: str
    description: str
    location: CodeLocation
    impact_description: str
    estimated_improvement: str
    priority: Priority


class Dependency(BaseModel):
    name: str
    version: str
    is_outdated: bool
    latest_version: Optional[str] = None
    has_known_vulnerabilities: bool = False
    vulnerability_count: int = 0


class ArchitectureRecommendation(BaseModel):
    area: str
    current_state: str
    recommended_change: str
    effort_estimate: str = Field(description="One of: small, medium, large, xlarge")
    impact: str
    dependencies: List[str] = Field(default_factory=list)


class ComplexResult(BaseModel):
    """Full code audit report — deeply nested, many fields."""

    project_name: str
    analysis_timestamp: str
    overall_health_score: int = Field(ge=0, le=100)
    summary: str
    detailed_description: str

    security_vulnerabilities: List[SecurityVulnerability]
    performance_issues: List[PerformanceIssue]
    dependencies: List[Dependency]
    architecture_recommendations: List[ArchitectureRecommendation]

    test_coverage_percent: float = Field(ge=0.0, le=100.0)
    suggested_test_cases: List[TestCase]

    metadata: Dict[str, Any] = Field(
        description="Arbitrary key-value metadata about the analysis"
    )
    tags: List[str]


# Level 4: Deeply nested — recursive-like structures
class TreeNode(BaseModel):
    name: str
    node_type: str = Field(
        description="One of: file, directory, module, class, function"
    )
    size_bytes: Optional[int] = None
    children: List["TreeNode"] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DependencyGraph(BaseModel):
    nodes: List[str]
    edges: List[Dict[str, str]] = Field(
        description="List of {source, target, relationship_type}"
    )


class DeeplyNestedResult(BaseModel):
    project_name: str
    file_tree: TreeNode
    dependency_graph: DependencyGraph
    module_summaries: Dict[str, str]
    cross_module_issues: List[Dict[str, Any]]
    overall_assessment: str


# Level 5: MASSIVE schema — >4K tokens, triggers file-based schema path
class HttpHeader(BaseModel):
    name: str
    value: str
    is_sensitive: bool = False
    description: Optional[str] = None


class ApiEndpoint(BaseModel):
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE, PATCH")
    path: str
    summary: str
    description: str
    request_headers: List[HttpHeader] = Field(default_factory=list)
    request_body_schema: Optional[Dict[str, Any]] = None
    response_status_codes: List[int]
    response_body_schema: Optional[Dict[str, Any]] = None
    query_parameters: List[Dict[str, Any]] = Field(default_factory=list)
    path_parameters: List[str] = Field(default_factory=list)
    authentication_required: bool = True
    rate_limit_per_minute: Optional[int] = None
    deprecated: bool = False
    deprecation_notice: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    example_request: Optional[Dict[str, Any]] = None
    example_response: Optional[Dict[str, Any]] = None


class DatabaseTable(BaseModel):
    name: str
    schema_name: str = "public"
    description: str
    columns: List[Dict[str, Any]] = Field(
        description="List of {name, type, nullable, default, description, constraints}"
    )
    primary_key: List[str]
    foreign_keys: List[Dict[str, str]] = Field(
        description="List of {column, references_table, references_column, on_delete}"
    )
    indexes: List[Dict[str, Any]] = Field(
        description="List of {name, columns, unique, type}"
    )
    row_count_estimate: Optional[int] = None
    size_bytes_estimate: Optional[int] = None


class ServiceHealth(BaseModel):
    service_name: str
    status: str = Field(description="One of: healthy, degraded, unhealthy, unknown")
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate_percent: float
    uptime_percent: float
    last_deployment: str
    version: str
    resource_usage: Dict[str, float] = Field(
        description="Keys: cpu_percent, memory_percent, disk_percent"
    )
    dependencies: List[Dict[str, str]] = Field(
        description="List of {name, status, latency_ms}"
    )


class DeploymentConfig(BaseModel):
    environment: str = Field(description="One of: development, staging, production")
    region: str
    replicas: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    auto_scaling: Dict[str, Any] = Field(
        description="Keys: enabled, min_replicas, max_replicas, target_cpu_percent"
    )
    environment_variables: List[Dict[str, str]] = Field(
        description="List of {name, value_source, description}"
    )
    secrets: List[str]
    volumes: List[Dict[str, str]] = Field(default_factory=list)
    health_check: Dict[str, Any] = Field(
        description="Keys: path, port, initial_delay_seconds, period_seconds"
    )


class TeamMember(BaseModel):
    name: str
    role: str
    email: str
    github_username: Optional[str] = None
    responsibilities: List[str]
    availability: str = Field(description="One of: full-time, part-time, on-call")


class IncidentReport(BaseModel):
    incident_id: str
    title: str
    severity: str = Field(description="One of: sev1, sev2, sev3, sev4")
    status: str = Field(description="One of: open, investigating, mitigated, resolved")
    started_at: str
    resolved_at: Optional[str] = None
    root_cause: Optional[str] = None
    affected_services: List[str]
    timeline: List[Dict[str, str]] = Field(
        description="List of {timestamp, action, actor}"
    )
    action_items: List[Dict[str, str]] = Field(
        description="List of {description, assignee, due_date, status}"
    )


class MassiveResult(BaseModel):
    """Full system documentation — designed to exceed 4K token threshold."""

    project_name: str
    version: str
    description: str
    architecture_overview: str

    api_endpoints: List[ApiEndpoint]
    database_tables: List[DatabaseTable]
    service_health: List[ServiceHealth]
    deployment_configs: List[DeploymentConfig]
    team_members: List[TeamMember]
    recent_incidents: List[IncidentReport]

    technical_debt_items: List[Dict[str, Any]]
    roadmap_items: List[Dict[str, str]]
    metrics_dashboard_url: Optional[str] = None
    documentation_urls: Dict[str, str]
    change_log: List[Dict[str, str]] = Field(
        description="List of {version, date, changes}"
    )
    tags: List[str]


# ────────────────────────────────────────────────────────────────────
# TEST PROMPTS
# ────────────────────────────────────────────────────────────────────

PROMPTS = {
    "simple": (
        SimpleResult,
        'Return exactly: greeting="Hello from harness" and number=42. '
        "Follow the OUTPUT REQUIREMENTS below precisely.",
    ),
    "medium": (
        MediumResult,
        "You are reviewing a Python file called `app.py` that contains a simple Flask web server "
        "with 3 routes: / (home), /api/users (returns user list), /api/health (health check). "
        "The code has some issues: no input validation, SQL injection risk in user query, "
        "no rate limiting, hardcoded database password. "
        "Generate a code review with summary, score 1-10, findings (at least 3), "
        "tags (at least 2), and the reviewed file name. "
        "Follow the OUTPUT REQUIREMENTS below precisely.",
    ),
    "complex": (
        ComplexResult,
        "You are performing a comprehensive code audit of a Python project called 'PaymentGateway'. "
        "The project is a payment processing microservice using FastAPI, SQLAlchemy, and Redis. "
        "Generate a COMPLETE audit report with ALL of the following:\n"
        "- project_name: 'PaymentGateway'\n"
        "- analysis_timestamp: current ISO timestamp\n"
        "- overall_health_score: 0-100\n"
        "- summary: 2-3 sentence overview\n"
        "- detailed_description: paragraph about the project state\n"
        "- security_vulnerabilities: at least 2 items, each with cve_id (optional), title, "
        "description, priority (low/medium/high/critical), location (file_path, start_line, end_line), "
        "affected_versions list, remediation_steps list, references list\n"
        "- performance_issues: at least 2 items with location, impact, estimated improvement\n"
        "- dependencies: at least 3 items with version info and vulnerability status\n"
        "- architecture_recommendations: at least 2 items with effort estimate and dependencies\n"
        "- test_coverage_percent: float 0-100\n"
        "- suggested_test_cases: at least 2 items with input_data dict and expected_output dict\n"
        "- metadata: dict with at least 3 keys (analyzer_version, language, framework)\n"
        "- tags: list of at least 3 strings\n\n"
        "Make the data realistic and detailed. Follow the OUTPUT REQUIREMENTS below precisely.",
    ),
    "massive": (
        MassiveResult,
        "You are generating comprehensive system documentation for a project called 'OrderService'. "
        "It is a microservice handling e-commerce orders, built with FastAPI + PostgreSQL + Redis.\n\n"
        "Generate ALL required data:\n"
        "- project_name: 'OrderService', version: '2.4.1'\n"
        "- description: paragraph about the service\n"
        "- architecture_overview: paragraph about architecture\n"
        "- api_endpoints: at least 3 endpoints, each with method, path, summary, description, "
        "request_headers (at least 1), response_status_codes, tags. Include example_request and "
        "example_response for at least one endpoint.\n"
        "- database_tables: at least 2 tables with columns (at least 3 per table), primary_key, "
        "foreign_keys, indexes\n"
        "- service_health: at least 2 services with latency stats, error rates, resource_usage dict, "
        "dependencies list\n"
        "- deployment_configs: at least 1 config with all fields including auto_scaling dict, "
        "environment_variables, secrets, health_check dict\n"
        "- team_members: at least 2 members with roles and responsibilities\n"
        "- recent_incidents: at least 1 incident with timeline entries and action_items\n"
        "- technical_debt_items: at least 2 items as dicts\n"
        "- roadmap_items: at least 2 items as dicts with keys: title, description, target_date\n"
        "- documentation_urls: dict with at least 3 key-value pairs\n"
        "- change_log: at least 2 entries with version, date, changes\n"
        "- tags: at least 4 strings\n\n"
        "Make all data realistic and detailed. Follow the OUTPUT REQUIREMENTS below precisely.",
    ),
    "deeply_nested": (
        DeeplyNestedResult,
        "You are analyzing a Python project called 'MicroserviceFramework' that has this structure:\n"
        "- src/ directory with 3 subdirectories: core/, api/, utils/\n"
        "- Each subdirectory has 2-3 Python files\n"
        "- core/ has: engine.py, config.py, events.py\n"
        "- api/ has: routes.py, middleware.py\n"
        "- utils/ has: helpers.py, validators.py\n\n"
        "Generate:\n"
        "- project_name: 'MicroserviceFramework'\n"
        "- file_tree: a TreeNode representing the full project structure with nested children "
        "(root -> src -> core/api/utils -> individual files). Each node needs name, node_type "
        "(file/directory/module), optional size_bytes, children list, and metadata dict.\n"
        "- dependency_graph: nodes (list of module names), edges (list of {source, target, relationship_type})\n"
        "- module_summaries: dict mapping module names to summary strings (at least 5 entries)\n"
        "- cross_module_issues: list of at least 2 dicts describing cross-cutting concerns\n"
        "- overall_assessment: paragraph summary\n\n"
        "Make the tree at least 3 levels deep with realistic data. "
        "Follow the OUTPUT REQUIREMENTS below precisely.",
    ),
}


# ────────────────────────────────────────────────────────────────────
# DEBUG RUNNER
# ────────────────────────────────────────────────────────────────────


def setup_work_dir() -> str:
    """Create a temp directory with a git repo (required by Codex)."""
    tmpdir = tempfile.mkdtemp(prefix="af_json_debug_")
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    dummy = os.path.join(tmpdir, ".gitkeep")
    with open(dummy, "w") as f:
        f.write("")
    subprocess.run(["git", "-C", tmpdir, "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", tmpdir, "commit", "-m", "init", "--allow-empty"],
        capture_output=True,
        check=True,
    )
    return tmpdir


async def run_single_test(
    provider: str,
    test_name: str,
    schema_cls: type,
    prompt: str,
    work_dir: str,
    verbose: bool = True,
) -> dict:
    """Run a single harness test and return diagnostic info."""
    from agentfield.harness._runner import HarnessRunner
    from agentfield.harness._schema import get_output_path, build_prompt_suffix

    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name} | PROVIDER: {provider}")
    print(f"{'=' * 80}")

    # Show the schema being tested
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    token_est = len(schema_json) // 4
    print(f"Schema tokens (est): {token_est}")
    print(f"Schema fields: {list(schema_cls.model_fields.keys())}")

    if verbose:
        print(
            f"\nFull JSON Schema:\n{schema_json[:2000]}{'...' if len(schema_json) > 2000 else ''}"
        )

    # Show the effective prompt suffix that will be added
    suffix = build_prompt_suffix(schema_cls, work_dir)
    print(f"\nPrompt suffix length: {len(suffix)} chars")
    if verbose:
        print(
            f"Prompt suffix preview:\n{suffix[:1000]}{'...' if len(suffix) > 1000 else ''}"
        )

    runner = HarnessRunner()

    diagnostics = {
        "test_name": test_name,
        "provider": provider,
        "schema_token_est": token_est,
        "success": False,
        "error": None,
        "parsed": None,
        "raw_result_preview": None,
        "output_file_exists": False,
        "output_file_content": None,
        "duration_ms": 0,
        "num_turns": 0,
        "cost_usd": None,
    }

    try:
        print("\n>>> Executing harness... (this may take 30-120s)")
        result = await runner.run(
            prompt,
            provider=provider,
            schema=schema_cls,
            cwd=work_dir,
            permission_mode="auto",
            max_retries=1,
        )

        diagnostics["duration_ms"] = result.duration_ms
        diagnostics["num_turns"] = result.num_turns
        diagnostics["cost_usd"] = result.cost_usd
        diagnostics["raw_result_preview"] = (result.result or "")[:500]

        # Check if the output file was created (before cleanup wipes it)
        # Note: cleanup happens in finally block of runner.run(), so file is already gone
        # But we can check from the result
        output_path = get_output_path(work_dir)
        diagnostics["output_file_exists"] = os.path.exists(output_path)
        if os.path.exists(output_path):
            with open(output_path) as f:
                diagnostics["output_file_content"] = f.read()[:2000]

        if result.is_error:
            diagnostics["error"] = result.error_message
            print(f"\n❌ FAILED: {result.error_message}")
            print(f"   Raw result: {(result.result or '')[:300]}")
            print(f"   Duration: {result.duration_ms}ms | Turns: {result.num_turns}")
            if result.cost_usd:
                print(f"   Cost: ${result.cost_usd:.4f}")

            # Dump messages for debugging
            if verbose and result.messages:
                print(f"\n   --- Message dump ({len(result.messages)} messages) ---")
                for i, msg in enumerate(result.messages[-5:]):  # Last 5 messages
                    msg_type = msg.get("type", "unknown")
                    content = str(
                        msg.get("content", msg.get("text", msg.get("result", "")))
                    )[:200]
                    print(f"   [{i}] type={msg_type}: {content}")
        else:
            diagnostics["success"] = True
            diagnostics["parsed"] = str(result.parsed)[:500] if result.parsed else None
            print("\n✅ SUCCESS!")
            print(f"   Parsed type: {type(result.parsed).__name__}")
            print(f"   Duration: {result.duration_ms}ms | Turns: {result.num_turns}")
            if result.cost_usd:
                print(f"   Cost: ${result.cost_usd:.4f}")
            if verbose and result.parsed:
                parsed_json = result.parsed.model_dump_json(indent=2)
                print(
                    f"\n   Parsed output:\n{parsed_json[:1000]}{'...' if len(parsed_json) > 1000 else ''}"
                )

    except Exception as e:
        diagnostics["error"] = f"Exception: {type(e).__name__}: {e}"
        print(f"\n💥 EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

    return diagnostics


async def run_manual_retry_test(
    provider: str,
    test_name: str,
    schema_cls: type,
    prompt: str,
    work_dir: str,
    max_retries: int = 2,
) -> dict:
    """Run harness with MANUAL retry on validation failure — simulates the fix we need.

    This demonstrates the behavior we want to implement in _runner.py:
    1. First attempt: normal harness call
    2. On validation failure: build follow-up prompt with error context
    3. Continue the conversation to fix the JSON
    """
    from agentfield.harness._schema import (
        build_prompt_suffix,
        get_output_path,
        parse_and_validate,
        build_followup_prompt,
        cleanup_temp_files,
    )
    from agentfield.harness.providers._factory import build_provider
    from agentfield.types import HarnessConfig

    print(f"\n{'=' * 80}")
    print(f"RETRY TEST: {test_name} | PROVIDER: {provider} | max_retries={max_retries}")
    print(f"{'=' * 80}")

    output_path = get_output_path(work_dir)
    effective_prompt = prompt + build_prompt_suffix(schema_cls, work_dir)

    # Build provider
    config = HarnessConfig(
        provider=provider,
        cwd=work_dir,
        permission_mode="auto",
    )
    provider_instance = build_provider(config)

    options: dict[str, object] = {
        "provider": provider,
        "cwd": work_dir,
        "permission_mode": "auto",
    }

    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")

        if attempt == 0:
            current_prompt = effective_prompt
        else:
            # Build follow-up prompt with error context
            error_msg = "JSON validation failed. "
            if not os.path.exists(output_path):
                error_msg += f"The output file {output_path} was NOT created. You MUST write valid JSON to this file."
            else:
                with open(output_path) as f:
                    content = f.read()
                if not content.strip():
                    error_msg += "The output file exists but is EMPTY."
                else:
                    try:
                        json.loads(content)
                        # JSON parses fine, but schema validation failed
                        error_msg += f"The JSON parses but does NOT match the required schema. File content:\n{content[:1000]}"
                    except json.JSONDecodeError as e:
                        error_msg += f"The file contains INVALID JSON. Parse error: {e}. File content:\n{content[:500]}"

            current_prompt = build_followup_prompt(error_msg, work_dir)
            print(f"   Follow-up prompt: {current_prompt[:300]}...")

        try:
            raw = await provider_instance.execute(current_prompt, options)
            print(
                f"   Provider returned: is_error={raw.is_error}, result_len={len(raw.result or '')}"
            )

            if raw.is_error:
                print(f"   Provider error: {raw.error_message}")
                continue

            # Check if file was written and try to validate
            validated = parse_and_validate(output_path, schema_cls)
            if validated is not None:
                print(f"\n✅ RETRY TEST SUCCESS on attempt {attempt + 1}!")
                print(f"   Parsed type: {type(validated).__name__}")
                parsed_json = validated.model_dump_json(indent=2)
                print(
                    f"   Output:\n{parsed_json[:1000]}{'...' if len(parsed_json) > 1000 else ''}"
                )
                cleanup_temp_files(work_dir)
                return {
                    "success": True,
                    "attempt": attempt + 1,
                    "parsed": str(validated)[:500],
                }

            # Validation failed — check what went wrong
            if os.path.exists(output_path):
                with open(output_path) as f:
                    content = f.read()
                print(f"   File exists ({len(content)} bytes) but validation failed")
                print(f"   Content preview: {content[:300]}...")
            else:
                print(f"   Output file NOT created at {output_path}")

        except Exception as e:
            print(f"   Exception: {type(e).__name__}: {e}")

    cleanup_temp_files(work_dir)
    print(f"\n❌ RETRY TEST FAILED after {max_retries + 1} attempts")
    return {
        "success": False,
        "attempt": max_retries + 1,
        "error": "All attempts failed",
    }


async def main():
    parser = argparse.ArgumentParser(description="Debug harness complex JSON handling")
    parser.add_argument(
        "--provider",
        default="claude-code",
        choices=["claude-code", "codex", "gemini", "opencode"],
        help="Harness provider to test",
    )
    parser.add_argument(
        "--test",
        default="all",
        choices=["simple", "medium", "complex", "deeply_nested", "massive", "all"],
        help="Which test to run",
    )
    parser.add_argument(
        "--retry-test",
        action="store_true",
        help="Also run the manual retry test for complex schemas",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (less output)",
    )
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    work_dir = setup_work_dir()
    print(f"Work directory: {work_dir}")

    tests_to_run = list(PROMPTS.keys()) if args.test == "all" else [args.test]
    all_results = []

    try:
        for test_name in tests_to_run:
            schema_cls, prompt = PROMPTS[test_name]

            # Create a subdirectory per test so output files don't clash
            test_dir = os.path.join(work_dir, test_name)
            os.makedirs(test_dir, exist_ok=True)

            result = await run_single_test(
                args.provider, test_name, schema_cls, prompt, test_dir, args.verbose
            )
            all_results.append(result)

            # Optionally run retry test for complex schemas
            if args.retry_test and test_name in ("complex", "deeply_nested"):
                retry_dir = os.path.join(work_dir, f"{test_name}_retry")
                os.makedirs(retry_dir, exist_ok=True)
                retry_result = await run_manual_retry_test(
                    args.provider,
                    test_name,
                    schema_cls,
                    prompt,
                    retry_dir,
                    max_retries=2,
                )
                all_results.append({"test_name": f"{test_name}_retry", **retry_result})

        # Summary
        print(f"\n\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        for r in all_results:
            status = "✅" if r.get("success") else "❌"
            name = r.get("test_name", "unknown")
            error = r.get("error", "")
            duration = r.get("duration_ms", 0)
            cost = r.get("cost_usd")
            cost_str = f" | ${cost:.4f}" if cost else ""
            print(
                f"  {status} {name}: {'PASS' if r.get('success') else 'FAIL'} "
                f"({duration}ms{cost_str})"
                f"{f' — {error[:80]}' if error else ''}"
            )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\nCleaned up: {work_dir}")


if __name__ == "__main__":
    asyncio.run(main())
