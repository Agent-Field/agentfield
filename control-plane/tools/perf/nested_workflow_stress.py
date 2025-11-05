#!/usr/bin/env python3
"""Lightweight load generator for Haxen durable execution flows.

This script exercises the /execute and /execute/async gateways with configurable
concurrency, nested payload parameters, and adaptive polling. It records latency
statistics, HTTP status distribution, and final execution states so you can spot
backpressure, retry storms, or payload issues under load.

Examples
========
Sync path with modest concurrency::

    python nested_workflow_stress.py \
        --base-url http://localhost:8080 \
        --target demo-agent.synthetic_nested \
        --mode sync \
        --requests 200 \
        --concurrency 16 \
        --depth 4 --width 3

Async path with bigger payloads and custom headers::

    python nested_workflow_stress.py \
        --mode async \
        --base-url http://localhost:8080 \
        --target demo-agent.synthetic_nested \
        --requests 300 \
        --concurrency 32 \
        --payload-bytes 65536 \
        --header "X-Workflow-Tags=perf,stress"

If you already have a request envelope that drives nested behaviour, provide it
via --body-template. The file must contain JSON and can reference placeholders
``{seq}``, ``{depth}``, ``{width}``, and ``{payload}`` which are substituted
before sending each request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUCCESS_STATUSES = {"success", "succeeded", "completed"}
FAILURE_STATUSES = {"error", "failed", "timeout", "cancelled"}

DEFAULT_METRIC_KEYS = [
    "process_resident_memory_bytes",
    "go_memstats_heap_alloc_bytes",
    "go_goroutines",
    "haxen_gateway_queue_depth",
    "haxen_worker_inflight",
]

import httpx


def parse_header(header: str) -> Tuple[str, str]:
    if ":" in header:
        key, value = header.split(":", 1)
    elif "=" in header:
        key, value = header.split("=", 1)
    else:
        raise argparse.ArgumentTypeError(
            f"invalid header '{header}', expected KEY:VALUE or KEY=VALUE",
        )
    return key.strip(), value.strip()


def load_template(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    text = path.read_text()
    # Basic validation for JSON-ness after formatting with empty payload
    try:
        json.loads(text.format(seq=0, depth=0, width=0, payload=""))
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Template at {path} is not valid JSON after formatting: {exc}") from exc
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haxen durable execution load tester")
    parser.add_argument("--base-url", default=os.getenv("HAXEN_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--target", required=True, help="Target in node.reasoner form")
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=1024, help="Size of filler payload inside input.payload")
    parser.add_argument("--depth", type=int, default=0, help="Hint for nested reasoners (added to payload)")
    parser.add_argument("--width", type=int, default=0, help="Hint for nested reasoners (added to payload)")
    parser.add_argument("--body-template", type=Path, help="Path to JSON template for request body")
    parser.add_argument("--header", dest="headers", action="append", help="Extra header KEY:VALUE", default=[])
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per HTTP request timeout in seconds")
    parser.add_argument("--execution-timeout", type=float, default=600.0, help="Max time to wait for async completion")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="Initial poll interval for async mode")
    parser.add_argument("--max-poll-interval", type=float, default=5.0, help="Max poll interval for async mode")
    parser.add_argument("--backoff-multiplier", type=float, default=1.7, help="Exponential backoff multiplier")
    parser.add_argument("--jitter", type=float, default=0.2, help="Random jitter applied to poll sleeps (0-1)")
    parser.add_argument("--save-metrics", type=Path, help="Optional path to dump metrics JSON")
    parser.add_argument("--print-failures", action="store_true", help="Log failing responses for inspection")
    parser.add_argument("--async-status-endpoint", default="/api/v1/executions", help="Status endpoint base path")
    parser.add_argument("--async-submit-prefix", default="/api/v1/execute/async", help="Async submission prefix")
    parser.add_argument("--sync-prefix", default="/api/v1/execute", help="Sync submission prefix")
    parser.add_argument("--verify-ssl", action="store_true", help="Verify TLS certificates (default off)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--scenario-file", type=Path, help="JSON file describing one or more load scenarios")
    parser.add_argument("--metrics-url", type=str, help="Prometheus metrics endpoint to sample before/after runs")
    parser.add_argument("--metrics", dest="metrics", action="append", default=[], help="Prometheus metric names to capture")
    parser.add_argument("--metrics-timeout", type=float, default=5.0, help="Timeout (seconds) for metrics scraping")
    return parser


class Metrics:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.latencies: List[float] = []
        self.status_counts: Counter[str] = Counter()
        self.http_codes: Counter[int] = Counter()
        self.exceptions: Counter[str] = Counter()
        self.total = 0

    def record(self, *, latency: Optional[float], status: Optional[str], http_code: Optional[int], exc: Optional[BaseException]) -> None:
        self.total += 1
        if latency is not None and math.isfinite(latency):
            self.latencies.append(latency)
        if status:
            self.status_counts[status] += 1
        if http_code is not None:
            self.http_codes[http_code] += 1
        if exc is not None:
            self.exceptions[type(exc).__name__] += 1

    def summary(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.started_at
        latencies = sorted(self.latencies)
        percentiles = {}
        for pct in (50, 75, 90, 95, 99):
            if latencies:
                idx = int(round((pct / 100.0) * (len(latencies) - 1)))
                percentiles[f"p{pct}"] = latencies[idx]
        result = {
            "total_requests": self.total,
            "elapsed_sec": elapsed,
            "throughput_rps": self.total / elapsed if elapsed else 0,
            "latency_ms_avg": statistics.mean(latencies) * 1000 if latencies else 0,
            "latency_ms_stddev": statistics.pstdev(latencies) * 1000 if len(latencies) > 1 else 0,
            "latency_ms_min": min(latencies) * 1000 if latencies else 0,
            "latency_ms_max": max(latencies) * 1000 if latencies else 0,
            "latency_ms_percentiles": {k: v * 1000 for k, v in percentiles.items()},
            "status_counts": dict(self.status_counts),
            "http_counts": dict(self.http_codes),
            "exceptions": dict(self.exceptions),
        }
        return result


def build_payload_factory(args: argparse.Namespace, template: Optional[str]):
    payload_seed = os.urandom(max(args.payload_bytes, 1)).hex() if args.payload_bytes > 0 else ""

    def default_body(seq: int) -> Dict[str, Any]:
        filler = (payload_seed * ((args.payload_bytes // len(payload_seed)) + 1))[: args.payload_bytes]
        body: Dict[str, Any] = {
            "input": {
                "sequence": seq,
                "payload": filler,
            }
        }
        if args.depth > 0:
            body["input"]["depth"] = args.depth
        if args.width > 0:
            body["input"]["width"] = args.width
        return body

    if template is None:
        return default_body

    def from_template(seq: int) -> Dict[str, Any]:
        filler = os.urandom(max(args.payload_bytes, 1)).hex()[: args.payload_bytes] if args.payload_bytes else ""
        formatted = template.format(seq=seq, depth=args.depth, width=args.width, payload=filler)
        return json.loads(formatted)

    return from_template


def load_scenarios(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    if not args.scenario_file:
        return [("default", {})]

    text = args.scenario_file.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Scenario file must contain valid JSON: {exc}") from exc

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit("Scenario file must describe a list of scenario objects")

    scenarios: List[Tuple[str, Dict[str, Any]]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or f"scenario_{idx + 1}"
        overrides = {k: v for k, v in entry.items() if k != "name"}
        scenarios.append((name, overrides))

    if not scenarios:
        scenarios.append(("default", {}))

    return scenarios


async def scrape_metrics(url: Optional[str], keys: Iterable[str], timeout: float, verify_ssl: bool) -> Optional[Dict[str, Optional[float]]]:
    if not url:
        return None

    result: Dict[str, Optional[float]] = {key: None for key in keys}
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            response = await client.get(url)
            response.raise_for_status()
            for line in response.text.splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                metric_name = parts[0]
                if "{" in metric_name:
                    metric_name = metric_name.split("{", 1)[0]
                if metric_name in result and result[metric_name] is None:
                    try:
                        result[metric_name] = float(parts[-1])
                    except ValueError:
                        continue
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": str(exc)}

    return result


async def wait_for_async_completion(
    client: httpx.AsyncClient,
    execution_id: str,
    headers: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[int]]:
    poll_interval = args.poll_interval
    deadline = time.perf_counter() + args.execution_timeout
    endpoint = f"{args.base_url.rstrip('/')}{args.async_status_endpoint.rstrip('/')}/{execution_id}"

    while True:
        if time.perf_counter() > deadline:
            raise asyncio.TimeoutError(f"execution {execution_id} exceeded timeout {args.execution_timeout}s")

        response = await client.get(endpoint, headers=headers)
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = None
        if payload:
            status = str(payload.get("status", "")).lower()
            if status in SUCCESS_STATUSES | FAILURE_STATUSES:
                return status, payload, response.status_code
        else:
            status = None

        await asyncio.sleep(poll_interval * (1 + random.uniform(-args.jitter, args.jitter)))
        poll_interval = min(poll_interval * args.backoff_multiplier, args.max_poll_interval)


async def invoke_request(
    seq: int,
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    payload_factory,
    metrics: Metrics,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    failures: List[Dict[str, Any]],
) -> None:
    async with semaphore:
        body = payload_factory(seq)
        url = f"{args.base_url.rstrip('/')}{(args.sync_prefix if args.mode == 'sync' else args.async_submit_prefix).rstrip('/')}/{args.target}"
        start = time.perf_counter()
        exc: Optional[BaseException] = None
        final_status: Optional[str] = None
        http_code: Optional[int] = None
        payload_snapshot: Optional[Dict[str, Any]] = None

        try:
            response = await client.post(url, json=body, headers=headers)
            http_code = response.status_code
            if args.mode == 'sync':
                payload_snapshot = response.json()
                final_status = str(payload_snapshot.get("status", "")).lower()
            else:
                if response.status_code >= 400:
                    payload_snapshot = safe_json(response)
                    final_status = f"http_{response.status_code}"
                else:
                    submission = response.json()
                    exec_id = submission.get("execution_id")
                    if not exec_id:
                        final_status = "missing_execution_id"
                        payload_snapshot = submission
                    else:
                        final_status, payload_snapshot, poll_code = await wait_for_async_completion(
                            client, exec_id, headers, args
                        )
                        if poll_code is not None:
                            http_code = poll_code
        except Exception as err:  # pylint: disable=broad-except
            exc = err
            final_status = "exception"
            payload_snapshot = {"error": str(err)}

        latency = time.perf_counter() - start
        metrics.record(latency=latency, status=final_status, http_code=http_code, exc=exc)
        if final_status and final_status not in SUCCESS_STATUSES:
            failures.append({
                "sequence": seq,
                "status": final_status,
                "http_status": http_code,
                "response": payload_snapshot,
            })


def safe_json(response: httpx.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"raw": response.text[:512]}


async def run_load(args: argparse.Namespace) -> Dict[str, Any]:
    headers = {key: value for key, value in (parse_header(h) for h in args.headers)} if args.headers else {}
    payload_factory = build_payload_factory(args, load_template(args.body_template))
    metrics = Metrics()
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: List[Dict[str, Any]] = []

    limits = httpx.Limits(max_connections=args.concurrency * 4, max_keepalive_connections=args.concurrency)
    metric_keys = args.metrics or DEFAULT_METRIC_KEYS
    pre_metrics = await scrape_metrics(args.metrics_url, metric_keys, args.metrics_timeout, args.verify_ssl)

    async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits, verify=args.verify_ssl) as client:
        tasks = [
            invoke_request(seq, client, headers, payload_factory, metrics, args, semaphore, failures)
            for seq in range(args.requests)
        ]
        await asyncio.gather(*tasks)

    post_metrics = await scrape_metrics(args.metrics_url, metric_keys, args.metrics_timeout, args.verify_ssl)

    summary = metrics.summary()
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["config"] = {
        "mode": args.mode,
        "base_url": args.base_url,
        "target": args.target,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "depth": args.depth,
        "width": args.width,
        "payload_bytes": args.payload_bytes,
        "poll_interval": args.poll_interval,
        "max_poll_interval": args.max_poll_interval,
        "backoff_multiplier": args.backoff_multiplier,
        "execution_timeout": args.execution_timeout,
        "request_timeout": args.request_timeout,
    }

    if args.print_failures and failures:
        print("\n--- Failures (truncated) ---")
        for failure in failures[:20]:
            print(json.dumps(failure, indent=2, default=str))
        if len(failures) > 20:
            print(f"... {len(failures) - 20} additional failures suppressed")
    summary["failures"] = failures

    if args.metrics_url:
        summary["metrics"] = {
            "url": args.metrics_url,
            "pre": pre_metrics,
            "post": post_metrics,
            "delta": {
                key: (
                    None
                    if not isinstance(pre_metrics, dict) or not isinstance(post_metrics, dict)
                    else None
                    if pre_metrics.get(key) is None or post_metrics.get(key) is None
                    else post_metrics.get(key) - pre_metrics.get(key)
                )
                for key in metric_keys
            },
        }

    if summary["status_counts"].get("exception") == args.requests and summary["exceptions"].get("ConnectError"):
        summary["note"] = "All requests failed with connection errors. Verify the gateway is reachable at the specified base URL."

    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")

    if args.seed is not None:
        random.seed(args.seed)

    scenarios = load_scenarios(args)

    async def orchestrate() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name, overrides in scenarios:
            scenario_args = argparse.Namespace(**vars(args))
            for key, value in overrides.items():
                if hasattr(scenario_args, key):
                    setattr(scenario_args, key, value)
            summary = await run_load(scenario_args)
            summary["scenario"] = name
            results.append(summary)
        return results

    results = asyncio.run(orchestrate())

    output = results[0] if len(results) == 1 else {"scenarios": results}
    print(json.dumps(output, indent=2, default=str))

    if args.save_metrics:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(json.dumps(output, indent=2, default=str))
        print(f"Metrics written to {args.save_metrics}")

    failed = any(res["status_counts"].get("exception") or res["exceptions"] for res in results)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
