#!/usr/bin/env python3
"""
AgentField Python SDK Benchmark

Measures: Agent init time, handler registration time (separate), memory footprint, request latency

IMPORTANT: This benchmark separates Agent initialization overhead from per-handler registration
to provide fair comparison with lightweight tool wrappers like LangChain's StructuredTool.
"""

import argparse
import asyncio
import gc
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'sdk', 'python'))

from agentfield import Agent


@dataclass
class Stats:
    mean: float
    stddev: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


def calculate_stats(data: list[float]) -> Stats:
    if not data:
        return Stats(0, 0, 0, 0, 0, 0, 0)

    sorted_data = sorted(data)
    n = len(data)
    mean = statistics.mean(data)
    stddev = statistics.stdev(data) if n > 1 else 0

    def percentile(p: float) -> float:
        idx = int((n - 1) * p)
        return sorted_data[idx]

    return Stats(
        mean=mean,
        stddev=stddev,
        min=sorted_data[0],
        max=sorted_data[-1],
        p50=percentile(0.50),
        p95=percentile(0.95),
        p99=percentile(0.99),
    )


def benchmark_agent_init(iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure Agent initialization time (WITHOUT handlers)."""
    if verbose:
        print("Benchmark: Agent Initialization (no handlers)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        time.sleep(0.01)

        start = time.perf_counter()

        agent = Agent(
            node_id=f"init-bench-{i}",
            agentfield_server="http://localhost:8080",
            auto_register=False,
            enable_mcp=False,  # MCP disabled by default
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.2f} ms")

        del agent
        gc.collect()

    return results


def benchmark_handler_registration(num_handlers: int, iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure ONLY handler registration time (Agent already created)."""
    if verbose:
        print(f"\nBenchmark: Handler Registration ONLY ({num_handlers} handlers)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        time.sleep(0.01)

        # Create Agent OUTSIDE the measurement
        agent = Agent(
            node_id=f"handler-bench-{i}",
            agentfield_server="http://localhost:8080",
            auto_register=False,
            enable_mcp=False,  # MCP disabled by default
        )

        # Measure ONLY handler registration
        start = time.perf_counter()

        for j in range(num_handlers):
            idx = j

            @agent.reasoner(f"handler-{j}")
            async def handler(input_data: dict, _idx=idx) -> dict:
                return {"id": _idx, "processed": True}

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                per_handler_ms = elapsed_ms / num_handlers
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.2f} ms ({per_handler_ms:.3f} ms/handler)")

        del agent
        gc.collect()

    return results


def benchmark_agent_memory(iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure memory for Agent ONLY (no handlers)."""
    if verbose:
        print("\nBenchmark: Agent Memory (no handlers)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        gc.collect()
        time.sleep(0.05)

        tracemalloc.start()

        agent = Agent(
            node_id=f"agent-mem-{i}",
            agentfield_server="http://localhost:8080",
            auto_register=False,
            enable_mcp=False,  # MCP disabled by default
        )

        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_mb = current / 1024 / 1024

        if i >= warmup:
            results.append(mem_mb)
            if verbose:
                print(f"  Run {i - warmup + 1}: {mem_mb:.2f} MB")

        del agent
        gc.collect()

    return results


def benchmark_handler_memory(num_handlers: int, iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure memory for handlers ONLY (Agent overhead excluded)."""
    if verbose:
        print(f"\nBenchmark: Handler Memory ONLY ({num_handlers} handlers)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        gc.collect()
        time.sleep(0.05)

        # Create Agent BEFORE starting memory tracking
        agent = Agent(
            node_id=f"handler-mem-{i}",
            agentfield_server="http://localhost:8080",
            auto_register=False,
            enable_mcp=False,  # MCP disabled by default
        )

        gc.collect()
        gc.collect()
        time.sleep(0.02)

        # Start tracking AFTER Agent is created
        tracemalloc.start()

        for j in range(num_handlers):
            idx = j

            @agent.reasoner(f"handler-{j}")
            async def handler(input_data: dict, _idx=idx) -> dict:
                return {"id": _idx}

        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_mb = current / 1024 / 1024

        if i >= warmup:
            results.append(mem_mb)
            if verbose:
                per_handler_kb = (mem_mb * 1024) / num_handlers
                print(f"  Run {i - warmup + 1}: {mem_mb:.2f} MB ({per_handler_kb:.2f} KB/handler)")

        del agent
        gc.collect()

    return results


def benchmark_cold_start(iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure agent + first handler time (traditional cold start)."""
    if verbose:
        print("\nBenchmark: Cold Start (Agent + 1 handler)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()

        start = time.perf_counter()

        agent = Agent(
            node_id=f"cold-{i}",
            agentfield_server="http://localhost:8080",
            auto_register=False,
            enable_mcp=False,  # MCP disabled by default
        )

        @agent.reasoner("ping")
        async def ping(input_data: dict) -> dict:
            return {"pong": True}

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.3f} ms")

        del agent

    return results


async def benchmark_request_processing(num_handlers: int, num_requests: int, verbose: bool) -> list[float]:
    """Measure request processing latency (handler invocation only)."""
    if verbose:
        print(f"\nBenchmark: Request Processing Latency ({num_requests} requests)")

    # Create handlers directly (measures raw async function overhead)
    handlers = []
    for i in range(num_handlers):
        idx = i

        async def handler(input_data: dict, _idx=idx) -> dict:
            return {
                "handler_id": _idx,
                "processed": True,
                "timestamp": time.time_ns(),
            }

        handlers.append(handler)

    # Warm up
    input_data = {"query": "test", "value": 42}
    for i in range(1000):
        await handlers[i % num_handlers](input_data)

    # Measure
    results = []
    for i in range(num_requests):
        handler_idx = i % num_handlers
        start = time.perf_counter()
        await handlers[handler_idx](input_data)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        results.append(elapsed_us)

    if verbose:
        stats = calculate_stats(results)
        print(f"  p50: {stats.p50:.2f} µs, p95: {stats.p95:.2f} µs, p99: {stats.p99:.2f} µs")

    return results


def main():
    parser = argparse.ArgumentParser(description="AgentField Python SDK Benchmark")
    parser.add_argument("--handlers", type=int, default=10000, help="Number of handlers")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    verbose = not args.json

    suite = {
        "framework": "AgentField",
        "language": "Python",
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "os": platform.system(),
            "arch": platform.machine(),
        },
        "results": [],
        "raw_data": {},
    }

    if verbose:
        print("AgentField Python SDK Benchmark (Fixed Methodology)")
        print("====================================================")
        print(f"Handlers: {args.handlers} | Iterations: {args.iterations} | Warmup: {args.warmup}")
        print("\nNOTE: Agent init and handler registration are measured SEPARATELY\n")

    # 1. Agent initialization time (no handlers)
    agent_init_times = benchmark_agent_init(args.iterations, args.warmup, verbose)
    agent_init_stats = calculate_stats(agent_init_times)
    suite["raw_data"]["agent_init_time_ms"] = agent_init_times
    suite["results"].extend([
        {"metric": "agent_init_time_mean_ms", "value": agent_init_stats.mean, "unit": "ms", "iterations": len(agent_init_times)},
        {"metric": "agent_init_time_p99_ms", "value": agent_init_stats.p99, "unit": "ms"},
    ])

    # 2. Handler registration time ONLY (Agent created outside measurement)
    reg_handlers = min(args.handlers, 10000)  # Can test up to 10K now
    reg_times = benchmark_handler_registration(reg_handlers, args.iterations, args.warmup, verbose)
    reg_stats = calculate_stats(reg_times)
    suite["raw_data"]["registration_time_ms"] = reg_times
    suite["results"].extend([
        {"metric": "registration_time_mean_ms", "value": reg_stats.mean, "unit": "ms", "iterations": len(reg_times), "handler_count": reg_handlers},
        {"metric": "registration_time_stddev_ms", "value": reg_stats.stddev, "unit": "ms"},
        {"metric": "registration_time_p50_ms", "value": reg_stats.p50, "unit": "ms"},
        {"metric": "registration_time_p99_ms", "value": reg_stats.p99, "unit": "ms"},
        {"metric": "registration_time_per_handler_ms", "value": reg_stats.mean / reg_handlers, "unit": "ms"},
    ])

    # 3. Agent memory (no handlers)
    agent_mem_data = benchmark_agent_memory(args.iterations, args.warmup, verbose)
    agent_mem_stats = calculate_stats(agent_mem_data)
    suite["raw_data"]["agent_memory_mb"] = agent_mem_data
    suite["results"].extend([
        {"metric": "agent_memory_mean_mb", "value": agent_mem_stats.mean, "unit": "MB", "iterations": len(agent_mem_data)},
    ])

    # 4. Handler memory ONLY (Agent excluded)
    mem_handlers = min(args.handlers, 10000)
    mem_data = benchmark_handler_memory(mem_handlers, args.iterations, args.warmup, verbose)
    mem_stats = calculate_stats(mem_data)
    suite["raw_data"]["memory_mb"] = mem_data
    suite["results"].extend([
        {"metric": "memory_mean_mb", "value": mem_stats.mean, "unit": "MB", "iterations": len(mem_data), "handler_count": mem_handlers},
        {"metric": "memory_stddev_mb", "value": mem_stats.stddev, "unit": "MB"},
        {"metric": "memory_per_handler_bytes", "value": (mem_stats.mean * 1024 * 1024) / mem_handlers, "unit": "bytes"},
    ])

    # 5. Cold start (Agent + 1 handler, for comparison with other benchmarks)
    cold_times = benchmark_cold_start(args.iterations, args.warmup, verbose)
    cold_stats = calculate_stats(cold_times)
    suite["raw_data"]["cold_start_ms"] = cold_times
    suite["results"].extend([
        {"metric": "cold_start_mean_ms", "value": cold_stats.mean, "unit": "ms", "iterations": len(cold_times)},
        {"metric": "cold_start_p99_ms", "value": cold_stats.p99, "unit": "ms"},
    ])

    # 6. Request latency benchmark
    req_times = asyncio.run(benchmark_request_processing(min(args.handlers, 1000), 10000, verbose))
    req_stats = calculate_stats(req_times)
    suite["raw_data"]["request_latency_us"] = req_times
    suite["results"].extend([
        {"metric": "request_latency_mean_us", "value": req_stats.mean, "unit": "us"},
        {"metric": "request_latency_p50_us", "value": req_stats.p50, "unit": "us"},
        {"metric": "request_latency_p95_us", "value": req_stats.p95, "unit": "us"},
        {"metric": "request_latency_p99_us", "value": req_stats.p99, "unit": "us"},
    ])

    if req_stats.mean > 0:
        suite["results"].append({
            "metric": "theoretical_single_thread_rps",
            "value": 1_000_000 / req_stats.mean,
            "unit": "req/s",
        })

    if args.json:
        print(json.dumps(suite, indent=2))
    else:
        print("\n=== Summary ===")
        print(f"Agent Init: {agent_init_stats.mean:.2f} ms (one-time overhead)")
        print(f"Agent Memory: {agent_mem_stats.mean:.2f} MB (one-time overhead)")
        print(f"Handler Registration ({reg_handlers}): {reg_stats.mean:.2f} ms ({reg_stats.mean / reg_handlers:.3f} ms/handler)")
        print(f"Handler Memory ({mem_handlers}): {mem_stats.mean:.2f} MB ({(mem_stats.mean * 1024 * 1024) / mem_handlers:.0f} bytes/handler)")
        print(f"Cold Start (Agent + 1 handler): {cold_stats.mean:.2f} ms")
        print(f"Request Latency p99: {req_stats.p99:.2f} µs")


if __name__ == "__main__":
    main()
