#!/usr/bin/env python3
"""
CrewAI Benchmark

Measures equivalent operations to AgentField for fair comparison.
Uses CrewAI's tool registration as the comparable operation.
"""

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Type

try:
    from crewai.tools import BaseTool, tool
    from pydantic import BaseModel, Field
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("Warning: crewai not installed. Install with: pip install crewai")


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


def benchmark_tool_registration(num_tools: int, iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure time to create CrewAI tools using BaseTool subclassing."""
    if not CREWAI_AVAILABLE:
        return []

    if verbose:
        print(f"Benchmark: Tool Registration ({num_tools} tools)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        time.sleep(0.01)

        start = time.perf_counter()

        # Create tools using BaseTool subclassing (similar to LangChain's StructuredTool)
        tools = []
        for j in range(num_tools):
            idx = j

            # Create input schema
            class ToolInput(BaseModel):
                query: str = Field(..., description="The query to process")

            # Create tool class dynamically
            tool_class = type(
                f"Tool_{idx}",
                (BaseTool,),
                {
                    "name": f"tool_{idx}",
                    "description": f"Tool number {idx}",
                    "args_schema": ToolInput,
                    "_run": lambda self, query: {"tool_id": idx, "processed": True},
                }
            )
            tools.append(tool_class())

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.2f} ms")

        del tools
        gc.collect()

    return results


def benchmark_tool_decorator(num_tools: int, iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure time to create CrewAI tools using @tool decorator."""
    if not CREWAI_AVAILABLE:
        return []

    if verbose:
        print(f"\nBenchmark: Tool Registration via @tool ({num_tools} tools)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        time.sleep(0.01)

        start = time.perf_counter()

        # Create tools using @tool decorator
        tools = []
        for j in range(num_tools):
            idx = j

            def make_tool(tool_idx):
                @tool(f"tool_{tool_idx}")
                def tool_func(query: str) -> dict:
                    """Process a query and return result."""
                    return {"tool_id": tool_idx, "processed": True}
                return tool_func

            tools.append(make_tool(idx))

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.2f} ms")

        del tools
        gc.collect()

    return results


def benchmark_memory(num_tools: int, iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure memory footprint of CrewAI tools."""
    if not CREWAI_AVAILABLE:
        return []

    if verbose:
        print(f"\nBenchmark: Memory Footprint ({num_tools} tools)")

    results = []

    for i in range(iterations + warmup):
        gc.collect()
        time.sleep(0.05)

        tracemalloc.start()

        # Create tools using @tool decorator
        tools = []
        for j in range(num_tools):
            idx = j

            def make_tool(tool_idx):
                @tool(f"tool_{tool_idx}")
                def tool_func(query: str) -> dict:
                    """Process a query."""
                    return {"tool_id": tool_idx}
                return tool_func

            tools.append(make_tool(idx))

        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_mb = current / 1024 / 1024

        if i >= warmup:
            results.append(mem_mb)
            if verbose:
                print(f"  Run {i - warmup + 1}: {mem_mb:.2f} MB")

        del tools
        gc.collect()

    return results


def benchmark_cold_start(iterations: int, warmup: int, verbose: bool) -> list[float]:
    """Measure time to create a minimal CrewAI tool setup."""
    if not CREWAI_AVAILABLE:
        return []

    if verbose:
        print("\nBenchmark: Cold Start Time")

    results = []

    for i in range(iterations + warmup):
        gc.collect()

        start = time.perf_counter()

        @tool("ping")
        def ping_tool(query: str) -> dict:
            """Ping tool."""
            return {"pong": True}

        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            results.append(elapsed_ms)
            if verbose:
                print(f"  Run {i - warmup + 1}: {elapsed_ms:.3f} ms")

        del ping_tool

    return results


def benchmark_tool_invocation(num_tools: int, num_invocations: int, verbose: bool) -> list[float]:
    """Measure tool invocation latency."""
    if not CREWAI_AVAILABLE:
        return []

    if verbose:
        print(f"\nBenchmark: Tool Invocation Latency ({num_invocations} invocations)")

    # Create tools
    tools = []
    for i in range(num_tools):
        idx = i

        def make_tool(tool_idx):
            @tool(f"tool_{tool_idx}")
            def tool_func(query: str) -> dict:
                """Process query."""
                return {
                    "tool_id": tool_idx,
                    "processed": True,
                    "timestamp": time.time_ns(),
                }
            return tool_func

        tools.append(make_tool(idx))

    # Warm up
    for i in range(1000):
        tools[i % num_tools].run(query="test")

    # Measure
    results = []
    for i in range(num_invocations):
        tool_idx = i % num_tools
        start = time.perf_counter()
        tools[tool_idx].run(query="test")
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        results.append(elapsed_us)

    if verbose:
        stats = calculate_stats(results)
        print(f"  p50: {stats.p50:.2f} us, p95: {stats.p95:.2f} us, p99: {stats.p99:.2f} us")

    return results


def main():
    parser = argparse.ArgumentParser(description="CrewAI Benchmark")
    parser.add_argument("--tools", type=int, default=1000, help="Number of tools")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    verbose = not args.json

    if not CREWAI_AVAILABLE:
        print("Error: crewai not available")
        return

    suite = {
        "framework": "CrewAI",
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
        print("CrewAI Benchmark")
        print("================")
        print(f"Tools: {args.tools} | Iterations: {args.iterations} | Warmup: {args.warmup}\n")

    # Registration benchmark (using @tool decorator - more common usage)
    reg_tools = min(args.tools, 1000)
    reg_times = benchmark_tool_decorator(reg_tools, args.iterations, args.warmup, verbose)
    if reg_times:
        reg_stats = calculate_stats(reg_times)
        suite["raw_data"]["registration_time_ms"] = reg_times
        suite["results"].extend([
            {"metric": "registration_time_mean_ms", "value": reg_stats.mean, "unit": "ms", "iterations": len(reg_times), "tool_count": reg_tools},
            {"metric": "registration_time_stddev_ms", "value": reg_stats.stddev, "unit": "ms"},
            {"metric": "registration_time_p50_ms", "value": reg_stats.p50, "unit": "ms"},
            {"metric": "registration_time_p99_ms", "value": reg_stats.p99, "unit": "ms"},
        ])

    # Memory benchmark
    mem_tools = min(args.tools, 1000)
    mem_data = benchmark_memory(mem_tools, args.iterations, args.warmup, verbose)
    if mem_data:
        mem_stats = calculate_stats(mem_data)
        suite["raw_data"]["memory_mb"] = mem_data
        suite["results"].extend([
            {"metric": "memory_mean_mb", "value": mem_stats.mean, "unit": "MB", "iterations": len(mem_data), "tool_count": mem_tools},
            {"metric": "memory_stddev_mb", "value": mem_stats.stddev, "unit": "MB"},
            {"metric": "memory_per_tool_bytes", "value": (mem_stats.mean * 1024 * 1024) / mem_tools, "unit": "bytes"},
        ])

    # Cold start benchmark
    cold_times = benchmark_cold_start(args.iterations, args.warmup, verbose)
    if cold_times:
        cold_stats = calculate_stats(cold_times)
        suite["raw_data"]["cold_start_ms"] = cold_times
        suite["results"].extend([
            {"metric": "cold_start_mean_ms", "value": cold_stats.mean, "unit": "ms", "iterations": len(cold_times)},
            {"metric": "cold_start_p99_ms", "value": cold_stats.p99, "unit": "ms"},
        ])

    # Invocation latency benchmark
    inv_times = benchmark_tool_invocation(min(args.tools, 100), 10000, verbose)
    if inv_times:
        inv_stats = calculate_stats(inv_times)
        suite["raw_data"]["invocation_latency_us"] = inv_times
        suite["results"].extend([
            {"metric": "invocation_latency_mean_us", "value": inv_stats.mean, "unit": "us"},
            {"metric": "invocation_latency_p50_us", "value": inv_stats.p50, "unit": "us"},
            {"metric": "invocation_latency_p95_us", "value": inv_stats.p95, "unit": "us"},
            {"metric": "invocation_latency_p99_us", "value": inv_stats.p99, "unit": "us"},
        ])

        if inv_stats.mean > 0:
            suite["results"].append({
                "metric": "theoretical_single_thread_rps",
                "value": 1_000_000 / inv_stats.mean,
                "unit": "req/s",
            })

    if args.json:
        print(json.dumps(suite, indent=2))
    else:
        print("\n=== Summary ===")
        if reg_times:
            print(f"Registration ({reg_tools}): {reg_stats.mean:.2f} ms (+/-{reg_stats.stddev:.2f})")
        if mem_data:
            print(f"Memory ({mem_tools}): {mem_stats.mean:.2f} MB ({(mem_stats.mean * 1024 * 1024) / mem_tools:.0f} bytes/tool)")
        if cold_times:
            print(f"Cold Start: {cold_stats.mean:.2f} ms")
        if inv_times:
            print(f"Invocation Latency p99: {inv_stats.p99:.2f} us")


if __name__ == "__main__":
    main()
