"""
Memory Performance Tests for AgentField Python SDK.

These tests validate memory efficiency of SDK components and establish
baseline metrics for regression testing.

Run with: pytest tests/test_memory_performance.py -v
"""

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.execution_state import ExecutionState, ExecutionStatus
from agentfield.result_cache import ResultCache
from agentfield.client import AgentFieldClient


@dataclass
class MemoryMetrics:
    """Memory measurement results."""
    name: str
    peak_mb: float
    current_mb: float
    iterations: int
    duration_sec: float

    @property
    def per_iteration_kb(self) -> float:
        """Memory per iteration in KB."""
        return (self.current_mb * 1024) / self.iterations if self.iterations > 0 else 0

    @property
    def reduction_pct(self) -> float:
        """Memory reduction from peak to current."""
        return ((self.peak_mb - self.current_mb) / self.peak_mb * 100) if self.peak_mb > 0 else 0


def measure_memory(func, iterations: int = 1000) -> MemoryMetrics:
    """Execute a function and measure memory usage."""
    gc.collect()
    tracemalloc.start()
    start_time = time.time()

    result = func(iterations)

    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MemoryMetrics(
        name=func.__name__,
        peak_mb=peak / 1024 / 1024,
        current_mb=current / 1024 / 1024,
        iterations=iterations,
        duration_sec=time.time() - start_time,
    )


class TestAsyncConfigDefaults:
    """Test that AsyncConfig has memory-optimized defaults."""

    def test_cache_ttl_is_optimized(self):
        """Cache TTL should be 2 minutes or less for memory efficiency."""
        config = AsyncConfig()
        assert config.result_cache_ttl <= 120.0, "Cache TTL should be <= 120s"

    def test_cache_max_size_is_bounded(self):
        """Cache max size should be bounded for memory efficiency."""
        config = AsyncConfig()
        assert config.result_cache_max_size <= 5000, "Cache max size should be <= 5000"

    def test_cleanup_interval_is_aggressive(self):
        """Cleanup interval should be short for memory efficiency."""
        config = AsyncConfig()
        assert config.cleanup_interval <= 30.0, "Cleanup interval should be <= 30s"

    def test_completed_execution_retention_is_short(self):
        """Completed execution retention should be short."""
        config = AsyncConfig()
        assert config.completed_execution_retention_seconds <= 120.0, \
            "Retention should be <= 120s"

    def test_max_completed_executions_is_bounded(self):
        """Max completed executions should be bounded."""
        config = AsyncConfig()
        assert config.max_completed_executions <= 2000, \
            "Max completed executions should be <= 2000"


class TestExecutionStateMemory:
    """Test ExecutionState memory management."""

    def _create_execution_states(self, count: int) -> List[ExecutionState]:
        """Create execution states with large payloads."""
        states = []
        for i in range(count):
            state = ExecutionState(
                execution_id=f"exec_{i:06d}",
                target=f"agent.reasoner_{i}",
                input_data={
                    "payload": "x" * 10000,  # ~10KB
                    "nested": {"items": list(range(500))},
                }
            )
            states.append(state)
        return states

    def test_input_data_cleared_on_success(self):
        """Input data should be cleared when execution succeeds."""
        state = ExecutionState(
            execution_id="test_exec",
            target="agent.reasoner",
            input_data={"large": "x" * 10000}
        )
        assert len(state.input_data) > 0

        state.set_result({"output": "result"})

        assert state.input_data == {}, "Input data should be cleared on success"
        assert state.status == ExecutionStatus.SUCCEEDED

    def test_input_data_cleared_on_error(self):
        """Input data should be cleared when execution fails."""
        state = ExecutionState(
            execution_id="test_exec",
            target="agent.reasoner",
            input_data={"large": "x" * 10000}
        )

        state.set_error("Test error")

        assert state.input_data == {}, "Input data should be cleared on error"
        assert state.status == ExecutionStatus.FAILED

    def test_input_data_cleared_on_cancel(self):
        """Input data should be cleared when execution is cancelled."""
        state = ExecutionState(
            execution_id="test_exec",
            target="agent.reasoner",
            input_data={"large": "x" * 10000}
        )

        state.cancel("User cancelled")

        assert state.input_data == {}, "Input data should be cleared on cancel"
        assert state.status == ExecutionStatus.CANCELLED

    def test_input_data_cleared_on_timeout(self):
        """Input data should be cleared when execution times out."""
        state = ExecutionState(
            execution_id="test_exec",
            target="agent.reasoner",
            input_data={"large": "x" * 10000},
            timeout=1.0
        )

        state.timeout_execution()

        assert state.input_data == {}, "Input data should be cleared on timeout"
        assert state.status == ExecutionStatus.TIMEOUT

    def test_memory_reduction_after_completion(self):
        """Memory should be significantly reduced after executions complete."""
        def benchmark(iterations: int):
            states = self._create_execution_states(iterations)
            # Complete 70% of executions
            for i in range(int(iterations * 0.7)):
                states[i].set_result({"output": f"result_{i}"})
            return states

        metrics = measure_memory(benchmark, iterations=1000)

        # Memory should be reduced by at least 50% after clearing input_data
        assert metrics.reduction_pct >= 50.0, \
            f"Expected >= 50% memory reduction, got {metrics.reduction_pct:.1f}%"


class TestResultCacheMemory:
    """Test ResultCache memory management."""

    def test_cache_respects_max_size(self):
        """Cache should not exceed max size."""
        config = AsyncConfig()
        cache = ResultCache(config)

        # Add more entries than max size
        for i in range(config.result_cache_max_size + 1000):
            cache.set(f"key_{i}", {"data": "x" * 100})

        assert len(cache) <= config.result_cache_max_size, \
            "Cache should not exceed max size"

    def test_cache_memory_is_bounded(self):
        """Cache memory should be bounded by max size."""
        def benchmark(iterations: int):
            config = AsyncConfig()
            cache = ResultCache(config)
            for i in range(iterations):
                cache.set(f"key_{i}", {"data": "x" * 1000})
            return cache

        metrics = measure_memory(benchmark, iterations=10000)

        # With 5000 max entries at ~1KB each, should be under 10MB
        assert metrics.current_mb < 10.0, \
            f"Cache memory should be bounded, got {metrics.current_mb:.2f} MB"


class TestClientSessionReuse:
    """Test HTTP session reuse in AgentFieldClient."""

    def test_shared_session_is_created(self):
        """Shared sync session should be created."""
        # Reset shared session
        AgentFieldClient._shared_sync_session = None

        client = AgentFieldClient(base_url="http://localhost:8080")

        assert AgentFieldClient._shared_sync_session is not None, \
            "Shared session should be created"

    def test_multiple_clients_share_session(self):
        """Multiple clients should share the same sync session."""
        # Reset shared session
        AgentFieldClient._shared_sync_session = None

        client1 = AgentFieldClient(base_url="http://localhost:8080")
        session1 = AgentFieldClient._shared_sync_session

        client2 = AgentFieldClient(base_url="http://localhost:8081")
        session2 = AgentFieldClient._shared_sync_session

        assert session1 is session2, "Clients should share session"

    def test_client_creation_memory_is_low(self):
        """Creating multiple clients should use minimal memory."""
        # Reset shared session
        AgentFieldClient._shared_sync_session = None

        def benchmark(iterations: int):
            clients = []
            for i in range(iterations):
                client = AgentFieldClient(base_url=f"http://localhost:808{i % 10}")
                clients.append(client)
            return clients

        metrics = measure_memory(benchmark, iterations=100)

        # 100 clients should use less than 1MB total
        assert metrics.current_mb < 1.0, \
            f"Client creation should be memory efficient, got {metrics.current_mb:.2f} MB"


class TestMemoryBenchmarkBaseline:
    """
    Baseline memory benchmark tests.

    These tests establish performance baselines and can be used for
    regression testing. The baseline simulates typical execution patterns.
    """

    def _create_baseline_workload(self, iterations: int) -> List[Dict[str, Any]]:
        """Create baseline workload simulating typical usage patterns."""
        workloads = []
        for i in range(iterations):
            workload = {
                "input": {
                    "payload": "x" * 10000,
                    "nested": {"items": list(range(500))},
                    "metadata": {"id": f"run_{i}"},
                },
                "history": [],
                "context": {},
            }
            # Simulate history accumulation (common pattern)
            for j in range(10):
                workload["history"].append({
                    "role": "user",
                    "content": f"Message {j}: " + "y" * 500,
                })
                workload["history"].append({
                    "role": "assistant",
                    "content": f"Response {j}: " + "z" * 500,
                })
            workloads.append(workload)
        return workloads

    def test_baseline_memory_usage(self):
        """Measure baseline memory usage for comparison."""
        def benchmark(iterations: int):
            return self._create_baseline_workload(iterations)

        metrics = measure_memory(benchmark, iterations=1000)

        print(f"\n{'=' * 60}")
        print("BASELINE MEMORY USAGE")
        print(f"{'=' * 60}")
        print(f"Iterations:     {metrics.iterations}")
        print(f"Peak Memory:    {metrics.peak_mb:.2f} MB")
        print(f"Current Memory: {metrics.current_mb:.2f} MB")
        print(f"Per Iteration:  {metrics.per_iteration_kb:.2f} KB")
        print(f"Duration:       {metrics.duration_sec:.3f}s")
        print(f"{'=' * 60}")

        # Baseline should be reasonable (under 100MB for 1000 iterations)
        assert metrics.current_mb < 100.0, "Baseline memory should be reasonable"

    def test_optimized_sdk_memory_usage(self):
        """Measure optimized SDK memory usage."""
        def benchmark(iterations: int):
            config = AsyncConfig()
            cache = ResultCache(config)
            states = []

            for i in range(iterations):
                state = ExecutionState(
                    execution_id=f"exec_{i:06d}",
                    target=f"agent.reasoner_{i}",
                    input_data={
                        "payload": "x" * 10000,
                        "nested": {"items": list(range(500))},
                        "metadata": {"id": f"run_{i}"},
                    }
                )
                state.set_result({"output": f"result_{i}"})
                cache.set_execution_result(state.execution_id, state.result)
                states.append(state)

            return states, cache

        metrics = measure_memory(benchmark, iterations=1000)

        print(f"\n{'=' * 60}")
        print("OPTIMIZED SDK MEMORY USAGE")
        print(f"{'=' * 60}")
        print(f"Iterations:     {metrics.iterations}")
        print(f"Peak Memory:    {metrics.peak_mb:.2f} MB")
        print(f"Current Memory: {metrics.current_mb:.2f} MB")
        print(f"Per Iteration:  {metrics.per_iteration_kb:.2f} KB")
        print(f"Duration:       {metrics.duration_sec:.3f}s")
        print(f"Reduction:      {metrics.reduction_pct:.1f}%")
        print(f"{'=' * 60}")

        # Optimized SDK should use significantly less memory
        assert metrics.current_mb < 10.0, \
            f"Optimized SDK should use < 10MB, got {metrics.current_mb:.2f} MB"
        assert metrics.per_iteration_kb < 10.0, \
            f"Per-iteration memory should be < 10KB, got {metrics.per_iteration_kb:.2f} KB"


@pytest.fixture
def memory_report():
    """Fixture to collect and report memory metrics."""
    metrics_list = []
    yield metrics_list

    if metrics_list:
        print("\n" + "=" * 70)
        print("MEMORY PERFORMANCE REPORT")
        print("=" * 70)
        print(f"{'Test Name':<40} {'Current':>10} {'Peak':>10} {'Per Iter':>12}")
        print("-" * 70)
        for m in metrics_list:
            print(f"{m.name:<40} {m.current_mb:>8.2f}MB {m.peak_mb:>8.2f}MB {m.per_iteration_kb:>10.2f}KB")
        print("=" * 70)


class TestMemoryPerformanceReport:
    """Generate comprehensive memory performance report."""

    def test_full_memory_report(self, memory_report):
        """Run all benchmarks and generate report."""
        config = AsyncConfig()

        # Test 1: ExecutionState with completion
        def exec_state_benchmark(n):
            states = []
            for i in range(n):
                s = ExecutionState(
                    execution_id=f"e_{i}",
                    target="a.r",
                    input_data={"p": "x" * 10000}
                )
                s.set_result({"o": i})
                states.append(s)
            return states

        m1 = measure_memory(exec_state_benchmark, 1000)
        m1.name = "ExecutionState (completed)"
        memory_report.append(m1)

        # Test 2: ResultCache bounded
        def cache_benchmark(n):
            c = ResultCache(config)
            for i in range(n):
                c.set(f"k_{i}", {"d": "x" * 1000})
            return c

        m2 = measure_memory(cache_benchmark, 10000)
        m2.name = "ResultCache (bounded)"
        memory_report.append(m2)

        # Test 3: Client session reuse
        AgentFieldClient._shared_sync_session = None

        def client_benchmark(n):
            clients = []
            for i in range(n):
                clients.append(AgentFieldClient(base_url=f"http://localhost:808{i%10}"))
            return clients

        m3 = measure_memory(client_benchmark, 100)
        m3.name = "AgentFieldClient (shared session)"
        memory_report.append(m3)

        # Assertions
        assert m1.current_mb < 5.0, "ExecutionState memory too high"
        assert m2.current_mb < 10.0, "ResultCache memory too high"
        assert m3.current_mb < 1.0, "Client memory too high"
