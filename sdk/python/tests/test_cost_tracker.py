"""Tests for the CostTracker module."""

from agentfield.cost_tracker import CostEntry, CostTracker


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_record_single_call(self):
        tracker = CostTracker()
        tracker.record(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
        )
        assert tracker.total_cost_usd == 0.005
        assert tracker.total_tokens == 150
        assert tracker.call_count == 1

    def test_record_multiple_calls(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4", total_tokens=100, cost_usd=0.003)
        tracker.record(model="claude-3", total_tokens=200, cost_usd=0.007)
        assert tracker.total_cost_usd == 0.010
        assert tracker.total_tokens == 300
        assert tracker.call_count == 2

    def test_summary_groups_by_model(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4", total_tokens=100, cost_usd=0.003)
        tracker.record(model="gpt-4", total_tokens=200, cost_usd=0.006)
        tracker.record(model="claude-3", total_tokens=150, cost_usd=0.004)
        summary = tracker.summary()
        assert summary["total_calls"] == 3
        assert summary["by_model"]["gpt-4"]["calls"] == 2
        assert summary["by_model"]["claude-3"]["calls"] == 1

    def test_reset_clears_entries(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4", total_tokens=100, cost_usd=0.003)
        tracker.reset()
        assert tracker.call_count == 0
        assert tracker.total_cost_usd == 0.0

    def test_cost_entry_with_reasoner_name(self):
        tracker = CostTracker()
        tracker.record(
            model="gpt-4",
            total_tokens=100,
            cost_usd=0.003,
            reasoner_name="evaluate_claim",
        )
        assert tracker.call_count == 1
        entry = CostEntry(
            model="gpt-4",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0.0,
            reasoner_name="evaluate_claim",
        )
        assert entry.reasoner_name == "evaluate_claim"
