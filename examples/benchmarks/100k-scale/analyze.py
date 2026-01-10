#!/usr/bin/env python3
"""
AgentField Benchmark Visualization

Creates a single, clean, publication-quality figure comparing frameworks.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Clean, minimal styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
})

# Professional color palette
COLORS = {
    "AgentField_Go": "#00ADD8",
    "AgentField_TypeScript": "#3178C6",
    "AgentField_Python": "#306998",
    "LangChain_Python": "#1C3C3C",
}

LABELS = {
    "AgentField_Go": "Go SDK",
    "AgentField_TypeScript": "TypeScript SDK",
    "AgentField_Python": "Python SDK",
    "LangChain_Python": "LangChain",
}


def load_results(results_dir: Path) -> dict:
    """Load benchmark results from JSON files."""
    results = {}
    for f in results_dir.glob("*.json"):
        if f.name.startswith(("AgentField", "LangChain")):
            with open(f) as fp:
                data = json.load(fp)
                key = f"{data.get('framework', 'unknown')}_{data.get('language', 'unknown')}"
                results[key] = data
    return results


def get_metric(results: dict, framework: str, metric: str) -> Optional[float]:
    """Extract a metric value."""
    if framework not in results:
        return None
    for r in results[framework].get("results", []):
        if r.get("metric") == metric:
            return r.get("value")
    return None


def create_benchmark_figure(results: dict, output_dir: Path):
    """
    Create a single clean figure with 4 key metrics.

    Uses horizontal bar charts with log scale for fair comparison
    across vastly different magnitudes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AgentField Framework Benchmark", fontsize=16, fontweight="bold", y=0.95)

    # Framework order (best to worst for visual hierarchy)
    frameworks = ["AgentField_Go", "AgentField_TypeScript", "AgentField_Python", "LangChain_Python"]

    def plot_metric(ax, metric_name, alt_metric, title, unit, lower_is_better=True):
        """Plot a single metric as horizontal bars."""
        values = []
        labels = []
        colors = []

        for fw in frameworks:
            v = get_metric(results, fw, metric_name) or get_metric(results, fw, alt_metric)
            if v is not None:
                values.append(v)
                labels.append(LABELS[fw])
                colors.append(COLORS[fw])

        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=1, height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(title, fontweight="bold", pad=10)

        # Use log scale for large range differences
        if max(values) / (min(values) + 0.001) > 10:
            ax.set_xscale("log")

        # Add value labels
        for bar, val in zip(bars, values):
            # Format based on magnitude
            if val >= 1_000_000:
                label = f"{val/1_000_000:.1f}M"
            elif val >= 1_000:
                label = f"{val/1_000:.1f}K"
            elif val >= 1:
                label = f"{val:.1f}"
            else:
                label = f"{val:.2f}"

            # Position label inside or outside bar
            x_pos = bar.get_width()
            ax.text(x_pos * 1.05, bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=10, fontweight="bold")

    # Plot 4 key metrics
    plot_metric(axes[0, 0],
                "registration_time_mean_ms", "registration_time_mean_ms",
                "Registration Time", "milliseconds")

    plot_metric(axes[0, 1],
                "memory_per_handler_bytes", "memory_per_tool_bytes",
                "Memory per Handler", "bytes")

    plot_metric(axes[1, 0],
                "request_latency_p99_us", "invocation_latency_p99_us",
                "Request Latency (p99)", "microseconds")

    plot_metric(axes[1, 1],
                "theoretical_single_thread_rps", "theoretical_single_thread_rps",
                "Throughput", "requests/second")

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # Add legend at bottom
    handles = [plt.Rectangle((0,0), 1, 1, color=COLORS[fw]) for fw in frameworks if fw in results]
    labels = [LABELS[fw] for fw in frameworks if fw in results]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=11)

    fig.savefig(output_dir / "benchmark_summary.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'benchmark_summary.png'}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)

    if not results:
        print("No benchmark results found.")
        return

    print(f"Loaded: {list(results.keys())}")
    create_benchmark_figure(results, results_dir)
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
