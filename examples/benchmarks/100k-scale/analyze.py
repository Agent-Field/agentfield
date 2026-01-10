#!/usr/bin/env python3
"""
AgentField Benchmark Analysis & Visualization

Creates technical, publication-quality plots for GitHub README.
Consolidated visualization with scientific styling.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Scientific plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
})

# Color palette - professional and accessible
COLORS = {
    "go": "#00ADD8",
    "typescript": "#3178C6",
    "python": "#3776AB",
    "langchain": "#6B7280",
}

LABELS = {
    "AgentField_Go": "Go SDK",
    "AgentField_TypeScript": "TypeScript SDK",
    "AgentField_Python": "Python SDK",
    "LangChain_Python": "LangChain",
}


def load_results(results_dir: Path) -> dict:
    """Load all benchmark results from JSON files."""
    results = {}
    for f in results_dir.glob("*.json"):
        if f.name.startswith(("AgentField", "LangChain")):
            with open(f) as fp:
                data = json.load(fp)
                key = f"{data.get('framework', 'unknown')}_{data.get('language', 'unknown')}"
                results[key] = data
    return results


def get_metric(results: dict, framework_lang: str, metric: str) -> Optional[float]:
    """Extract a specific metric from results."""
    if framework_lang not in results:
        return None
    for r in results[framework_lang].get("results", []):
        if r.get("metric") == metric:
            return r.get("value")
    return None


def get_raw_data(results: dict, framework_lang: str, key: str) -> Optional[list]:
    """Extract raw data array from results."""
    if framework_lang not in results:
        return None
    return results[framework_lang].get("raw_data", {}).get(key)


def plot_consolidated_summary(results: dict, output_dir: Path):
    """
    Create a single consolidated figure with 4 key metrics.

    Layout: 2x2 grid showing:
    - Registration time (ms)
    - Memory per handler (bytes/KB)
    - Latency p99 (µs)
    - Throughput (req/s)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("AgentField Benchmark Summary", fontsize=14, fontweight="bold", y=0.98)

    frameworks_order = [
        ("AgentField_Go", "go"),
        ("AgentField_TypeScript", "typescript"),
        ("AgentField_Python", "python"),
        ("LangChain_Python", "langchain"),
    ]

    # 1. Registration Time (top-left)
    ax = axes[0, 0]
    names, times, colors = [], [], []
    for key, color_key in frameworks_order:
        t = get_metric(results, key, "registration_time_mean_ms")
        if t is not None:
            names.append(LABELS[key])
            times.append(t)
            colors.append(COLORS[color_key])

    if names:
        bars = ax.barh(names, times, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_title("Handler Registration Time", fontweight="bold")
        ax.invert_yaxis()
        for bar, t in zip(bars, times):
            ax.text(t + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{t:.1f} ms", va="center", fontsize=9)
        ax.set_xlim(0, max(times) * 1.25)

    # 2. Memory per Handler (top-right)
    ax = axes[0, 1]
    names, memory, colors = [], [], []
    for key, color_key in frameworks_order:
        m = get_metric(results, key, "memory_per_handler_bytes") or \
            get_metric(results, key, "memory_per_tool_bytes")
        if m is not None:
            names.append(LABELS[key])
            memory.append(m)
            colors.append(COLORS[color_key])

    if names:
        bars = ax.barh(names, memory, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Bytes per Handler")
        ax.set_title("Memory Efficiency", fontweight="bold")
        ax.invert_yaxis()
        for bar, m in zip(bars, memory):
            label = f"{m:.0f} B" if m < 1024 else f"{m/1024:.1f} KB"
            ax.text(m + max(memory) * 0.02, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=9)
        ax.set_xlim(0, max(memory) * 1.25)

    # 3. Latency p99 (bottom-left)
    ax = axes[1, 0]
    names, latency, colors = [], [], []
    for key, color_key in frameworks_order:
        l = get_metric(results, key, "request_latency_p99_us") or \
            get_metric(results, key, "invocation_latency_p99_us")
        if l is not None:
            names.append(LABELS[key])
            latency.append(l)
            colors.append(COLORS[color_key])

    if names:
        bars = ax.barh(names, latency, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Latency (µs)")
        ax.set_title("Request Latency (p99)", fontweight="bold")
        ax.invert_yaxis()
        ax.set_xscale("log")
        for bar, l in zip(bars, latency):
            ax.text(l * 1.1, bar.get_y() + bar.get_height() / 2,
                    f"{l:.2f} µs", va="center", fontsize=9)

    # 4. Throughput (bottom-right)
    ax = axes[1, 1]
    names, throughput, colors = [], [], []
    for key, color_key in frameworks_order:
        t = get_metric(results, key, "theoretical_single_thread_rps")
        if t is not None:
            names.append(LABELS[key])
            throughput.append(t)
            colors.append(COLORS[color_key])

    if names:
        bars = ax.barh(names, throughput, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Requests per Second")
        ax.set_title("Theoretical Throughput", fontweight="bold")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"
        ))
        for bar, t in zip(bars, throughput):
            label = f"{t/1e6:.1f}M" if t >= 1e6 else f"{t/1e3:.0f}K"
            ax.text(t * 1.1, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "benchmark_summary.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'benchmark_summary.png'}")


def plot_latency_comparison(results: dict, output_dir: Path):
    """
    Create a detailed latency comparison with CDF curves.
    Scientific visualization showing distribution characteristics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Latency Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)

    frameworks_order = [
        ("AgentField_Go", "go", "Go SDK"),
        ("AgentField_TypeScript", "typescript", "TypeScript SDK"),
        ("AgentField_Python", "python", "Python SDK"),
        ("LangChain_Python", "langchain", "LangChain"),
    ]

    # Left: CDF plot
    for key, color_key, label in frameworks_order:
        raw = get_raw_data(results, key, "request_latency_us") or \
              get_raw_data(results, key, "invocation_latency_us")
        if raw:
            sorted_data = np.sort(raw)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax1.plot(sorted_data, cdf * 100, label=label,
                     color=COLORS[color_key], linewidth=2)

    ax1.set_xlabel("Latency (µs)")
    ax1.set_ylabel("Percentile (%)")
    ax1.set_title("Cumulative Distribution (CDF)", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 100)
    ax1.axhline(y=99, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(y=95, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax1.text(ax1.get_xlim()[1] * 0.7, 99.5, "p99", fontsize=8, color="gray")
    ax1.text(ax1.get_xlim()[1] * 0.7, 95.5, "p95", fontsize=8, color="gray")
    ax1.legend(loc="lower right", fontsize=10)

    # Right: Box plot
    data_to_plot = []
    labels = []
    colors_list = []

    for key, color_key, label in frameworks_order:
        raw = get_raw_data(results, key, "request_latency_us") or \
              get_raw_data(results, key, "invocation_latency_us")
        if raw:
            # Sample if too many points for cleaner visualization
            if len(raw) > 1000:
                raw = np.random.choice(raw, 1000, replace=False).tolist()
            data_to_plot.append(raw)
            labels.append(label)
            colors_list.append(COLORS[color_key])

    if data_to_plot:
        bp = ax2.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("Latency (µs)")
        ax2.set_title("Latency Distribution (Box Plot)", fontweight="bold")
        ax2.set_yscale("log")

        # Add median annotations
        for i, data in enumerate(data_to_plot):
            median = np.median(data)
            ax2.annotate(
                f"{median:.2f}µs",
                xy=(i + 1, median),
                xytext=(15, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
            )

    plt.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'latency_comparison.png'}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)

    if not results:
        print("No benchmark results found. Run the benchmarks first.")
        return

    print(f"Loaded results for: {list(results.keys())}")
    print()

    # Generate consolidated visualizations (only 2 images)
    plot_consolidated_summary(results, results_dir)
    plot_latency_comparison(results, results_dir)

    print("\nVisualization complete! Generated 2 publication-quality figures.")


if __name__ == "__main__":
    main()
