#!/usr/bin/env python3
"""
AgentField Benchmark Analysis & Visualization

Creates technical, publication-quality plots suitable for GitHub README.
Uses seaborn for aesthetic, professional visualizations.
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Set seaborn style for clean, technical aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Custom color palette - accessible and professional
COLORS = {
    "go": "#00ADD8",        # Go blue
    "typescript": "#3178C6", # TypeScript blue
    "python": "#3776AB",     # Python blue
    "langchain": "#1C3C3C",  # LangChain dark
    "highlight": "#FF6B6B",  # Accent for callouts
}

# Figure settings
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_results(results_dir: Path) -> dict:
    """Load all benchmark results from JSON files."""
    results = {}
    for f in results_dir.glob("*.json"):
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


def plot_registration_comparison(results: dict, output_dir: Path):
    """Bar chart comparing registration time across frameworks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    frameworks = []
    times = []
    errors = []
    colors = []
    handler_counts = []

    # Collect data
    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        mean = get_metric(results, key, "registration_time_mean_ms")
        std = get_metric(results, key, "registration_time_stddev_ms")
        if mean is not None:
            # Get handler count for annotation
            for r in results[key].get("results", []):
                if r.get("metric") == "registration_time_mean_ms":
                    handler_counts.append(r.get("handler_count", r.get("tool_count", "N/A")))
                    break
            else:
                handler_counts.append("N/A")

            frameworks.append(key.replace("_", "\n"))
            times.append(mean)
            errors.append(std or 0)
            colors.append(color)

    if not frameworks:
        print("No registration data available")
        return

    x = np.arange(len(frameworks))
    bars = ax.bar(x, times, yerr=errors, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, handler_counts)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f} ms\n({count} handlers)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Registration Time (ms)", fontweight="bold")
    ax.set_title("Handler Registration Time Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks)
    ax.set_ylim(0, max(times) * 1.3)

    plt.tight_layout()
    fig.savefig(output_dir / "registration_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'registration_comparison.png'}")


def plot_memory_comparison(results: dict, output_dir: Path):
    """Bar chart comparing memory footprint."""
    fig, ax = plt.subplots(figsize=(10, 6))

    frameworks = []
    memory = []
    per_handler = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        mem = get_metric(results, key, "memory_mean_mb")
        per_h = get_metric(results, key, "memory_per_handler_bytes") or get_metric(results, key, "memory_per_tool_bytes")
        if mem is not None:
            frameworks.append(key.replace("_", "\n"))
            memory.append(mem)
            per_handler.append(per_h or 0)
            colors.append(color)

    if not frameworks:
        print("No memory data available")
        return

    x = np.arange(len(frameworks))
    bars = ax.bar(x, memory, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, per_h in zip(bars, per_handler):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f} MB\n({per_h:.0f} B/handler)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Memory Usage (MB)", fontweight="bold")
    ax.set_title("Memory Footprint Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks)
    ax.set_ylim(0, max(memory) * 1.4)

    plt.tight_layout()
    fig.savefig(output_dir / "memory_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'memory_comparison.png'}")


def plot_latency_distribution(results: dict, output_dir: Path):
    """Box plot showing latency distribution across frameworks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []
    colors_list = []

    for key, color, label in [
        ("AgentField_Go", COLORS["go"], "AgentField\n(Go)"),
        ("AgentField_TypeScript", COLORS["typescript"], "AgentField\n(TypeScript)"),
        ("AgentField_Python", COLORS["python"], "AgentField\n(Python)"),
        ("LangChain_Python", COLORS["langchain"], "LangChain\n(Python)"),
    ]:
        raw = get_raw_data(results, key, "request_latency_us") or get_raw_data(results, key, "invocation_latency_us")
        if raw:
            # Sample if too many points
            if len(raw) > 1000:
                raw = np.random.choice(raw, 1000, replace=False).tolist()
            data_to_plot.append(raw)
            labels.append(label)
            colors_list.append(color)

    if not data_to_plot:
        print("No latency data available")
        return

    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        showfliers=False,  # Hide outliers for cleaner look
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (µs)", fontweight="bold")
    ax.set_title("Request Processing Latency Distribution", fontweight="bold", fontsize=14)
    ax.set_yscale("log")

    # Add median annotations
    for i, data in enumerate(data_to_plot):
        median = np.median(data)
        ax.annotate(
            f"p50: {median:.1f}µs",
            xy=(i + 1, median),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "latency_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'latency_distribution.png'}")


def plot_latency_cdf(results: dict, output_dir: Path):
    """CDF plot for latency - standard for performance analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, color, label in [
        ("AgentField_Go", COLORS["go"], "AgentField (Go)"),
        ("AgentField_TypeScript", COLORS["typescript"], "AgentField (TypeScript)"),
        ("AgentField_Python", COLORS["python"], "AgentField (Python)"),
        ("LangChain_Python", COLORS["langchain"], "LangChain (Python)"),
    ]:
        raw = get_raw_data(results, key, "request_latency_us") or get_raw_data(results, key, "invocation_latency_us")
        if raw:
            sorted_data = np.sort(raw)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf * 100, label=label, color=color, linewidth=2)

    ax.set_xlabel("Latency (µs)", fontweight="bold")
    ax.set_ylabel("Percentile (%)", fontweight="bold")
    ax.set_title("Latency CDF (Cumulative Distribution)", fontweight="bold", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.axhline(y=99, color="gray", linestyle="--", alpha=0.5, label="p99")
    ax.axhline(y=95, color="gray", linestyle=":", alpha=0.5, label="p95")

    plt.tight_layout()
    fig.savefig(output_dir / "latency_cdf.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'latency_cdf.png'}")


def plot_scaling_analysis(results: dict, output_dir: Path):
    """Plot showing Go SDK scaling characteristics."""
    # This uses the raw scaling data from Go benchmark if available
    go_data = results.get("AgentField_Go", {})
    raw_reg = go_data.get("raw_data", {}).get("registration_time_ms")
    raw_mem = go_data.get("raw_data", {}).get("memory_mb")

    if not raw_reg or not raw_mem:
        print("No scaling data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # If we have scaling test data (multiple handler counts)
    # For now, show the distribution of measurements
    ax1.hist(raw_reg, bins=20, color=COLORS["go"], edgecolor="black", alpha=0.7)
    ax1.axvline(np.mean(raw_reg), color=COLORS["highlight"], linestyle="--", linewidth=2, label=f"Mean: {np.mean(raw_reg):.2f}ms")
    ax1.set_xlabel("Registration Time (ms)", fontweight="bold")
    ax1.set_ylabel("Frequency", fontweight="bold")
    ax1.set_title("Registration Time Distribution", fontweight="bold")
    ax1.legend()

    ax2.hist(raw_mem, bins=20, color=COLORS["go"], edgecolor="black", alpha=0.7)
    ax2.axvline(np.mean(raw_mem), color=COLORS["highlight"], linestyle="--", linewidth=2, label=f"Mean: {np.mean(raw_mem):.2f}MB")
    ax2.set_xlabel("Memory Usage (MB)", fontweight="bold")
    ax2.set_ylabel("Frequency", fontweight="bold")
    ax2.set_title("Memory Usage Distribution", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "go_scaling_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'go_scaling_distribution.png'}")


def plot_summary_dashboard(results: dict, output_dir: Path):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Registration time comparison (bar)
    ax1 = fig.add_subplot(gs[0, 0])
    frameworks = []
    times = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        t = get_metric(results, key, "registration_time_mean_ms")
        if t is not None:
            frameworks.append(key.split("_")[1] if "AgentField" in key else "LangChain")
            times.append(t)
            colors.append(color)

    if frameworks:
        ax1.barh(frameworks, times, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Registration Time", fontweight="bold")
        for i, t in enumerate(times):
            ax1.text(t + max(times) * 0.02, i, f"{t:.1f}ms", va="center", fontsize=9)

    # 2. Memory per handler (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    frameworks = []
    memory = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        m = get_metric(results, key, "memory_per_handler_bytes") or get_metric(results, key, "memory_per_tool_bytes")
        if m is not None:
            frameworks.append(key.split("_")[1] if "AgentField" in key else "LangChain")
            memory.append(m)
            colors.append(color)

    if frameworks:
        ax2.barh(frameworks, memory, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Bytes per Handler")
        ax2.set_title("Memory Efficiency", fontweight="bold")
        for i, m in enumerate(memory):
            ax2.text(m + max(memory) * 0.02, i, f"{m:.0f}B", va="center", fontsize=9)

    # 3. Cold start time (bar)
    ax3 = fig.add_subplot(gs[0, 2])
    frameworks = []
    cold = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        c = get_metric(results, key, "cold_start_mean_ms")
        if c is not None:
            frameworks.append(key.split("_")[1] if "AgentField" in key else "LangChain")
            cold.append(c)
            colors.append(color)

    if frameworks:
        ax3.barh(frameworks, cold, color=colors, edgecolor="black", linewidth=0.5)
        ax3.set_xlabel("Time (ms)")
        ax3.set_title("Cold Start Time", fontweight="bold")
        for i, c in enumerate(cold):
            ax3.text(c + max(cold) * 0.02, i, f"{c:.1f}ms", va="center", fontsize=9)

    # 4. Latency p99 (bar)
    ax4 = fig.add_subplot(gs[1, 0])
    frameworks = []
    latency = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        l = get_metric(results, key, "request_latency_p99_us") or get_metric(results, key, "invocation_latency_p99_us")
        if l is not None:
            frameworks.append(key.split("_")[1] if "AgentField" in key else "LangChain")
            latency.append(l)
            colors.append(color)

    if frameworks:
        ax4.barh(frameworks, latency, color=colors, edgecolor="black", linewidth=0.5)
        ax4.set_xlabel("Latency (µs)")
        ax4.set_title("Request Latency (p99)", fontweight="bold")
        for i, l in enumerate(latency):
            ax4.text(l + max(latency) * 0.02, i, f"{l:.1f}µs", va="center", fontsize=9)

    # 5. Theoretical throughput (bar)
    ax5 = fig.add_subplot(gs[1, 1])
    frameworks = []
    throughput = []
    colors = []

    for key, color in [
        ("AgentField_Go", COLORS["go"]),
        ("AgentField_TypeScript", COLORS["typescript"]),
        ("AgentField_Python", COLORS["python"]),
        ("LangChain_Python", COLORS["langchain"]),
    ]:
        t = get_metric(results, key, "theoretical_single_thread_rps")
        if t is not None:
            frameworks.append(key.split("_")[1] if "AgentField" in key else "LangChain")
            throughput.append(t)
            colors.append(color)

    if frameworks:
        ax5.barh(frameworks, throughput, color=colors, edgecolor="black", linewidth=0.5)
        ax5.set_xlabel("Requests/sec")
        ax5.set_title("Theoretical Throughput", fontweight="bold")
        ax5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
        for i, t in enumerate(throughput):
            label = f"{t/1000:.0f}K" if t >= 1000 else f"{t:.0f}"
            ax5.text(t + max(throughput) * 0.02, i, label, va="center", fontsize=9)

    # 6. Legend / Key metrics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    legend_elements = [
        Patch(facecolor=COLORS["go"], edgecolor="black", label="AgentField (Go)"),
        Patch(facecolor=COLORS["typescript"], edgecolor="black", label="AgentField (TypeScript)"),
        Patch(facecolor=COLORS["python"], edgecolor="black", label="AgentField (Python)"),
        Patch(facecolor=COLORS["langchain"], edgecolor="black", label="LangChain (Python)"),
    ]
    ax6.legend(handles=legend_elements, loc="center", fontsize=12)

    # Add key insight text
    go_mem = get_metric(results, "AgentField_Go", "memory_per_handler_bytes")
    lc_mem = get_metric(results, "LangChain_Python", "memory_per_tool_bytes")
    if go_mem and lc_mem:
        ratio = lc_mem / go_mem
        ax6.text(
            0.5, 0.2,
            f"Go is {ratio:.0f}x more memory efficient",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["highlight"],
            transform=ax6.transAxes,
        )

    fig.suptitle("AgentField Benchmark Summary", fontsize=16, fontweight="bold", y=1.02)

    plt.savefig(output_dir / "benchmark_summary.png", bbox_inches="tight", facecolor="white")
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
        print("No benchmark results found. Run the benchmarks first.")
        return

    print(f"Loaded results for: {list(results.keys())}")
    print()

    # Generate all plots
    plot_registration_comparison(results, results_dir)
    plot_memory_comparison(results, results_dir)
    plot_latency_distribution(results, results_dir)
    plot_latency_cdf(results, results_dir)
    plot_scaling_analysis(results, results_dir)
    plot_summary_dashboard(results, results_dir)

    print("\nAll visualizations generated!")


if __name__ == "__main__":
    main()
