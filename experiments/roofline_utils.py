#!/usr/bin/env python
"""
Roofline Model Visualization Utilities.

This module provides functions to generate roofline plots from benchmark data.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Conditional matplotlib import for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cluster
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# A100 GPU specifications
A100_PEAK_FLOPS = 19.5e12  # 19.5 TFLOPS FP32
A100_PEAK_BANDWIDTH = 2039e9  # 2039 GB/s HBM2e
A100_RIDGE_POINT = A100_PEAK_FLOPS / A100_PEAK_BANDWIDTH


def load_benchmark_results(json_path: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_roofline(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "FlashSpread Roofline Analysis",
    peak_flops: float = A100_PEAK_FLOPS,
    peak_bandwidth: float = A100_PEAK_BANDWIDTH,
    figsize: tuple = (12, 8),
) -> None:
    """
    Generate roofline plot from benchmark results.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Path to save the plot.
        title: Plot title.
        peak_flops: GPU peak FLOPS (default: A100).
        peak_bandwidth: GPU memory bandwidth (default: A100).
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping roofline plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Compute ridge point
    ridge_point = peak_flops / peak_bandwidth

    # Generate roofline ceiling
    ai_range = np.logspace(-2, 3, 1000)  # 0.01 to 1000 FLOPs/byte
    memory_ceiling = peak_bandwidth * ai_range / 1e9  # GFLOPS
    compute_ceiling = np.full_like(ai_range, peak_flops / 1e9)
    roofline = np.minimum(memory_ceiling, compute_ceiling)

    # Plot roofline ceiling
    ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline Ceiling (100%)')

    # Add efficiency bands (10%, 5%, 1%)
    for eff, color, style in [(0.10, 'green', '--'), (0.05, 'orange', ':'), (0.01, 'red', ':')]:
        ax.loglog(ai_range, roofline * eff, color=color, linestyle=style, linewidth=1,
                  alpha=0.6, label=f'{int(eff*100)}% Efficiency')

    ax.axvline(x=ridge_point, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label=f'Ridge Point ({ridge_point:.1f})')

    # Color map for different engine types
    colors = {
        'renewal': 'blue',
        'renewal_cuda_graph': 'green',
        'markovian': 'red',
    }

    markers = {
        'renewal': 'o',
        'renewal_cuda_graph': 's',
        'markovian': '^',
    }

    # Plot data points
    for result in results:
        ai = result['arithmetic_intensity']
        gflops = result['achieved_gflops']
        engine_type = result['engine_type']
        name = result['config_name']

        color = colors.get(engine_type, 'gray')
        marker = markers.get(engine_type, 'x')

        ax.scatter(ai, gflops, c=color, marker=marker, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.annotate(
            name.replace('renewal_', 'R:').replace('markov_', 'M:'),
            (ai, gflops),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.8,
        )

    # Add legend combining lines and data points
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2, label='Roofline (100%)'),
        Line2D([0], [0], color='green', linestyle='--', label='10% Efficiency'),
        Line2D([0], [0], color='orange', linestyle=':', label='5% Efficiency'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Renewal'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='Renewal+CUDA Graph'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Markovian'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Labels and title
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set axis limits
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.1, peak_flops/1e9 * 1.5)

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Add annotations for regions
    ax.text(0.05, peak_flops/1e9 * 0.3, 'Memory\nBound', fontsize=10, alpha=0.5)
    ax.text(ridge_point * 3, peak_flops/1e9 * 0.8, 'Compute\nBound', fontsize=10, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved roofline plot: {output_path}")


def plot_speedup_comparison(
    results: List[Dict[str, Any]],
    output_path: str,
    baseline_name: str = "renewal_baseline",
) -> None:
    """
    Generate speedup comparison bar chart.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Path to save the plot.
        baseline_name: Name of the baseline configuration.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping speedup plot")
        return

    # Find baseline
    baseline = None
    for r in results:
        if r['config_name'] == baseline_name:
            baseline = r
            break

    if baseline is None:
        print(f"Warning: Baseline '{baseline_name}' not found")
        return

    baseline_time = baseline['time_per_step_ms']

    # Compute speedups
    names = []
    speedups = []
    colors = []

    color_map = {
        'renewal': 'steelblue',
        'renewal_cuda_graph': 'forestgreen',
        'markovian': 'indianred',
    }

    for r in results:
        if r['config_name'] == baseline_name:
            continue
        names.append(r['config_name'].replace('renewal_', '').replace('markov_', 'm_'))
        speedups.append(baseline_time / r['time_per_step_ms'])
        colors.append(color_map.get(r['engine_type'], 'gray'))

    # Sort by speedup
    sorted_indices = np.argsort(speedups)[::-1]
    names = [names[i] for i in sorted_indices]
    speedups = [speedups[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(names)), speedups, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Baseline')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Speedup vs Baseline')
    ax.set_title(f'Speedup Comparison (Baseline: {baseline_name})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved speedup plot: {output_path}")


def plot_timing_breakdown(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Generate timing breakdown chart.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Path to save the plot.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping timing plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r['config_name'].replace('renewal_', 'R:').replace('markov_', 'M:')
             for r in results]
    times = [r['time_per_step_ms'] for r in results]

    colors = []
    for r in results:
        if 'markov' in r['config_name']:
            colors.append('indianred')
        elif 'cuda_graph' in r['engine_type']:
            colors.append('forestgreen')
        else:
            colors.append('steelblue')

    bars = ax.bar(range(len(names)), times, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Time per Step (ms)')
    ax.set_title('Step Time Comparison')

    # Add value labels
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved timing plot: {output_path}")


def generate_all_plots(
    results_dir: str = "results",
) -> None:
    """
    Generate all plots from benchmark results.

    Args:
        results_dir: Directory containing roofline_data.json.
    """
    json_path = os.path.join(results_dir, "roofline_data.json")

    if not os.path.exists(json_path):
        print(f"Error: Results file not found: {json_path}")
        return

    results = load_benchmark_results(json_path)

    if not results:
        print("No results to plot")
        return

    print(f"Loaded {len(results)} benchmark results")

    # Generate roofline plot
    plot_roofline(
        results,
        os.path.join(results_dir, "roofline_plot.png"),
    )

    # Generate speedup comparison
    plot_speedup_comparison(
        results,
        os.path.join(results_dir, "speedup_comparison.png"),
    )

    # Generate timing breakdown
    plot_timing_breakdown(
        results,
        os.path.join(results_dir, "timing_breakdown.png"),
    )

    print("All plots generated!")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_all_plots(results_dir)
