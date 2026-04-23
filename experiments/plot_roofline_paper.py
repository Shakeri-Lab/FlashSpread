#!/usr/bin/env python
"""
Publication-quality Roofline Plot for two-column papers.

Features:
- Unique marker for each configuration
- Large fonts for two-column format
- LaTeX-style rendering (mathtext fallback if latex unavailable)
- No figure title (caption provided separately)
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Check if LaTeX is available
import shutil
HAS_LATEX = shutil.which('latex') is not None

if HAS_LATEX:
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })
else:
    # Use mathtext for LaTeX-like rendering without latex binary
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # Computer Modern
    })

# Common font settings for publication
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
})

# A100 GPU specifications (CUDA cores only)
A100_PEAK_FLOPS = 19.5e12  # 19.5 TFLOPS FP32
A100_PEAK_BANDWIDTH = 2039e9  # 2039 GB/s HBM2e
A100_RIDGE_POINT = A100_PEAK_FLOPS / A100_PEAK_BANDWIDTH


# Unique markers and colors for each configuration
# Using shorter labels for publication clarity
CONFIG_STYLES = {
    # Renewal baseline - filled marker (plotted last for visibility)
    'renewal_baseline': {'marker': 'o', 'color': '#1f77b4', 'label': 'Baseline',
                         'size': 140, 'edgecolor': 'black', 'linewidth': 1.5, 'zorder': 10},

    # Epsilon variants - UNFILLED markers to distinguish from baseline at same position
    'renewal_small_tau': {'marker': 'D', 'color': 'none', 'label': r'$\epsilon{=}0.01$',
                          'edgecolor': '#9467bd', 'linewidth': 2.0, 'size': 100},
    'renewal_large_tau': {'marker': 's', 'color': 'none', 'label': r'$\epsilon{=}0.1$',
                          'edgecolor': '#d62728', 'linewidth': 2.0, 'size': 110},

    # Dense hazard
    'renewal_dense': {'marker': 'p', 'color': '#17becf', 'label': 'Dense hazard'},

    # CUDA Graph batched configs (green tones)
    'renewal_batched_10': {'marker': '^', 'color': '#2ca02c', 'label': r'CG $b{=}10$'},
    'renewal_batched_50': {'marker': 'v', 'color': '#98df8a', 'label': r'CG $b{=}50$', 'edgecolor': '#2ca02c'},
    'renewal_batched_100': {'marker': '<', 'color': '#ff7f0e', 'label': r'CG $b{=}100$'},

    # Compute-heavy configs (orange/red tones)
    'renewal_compute_heavy': {'marker': '>', 'color': '#e377c2', 'label': r'$m{\times}2$'},
    'renewal_ridge_8': {'marker': 'h', 'color': '#bcbd22', 'label': r'$m{\times}8$'},
    'renewal_ridge_16': {'marker': 'H', 'color': '#7f7f7f', 'label': r'$m{\times}16$'},
    'renewal_compute_bound': {'marker': '*', 'color': '#000000', 'label': r'$m{\times}20$', 'size': 200},

    # Markovian configs
    'markov_baseline': {'marker': 'X', 'color': '#8c564b', 'label': 'Markov'},
    'markov_aggressive': {'marker': 'P', 'color': '#984ea3', 'label': 'Markov aggr.'},
}


def plot_roofline_paper(
    results_path: str,
    output_path: str,
    figsize: tuple = (7.0, 4.5),  # Fits single column or spans two columns
    dpi: int = 300,
) -> None:
    """
    Generate publication-quality roofline plot.

    Args:
        results_path: Path to roofline_data.json
        output_path: Path for output figure (PDF recommended)
        figsize: Figure size in inches
        dpi: Resolution for raster output
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute ridge point
    ridge_point = A100_PEAK_FLOPS / A100_PEAK_BANDWIDTH

    # Generate roofline ceiling
    ai_range = np.logspace(-2, 2, 500)  # 0.01 to 100 FLOPs/byte
    memory_ceiling = A100_PEAK_BANDWIDTH * ai_range / 1e9  # GFLOPS
    compute_ceiling = np.full_like(ai_range, A100_PEAK_FLOPS / 1e9)
    roofline = np.minimum(memory_ceiling, compute_ceiling)

    # Plot roofline ceiling (thick black line)
    ax.loglog(ai_range, roofline, 'k-', linewidth=2.0, zorder=1)

    # Add efficiency bands
    for eff, color, style, lw in [
        (0.10, '#2ca02c', '--', 1.2),
        (0.05, '#ff7f0e', ':', 1.2),
    ]:
        ax.loglog(ai_range, roofline * eff, color=color, linestyle=style,
                  linewidth=lw, alpha=0.7, zorder=1)

    # Ridge point vertical line
    ax.axvline(x=ridge_point, color='gray', linestyle=':', linewidth=1.0,
               alpha=0.6, zorder=1)

    # Plot data points with unique markers
    legend_handles = []

    # Sort results so baseline is plotted last (on top of overlapping markers)
    def sort_key(r):
        if r['config_name'] == 'renewal_baseline':
            return 1  # Plot last
        return 0
    results_sorted = sorted(results, key=sort_key)

    for result in results_sorted:
        config_name = result['config_name']
        ai = result['arithmetic_intensity']
        gflops = result['achieved_gflops']

        style = CONFIG_STYLES.get(config_name, {
            'marker': 'o', 'color': 'gray', 'label': config_name
        })

        size = style.get('size', 120)
        edgecolor = style.get('edgecolor', 'black')
        linewidth = style.get('linewidth', 1.0)
        zorder = style.get('zorder', 3)

        scatter = ax.scatter(
            ai, gflops,
            marker=style['marker'],
            c=style['color'],
            s=size,
            edgecolors=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            alpha=0.9,
        )

        # Create legend handle
        handle = Line2D(
            [0], [0],
            marker=style['marker'],
            color='w',
            markerfacecolor=style['color'],
            markeredgecolor=edgecolor,
            markeredgewidth=linewidth,
            markersize=10 if config_name == 'renewal_baseline' else (9 if style['marker'] != '*' else 13),
            label=style['label'],
            linestyle='None',
        )
        legend_handles.append((config_name, handle))

    # Sort legend handles by category
    order = [
        'renewal_baseline', 'renewal_dense', 'renewal_small_tau', 'renewal_large_tau',
        'renewal_batched_10', 'renewal_batched_50', 'renewal_batched_100',
        'renewal_compute_heavy', 'renewal_ridge_8', 'renewal_ridge_16', 'renewal_compute_bound',
        'markov_baseline', 'markov_aggressive',
    ]

    sorted_handles = []
    for name in order:
        for cfg_name, handle in legend_handles:
            if cfg_name == name:
                sorted_handles.append(handle)
                break

    # Add roofline lines to legend
    roofline_handles = [
        Line2D([0], [0], color='k', linewidth=2.0, label='Roofline (100%)'),
        Line2D([0], [0], color='#2ca02c', linestyle='--', linewidth=1.2, label='10% efficiency'),
        Line2D([0], [0], color='#ff7f0e', linestyle=':', linewidth=1.2, label='5% efficiency'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0, label=f'Ridge ({ridge_point:.1f} F/B)'),
    ]

    # Create legend with three columns for compact display
    all_handles = roofline_handles + sorted_handles
    ax.legend(
        handles=all_handles,
        loc='lower right',
        ncol=3,
        framealpha=0.95,
        fontsize=9,
        columnspacing=0.5,
        handletextpad=0.3,
        borderpad=0.4,
    )

    # Axis labels with LaTeX
    ax.set_xlabel(r'Arithmetic Intensity (FLOPs/byte)')
    ax.set_ylabel(r'Performance (GFLOPS)')

    # Set axis limits
    ax.set_xlim(0.1, 50)
    ax.set_ylim(10, A100_PEAK_FLOPS / 1e9 * 1.2)

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, zorder=0)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1, zorder=0)

    # Region annotations
    ax.text(0.25, 5000, 'Memory-bound', fontsize=12, alpha=0.5,
            rotation=0, ha='center', style='italic')
    ax.text(25, 12000, 'Compute-bound', fontsize=12, alpha=0.5,
            rotation=0, ha='center', style='italic')

    plt.tight_layout()

    # Save as both PDF and PNG
    base_path = output_path.rsplit('.', 1)[0]
    plt.savefig(f'{base_path}.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{base_path}.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved: {base_path}.pdf and {base_path}.png")


def get_suggested_caption() -> str:
    """Return suggested LaTeX caption for the figure."""
    return r"""
\caption{Roofline analysis of \textsc{FlashSpread} on NVIDIA A100 (CUDA cores only,
19.5 TFLOPS FP32, 2039 GB/s). The ridge point at 9.6 FLOPs/byte separates memory-bound
(left) from compute-bound (right) regimes. Standard renewal configurations operate in
the memory-bound regime with 3--10\% efficiency, typical for sparse irregular workloads.
CUDA Graph batching ($b$) reduces kernel launch overhead, achieving the highest efficiency
(10\%) at the same arithmetic intensity. Artificial compute multipliers (mult$\times$8--20)
demonstrate the transition to compute-bound behavior, though these are not used in
practice. The algorithm's memory-bound nature motivates our CUDA Graph optimization
for amortizing launch overhead rather than increasing compute intensity.}
"""


if __name__ == '__main__':
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    results_path = os.path.join(results_dir, 'roofline_data.json')
    output_path = os.path.join(results_dir, 'roofline_paper.pdf')

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)

    plot_roofline_paper(results_path, output_path)

    print("\n" + "="*60)
    print("Suggested LaTeX caption:")
    print("="*60)
    print(get_suggested_caption())
