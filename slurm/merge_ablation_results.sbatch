#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -J merge_ablation
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Merging ablation study results..."

python << 'EOF'
import json
import glob
import os

results = []
for f in sorted(glob.glob('results/ablation/ablation_*.json')):
    with open(f) as fp:
        results.append(json.load(fp))
    print(f'Loaded: {f}')

if not results:
    print("No results found!")
    exit(1)

# Sort by config_id
results.sort(key=lambda x: x['config_id'])

# Save merged JSON
with open('results/ablation/ablation_all.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Merged {len(results)} results to results/ablation/ablation_all.json')

# Generate summary report
baseline = [r for r in results if r['config_id'] == 0]
baseline = baseline[0] if baseline else None

with open('results/ablation/ablation_summary.md', 'w') as f:
    f.write('# FlashSpread Ablation Study Results\n\n')

    f.write('## Configuration Summary\n\n')
    f.write('| ID | Config | RCM | Fused | Block | CUDA Graph | Steps/Launch |\n')
    f.write('|----|--------|-----|-------|-------|------------|---------------|\n')
    for r in results:
        f.write(f'| {r["config_id"]} | {r["config_name"]} | ')
        f.write(f'{"Y" if r["use_rcm_reordering"] else "N"} | ')
        f.write(f'{"Y" if r["use_fused_ops"] else "N"} | ')
        f.write(f'{r["flash_neighbor_block_size"]} | ')
        f.write(f'{"Y" if r["use_cuda_graph"] else "N"} | ')
        f.write(f'{r["steps_per_launch"]} |\n')

    f.write('\n## Performance Results\n\n')
    f.write('| Config | ms/step | Steps/sec | Speedup | Accuracy | Status |\n')
    f.write('|--------|---------|-----------|---------|----------|--------|\n')
    for r in results:
        status = '✓' if r['accuracy_passed'] else '✗'
        f.write(f'| {r["config_name"]} | {r["time_per_step_ms"]:.3f} | ')
        f.write(f'{r["steps_per_second"]:.1f} | ')
        f.write(f'{r["speedup_vs_baseline"]:.2f}x | ')
        f.write(f'{r["accuracy_error_percent"]:.1f}% | {status} |\n')

    if baseline:
        f.write('\n## Key Findings\n\n')

        # Find best speedup with accuracy
        valid = [r for r in results if r['accuracy_passed']]
        if valid:
            best = max(valid, key=lambda x: x['speedup_vs_baseline'])
            f.write(f'**Best configuration:** `{best["config_name"]}` with {best["speedup_vs_baseline"]:.2f}x speedup\n\n')

        # Individual optimization effects
        f.write('### Individual Optimization Effects\n\n')
        for r in results:
            if r['config_id'] == 0:
                continue
            if r['config_id'] <= 4:  # Individual opts
                f.write(f'- **{r["config_name"]}**: {r["speedup_vs_baseline"]:.2f}x speedup')
                if r['use_rcm_reordering'] and r['graph_bandwidth_before'] > 0:
                    reduction = (1 - r['graph_bandwidth_after']/r['graph_bandwidth_before']) * 100
                    f.write(f' (bandwidth reduced {reduction:.0f}%)')
                f.write('\n')

        f.write('\n### Combined Optimization Effects\n\n')
        for r in results:
            if r['config_id'] >= 5:
                f.write(f'- **{r["config_name"]}**: {r["speedup_vs_baseline"]:.2f}x speedup\n')

    f.write('\n## Accuracy Validation\n\n')
    f.write('All configurations compared against baseline epidemic trajectory.\n')
    f.write('Accuracy passes if final infected count within 10% of baseline.\n\n')

    passed = sum(1 for r in results if r['accuracy_passed'])
    f.write(f'**{passed}/{len(results)} configurations passed accuracy validation.**\n')

print('Wrote results/ablation/ablation_summary.md')

# Generate CSV
import csv
with open('results/ablation/ablation_all.csv', 'w', newline='') as f:
    if results:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
print('Wrote results/ablation/ablation_all.csv')
EOF

echo "Done! Results in results/ablation/"
