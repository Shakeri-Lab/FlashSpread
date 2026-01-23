#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -J merge_roofline
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Merging roofline results..."

# Merge individual JSON files into combined results
python -c "
import json
import glob
import os

results = []
for f in sorted(glob.glob('results/roofline_*.json')):
    if 'data' not in f:  # Skip roofline_data.json if exists
        with open(f) as fp:
            results.append(json.load(fp))
        print(f'Loaded: {f}')

if results:
    with open('results/roofline_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Merged {len(results)} results to results/roofline_data.json')

    # Also write CSV
    import csv
    with open('results/roofline_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('Wrote results/roofline_data.csv')
"

# Generate plots
echo "Generating plots..."
python experiments/roofline_utils.py results

# Generate summary
python -c "
import json

A100_RIDGE = 9.6  # FLOPs/byte

with open('results/roofline_data.json') as f:
    results = json.load(f)

renewal = [r for r in results if 'renewal' in r['engine_type']]
markov = [r for r in results if 'markov' in r['engine_type']]

best_renewal = max(renewal, key=lambda r: r['achieved_gflops']) if renewal else None
best_markov = max(markov, key=lambda r: r['achieved_gflops']) if markov else None

with open('results/roofline_summary.md', 'w') as f:
    f.write('# Roofline Benchmark Summary\n\n')
    f.write('## Best Performers\n\n')

    if best_renewal:
        f.write(f'**Best Renewal**: \`{best_renewal[\"config_name\"]}\`\n')
        f.write(f'- {best_renewal[\"achieved_gflops\"]:.2f} GFLOPS\n')
        f.write(f'- AI: {best_renewal[\"arithmetic_intensity\"]:.2f} FLOPs/byte\n')
        f.write(f'- Efficiency: {best_renewal[\"efficiency_percent\"]:.1f}%\n\n')

    if best_markov:
        f.write(f'**Best Markovian**: \`{best_markov[\"config_name\"]}\`\n')
        f.write(f'- {best_markov[\"achieved_gflops\"]:.2f} GFLOPS\n')
        f.write(f'- AI: {best_markov[\"arithmetic_intensity\"]:.2f} FLOPs/byte\n\n')

    f.write('## All Results\n\n')
    f.write('| Config | AI | GFLOPS | Bound | Efficiency |\n')
    f.write('|--------|-----|--------|-------|------------|\n')
    for r in sorted(results, key=lambda x: -x['achieved_gflops']):
        bound = 'Compute' if r['arithmetic_intensity'] >= A100_RIDGE else 'Memory'
        f.write(f'| {r[\"config_name\"]} | {r[\"arithmetic_intensity\"]:.2f} | ')
        f.write(f'{r[\"achieved_gflops\"]:.2f} | {bound} | {r[\"efficiency_percent\"]:.1f}% |\n')

print('Wrote results/roofline_summary.md')
"

echo "Done! Results in results/"
