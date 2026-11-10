#!/usr/bin/env python3
"""Plot every registered independent gain, including exact zero and negatives."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIXED = ('fixed_native_full', 'fixed_unbounded_best', 'fixed_mean_budget',
         'fixed_minimax_validation_regret')
LABELS = ('native full', 'unbounded best', 'mean cap', 'minimax regret')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.run_root
    out = args.output or root / 'reports/figures'
    sources = [root / 'confirmation/primary_confirmation_results.csv',
               root / 'rich_behavior/results/primary_confirmation.csv']
    for marker, flag in [('confirmation/confirmation_complete.json', 'completed'),
                         ('rich_behavior/results/evaluation_complete.json', 'complete')]:
        assert json.loads((root / marker).read_text()).get(flag) is True
    mcq, rich = [pd.read_csv(p) for p in sources]
    rich = rich[rich.fixed_policy.isin(FIXED)].rename(columns={
        'target_index': 'target_id', 'fixed_policy': 'comparator',
        'budget_averaged_gain_tv': 'gain_tv', 'simultaneous_low': 'simultaneous_ci_low',
        'simultaneous_high': 'simultaneous_ci_high', 'clusters': 'independent_test_clusters'})
    rich['target_id'] = rich.target_id.map({3: 'Qwen3-8B', 4: 'Qwen3-14B'})
    assert len(mcq) == len(rich) == 8
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    records = []
    for ax, data, title, caps in zip(axes, [mcq, rich],
                                    ['External academic MCQ', 'Full-vocabulary next-token'], [4, 8]):
        labels = []
        for target, group in data.groupby('target_id', sort=True):
            group = group.set_index('comparator').loc[list(FIXED)]
            for comp, label in zip(FIXED, LABELS):
                row = group.loc[comp]
                y = len(labels)
                labels.append(f"{str(target).split('/')[-1]} / {label}")
                low, high, value = row.simultaneous_ci_low, row.simultaneous_ci_high, row.gain_tv
                ax.hlines(y, low, high, color='#2166ac', linewidth=2)
                ax.plot(value, y, 'o', color='#2166ac', markersize=5)
                records.append({'route': title, 'target': target, 'comparator': comp,
                                'gain_tv': value, 'simultaneous_low': low,
                                'simultaneous_high': high,
                                'test_clusters': int(row.independent_test_clusters)})
        extent = max(.006, float(data.simultaneous_ci_low.abs().max()) * 1.2,
                     float(data.simultaneous_ci_high.abs().max()) * 1.2)
        ax.set_xlim(-extent, extent)
        ax.axvline(0, color='black', linewidth=.8)
        ax.axvline(.005, color='#b2182b', linestyle='--', linewidth=1)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('TV gain: fixed peer set − joint selection')
        ax.set_title(f'{title}\nWithin-cluster average over {caps} caps', fontsize=11)
    fig.suptitle('Independent confirmation: every registered strong-fixed comparison\n'
                 'Simultaneous 95% intervals within each eight-comparison family; dashed line: .005 design effect',
                 fontsize=11)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'fresh_confirmation_gains.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(records).to_csv(out / 'fresh_confirmation_gains.csv', index=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'fresh_confirmation_figure_manifest.json').write_text(json.dumps({
        'inputs': {str(p): digest(p) for p in sources},
        'outputs': {p.name: digest(p) for p in out.glob('fresh_confirmation_gains.*')},
        'scope': 'Frozen comparisons; query clusters are independent units. Overlapping seeds and caps are not extra replicates.'
    }, indent=2) + '\n')
    print(out / 'fresh_confirmation_gains.png')


if __name__ == '__main__':
    main()
