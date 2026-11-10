"""Scientific confirmation curves; run using the existing plotting environment."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--summary', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); frame = pd.read_csv(args.summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = ['#4974a5', '#ba7042', '#328266']
    for ax, target in zip(axes, (3, 4)):
        data = frame[frame.target_index == target]
        for color, pair in zip(colors, ('0_1', '0_2', '1_2')):
            rows = data[data.policy.eq('fixed_pair_' + pair)].sort_values('cap_bytes')
            cheap = data[data.policy.eq('cheap_pair_' + pair)].sort_values('cap_bytes')
            ax.plot(rows.cap_bytes, rows.tv, marker='o', markersize=3, color=color, label='Pair ' + pair.replace('_', ','))
            ax.plot(cheap.cap_bytes, cheap.tv, linestyle=':', color=color, alpha=.7)
        rows = data[data.policy.eq('joint')].sort_values('cap_bytes')
        ax.plot(rows.cap_bytes, rows.tv, color='black', linewidth=1.2, linestyle='--', label='Joint')
        ax.set_xscale('log', base=2); ax.set_xlabel('Target-specific state cap (bytes)')
        ax.set_title('Qwen3-' + ('8B' if target == 3 else '14B')); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Fresh confirmation mean TV (lower is better)')
    fig.suptitle('Identical fully retuned menus; dotted lines: strong cheap calibration', fontsize=11)
    fig.tight_layout(); path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == '__main__': main()
