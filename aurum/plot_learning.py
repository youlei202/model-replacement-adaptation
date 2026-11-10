"""Scientific learning-curve figure from the signed primary-seed table."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/aurum'
    d = pd.read_csv(run / 'RICH_LEARNING_CURVE_SUMMARY.csv')
    d = d[d.seed.eq(20261110)]
    if len(d) != 24: raise ValueError('All primary sample-size cells required')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout='constrained')
    for row, target in enumerate((3, 4)):
        for col, pair in enumerate(('0,1', '0,2', '1,2')):
            part = d[d.target_index.eq(target) & d.pair.eq(pair)].sort_values('n_train')
            axis = axes[row, col]
            axis.plot(part.n_train, part.cheap_validation_tv, 'o-', color='#757575', label='Strong cheap calibration')
            axis.plot(part.n_train, part.residual_validation_tv, 's-', color='#0072B2', label='Richer adapter')
            axis.set_xscale('log', base=2); axis.set_xticks(part.n_train, part.n_train.astype(str))
            axis.set_title(f'Target {target}: Qwen3-{8 if target == 3 else 14}B; peers {pair}')
            axis.set_ylabel('Validation total variation'); axis.set_xlabel('Training documents')
            axis.grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Additional development data: fixed representation, refitted strong controls\nExploratory validation-selected endpoints; no independent confirmation', fontsize=13)
    path = run / 'figures'; path.mkdir(exist_ok=True)
    fig.savefig(path / 'rich_learning_curve.png', dpi=180)
    fig.savefig(path / 'rich_learning_curve.pdf')
    plt.close(fig)


if __name__ == '__main__': main()
