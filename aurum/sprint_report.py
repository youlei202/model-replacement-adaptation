"""Reproduce first-six AURUM findings from decoded development ledgers only.

No response cache, historical test, or independent confirmation data is opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/aurum'
LEGACY = ROOT / 'experiments/runs/v2/rich_behavior'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(frame, metric):
    return frame.sort_values([metric, 'actual_bytes', 'candidate_id']).iloc[0]


def markdown(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))


def report(run):
    run = Path(run).resolve()
    cells = sorted((run / 'development/cells').glob('*/candidate_ledger.csv'))
    assert len(cells) == 6, 'All six complete cells required'
    inputs = {}
    summaries, ablations, all_rows, resources = [], [], [], []
    status = json.loads((run / 'execution/first_six/status.json').read_text())
    assert status['complete'] and all(j['returncode'] == 0 for j in status['jobs'])
    for path in cells:
        done_path = path.parent / 'complete.json'
        done = json.loads(done_path.read_text())
        assert done['complete'] and done['outputs']['candidate_ledger.csv'] == sha(path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        frame = pd.read_csv(path)
        assert frame.status.eq('scored_decoded').all()
        target = int(frame.target_index.iloc[0]); pair = frame.pair.iloc[0]
        p0, p1 = map(int, pair.split(','))
        cheap = frame[frame.cheap.eq(True)]
        residual = frame[frame.ablation_variant.notna()]
        cval, cinner = choose(cheap, 'validation_tv'), choose(cheap, 'inner_stop_tv')
        new = choose(residual, 'inner_stop_tv')
        nnls = frame[frame.label.eq('legacy_inner_nnls')].iloc[0]
        old_path = LEGACY / 'fits' / f'target_{target}' / f'pair_{p0}_{p1}' / 'validation_candidates.parquet'
        old_frame = pd.read_parquet(old_path)
        inputs[str(old_path.relative_to(ROOT))] = sha(old_path)
        old = choose(old_frame[old_frame.family.isin(['ridge_residual', 'factored_residual'])], 'validation_tv')
        summary = {
            'cell': path.parent.name, 'target_index': target, 'target_id': frame.target_id.iloc[0], 'pair': pair,
            'cheap_outer_selected_label': cval.label, 'cheap_outer_selected_candidate': cval.candidate_id,
            'cheap_outer_validation_tv': cval.validation_tv, 'cheap_bytes': int(cval.actual_bytes),
            'cheap_inner_selected_label': cinner.label, 'cheap_inner_selected_validation_tv': cinner.validation_tv,
            'legacy_nnls_validation_tv': nnls.validation_tv,
            'old_residual_outer_selected_family': old.family, 'old_residual_outer_selected_rank': int(old.basis_rank),
            'old_residual_outer_selected_codec': old.codec, 'old_residual_outer_selected_validation_tv': old.validation_tv,
            'old_residual_bytes': int(old.actual_bytes), 'old_residual_fit_scope': 'historical_all2048_fit_outer_validation_selected',
            'new_residual_inner_selected_label': new.label, 'new_residual_rank': int(new.basis_rank),
            'new_residual_validation_tv': new.validation_tv, 'new_residual_inner_stop_tv': new.inner_stop_tv,
            'new_residual_fit_tv': new.fit_tv, 'new_residual_bytes': int(new.actual_bytes),
            'new_residual_public_numeric_bytes': int(new.public_numeric_bytes),
            'new_residual_candidate': new.candidate_id,
            'gain_vs_strongest_cheap_validation_tv': cval.validation_tv - new.validation_tv,
            'gain_vs_inner_selected_cheap_validation_tv': cinner.validation_tv - new.validation_tv,
            'relative_gain_vs_strongest_cheap_percent': 100 * (cval.validation_tv - new.validation_tv) / cval.validation_tv,
            'fit_gain_vs_strongest_cheap': cval.fit_tv - new.fit_tv,
            'gain_vs_legacy_nnls_validation_tv': nnls.validation_tv - new.validation_tv,
            'gain_vs_old_residual_validation_tv': old.validation_tv - new.validation_tv,
            'new_residual_selection': 'inner_stop_TV_only_among_L0_L4_rank32_128',
            'decoded_candidate_rows': len(frame), 'unique_decoded_states': frame.candidate_id.nunique(),
            'data_scope': 'development_original_fit_validation_not_independent_confirmation',
        }
        summaries.append(summary)
        for (variant, rank), group in residual.groupby(['ablation_variant', 'basis_rank']):
            selected = choose(group, 'inner_stop_tv').to_dict()
            selected.update(cell=path.parent.name, selection_rule='inner_stop_tv_then_bytes_then_candidate_digest',
                            strongest_cheap_validation_tv=cval.validation_tv,
                            gain_vs_strongest_cheap_validation_tv=cval.validation_tv-selected['validation_tv'])
            ablations.append(selected)
        frame['cell'] = path.parent.name; all_rows.append(frame)
        job = next(j for j in status['jobs'] if j['label'] == path.parent.name)
        resources.append({'cell':path.parent.name, 'GPU_uuid':job['GPU_uuid'], 'worker_pid':job['pid'],
                          'worker_wall_seconds':job['wall_seconds'], 'scientific_cell_seconds':done['seconds'],
                          'peak_torch_allocated_bytes':done['maximum_allocated_GPU_bytes'],
                          'candidate_rows':done['candidate_rows'], 'unique_states':done['unique_states'],
                          'scope':'cached_logit_adapter_training_and_validation; excludes_collection_and_serving'})
    summary = pd.DataFrame(summaries); ablation = pd.DataFrame(ablations); catalog = pd.concat(all_rows, ignore_index=True)
    assert set(zip(summary.target_index, summary.pair)) == {(t, p) for t in (3,4) for p in ('0,1','0,2','1,2')}
    summary.to_csv(run/'SIX_CELL_SUMMARY.csv', index=False)
    ablation.to_csv(run/'ANCHOR_LOSS_ABLATION.csv', index=False)
    catalog.to_csv(run/'FIRST_SIX_ALL_CANDIDATES.csv', index=False)
    pd.DataFrame(resources).to_csv(run/'audit/first_six_training_resources.csv', index=False)
    # The same catalog defines every pair's deployment frontier. All model/rank/
    # codec/anchor choices remain available to a fixed-peer comparator at each cap.
    frontier = []
    caps = sorted(catalog.actual_bytes.unique())
    for (target, pair), group in catalog.groupby(['target_index','pair']):
        for cap in caps:
            feasible = group[group.actual_bytes.le(cap)]
            if feasible.empty:
                continue
            row = choose(feasible, 'validation_tv').to_dict()
            frontier.append({'target_index':target,'pair':pair,'cap_bytes':int(cap),
                             'candidate_id':row['candidate_id'],'label':row['label'],
                             'selected_state_bytes':int(row['actual_bytes']),
                             'validation_tv':row['validation_tv'],
                             'selection_scope':'outer_validation_exploratory_shared_catalog'})
    frontier = pd.DataFrame(frontier)
    frontier.to_csv(run/'FIRST_SIX_STATE_FRONTIERS.csv', index=False)
    winners = []
    for (target,cap), group in frontier.groupby(['target_index','cap_bytes']):
        row=group.sort_values(['validation_tv','selected_state_bytes','pair']).iloc[0]
        winners.append(row.to_dict())
    winners=pd.DataFrame(winners); winners.to_csv(run/'FIRST_SIX_PAIR_WINNERS.csv', index=False)
    make_figures(run, ablation, summary, frontier)
    table=summary[['cell','cheap_outer_validation_tv','new_residual_validation_tv','gain_vs_strongest_cheap_validation_tv','relative_gain_vs_strongest_cheap_percent']].copy()
    table.columns=['Cell','Strong cheap TV','New residual TV','TV gain','Relative gain (%)']
    for column in table.columns[1:]: table[column]=table[column].map(lambda x:f'{x:.6f}')
    positive=int(summary.gain_vs_strongest_cheap_validation_tv.gt(0).sum())
    best=summary.sort_values('gain_vs_strongest_cheap_validation_tv',ascending=False).iloc[0]
    dominant = {str(t): sorted(g.pair.unique()) for t,g in winners.groupby('target_index')}
    resource=pd.DataFrame(resources)
    text=f'''# AURUM first-six findings

All six cells completed and every reported candidate was decoded before scoring. The strongest positive first-sprint result is **{best['cell']}**: the inner-selected rank-{int(best['new_residual_rank'])} residual lowers validation TV from {best['cheap_outer_validation_tv']:.6f} for the strongest cheap calibration to {best['new_residual_validation_tv']:.6f}, an absolute gain of {best['gain_vs_strongest_cheap_validation_tv']:.6f} ({best['relative_gain_vs_strongest_cheap_percent']:.3f}% relative). This is exploratory development evidence on existing cached documents, not independent confirmation.

{markdown(table)}

Positive, zero, and negative values are retained: {positive}/6 cells improve over the strongest cheap control; {6-positive}/6 do not. The strongest cheap control is selected on outer validation from every decoded cheap candidate, including continuous KL/TV calibration and retained historical full-fit anchors. Residual variant and rank in the main table are chosen only on inner-stop TV. `SIX_CELL_SUMMARY.csv` also reports a symmetric inner-selected cheap comparison, the old residual result, state bytes, and fit gains.

## Mechanism and what to do next

The new distribution objective and better starting anchor remove much of the old projected-MSE residual's degradation. `ANCHOR_LOSS_ABLATION.csv` gives all five factorial variants at both ranks in every cell, selecting ridge/TV-refinement alternatives only on inner-stop TV. The retained `FIRST_SIX_ALL_CANDIDATES.csv` includes every intermediate positive, zero, and negative candidate, including alternatives not chosen for the main table.

However, best new residual fit-TV improvements over the strongest cheap anchor span {summary.fit_gain_vs_strongest_cheap.min():.6f}–{summary.fit_gain_vs_strongest_cheap.max():.6f}, while validation gains span {summary.gain_vs_strongest_cheap_validation_tv.min():.6f}–{summary.gain_vs_strongest_cheap_validation_tv.max():.6f}. The added linear capacity therefore fits a signal, but much less of it transfers to held-out documents. This motivates a focused distinction between limited output-basis coverage and statistical overfitting: measure target-residual projection coverage on inner-fit versus validation, then test a fit-only target-residual output basis with its basis bytes explicitly charged. A low-rank conditional peer-weight gate is a complementary small-state mechanism if corrections mostly reweight existing peer distributions. These diagnostics should retain the same calibration menu for all peers; changing seeds or dropping strong controls would not resolve the mechanism.

The first state frontier selects pair(s) {json.dumps(dominant, sort_keys=True)} across the observed caps. `FIRST_SIX_STATE_FRONTIERS.csv` lets each fixed pair reselect its entire shared menu at every cap. A common best pair across caps implies zero peer-selection penalty for that target within this first-sprint catalog; the small within-pair residual gains alone do not establish a practical resource disadvantage for a strong fixed peer choice. This frontier is exploratory and does not substitute for full tuned catalog comparison or fresh confirmation.

## Scope, state accounting, and measured training cost

The main factorial uses 1364 inner-fit documents and 684 inner-stop documents; both originate in the original 2048-document fit role. Outer validation has 1024 documents. Historical residual comparison uses the old validation-only ledger and was fitted on all 2048 fit documents, so it is a historical reference with a distinct fit/selection scope, not a controlled ablation. No historical test responses or independent new test responses were opened by this reporting script.

The first sprint uses FP32 rank 32/128 adapters in the original peer-only output basis. Encoded adapter sizes are 8,420 and 131,684 bytes; cheap anchors use 60 bytes. Public context storage is separately recorded in every row and must be charged in applicable deployment totals; it is not silently removed from a real resource claim. Encoded adapter state and cached training GPU memory are different quantities.

The six cached-logit workers consumed {resource.worker_wall_seconds.sum():.3f} summed GPU-worker wall seconds, with individual cells taking {resource.worker_wall_seconds.min():.3f}–{resource.worker_wall_seconds.max():.3f} seconds and peak PyTorch allocation {resource.peak_torch_allocated_bytes.max()/2**30:.3f} GiB. These measurements exclude language-model response collection and serving. Exact per-cell measurements are in `audit/first_six_training_resources.csv`; sampled device utilization/memory/power observations are in `audit/first_six_resource_samples.jsonl`. Serving latency, deployment residency, quality-matched resource crossings, and independent confirmation remain separate experiments.

Reproduce this report with `.venv-benchmarks/bin/python -B -m aurum.sprint_report`. Scientific plots are in `figures/first_six_anchor_loss.png` and `figures/first_six_state_frontiers.png`.
'''
    (run/'SPRINT_FINDINGS.md').write_text(text)
    (run/'SPRINT_REPORT_PROVENANCE.json').write_text(json.dumps({'report_source_sha256':sha(Path(__file__)),
        'inputs':inputs,'candidate_rows':len(catalog),'unique_states':catalog.candidate_id.nunique(),
        'historical_test_opened':False,'confirmation_opened':False,'scope':'exploratory_original_fit_validation'},indent=2)+'\n')
    print(json.dumps({'cells':len(summary),'candidate_rows':len(catalog),'positive_strong_control_cells':positive,
                      'report':str(run/'SPRINT_FINDINGS.md'),'pair_winners':dominant}))


def make_figures(run, ablation, summary, frontier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    folder=run/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    labels=['L0','L1','L2','L3','L4']
    for axis,(_,row) in zip(axes.flat,summary.iterrows()):
        group=ablation[ablation.cell.eq(row.cell)]
        for rank,sub in group.groupby('basis_rank'):
            ordered=sub.set_index('ablation_variant').reindex(labels)
            axis.plot(range(5),ordered.validation_tv,marker='o',label=f'rank {rank}')
        axis.axhline(row.cheap_outer_validation_tv,color='black',ls='--',label='strong cheap')
        axis.set_title(row.cell);axis.set_xticks(range(5),labels);axis.set_ylabel('Validation TV (lower better)')
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Anchor × objective: decoded states, inner-stop selection within each variant\nL0 NNLS+MSE; L1 better anchor+MSE; L2 NNLS+KL; L3 better anchor+KL; L4 TV refinement',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'first_six_anchor_loss.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,(target,group) in zip(axes,frontier.groupby('target_index')):
        for pair,sub in group.groupby('pair'):
            axis.step(sub.cap_bytes,sub.validation_tv,where='post',marker='o',label=f'peers {pair}')
        axis.set_xscale('log');axis.set_xlabel('Encoded adapter state cap (bytes)')
        axis.set_ylabel('Best validation TV');axis.set_title(f'Target {target}: shared menu for all pairs')
        axis.legend();axis.grid(alpha=.2)
    fig.suptitle('Exploratory first-sprint frontier; public context and serving costs separate',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'first_six_state_frontiers.png',dpi=200);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUN)
    report(parser.parse_args().run_root)
