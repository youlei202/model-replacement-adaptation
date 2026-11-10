"""Reproduce all signed rich learning-curve and split-sensitivity summaries."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .sprint import ROOT, dump, sha


def main():
    run = ROOT / 'runs/aurum'
    summary = []; frontiers = []; hashes = {}
    caps = [128, 512, 2048, 8192, 32768, 131072, 524288, 2097152]
    for folder in sorted((run / 'development/learning_curve_v1').glob('t*/n*')):
        if not (folder / 'complete.json').exists(): continue
        spec = json.loads((folder / 'run_spec.json').read_text())
        d = pd.read_csv(folder / 'candidate_ledger.csv')
        hashes[str(folder / 'candidate_ledger.csv')] = sha(folder / 'candidate_ledger.csv')
        cheap = d[d.cheap]; residual = d[~d.cheap]
        def select(frame, role):
            return frame.sort_values([role + '_tv', 'actual_bytes', 'candidate_id']).iloc[0]
        c = select(cheap, 'validation'); r = select(residual, 'validation')
        ci = select(cheap, 'inner_stop'); ri = select(residual, 'inner_stop')
        record = {'target_index': int(d.iloc[0].target_index), 'pair': d.iloc[0].pair,
                  'n_train': spec['n_train'], 'seed': spec['seed'],
                  'primary_or_sensitivity': 'primary_sample_size' if spec['seed'] == 20261110 else 'split_optimizer_sensitivity',
                  'cheap_validation_tv': c.validation_tv, 'residual_validation_tv': r.validation_tv,
                  'outer_envelope_gain': c.validation_tv - r.validation_tv,
                  'inner_selected_cheap_validation_tv': ci.validation_tv,
                  'inner_selected_residual_validation_tv': ri.validation_tv,
                  'inner_selected_gain': ci.validation_tv - ri.validation_tv,
                  'residual_fit_tv': ri.fit_tv, 'cheap_fit_tv': ci.fit_tv,
                  'residual_inner_stop_tv': ri.inner_stop_tv, 'cheap_inner_stop_tv': ci.inner_stop_tv,
                  'residual_label': r.label, 'residual_codec': r.codec, 'residual_bytes': int(r.actual_bytes),
                  'cheap_candidate_id': c.candidate_id, 'residual_candidate_id': r.candidate_id,
                  'inner_selected_cheap_id': ci.candidate_id, 'inner_selected_residual_id': ri.candidate_id,
                  'candidate_rows': len(d), 'folder': str(folder),
                  'role': 'development_only_not_independent_confirmation'}
        summary.append(record)
        for cap in caps:
            for menu, frame in [('cheap', cheap), ('full', d)]:
                chosen = select(frame[frame.actual_bytes.le(cap)], 'validation')
                frontiers.append({'target_index': record['target_index'], 'pair': record['pair'],
                    'n_train': spec['n_train'], 'seed': spec['seed'], 'menu': menu, 'cap_bytes': cap,
                    'candidate_id': chosen.candidate_id, 'validation_tv': chosen.validation_tv,
                    'actual_bytes': int(chosen.actual_bytes), 'label': chosen.label})
    frame = pd.DataFrame(summary)
    frame.to_csv(run / 'RICH_LEARNING_CURVE_SUMMARY.csv', index=False)
    if len(frame):
        frame[frame.n_train.eq(3412)].to_csv(run / 'RICH_SEED_SENSITIVITY.csv', index=False)
    pd.DataFrame(frontiers).to_csv(run / 'RICH_LEARNING_CURVE_FRONTIERS.csv', index=False)
    dump(run / 'RICH_LEARNING_CURVE_SUMMARY.provenance.json', {'completed_cells': len(summary),
         'expected_cells': 36, 'complete': len(summary) == 36, 'input_hashes': hashes,
         'primary_seed': 20261110, 'extra_seeds_are_sensitivity_not_independent_observations': True,
         'outer_envelopes_selected_on_same_validation_data_and_exploratory': True,
         'main_catalog_must_not_select_best_fit_seed': True})
    print(frame[['target_index', 'pair', 'n_train', 'seed', 'outer_envelope_gain', 'inner_selected_gain']].to_string(index=False))


if __name__ == '__main__': main()
