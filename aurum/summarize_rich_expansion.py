"""Full six-cell stronger-menu summaries and exploratory source decomposition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .sprint import ROOT, dump, sha
from .data import RichDevelopmentCache


def main():
    run = ROOT / 'runs/aurum'; items = RichDevelopmentCache().items('validation')
    rows = []; sources = []; hashes = {}
    for folder in sorted((run / 'development/expanded_rich_v1').glob('t*')):
        if not (folder / 'complete.json').exists(): raise ValueError('Incomplete expansion cell')
        d = pd.read_csv(folder / 'candidate_ledger.csv')
        c = d[d.cheap].sort_values(['validation_tv', 'actual_bytes']).iloc[0]
        r = d[~d.cheap].sort_values(['validation_tv', 'actual_bytes']).iloc[0]
        i = d[~d.cheap].sort_values(['inner_stop_tv', 'actual_bytes']).iloc[0]
        meta = json.loads((folder / 'complete.json').read_text())
        hashes[str(folder / 'candidate_ledger.csv')] = sha(folder / 'candidate_ledger.csv')
        rows.append({'cell': folder.name, 'cheap_tv': c.validation_tv, 'cheap_label': c.label,
            'best_residual_tv': r.validation_tv, 'best_residual_label': r.label, 'codec': r.codec,
            'bytes': int(r.actual_bytes), 'validation_selected_gain': c.validation_tv - r.validation_tv,
            'inner_selected_residual_tv': i.validation_tv, 'inner_selected_gain': c.validation_tv - i.validation_tv,
            'seconds': meta['seconds'], 'maximum_allocated_GPU_bytes': meta['maximum_allocated_GPU_bytes'],
            'candidate_rows': len(d), 'cheap_state': c.candidate_id, 'residual_state': r.candidate_id,
            'inner_selected_residual_state': i.candidate_id, 'scope': 'exploratory_original_fit_validation'})
        with np.load(Path(c.state_path).with_suffix('.losses.npz')) as data: a = data['validation_tv']
        with np.load(Path(r.state_path).with_suffix('.losses.npz')) as data: b = data['validation_tv']
        for source, idx in items.groupby('source').groups.items():
            sources.append({'cell': folder.name, 'source': source, 'n': len(idx),
                'cheap_tv': float(a[idx].mean()), 'residual_tv': float(b[idx].mean()),
                'gain': float((a - b)[idx].mean()), 'cheap_state': c.candidate_id, 'residual_state': r.candidate_id,
                'scope': 'posthoc_source_breakdown_of_global_validation_selected_candidates'})
    pd.DataFrame(rows).to_csv(run / 'RICH_EXPANSION_SUMMARY.csv', index=False)
    pd.DataFrame(sources).to_csv(run / 'RICH_EXPANSION_BY_SOURCE.csv', index=False)
    dump(run / 'RICH_EXPANSION_SUMMARY.provenance.json', {'input_hashes': hashes,
        'complete_cells': len(rows), 'expected_cells': 6, 'complete': len(rows) == 6,
        'independent_confirmation': False, 'source_strata_were_not_used_to_select_candidates': True})


if __name__ == '__main__': main()
