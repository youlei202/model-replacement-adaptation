#!/usr/bin/env python3
"""Explain retained full-audit CPU/GPU argmax discrepancies without changing losses."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rich_setup import OUT, digest, dump
from rich_codec import decode
SUP = OUT / 'matched_resource_supplement'


def main():
    lock = json.load(open(SUP / 'prospective_lock.json'))
    discrepancy = json.load(open(SUP / 'independent_CPU_discrete_discrepancies.json'))
    states = {r['candidate_id']: r for r in lock['decisions'] if r['feasible_on_development']}
    indices = {item: i for i, item in enumerate(lock['test_item_ids'])}
    banks = {i: np.load(SUP / 'logits' / f'model_{i}.npy', mmap_mode='r') for i in range(3)}
    rows = []
    for d in discrepancy:
        if d['metric'] != 'prediction_argmax':
            continue
        state = decode((OUT / states[d['candidate_id']]['state_path']).read_bytes())
        idx = indices[d['item_id']]
        p32 = sum(np.float32(w) * banks[p][idx] for p, w in zip(state['peers'], state['arrays'][0]))
        p64 = sum(float(w) * np.asarray(banks[p][idx], dtype=np.float64) for p, w in zip(state['peers'], state['arrays'][0]))
        a, b = d['CPU_FP64_value'], d['recorded_GPU_FP32_value']
        rows.append({**d, 'CPU_FP32_prediction_argmax': int(p32.argmax()),
                     'CPU_FP32_matches_recorded_GPU_FP32': int(p32.argmax()) == b,
                     'two_candidate_FP32_logits_exactly_tied': bool(p32[a] == p32[b]),
                     'GPU_choice_is_lower_token_index': b < a,
                     'CPU_FP64_logit_gap_between_disputed_tokens': float(p64[a] - p64[b]),
                     'FP32_logit_at_GPU_choice': float(p32[b]),
                     'FP32_logit_at_CPU_FP64_choice': float(p32[a]),
                     'scope': 'Same saved peer logits and decoded weights; only arithmetic precision differs. Frozen scientific losses are unchanged.'})
    dump(SUP / 'independent_argmax_tie_diagnostics.json', {
        'posthoc_diagnostic_only_no_reselection': True, 'cases': rows,
        'all_four_reproduced_as_exact_FP32_ties': len(rows) == 4 and all(r['two_candidate_FP32_logits_exactly_tied'] and r['CPU_FP32_matches_recorded_GPU_FP32'] for r in rows),
        'audit_source_sha256': digest(__file__),
        'discrepancy_source_sha256': digest(SUP / 'independent_CPU_discrete_discrepancies.json')})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
