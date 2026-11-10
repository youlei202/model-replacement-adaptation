"""Reporting-only guard must never rescue a frozen resource failure."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import global_resource_cap_guard as guard


def test_bound_is_monotone_and_requires_complete_blocks():
    assert guard.upper_bound([1.] * 10, 2.) == 2.
    assert guard.upper_bound([1.] * 9, 2.) is None
    assert guard.upper_bound([np.nan] * 10, 2.) is None
    assert guard.upper_bound(range(1, 11), 6.) >= 6.


def test_both_routes_preserve_frozen_flags_and_tighten(tmp_path):
    def dump(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    dump(tmp_path/'protocol/global_resource_cap_report_guard.json',
         {'analysis_script_sha256': guard.sha(guard.__file__)})
    for rich in [False, True]:
        base = tmp_path / ('rich_behavior/matched_resource_supplement' if rich else 'resource_cap_confirmation')
        base.mkdir(parents=True)
        decisions, results, power = [], [], []
        for target in [1, 3]:
            for policy in ['joint' if rich else 'budget_aware', *guard.FIXED]:
                decisions.append({'target_index': target, 'policy': policy, 'cap_id': 'C',
                    'candidate_id' if rich else 'system_id': 's',
                    'mean_wall_latency_cap_ms' if rich else 'latency_cap_ms_per_context': 6.,
                    'allocated_VRAM_cap_bytes': 1000})
            for i, policy in enumerate(guard.FIXED):
                results.append({'target_index': target, 'fixed_policy' if rich else 'comparator': policy,
                    'common_feasible_caps': 'C', 'same_resource_advantage_confirmed': i != 0})
                power.append({'target_index': target, 'comparator': policy, 'common_feasible_cap_ids': ['C']})
        dump(base/('prospective_lock.json' if rich else 'selection_lock.json'),
             {'decisions' if rich else 'choices': decisions, 'power': power})
        pd.DataFrame(results).to_csv(base/('primary_confirmation.csv' if rich else 'guarded_confirmation_results.csv'), index=False)
        pd.DataFrame([{'candidate_id' if rich else 'system_id': 's', 'batch_size': 1, 'context_tokens': 0,
            'mean_wall_upper_simultaneous' if rich else 'latency_familywise_upper_one_sided': 5.5,
            'peak_allocated_bytes': 900}]).to_csv(base/'resource_summary.csv', index=False)
        rows = []
        for block in range(10):
            if rich:
                rows.extend({'system_digest': 's', 'batch_size': 1, 'context_tokens': 0,
                    'timing_block': block, 'request': request, 'end_to_end_ms': block+1.} for request in range(20))
            else:
                rows.append({'system_id': 's', 'batch_size': 1, 'context_tokens': 0,
                    'timing_block': block, 'failure_status': None,
                    'traces': [{'end_to_end_ms': block+1., 'context_count': 1} for _ in range(20)]})
        raw = base/'timing_raw.jsonl'
        raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        dump(base/('supplement_complete.json' if rich else 'complete.json'),
             {'complete': True, 'raw_measurements': {str(raw): guard.sha(raw)}})
    claims = guard.run_audit(tmp_path)
    assert claims.primary_registered.sum() == 16
    assert claims.frozen_route_same_resource_advantage_confirmed.sum() == 12
    assert not claims.global_all_common_resource_caps_confirmed.any()
    assert not claims.global_same_resource_advantage_confirmed.any()
