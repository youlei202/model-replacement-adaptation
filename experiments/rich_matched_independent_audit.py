#!/usr/bin/env python3
"""CPU-only posthoc full-table and numerical audit; never changes frozen outcomes."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rich_setup import OUT, digest, dump
from rich_codec import decode
SUP = OUT / 'matched_resource_supplement'


def main():
    start = time.time()
    lock = json.load(open(SUP / 'prospective_lock.json'))
    completion = json.load(open(SUP / 'supplement_complete.json'))
    for path, h in lock['inputs_sha256'].items():
        assert digest(ROOT / path) == h, path
    for path, h in completion['outputs_sha256'].items():
        assert digest(SUP / path) == h, path
    docs = pd.read_parquet(OUT / 'documents.parquet')
    cohort = pd.read_parquet(SUP / 'cohort.parquet')
    old = json.load(open(OUT / 'selection_lock.json'))
    exposed = docs[~docs.role.eq('test_pool') | docs.item_id.isin(old['test_item_ids'])]
    assert not set(cohort.cluster_id) & set(exposed.cluster_id)
    assert not set(cohort.item_id) & set(exposed.item_id)
    assert cohort.item_id.tolist() == lock['test_item_ids']
    assert cohort.cluster_id.tolist() == lock['test_cluster_ids']
    assert cohort.cluster_id.nunique() == len(cohort) == 2048
    assert not cohort.historical_overlap.any()
    reservoir = docs[docs.role.eq('test_pool') & ~docs.cluster_id.isin(exposed.cluster_id)].copy()
    reservoir['supplement_order_hash'] = reservoir.item_id.map(lambda x: hashlib.sha256(f'81724061|{x}'.encode()).hexdigest())
    expected = pd.concat([g.sort_values('supplement_order_hash').head(1024) for _, g in reservoir.groupby('source')]).sort_values('supplement_order_hash')
    assert expected.item_id.tolist() == cohort.item_id.tolist()
    cross = json.load(open(SUP / 'concurrent_new_mcq_cross_audit.json'))
    assert cross['passed'] and cross['cross_collision_count'] == 0
    assert digest(ROOT / cross['mcq_reservoir_path']) == cross['mcq_reservoir_sha256']
    selected = {(r['target_index'], r['candidate_id']): r for r in lock['decisions'] if r['feasible_on_development']}
    assert len(selected) == 12
    states = {}
    for (_, sid), row in selected.items():
        blob = (OUT / row['state_path']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == sid
        states[sid] = decode(blob)
        assert states[sid]['basis_rank'] == 0 and set(states[sid]['peers']) <= {0, 1, 2}
    assert len(states) == 8
    losses = pd.read_parquet(SUP / 'item_losses.parquet')
    assert len(losses) == 24576 and not losses.duplicated(['target_index', 'candidate_id', 'item_id']).any()
    assert np.isfinite(losses[['tv', 'kl', 'target_margin']]).all().all()
    assert losses.tv.between(0, 1).all() and losses.kl.ge(0).all()
    assert set(zip(losses.target_index, losses.candidate_id)) == set(selected)
    lookup = {}
    for key, frame in losses.groupby(['target_index', 'candidate_id']):
        aligned = frame.set_index('item_id').loc[lock['test_item_ids']].reset_index()
        assert len(aligned) == 2048 and aligned.cluster_id.tolist() == cohort.cluster_id.tolist()
        assert np.array_equal(aligned.argmax_disagreement, (aligned.prediction_argmax != aligned.target_argmax).astype(float))
        lookup[key] = aligned
    policies = pd.read_parquet(SUP / 'policy_item_losses.parquet')
    assert len(policies) == 344064 and not policies.duplicated(['target_index', 'policy', 'cap_id', 'item_id']).any()
    assert policies.groupby(['target_index', 'policy', 'cap_id']).size().eq(2048).all()
    decisions = {(r['target_index'], r['policy'], r['cap_id']): r for r in lock['decisions']}
    for key, frame in policies.groupby(['target_index', 'policy', 'cap_id']):
        decision = decisions[key]
        assert frame.candidate_id.nunique() == 1 and frame.candidate_id.iloc[0] == decision['candidate_id']
        a = frame.set_index('item_id').loc[lock['test_item_ids']]
        b = lookup[(key[0], decision['candidate_id'])]
        for metric in ['tv', 'kl', 'argmax_disagreement', 'prediction_argmax', 'target_argmax', 'target_margin']:
            assert np.array_equal(a[metric].to_numpy(), b[metric].to_numpy()), (key, metric)
    raw = [json.loads(line) for line in (SUP / 'timing_raw.jsonl').read_text().splitlines()]
    assert len(raw) == 9600
    ids = {(r['system_digest'], r['batch_size'], r['context_tokens'], r['timing_block'], r['request']) for r in raw}
    expected_ids = {(sid, b, c, block, req) for sid in states for b in [1, 8] for c in [0, 128, 512] for block in range(10) for req in range(20)}
    assert ids == expected_ids and len(ids) == len(raw)
    for row in raw:
        state = states[row['system_digest']]
        active = [p for p, w in zip(state['peers'], state['arrays'][0]) if w != 0]
        assert row['peer_ids'] == active and row['actual_call_q'] == len(active)
        assert row['target_absent'] and row['failure_status'] is None
        assert row['complete_pipeline'] and row['inference_microbatch'] == 1
        assert row['end_to_end_ms'] > 0 and row['GPU_ms'] > 0
    cohort_hash = digest(SUP / 'cohort.parquet')
    assert all(row['fresh_timing_cohort_sha256'] == cohort_hash for row in raw)
    replay = json.load(open(SUP / 'deployment_replay.json'))
    assert replay['complete'] and {r['candidate_id'] for r in replay['records']} == set(states)
    assert all(r['max_full_support_TV'] == 0 and not r['target_loaded'] for r in replay['records'])
    for folder in [SUP / 'mini_replay', OUT / 'reproduce/mini_replay']:
        manifest = json.load(open(folder / 'manifest.json'))
        for path, h in manifest['hashes'].items():
            assert digest(folder / path) == h, path
    banks = {}
    for i in range(5):
        path = SUP / 'logits' / f'model_{i}.npy'; meta = json.load(open(path.with_suffix('.json')))
        assert meta['complete'] and meta['completed'] == 2048 and meta['sha256'] == digest(path)
        banks[i] = np.load(path, mmap_mode='r')
        assert banks[i].shape == (2048, 151936) and banks[i].dtype == np.float32
    diagnostics = {key: {'max_abs_TV_difference': 0., 'max_abs_KL_difference': 0., 'max_abs_target_margin_difference': 0.,
                         'target_argmax_mismatches': 0, 'prediction_argmax_mismatches': 0,
                         'argmax_disagreement_mismatches': 0, 'rows_checked': 0} for key in selected}
    discrepancies = []
    for begin in range(0, 2048, 16):
        stop = min(begin + 16, 2048)
        x = {i: np.asarray(bank[begin:stop], dtype=np.float64) for i, bank in banks.items()}
        teacher_logp = {i: x[i] - logsumexp(x[i], axis=1, keepdims=True) for i in [3, 4]}
        teacher_p = {i: np.exp(teacher_logp[i]) for i in [3, 4]}
        for sid, state in states.items():
            assert state['family'] in ['convex', 'nnls', 'temperature']
            # Predictor consumes peers only; teacher arrays are accessed below
            # solely to independently verify the already recorded losses.
            prediction = sum(float(w) * x[p] for p, w in zip(state['peers'], state['arrays'][0]))
            logq = prediction - logsumexp(prediction, axis=1, keepdims=True)
            q = np.exp(logq); predicted = prediction.argmax(axis=1)
            for target in [3, 4]:
                key = (target, sid)
                if key not in selected:
                    continue
                saved = lookup[key].iloc[begin:stop]
                p = teacher_p[target]; lp = teacher_logp[target]
                tv = .5 * np.abs(p - q).sum(axis=1); kl = np.maximum((p * (lp - logq)).sum(axis=1), 0)
                target_argmax = x[target].argmax(axis=1); top = np.partition(p, -2, axis=1)[:, -2:]
                margin = top.max(axis=1) - top.min(axis=1)
                d = diagnostics[key]
                d['max_abs_TV_difference'] = max(d['max_abs_TV_difference'], float(np.max(np.abs(tv - saved.tv.to_numpy()))))
                d['max_abs_KL_difference'] = max(d['max_abs_KL_difference'], float(np.max(np.abs(kl - saved.kl.to_numpy()))))
                d['max_abs_target_margin_difference'] = max(d['max_abs_target_margin_difference'], float(np.max(np.abs(margin - saved.target_margin.to_numpy()))))
                for name, actual, reference in [('target_argmax', target_argmax, saved.target_argmax.to_numpy()),
                                                ('prediction_argmax', predicted, saved.prediction_argmax.to_numpy()),
                                                ('argmax_disagreement', predicted != target_argmax, saved.argmax_disagreement.to_numpy())]:
                    mismatches = np.flatnonzero(actual != reference)
                    d[name + '_mismatches'] += len(mismatches)
                    for j in mismatches:
                        discrepancies.append({'target_index': target, 'candidate_id': sid, 'item_id': saved.item_id.iloc[j],
                                              'metric': name, 'CPU_FP64_value': int(actual[j]), 'recorded_GPU_FP32_value': int(reference[j])})
                d['rows_checked'] += len(saved)
        if stop % 256 == 0:
            print('Full CPU numerical audit', stop, '/2048 documents x12 states', flush=True)
    numerical = [{'target_index': key[0], 'candidate_id': key[1], **d,
                  'strict_TV_1e_6_passed': d['max_abs_TV_difference'] <= 1e-6} for key, d in diagnostics.items()]
    pd.DataFrame(numerical).to_csv(SUP / 'independent_CPU_numerical_audit.csv', index=False)
    dump(SUP / 'independent_CPU_discrete_discrepancies.json', discrepancies)
    policy_resources = pd.read_csv(SUP / 'policy_quality_resources.csv')
    audit = {'complete': True, 'posthoc_independent_audit_no_reselection': True,
             'CPU_only': True, 'numpy': np.__version__, 'elapsed_wall_seconds': time.time() - start,
             'frozen_input_hashes_verified': len(lock['inputs_sha256']), 'scientific_output_hashes_verified': len(completion['outputs_sha256']),
             'all_five_full_bank_hashes_verified': True, 'new_independent_clusters': 2048,
             'old_exposed_cluster_overlap': 0, 'deterministic_reservoir_sampling_reproduced': True,
             'source_counts': cohort.source.value_counts().to_dict(), 'cross_audit_MCQs': cross['mcq_reservoir_rows'],
             'target_states': len(selected), 'unique_state_blobs': len(states), 'query_state_rows': len(losses),
             'policy_query_rows': len(policies), 'timing_rows': len(raw), 'timing_settings_per_state': 6,
             'same_B200_fresh_live_replay_max_TV': 0., 'both_mini_replay_hash_manifests_verified': True,
             'full_CPU_FP64_reconstruction_rows': sum(d['rows_checked'] for d in numerical),
             'max_CPU_FP64_vs_GPU_FP32_TV_difference': max(d['max_abs_TV_difference'] for d in numerical),
             'max_CPU_FP64_vs_GPU_FP32_KL_difference': max(d['max_abs_KL_difference'] for d in numerical),
             'strict_TV_1e_6_all_passed': all(d['strict_TV_1e_6_passed'] for d in numerical),
             'discrete_cross_backend_discrepancies': len(discrepancies),
             'retained_policy_rows_observed_latency_cap_breach': int((~policy_resources.fresh_latency_cap_met_observed).sum()),
             'retained_policy_rows_upper_latency_cap_breach': int((~policy_resources.fresh_latency_cap_met_upper_bound).sum()),
             'retained_policy_rows_memory_cap_breach': int((~policy_resources.fresh_memory_cap_met).sum()),
             'breach_count_scope': 'Policy rows share states/caps and are not independent failed runs.',
             'numerical_scope': 'NumPy FP64 canonical full-support probabilities versus frozen PyTorch/B200 FP32 metric arithmetic. Any strict tolerance or argmax disagreement is recorded without editing scientific losses or relaxing tolerance.',
             'completion_marker_sha256': digest(SUP / 'supplement_complete.json'), 'audit_code_sha256': digest(__file__),
             'numerical_csv_sha256': digest(SUP / 'independent_CPU_numerical_audit.csv'),
             'discrete_discrepancy_file_sha256': digest(SUP / 'independent_CPU_discrete_discrepancies.json')}
    dump(SUP / 'independent_audit.json', audit)
    primary = pd.read_csv(SUP / 'primary_confirmation.csv')
    lines = ['# Final rich matched-resource supplement summary', '',
             'The requested measured-resource-cap experiment is complete. It was prospectively registered after the original experiments, uses2048 previously uncollected document clusters, and retains all original results.', '',
             'Across12 frozen latency/allocated-VRAM caps, the eight primary comparisons show no design-sized positive advantage. For Qwen3-8B, fixed-minus-joint TV averages -0.00031360; family16 simultaneous interval approximately[-0.000674,0.000036]. For Qwen3-14B the gain and interval are exactly0. The original four fixed policies and the additional once-across-grid fixed pair use peers1/2; every adapter is independently retuned under the same caps. The cheap joint menu matches joint selection.', '',
             'Eight distinct45–48-byte scalar states cover12 target/state combinations. The actual memory/latency frontier comparison does not establish retained-state-budget consequence. Absolute fidelity, decision losses, state/model/shared memory and every cap are available in policy_quality_resources.csv.', '',
             'All9600 complete live requests finished:8 systems x6 settings x200 requests in10 paired blocks, with20 warmups per20-request block. All8 same-B200 target-free live/cache replays have TV0. The primary canonical batch1 profile retains7 observed-mean latency breaches and21 latency-upper-bound breaches among168 policy/cap rows; these are shared policy rows, not21 independent failures. No allocated-memory cap breach occurred. Cap failures remain in the frozen quality average and block an operational advantage claim.', '',
             f"Independent audit: {audit['query_state_rows']} item/state losses, {audit['policy_query_rows']} policy losses, all frozen hashes, all five full-bank hashes, deterministic cohort selection and both mini manifests verified. Full CPU FP64 recomputation covers all24576 predictions; maximum TV difference={audit['max_CPU_FP64_vs_GPU_FP32_TV_difference']:.9g}, strict1e-6 all passed={audit['strict_TV_1e_6_all_passed']}, discrete cross-backend differences={len(discrepancies)}. The original separate Torch CPU precision failure remains preserved.", '',
             'One-off ledger:10240 newly collected scientific model/context responses;64800 timed serving model/context evaluations and64800 registered warmup evaluations. Inclusive process duration overlaps stage/request timers. No new fits were trained; historical fitting costs remain charged in the original ledger. Network/source preparation, energy and monetary charges were not fabricated.', '',
             'The12.22MB mini_replay directory contains four already-evaluated new-cohort documents and all eight states, not a new independent replication. Full6.2GB response banks and all old outputs remain in the workspace. Full GPU process ownership has returned to the root MCQ resource queue; no rich jobs remain.', '',
             primary.to_string(index=False), '']
    (SUP / 'FINAL_STANDALONE_SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
