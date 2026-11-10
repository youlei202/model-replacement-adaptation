#!/usr/bin/env python3
"""Prospective supplement: fresh fidelity under measured latency/VRAM caps.

Original scientific locks and results are inputs only. This is a separately
registered missing experiment, conducted after those experiments finished.
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/experiment_config'))
from rich_setup import OUT, digest, dump
from reference.statistics import paired_cluster_interval, paired_power_n

SUP = OUT / 'matched_resource_supplement'
FIXED = ['fixed_native_full', 'fixed_unbounded_best', 'fixed_mean_budget', 'fixed_minimax_validation_regret']
EXTRA = ['fixed_resource_mean_frontier', 'joint_cheap']
CHEAP = {'nnls', 'convex', 'temperature', 'probability_mean', 'logit_mean'}
SEED = 81724061


def candidates(target):
    return {r['candidate_id']: r for p in (OUT / f'fits/target_{target}').glob('*/envelope.json') for r in json.load(open(p))}


def validate_lock():
    lock = json.load(open(SUP / 'prospective_lock.json'))
    for path, expected in lock['inputs_sha256'].items():
        assert digest(ROOT / path) == expected, path
    return lock


def freeze():
    SUP.mkdir(exist_ok=True)
    if (SUP / 'prospective_lock.json').exists():
        validate_lock()
        print('Already frozen; preserving existing supplement')
        return
    amendment = {
        'created_unix': time.time(), 'supplement_id': 'rich_actual_resource_caps_v1',
        'reason': 'Contract audit found Experiment 1 absent: prior operating points minimized measured cost under fidelity/state-byte caps, not fidelity under measured latency/VRAM caps.',
        'chronology': 'Original test results already available; this amendment is not retroactive preregistration. Existing fit/validation artifacts and development timings only determine new choices; a previously uncollected reservoir cohort provides independent prospective confirmation.',
        'unchanged_original_experiments': True,
        'cap_rule': 'Cartesian product of quantiles .25/.5/.75/1 of121 pre-original-test complete-predictor mean wall latency and maximum allocated VRAM; round latency up to0.5ms and VRAM up to0.25GiB; collapse duplicate thresholds.',
        'not_external_requirement': 'These are development-derived research resource caps, not documented production SLAs or measured hardware OOM limits.',
        'catalog_limitation': 'All retained family/rank/state-cap validation champions and direct/native/unbounded controls,133 target-state entries121 unique states. Timings do not cover all2196 originally fitted encoded candidates. Equal retained menu for every policy.',
        'selection': 'Minimize validation full-support TV independently per actual resource cap; tie-break smaller literal state then peer mask then digest. Four original fixed pair choices retained. Extra fixed pair maximizes feasible cap coverage then minimizes equal-cap mean validationTV; cheap menu uses only means/scalar calibration/direct peers.',
        'primary_endpoint': 'Equal average of per-document fixedTV minus jointTV across the prelocked common-feasible caps; two targets x four registered fixed comparators. Missing feasibility is reported, not imputed as a quality advantage.',
        'joint_quality_guard': 'A same-resource advantage additionally requires the paired upper bound on joint-minus-fixed argmax disagreement to be <=0.01; equivalently fixed-minus-joint agreement-gain lower bound >=-0.01. TV superiority and this noninferiority guard form an intersection-union claim.',
        'power': 'Paired validation cap-average SD, effect0.005,90% normal-approx power,16 simultaneous comparisons across both supplemental routes; minimum2048 new clusters, balanced sources, capped by untouched available reservoir. Degenerate SD is not positive-effect evidence.',
        'timing': 'New selected systems:20warmup before each20timed block,10 randomized paired blocks,200 timed requests per system/setting. Batch1 canonical untruncated is primary; logical batch8 serializes8 microbatch1 requests. Controlled128/512 are secondary and have no transferred canonical-quality guarantee.',
        'final_cap_check': 'Fresh primary complete-predictor block mean latency plus Bonferroni one-sided95% bootstrap upper bound, maximum observed allocated VRAM. Record observed breaches; never reselect after new results.',
        'interpretation': 'A resource frontier advantage does not by itself establish retained-state-budget coupling; actual state bytes and stronger fixed frontier control remain visible.'}
    dump(SUP / 'amendment.json', amendment)
    profiles = json.load(open(OUT / 'resources/development_profiles.json'))
    profile = {p['candidate_id']: p for p in profiles}
    latency = sorted(set(float(x) for x in np.ceil(np.quantile([p['mean_end_to_end_ms'] for p in profiles], [.25, .5, .75, 1]) * 2) / 2))
    memory = sorted(set(int(x) for x in np.ceil(np.quantile([max(z['peak_allocated_bytes'] for z in p['raw_timings']) for p in profiles], [.25, .5, .75, 1]) / 2**28) * 2**28))
    caps = [{'cap_id': f'L{li}_M{mi}', 'mean_wall_latency_cap_ms': l, 'allocated_VRAM_cap_bytes': m}
            for li, l in enumerate(latency) for mi, m in enumerate(memory)]
    old = json.load(open(OUT / 'selection_lock.json'))
    decisions, catalog, fixed_diagnostics, power_rows = [], [], [], []
    choose_key = lambda r: (r['validation_tv'], r['actual_bytes'], r['peer_mask'], r['candidate_id'])
    for target in [3, 4]:
        rows = list(candidates(target).values())
        for r in rows:
            p = profile[r['candidate_id']]
            r.update({'development_mean_wall_ms': p['mean_end_to_end_ms'],
                      'development_peak_allocated_bytes': max(z['peak_allocated_bytes'] for z in p['raw_timings']),
                      'actual_call_q': p['actual_call_q'], 'active_peer_indices': p['peer_ids']})
        catalog.extend(rows)
        pools = {cap['cap_id']: [r for r in rows if r['development_mean_wall_ms'] <= cap['mean_wall_latency_cap_ms'] and r['development_peak_allocated_bytes'] <= cap['allocated_VRAM_cap_bytes']] for cap in caps}
        fixed_pairs = {r['policy']: r['peer_mask'] for r in old['choices'] if r['target_index'] == target and r['policy'] in FIXED}
        diagnostics = []
        for mask in [3, 5, 6]:
            wins = [min([r for r in pool if r['peer_mask'] == mask], key=choose_key) for pool in pools.values() if any(r['peer_mask'] == mask for r in pool)]
            diagnostics.append({'peer_mask': mask, 'feasible_caps': len(wins), 'mean_validation_TV': float(np.mean([r['validation_tv'] for r in wins])) if wins else None})
        best_pair = min(diagnostics, key=lambda r: (-r['feasible_caps'], r['mean_validation_TV'] if r['mean_validation_TV'] is not None else float('inf'), r['peer_mask']))
        fixed_pairs['fixed_resource_mean_frontier'] = best_pair['peer_mask']
        fixed_diagnostics.append({'target_index': target, 'selected': best_pair, 'all_pairs': diagnostics})
        for cap in caps:
            for policy in ['joint', *FIXED, *EXTRA]:
                pool = pools[cap['cap_id']]
                if policy in fixed_pairs:
                    pool = [r for r in pool if r['peer_mask'] == fixed_pairs[policy]]
                elif policy == 'joint_cheap':
                    pool = [r for r in pool if r['family'] in CHEAP]
                rec = {'target_index': target, 'policy': policy, **cap, 'feasible_on_development': bool(pool)}
                if pool:
                    rec.update(min(pool, key=choose_key))
                else:
                    rec['failure_status'] = 'no_retained_candidate_meets_both_measured_development_caps'
                decisions.append(rec)
        target_decisions = {(r['policy'], r['cap_id']): r for r in decisions if r['target_index'] == target}
        for policy in FIXED:
            differences, common = [], []
            for cap in caps:
                a, b = target_decisions[(policy, cap['cap_id'])], target_decisions[('joint', cap['cap_id'])]
                if a['feasible_on_development'] and b['feasible_on_development']:
                    av = np.load(OUT / a['state_path'].replace('.bin', '.validation.npz'))['tv']
                    bv = np.load(OUT / b['state_path'].replace('.bin', '.validation.npz'))['tv']
                    differences.append(av - bv); common.append(cap['cap_id'])
            sd = float(np.std(np.mean(differences, axis=0), ddof=1)) if differences else None
            planned = paired_power_n(sd, .005, comparisons=16, power=.9) if sd and sd > 0 else 2
            power_rows.append({'target_index': target, 'fixed_policy': policy, 'common_feasible_caps': common,
                               'paired_validation_cap_average_sd': sd, 'normal_approximation_n': planned,
                               'zero_sd_not_evidence_of_gain': sd == 0})
    documents = pd.read_parquet(OUT / 'documents.parquet')
    old_test = documents[documents.item_id.isin(old['test_item_ids'])]
    reservoir = documents[documents.role.eq('test_pool') & ~documents.cluster_id.isin(old_test.cluster_id)].copy()
    reservoir['supplement_order_hash'] = reservoir.item_id.map(lambda x: hashlib.sha256(f'{SEED}|{x}'.encode()).hexdigest())
    requested = max(2048, max(p['normal_approximation_n'] for p in power_rows))
    available = 2 * int(reservoir.groupby('source').size().min())
    actual = min(2 * math.ceil(requested / 2), available)
    cohort = pd.concat([group.sort_values('supplement_order_hash').head(actual // 2) for _, group in reservoir.groupby('source')]).sort_values('supplement_order_hash').reset_index(drop=True)
    assert len(cohort) == actual and cohort.cluster_id.nunique() == actual
    assert int(cohort.context_tokens.ge(512).sum()) >= 208, 'Insufficient fresh full-length timing contexts'
    assert not set(cohort.cluster_id) & set(documents.loc[~documents.role.eq('test_pool'), 'cluster_id'])
    assert not set(cohort.cluster_id) & set(old_test.cluster_id)
    cohort.to_parquet(SUP / 'cohort.parquet', index=False)
    pd.DataFrame(catalog).to_csv(SUP / 'candidate_catalog.csv', index=False)
    pd.DataFrame(decisions).to_csv(SUP / 'validation_decisions.csv', index=False)
    selected = {(r['target_index'], r['candidate_id']): r for r in decisions if r['feasible_on_development']}
    paths = [ROOT / 'experiments/runs/v2/protocol/matched_resource_completion_amendment.json', SUP / 'concurrent_new_mcq_cross_audit.json', SUP / 'amendment.json', SUP / 'cohort.parquet', SUP / 'candidate_catalog.csv', SUP / 'validation_decisions.csv',
             OUT / 'documents.parquet', OUT / 'data_lock.json', OUT / 'model_registry.json', OUT / 'compatible_tokenizer_audit.json', OUT / 'selection_lock.json', OUT / 'concurrent_mcq_cross_audit.json',
             OUT / 'resources/development_profiles.json', OUT / 'resources/development_profiles_freeze.json']
    paths += [ROOT / 'experiments' / name for name in ['rich_matched_resource.py', 'rich_matched_resource_live.py', 'rich_matched_cross_audit.py', 'rich_setup.py', 'rich_codec.py', 'rich_inference.py', 'rich_deploy.py', 'rich_collect.py']]
    for r in catalog:
        paths += [OUT / r['state_path'], OUT / r['state_path'].replace('.bin', '.validation.npz')]
    paths += list((OUT / 'public_contexts').glob('*.npy'))
    lock = {'all_choices_frozen_before_supplement_outputs': True, 'created_unix': time.time(),
            'prior_test_known_not_reused': True, 'amendment': 'amendment.json', 'caps': caps,
            'decisions': decisions, 'fixed_resource_pair_diagnostics': fixed_diagnostics,
            'power_plan': power_rows, 'requested_test_clusters': requested, 'available_balanced_clusters': available,
            'actual_test_clusters': actual, 'underpowered_against_normal_plan': actual < requested,
            'test_item_ids': cohort.item_id.tolist(), 'test_cluster_ids': cohort.cluster_id.tolist(),
            'candidate_target_states': len(catalog), 'unique_development_profiles': len(profile),
            'selected_target_states': len(selected), 'selected_unique_states': len({k[1] for k in selected}),
            'required_concurrent_crossaudit': 'concurrent_new_mcq_cross_audit.json; must pass before collection',
            'primary_comparisons': 16, 'per_cap_comparisons': 16 * len(caps), 'bootstrap_draws': 10000,
            'bootstrap_seed': SEED + 1, 'design_effect_TV': .005, 'secondary_argmax_margin': .01,
            'benchmark_settings': [{'batch_size': b, 'context_tokens': l} for b in [1, 8] for l in [0, 128, 512]],
            'benchmark_blocks': 10, 'warmups_per_block': 20, 'timed_requests_per_block': 20,
            'resource_scope': 'Primary natural batch1; complete predictor. Actual allocated VRAM cap is standalone allocation, not only per-target bytes.',
            'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}}
    dump(SUP / 'prospective_lock.json', lock)
    print(json.dumps({k: lock[k] for k in ['actual_test_clusters', 'requested_test_clusters', 'selected_target_states', 'selected_unique_states', 'caps']}, indent=2))


def evaluate():
    import torch
    from rich_codec import decode
    from rich_inference import predict, metrics
    lock = validate_lock()
    torch.set_num_threads(4)
    cohort = pd.read_parquet(SUP / 'cohort.parquet')
    unique = {(r['target_index'], r['candidate_id']): r for r in lock['decisions'] if r['feasible_on_development']}
    losses = []
    for (target, sid), row in unique.items():
        state = decode((OUT / row['state_path']).read_bytes())
        assert target not in state['peers']
        indices = [target, *state['peers']]
        banks = {}
        for i in indices:
            path = SUP / 'logits' / f'model_{i}.npy'
            meta = json.load(open(path.with_suffix('.json')))
            assert meta['complete'] and meta['sha256'] == digest(path)
            banks[i] = np.load(path, mmap_mode='r')
        basis = None
        if state['basis_rank']:
            p = OUT / 'public_contexts' / f"pair_{state['peers'][0]}_{state['peers'][1]}.npy"
            basis = torch.as_tensor(np.load(p), device='cuda:0')
        vals = {}
        with torch.inference_mode():
            for start in range(0, len(cohort), 32):
                t = torch.as_tensor(np.asarray(banks[target][start:start+32]).copy(), device='cuda:0')
                peer = [torch.as_tensor(np.asarray(banks[i][start:start+32]).copy(), device='cuda:0') for i in state['peers']]
                prediction, isp = predict(state, peer, basis)
                for key, value in metrics(t, prediction, isp).items():
                    vals.setdefault(key, []).append(value.cpu().numpy())
        frame = cohort[['item_id', 'cluster_id', 'source']].copy()
        frame['target_index'] = target; frame['candidate_id'] = sid
        for key, value in vals.items():
            frame[key] = np.concatenate(value)
        losses.append(frame)
        del basis, banks, t, peer, prediction
        torch.cuda.empty_cache()
    frame = pd.concat(losses, ignore_index=True)
    frame.to_parquet(SUP / 'item_losses.parquet', index=False)
    dump(SUP / 'quality_complete.json', {'complete': True, 'fresh_clusters': len(cohort), 'target_states': len(unique),
         'item_losses_sha256': digest(SUP / 'item_losses.parquet'), 'prospective_lock_sha256': digest(SUP / 'prospective_lock.json')})
    print('Fresh supplement quality complete', len(frame), flush=True)


def report():
    lock = validate_lock()
    losses = pd.read_parquet(SUP / 'item_losses.parquet')
    lookup = {(t, s): g.set_index('item_id').loc[lock['test_item_ids']].reset_index() for (t, s), g in losses.groupby(['target_index', 'candidate_id'])}
    timings = pd.read_json(SUP / 'timing_raw.jsonl', lines=True)
    assert len(timings) == lock['selected_unique_states'] * len(lock['benchmark_settings']) * 200
    summaries = []
    for (sid, batch, length), g in timings.groupby(['system_digest', 'batch_size', 'context_tokens']):
        block = g.groupby('timing_block').end_to_end_ms.mean().to_numpy()
        assert len(block) == 10 and len(g) == 200
        rng = np.random.default_rng(SEED + 2)
        boot = block[rng.integers(10, size=(10000, 10))].mean(1)
        summaries.append({'candidate_id': sid, 'batch_size': batch, 'context_tokens': length,
                          'mean_wall_ms': g.end_to_end_ms.mean(), 'p95_wall_ms': g.end_to_end_ms.quantile(.95),
                          'mean_GPU_event_ms': g.GPU_ms.mean(), 'mean_wall_upper_simultaneous': np.quantile(boot, 1 - .05 / lock['selected_unique_states']),
                          'peak_allocated_bytes': int(g.peak_allocated_bytes.max()), 'peak_reserved_bytes': int(g.peak_reserved_bytes.max()),
                          'peak_host_RSS_bytes': int(g.host_RSS.max()), 'timed_requests': len(g), 'independent_blocks': len(block)})
    summary = pd.DataFrame(summaries); summary.to_csv(SUP / 'resource_summary.csv', index=False)
    natural = summary[(summary.batch_size == 1) & (summary.context_tokens == 0)].set_index('candidate_id')
    policy_rows, policy_losses, comparisons = [], [], []
    mapping = {(r['target_index'], r['policy'], r['cap_id']): r for r in lock['decisions']}
    for r in lock['decisions']:
        rec = dict(r)
        if r['feasible_on_development']:
            loss = lookup[(r['target_index'], r['candidate_id'])]
            resource = natural.loc[r['candidate_id']].to_dict()
            rec.update(resource)
            rec.update({'test_TV': loss.tv.mean(), 'test_KL': loss.kl.mean(), 'test_argmax_disagreement': loss.argmax_disagreement.mean(),
                        'fresh_latency_cap_met_observed': resource['mean_wall_ms'] <= r['mean_wall_latency_cap_ms'],
                        'fresh_latency_cap_met_upper_bound': resource['mean_wall_upper_simultaneous'] <= r['mean_wall_latency_cap_ms'],
                        'fresh_memory_cap_met': resource['peak_allocated_bytes'] <= r['allocated_VRAM_cap_bytes']})
            rec['fresh_both_caps_confirmed'] = rec['fresh_latency_cap_met_upper_bound'] and rec['fresh_memory_cap_met']
            copy = loss.copy(); copy['policy'] = r['policy']; copy['cap_id'] = r['cap_id']; policy_losses.append(copy)
        policy_rows.append(rec)
    pd.DataFrame(policy_rows).to_csv(SUP / 'policy_quality_resources.csv', index=False)
    pd.concat(policy_losses, ignore_index=True).to_parquet(SUP / 'policy_item_losses.parquet', index=False)
    policy_lookup = {(r['target_index'], r['policy'], r['cap_id']): r for r in policy_rows}
    primary = []
    for target in [3, 4]:
        for policy in FIXED + EXTRA:
            differences, argdiff, common, confirmed = [], [], [], []
            for cap in lock['caps']:
                a = mapping[(target, policy, cap['cap_id'])]; b = mapping[(target, 'joint', cap['cap_id'])]
                rec = {'target_index': target, 'fixed_policy': policy, **cap, 'both_development_feasible': a['feasible_on_development'] and b['feasible_on_development']}
                if rec['both_development_feasible']:
                    av, bv = lookup[(target, a['candidate_id'])], lookup[(target, b['candidate_id'])]
                    d = av.tv.to_numpy() - bv.tv.to_numpy()
                    ci = paired_cluster_interval(d, av.cluster_id, comparisons=lock['per_cap_comparisons'] if policy in FIXED else 4 * len(lock['caps']), draws=10000, seed=SEED + 1)
                    ok = policy_lookup[(target, policy, cap['cap_id'])]['fresh_both_caps_confirmed'] and policy_lookup[(target, 'joint', cap['cap_id'])]['fresh_both_caps_confirmed']
                    rec.update({'gain_TV': ci.estimate, 'simultaneous_low': ci.low, 'simultaneous_high': ci.high,
                                'fixed_candidate_id': a['candidate_id'], 'joint_candidate_id': b['candidate_id'], 'both_fresh_caps_confirmed': ok})
                    differences.append(d); argdiff.append(av.argmax_disagreement.to_numpy() - bv.argmax_disagreement.to_numpy())
                    common.append(cap['cap_id']); confirmed.append(ok)
                comparisons.append(rec)
            rec = {'target_index': target, 'fixed_policy': policy, 'common_feasible_caps': '|'.join(common), 'number_common_caps': len(common),
                   'fresh_clusters': lock['actual_test_clusters'], 'primary_registered': policy in FIXED,
                   'all_common_caps_fresh_confirmed': bool(confirmed) and all(confirmed)}
            if differences:
                ci = paired_cluster_interval(np.mean(differences, axis=0), av.cluster_id, comparisons=lock['primary_comparisons'] if policy in FIXED else 4, draws=10000, seed=SEED + 1)
                argci = paired_cluster_interval(np.mean(argdiff, axis=0), av.cluster_id, comparisons=lock['primary_comparisons'] if policy in FIXED else 4, draws=10000, seed=SEED + 1)
                rec.update({'cap_averaged_gain_TV': ci.estimate, 'simultaneous_low': ci.low, 'simultaneous_high': ci.high,
                            'cap_averaged_argmax_agreement_gain': argci.estimate,
                            'argmax_agreement_gain_simultaneous_low': argci.low,
                            'argmax_agreement_gain_simultaneous_high': argci.high,
                            'argmax_noninferiority_guard_passed': argci.low >= -.01,
                            'design_effect_and_positive_lower': ci.estimate >= .005 and ci.low > 0,
                            'same_resource_advantage_confirmed': ci.estimate >= .005 and ci.low > 0 and argci.low >= -.01 and rec['all_common_caps_fresh_confirmed']})
            primary.append(rec)
    pd.DataFrame(comparisons).to_csv(SUP / 'per_cap_comparisons.csv', index=False)
    pd.DataFrame(primary).to_csv(SUP / 'primary_confirmation.csv', index=False)
    text = ['# Prospective measured-resource-cap supplement', '',
            'This additional experiment was registered after the original confirmation finished. It uses untouched document clusters; original tests are excluded. The same retained adapter catalog and original four fixed peer choices are retuned under measured development latency and allocated-VRAM caps.', '',
            f"New independent clusters: {lock['actual_test_clusters']}; requested normal-plan/minimum size: {lock['requested_test_clusters']}. Caps are research thresholds derived from development profiling, not production requirements.", '',
            pd.DataFrame(primary).to_string(index=False), '',
            'A fresh cap breach remains a result and invalidates a same-resource claim at that operating point. No systems are reselected. Batch1 canonical inputs link quality and resources; batch8 and controlled128/512 timings remain secondary. Numeric payload is distinct from complete process memory. This resource-cap comparison alone does not demonstrate a retained-state-cap benefit.', '']
    (SUP / 'RESULTS.md').write_text('\n'.join(text))
    outputs = ['item_losses.parquet', 'policy_item_losses.parquet', 'policy_quality_resources.csv', 'primary_confirmation.csv', 'per_cap_comparisons.csv', 'resource_summary.csv', 'timing_raw.jsonl', 'quality_complete.json', 'benchmark_complete.json', 'RESULTS.md']
    dump(SUP / 'supplement_complete.json', {'complete': True, 'prospective_lock_sha256': digest(SUP / 'prospective_lock.json'),
         'independent_new_clusters': lock['actual_test_clusters'], 'primary_comparisons': lock['primary_comparisons'],
         'same_resource_positive_primary_comparisons': sum(r.get('same_resource_advantage_confirmed', False) for r in primary if r['primary_registered']),
         'outputs_sha256': {name: digest(SUP / name) for name in outputs}})
    print(pd.DataFrame(primary).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'evaluate', 'report'])
    action = parser.parse_args().action
    {'freeze': freeze, 'evaluate': evaluate, 'report': report}[action]()
