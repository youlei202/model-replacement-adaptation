"""Freeze a complete rich candidate frontier, then replay on newly collected documents."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .sprint import ROOT, sha, dump
from .catalog import CAPS, candidate_key
from .confirmation_data import _check_freeze
from .deployment import DeployedPredictor
from .train import ArrayData, evaluate
from .statistics import simultaneous_intervals


def freeze(catalog_dir, output, documents=4096):
    catalog_dir = Path(catalog_dir).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep existing lock; new specification requires a new lock')
    decisions = json.loads((catalog_dir / 'POLICY_DECISIONS.json').read_text())
    catalog = pd.read_csv(catalog_dir / 'ALL_CANDIDATES.csv')
    thresholds = []; extra = []
    for target, frame in catalog.groupby('target_index'):
        cheap = float(frame[frame.cheap].validation_tv.min())
        for fraction in (1.0, .99, .975, .95):
            epsilon = cheap * fraction
            for pair in sorted(frame.pair.unique()) + ['joint']:
                allowed = frame if pair == 'joint' else frame[frame.pair.eq(pair)]
                feasible = allowed[allowed.validation_tv.le(epsilon)]
                row = {'target_index': int(target), 'pair_scope': pair, 'epsilon': epsilon,
                       'epsilon_fraction_of_dev_bestcheap': fraction, 'feasible_on_development': not feasible.empty}
                if not feasible.empty:
                    selected = feasible.sort_values(['actual_bytes', 'validation_tv', 'candidate_id']).iloc[0].to_dict()
                    extra.append(selected); row.update(candidate_id=selected['candidate_id'], pair=selected['pair'], actual_bytes=int(selected['actual_bytes']))
                thresholds.append(row)
    dump(catalog_dir / 'MATCHED_FIDELITY_SELECTIONS.json', thresholds)
    required = pd.DataFrame(decisions + extra).drop_duplicates(['target_index', 'pair', 'candidate_id', 'state_format'])
    dump(catalog_dir / 'CONFIRMATION_STATES.json', json.loads(required.to_json(orient='records')))
    paths = set()
    for name in ('POLICY_DECISIONS.json', 'CONFIRMATION_STATES.json', 'MATCHED_FIDELITY_SELECTIONS.json', 'ALL_CANDIDATES.csv', 'ADAPTER_CAPACITY_CURVES.csv'):
        paths.add(catalog_dir / name)
    for row in required.to_dict('records'):
        paths.add(Path(row['state_path']))
        manifest = row.get('public_context_manifest')
        if isinstance(manifest, str) and manifest:
            manifest = Path(manifest); paths.add(manifest)
            meta = json.loads(manifest.read_text())
            if row['state_format'] == 'RBV2':
                paths.add(manifest.with_suffix('.npy'))
            else:
                paths.update(manifest.parent / name for name in meta['files'])
    for name in ('confirmation.py', 'confirmation_data.py', 'deployment.py', 'statistics.py', 'codec.py', 'inference.py', 'models.py', 'session.py', 'train.py', 'losses.py', 'sprint.py', 'data.py', 'catalog.py'):
        paths.add(ROOT / 'aurum' / name)
    paths.update(ROOT / 'experiments' / name for name in ('rich_codec.py', 'rich_inference.py'))
    paths.update(ROOT / 'experiments/runs/v2/rich_behavior' / name for name in ('model_registry.json', 'data_lock.json'))
    contrasts = []
    for target in sorted(catalog.target_index.unique()):
        contrasts += [f't{target}:fixed_pair_{p}_minus_joint' for p in ('0_1', '0_2', '1_2')]
        contrasts += [f't{target}:joint_cheap_minus_joint']
        contrasts += [f't{target}:cheap_pair_{p}_minus_fixed_pair_{p}' for p in ('0_1', '0_2', '1_2')]
    specification = {
        'campaign': 'AURUM', 'population': 'Equal whole-document mixture of canonical WikiText103 and arxiv_train_shard0, existing pinned tokenizer contexts, previously unexposed clusters only',
        'targets': [3, 4], 'admissible_pairs': [[0, 1], [0, 2], [1, 2]],
        'budgets_bytes': list(CAPS), 'budget_weights': [1 / len(CAPS)] * len(CAPS),
        'primary_endpoint': 'full-support TV; fixed-choice price Gamma=min_pair mean_document mean_frozen_budget(fixed_retuned minus joint)',
        'contrast_family': [population + ':' + contrast for population in ('mixture', 'arxiv_train_shard0', 'wikitext103') for contrast in contrasts], 'inference': {'alpha': .05, 'draws': 10000, 'seed': 20261110, 'method': 'source-stratified Gaussian multiplier max-t over forty-two paired document contrasts (mixture and both predeclared source populations); Gamma interval min endpoints'},
        'historical_access_statement': 'All earlier campaigns and current fit/validation outcomes are historical development. No old-cache reshuffle is confirmation. AURUM new development reservations excluded by access registry.',
        'confirmation_design': {'n_documents': documents, 'selection_seed': 20261110, 'inference_microbatch': 1, 'inference_dtype': 'bfloat16', 'vocab': 151936},
        'selection': 'Same decoded catalog and per-budget adapters for every fixed and joint policy; no retraining after this freeze',
        'matched_fidelity': 'Development thresholds 1,.99,.975,.95 times best cheap validation TV; min state meeting threshold selected before fresh outcomes; conservative Hoeffding upper bounds for threshold claims',
        'secondary': 'KL and argmax disagreement descriptive; both source populations predeclared after observed development heterogeneity and included in the full simultaneous family; adapter timing measured separately from peer calls',
        'catalog_dir': str(catalog_dir.relative_to(ROOT)), 'all_signed_outcomes_retained': True,
    }
    lock = {'locked_at_utc': datetime.now(timezone.utc).isoformat(), 'specification': specification,
            'artifacts': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in sorted(paths)]}
    dump(output, lock); _check_freeze(output)
    return lock


def score(method_lock, cohort_lock, output, device='cuda:0'):
    lock, spec, _, _ = _check_freeze(method_lock)
    cohort_lock = Path(cohort_lock); membership = json.loads(cohort_lock.read_text())
    if membership['method_lock_sha256'] != sha(method_lock):
        raise ValueError('Cohort belongs to a different method lock')
    folder = cohort_lock.parent; documents = pd.read_parquet(folder / 'cohort.parquet')
    if sha(folder / 'cohort.parquet') != membership['cohort_sha256']:
        raise ValueError('Cohort changed')
    banks = {}
    for model in range(5):
        path = folder / 'logits' / f'model_{model}.npy'; meta = json.loads(path.with_suffix('.json').read_text())
        if not meta['complete'] or sha(path) != meta['sha256'] or meta['cohort_sha256'] != membership['cohort_sha256']:
            raise ValueError('Incomplete or mismatched new response bank')
        banks[model] = np.load(path, mmap_mode='r')
    catalog_dir = ROOT / spec['catalog_dir']
    selected = json.loads((catalog_dir / 'CONFIRMATION_STATES.json').read_text())
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Confirmation already scored')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    rows = []; started = time.time()
    for number, row in enumerate(selected):
        pair = tuple(map(int, row['pair'].split(','))); target = int(row['target_index'])
        model = DeployedPredictor(row, device)
        data = ArrayData(tuple(banks[p] for p in pair), None, banks[target], role='confirmation')
        result = evaluate(model, data, batch_size=64)
        frame = documents[['item_id', 'cluster_id', 'source']].copy()
        frame['target_index'] = target; frame['pair'] = row['pair']; frame['candidate_id'] = row['candidate_id']
        frame['state_format'] = row['state_format']; frame['actual_bytes'] = row['actual_bytes']
        for name, values in result['per_item'].items():
            frame[name] = values
        rows.append(frame)
        dump(out / 'progress.json', {'scored': number + 1, 'states': len(selected), 'candidate_id': row['candidate_id'], 'seconds': time.time() - started})
        print(f'scored {number+1}/{len(selected)} t{target} pair{pair} TV={result["tv"]:.8f}', flush=True)
        del model
    pd.concat(rows, ignore_index=True).to_parquet(out / 'CONFIRMATION_PER_CLUSTER.parquet', index=False)
    dump(out / 'confirmation_provenance.json', {'method_lock_sha256': sha(method_lock), 'cohort_lock_sha256': sha(cohort_lock),
        'cohort_sha256': membership['cohort_sha256'], 'documents': len(documents), 'states': len(selected),
        'state_identity_join': 'target_index,pair,candidate_id,state_format join frozen POLICY_DECISIONS.json; budgets and policies may share identical predictions',
        'seconds': time.time() - started})
    analyze(method_lock, out)
    dump(out / 'complete.json', {'complete': True, 'method_lock_sha256': sha(method_lock), 'seconds': time.time() - started})


def analyze(method_lock, output):
    _, spec, _, _ = _check_freeze(method_lock); out = Path(output)
    catalog = ROOT / spec['catalog_dir']
    decisions = pd.DataFrame(json.loads((catalog / 'POLICY_DECISIONS.json').read_text()))
    data = pd.read_parquet(out / 'CONFIRMATION_PER_CLUSTER.parquet')
    states = {(int(t), p, c): group.sort_values('cluster_id') for (t, p, c), group in data.groupby(['target_index', 'pair', 'candidate_id'])}
    reference = next(iter(states.values())); clusters = reference.cluster_id.to_numpy(); strata = reference.source.to_numpy()
    policy_arrays = {}; policy_rows = []
    for (target, policy), group in decisions.groupby(['target_index', 'policy']):
        vector = np.zeros((len(clusters), 3))
        for row in group.to_dict('records'):
            prediction = states[(int(target), row['pair'], row['candidate_id'])]
            assert np.array_equal(prediction.cluster_id, clusters)
            values = prediction[['tv', 'kl', 'argmax_disagreement']].to_numpy(float)
            vector += row['budget_weight'] * values
            policy_rows.append({'target_index': target, 'policy': policy, 'cap_bytes': row['cap_bytes'],
                'pair': row['pair'], 'candidate_id': row['candidate_id'], 'actual_bytes': row['actual_bytes'],
                'tv': values[:, 0].mean(), 'kl': values[:, 1].mean(), 'argmax_disagreement': values[:, 2].mean(),
                'documents': len(clusters)})
        policy_arrays[(int(target), policy)] = vector
    contrasts = []; labels = []
    for target in spec['targets']:
        joint = policy_arrays[(target, 'joint')][:, 0]
        for pair in ('0_1', '0_2', '1_2'):
            labels.append((target, 'fixed_pair_' + pair, 'joint'))
            contrasts.append(policy_arrays[(target, 'fixed_pair_' + pair)][:, 0] - joint)
        labels.append((target, 'joint_cheap', 'joint'))
        contrasts.append(policy_arrays[(target, 'joint_cheap')][:, 0] - joint)
        for pair in ('0_1', '0_2', '1_2'):
            labels.append((target, 'cheap_pair_' + pair, 'fixed_pair_' + pair))
            contrasts.append(policy_arrays[(target, 'cheap_pair_' + pair)][:, 0] - policy_arrays[(target, 'fixed_pair_' + pair)][:, 0])
    base_contrasts = contrasts[:]; base_labels = labels[:]
    labels = [(t, c, r, 'mixture') for t, c, r in base_labels]
    for source in sorted(np.unique(strata)):
        mask = strata == source
        for label, vector in zip(base_labels, base_contrasts):
            contrasts.append(vector * mask / mask.mean())
            labels.append((*label, str(source)))
    settings = spec['inference']
    intervals = simultaneous_intervals(np.stack(contrasts, 1), strata, alpha=settings['alpha'], draws=settings['draws'], seed=settings['seed'])
    rows = []
    for index, (target, comparator, reference_policy, population) in enumerate(labels):
        rows.append({'target_index': target, 'comparator': comparator, 'reference': reference_policy, 'population': population,
            'difference_tv': intervals['mean'][index], 'lower_simultaneous': intervals['lower'][index],
            'upper_simultaneous': intervals['upper'][index], 'se': intervals['se'][index],
            'documents': len(clusters) if population == 'mixture' else int((strata == population).sum()), 'family_size': len(labels), 'scope': 'fresh_cluster_confirmation'})
    for target in spec['targets']:
        for population in ('mixture', *sorted(np.unique(strata))):
            fixed = [r for r in rows if r['target_index'] == target and r.get('population') == population and r['comparator'].startswith('fixed_pair_')]
            rows.append({'target_index': target, 'comparator': 'Gamma_best_fixed_pair', 'reference': 'joint', 'population': population,
                'difference_tv': min(r['difference_tv'] for r in fixed), 'lower_simultaneous': min(r['lower_simultaneous'] for r in fixed),
                'upper_simultaneous': min(r['upper_simultaneous'] for r in fixed), 'documents': fixed[0]['documents'],
                'family_size': len(labels), 'scope': 'min endpoints of simultaneous fixed-pair intervals'})
    pd.DataFrame(rows).to_csv(out / 'FIXED_CHOICE_PRICE.csv', index=False)
    pd.DataFrame(policy_rows).to_csv(out / 'CONFIRMATION_POLICY_SUMMARY.csv', index=False)
    dump(out / 'inference_method.json', {k: v for k, v in intervals.items() if not isinstance(v, np.ndarray)})
    thresholds = json.loads((catalog / 'MATCHED_FIDELITY_SELECTIONS.json').read_text())
    feasible_count = sum(row['feasible_on_development'] for row in thresholds)
    for row in thresholds:
        if row['feasible_on_development']:
            losses = states[(row['target_index'], row['pair'], row['candidate_id'])].tv.to_numpy(float)
            upper = min(1., float(losses.mean()) + np.sqrt(np.log(max(1, feasible_count) / .05) / (2 * len(losses))))
            row.update(confirmation_tv=float(losses.mean()), simultaneous_hoeffding_upper=upper,
                       threshold_supported_by_upper_bound=upper <= row['epsilon'])
    pd.DataFrame(thresholds).to_csv(out / 'MATCHED_FIDELITY_STATE.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--catalog', required=True); p.add_argument('--output', required=True); p.add_argument('--documents', type=int, default=4096)
    p = sub.add_parser('score'); p.add_argument('--method-lock', required=True); p.add_argument('--cohort-lock', required=True); p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('analyze'); p.add_argument('--method-lock', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.catalog, args.output, args.documents)
    elif args.action == 'score': score(args.method_lock, args.cohort_lock, args.output, args.device)
    else: analyze(args.method_lock, args.output)
