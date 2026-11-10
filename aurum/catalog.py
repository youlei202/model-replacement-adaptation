"""Common literal-state catalog for every rich fixed and joint policy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .sprint import ROOT, sha, dump
from .data import RichDevelopmentCache, _legacy_codec

CAPS = (128, 512, 2048, 8192, 32768, 131072, 524288, 2097152)


def load_catalog(paths, *, historical=True):
    tables = []
    for folder in map(Path, paths):
        if not (folder / 'complete.json').exists():
            raise ValueError('Only complete runs may enter the selection catalog: ' + str(folder))
        frame = pd.read_csv(folder / 'candidate_ledger.csv')
        frame['state_path'] = frame.state_path.map(lambda p: str((folder / p).resolve()) if not Path(p).is_absolute() else p)
        frame['losses_path'] = frame.state_path.map(lambda p: str(Path(p).with_suffix('.losses.npz')))
        frame['state_format'] = 'AUR1'
        frame['origin_run'] = str(folder.resolve())
        frame['historical_fit_scope'] = False
        tables.append(frame)
    if historical:
        cache = RichDevelopmentCache(); decoder = _legacy_codec(); rows = []
        for target in (3, 4):
            for pair in ((0, 1), (0, 2), (1, 2)):
                folder = cache.root / 'fits' / f'target_{target}' / f'pair_{pair[0]}_{pair[1]}'
                ledger = pd.read_parquet(folder / 'validation_candidates.parquet')
                # The historical run kept every envelope/family/native winner;
                # dominated old rows remain in its unchanged 366-row ledger.
                for file in sorted((folder / 'states').glob('*.bin')):
                    matching = ledger[ledger.candidate_id.eq(file.stem)]
                    if matching.empty:
                        raise ValueError('Historical state not found in source ledger')
                    record = matching.iloc[0].to_dict()
                    state = decoder.decode(file.read_bytes())
                    assert sha(file) == file.stem and len(file.read_bytes()) == record['actual_bytes']
                    rank = state['basis_rank']
                    context = cache.root / 'public_contexts' / f'pair_{pair[0]}_{pair[1]}.json' if rank else None
                    lossfile = file.with_suffix('.validation.npz')
                    with np.load(lossfile) as losses:
                        assert abs(losses['tv'].astype(float).mean() - record['validation_tv']) < 2e-6
                    rows.append({
                        **record, 'pair': ','.join(map(str, pair)), 'target_id': cache.registry[target]['model_id'],
                        'state_path': str(file), 'losses_path': str(lossfile), 'state_format': 'RBV2',
                        'label': 'historical_' + record['family'], 'origin_run': str(folder),
                        'cheap': record['family'] in ('nnls', 'convex', 'temperature', 'logit_mean', 'probability_mean'),
                        'public_context_manifest': str(context) if context else None,
                        'public_context_digest': sha(context) if context else None,
                        'public_numeric_bytes': 151936 * rank * 4,
                        'resident_parameter_bytes': sum(x.nbytes for x in state['arrays']),
                        'historical_fit_scope': True, 'scored_decoded': True,
                        'selection_data': 'historical_fit_validation_development',
                        'context': 'legacy_raw_logit_projection' if rank else 'none',
                    })
        tables.append(pd.DataFrame(rows))
    combined = pd.concat(tables, ignore_index=True)
    combined['codec'] = combined.codec.astype(str)
    for row in combined.itertuples():
        if Path(row.state_path).stat().st_size != row.actual_bytes or sha(row.state_path) != row.candidate_id:
            raise ValueError('State size/hash mismatch')
    # Keep duplicate recipe entries in ALL_CANDIDATES, but choose unique decoded
    # predictors when building frontiers. Losses are target-specific.
    return combined


def candidate_key(frame):
    return frame.sort_values(['validation_tv', 'actual_bytes', 'family', 'codec', 'candidate_id', 'pair'], kind='stable')


def selected_frontiers(catalog, caps=CAPS):
    rows = []
    for (target, pair), candidates in catalog.groupby(['target_index', 'pair'], sort=True):
        for cap in caps:
            for menu in ('full', 'cheap'):
                feasible = candidates[candidates.actual_bytes.le(cap)]
                if menu == 'cheap':
                    feasible = feasible[feasible.cheap]
                if feasible.empty:
                    raise ValueError(f'Missing feasible {target}/{pair}/{cap}/{menu}')
                row = candidate_key(feasible).iloc[0].to_dict()
                row.update(cap_bytes=cap, menu=menu)
                rows.append(row)
    return pd.DataFrame(rows)


def build_policies(catalog, frontiers, weights=None):
    caps = sorted(frontiers.cap_bytes.unique().tolist())
    weights = np.ones(len(caps)) / len(caps) if weights is None else np.asarray(weights, float)
    assert len(weights) == len(caps) and np.all(weights >= 0) and abs(weights.sum() - 1) < 1e-12
    decisions = []
    for target, frame in frontiers.groupby('target_index', sort=True):
        full = frame[frame.menu.eq('full')]
        cheap = frame[frame.menu.eq('cheap')]
        candidates = catalog[catalog.target_index.eq(target)]
        pairs = sorted(full.pair.unique())
        joint = {cap: candidate_key(full[full.cap_bytes.eq(cap)]).iloc[0] for cap in caps}
        cheap_joint = {cap: candidate_key(cheap[cheap.cap_bytes.eq(cap)]).iloc[0] for cap in caps}
        best = {pair: full[full.pair.eq(pair)].set_index('cap_bytes').loc[caps] for pair in pairs}
        native = candidates[candidates.codec.eq('fp32')]
        if native.empty:
            raise ValueError('No full precision native candidate')
        fixed = {
            'fixed_native_full': candidate_key(native).iloc[0]['pair'],
            'fixed_unbounded_menu': candidate_key(candidates).iloc[0]['pair'],
            'fixed_mean_budget': min(pairs, key=lambda p: (float(best[p].validation_tv.to_numpy() @ weights), p)),
            'fixed_minimax_regret': min(pairs, key=lambda p: (max(float(best[p].loc[c].validation_tv - joint[c].validation_tv) for c in caps), p)),
            **{'fixed_pair_' + pair.replace(',', '_'): pair for pair in pairs},
        }
        for cap, weight in zip(caps, weights):
            cheap_fixed = [('cheap_pair_' + p.replace(',', '_'), cheap[cheap.pair.eq(p) & cheap.cap_bytes.eq(cap)].iloc[0]) for p in pairs]
            for policy, row in [('joint', joint[cap]), ('joint_cheap', cheap_joint[cap])] + [(name, best[pair].loc[cap]) for name, pair in fixed.items()] + cheap_fixed:
                decision = row.to_dict()
                decision.update(target_index=int(target), policy=policy, cap_bytes=int(cap), budget_weight=float(weight))
                decisions.append(decision)
    # JSON roundtrip converts numpy scalars and missing table values safely.
    return json.loads(pd.DataFrame(decisions).to_json(orient='records'))


def development_losses(row):
    with np.load(row['losses_path']) as data:
        prefix = '' if row['state_format'] == 'RBV2' else 'validation_'
        return {key: data[prefix + key].astype(float) for key in ('tv', 'kl', 'argmax_disagreement')}


def summarize_development(decisions):
    frame = pd.DataFrame(decisions); rows = []
    for target, group in frame.groupby('target_index'):
        arrays = {}
        for policy, part in group.groupby('policy'):
            values = [development_losses(row)['tv'] * row['budget_weight'] for row in part.to_dict('records')]
            arrays[policy] = np.sum(values, axis=0)
        for policy, losses in arrays.items():
            delta = losses - arrays['joint']
            rows.append({'target_index': target, 'policy': policy,
                         'weighted_validation_tv': float(losses.mean()), 'fixed_minus_joint_tv': float(delta.mean()),
                         'cluster_sd': float(delta.std(ddof=1)), 'clusters': len(delta),
                         'role': 'exploratory_development_selection_estimate_not_confirmation'})
    return pd.DataFrame(rows)


def save_catalog(paths, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(paths)
    catalog.to_csv(out / 'ALL_CANDIDATES.csv', index=False)
    unique = catalog.drop_duplicates(['target_index', 'pair', 'candidate_id', 'state_format'])
    frontiers = selected_frontiers(unique)
    frontiers.to_csv(out / 'ADAPTER_CAPACITY_CURVES.csv', index=False)
    frontiers.to_csv(out / 'ALL_PEER_PAIRS.csv', index=False)
    decisions = build_policies(unique, frontiers)
    dump(out / 'POLICY_DECISIONS.json', decisions)
    summarize_development(decisions).to_csv(out / 'DEVELOPMENT_FIXED_CHOICE_PRICE.csv', index=False)
    dump(out / 'catalog_provenance.json', {'inputs': [str(Path(p).resolve()) for p in paths],
         'input_ledgers': {str(Path(p) / 'candidate_ledger.csv'): sha(Path(p) / 'candidate_ledger.csv') for p in paths},
         'all_old_retained_states_included_at_original_bytes': True,
         'old_dominated_nonretained_rows_remain_in_original_ledgers': True,
         'unique_candidate_states_per_target': len(unique), 'recipe_rows': len(catalog), 'caps_bytes': list(CAPS),
         'same_catalog_for_every_fixed_and_joint_policy': True,
         'principle': 'a fixed pair freely retunes its entire shared adapter menu at every budget'})
    return unique, frontiers, decisions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); save_catalog(args.runs, args.output)
