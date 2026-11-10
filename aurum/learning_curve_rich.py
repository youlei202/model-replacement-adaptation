"""Controlled rich data scaling, with fixed peer representation and strong controls.

All new rows are registered development. The original inner stopping documents
and outer validation remain unchanged for the four primary sample sizes. Two
additional declared split/optimizer seeds check the largest-size recipe; they
are sensitivity analyses, never independent data replications.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .session import DevelopmentSession, response_statistics
from .sprint import ROOT, dump, sha, gpu_bank, numpy
from .discover_rich import strengthened_cheap_menu, make_model, offer_codecs


class ExtendedCache:
    def __init__(self, base, combined):
        self.base = base; self.combined = combined
    def __getattr__(self, name): return getattr(self.base, name)
    def logits(self, role, model):
        return self.combined[model] if role == 'fit' else self.base.logits(role, model)


def train_indices(items, candidates, n, seed):
    """Nested, source-balanced ordering using document IDs rather than outcomes."""
    frame = items.iloc[candidates].copy(); frame['row'] = candidates
    ordering = []
    for source, group in frame.groupby('source', sort=True):
        group = group.copy()
        group['order'] = [hashlib.sha256(f'aurum-learning-curve|{seed}|{x}'.encode()).hexdigest() for x in group.cluster_id]
        ordering.append(group.sort_values(['order', 'cluster_id']).row.tolist())
    if n % 2 or len(ordering) != 2 or any(len(x) < n // 2 for x in ordering):
        raise ValueError('Balanced complete nested subset required')
    return np.asarray([value for pair in zip(ordering[0][:n // 2], ordering[1][:n // 2]) for value in pair], dtype=np.int64)


def extend_session(session, new_folder, n_train):
    new_folder = Path(new_folder)
    lock = json.loads((new_folder / 'cohort_lock.json').read_text())
    if lock['purpose'] != 'additional_development' or sha(new_folder / 'cohort.parquet') != lock['cohort_sha256']:
        raise ValueError('Only registered additional development may enter training')
    new_items = pd.read_parquet(new_folder / 'cohort.parquet')
    old_items = session.cache.items('fit')
    if set(new_items.cluster_id) & (set(old_items.cluster_id) | set(session.cache.items('validation').cluster_id)):
        raise ValueError('Additional development overlaps old fit/validation')
    for key in ('none', 'logit32', 'logit128', 'stats'):
        session.context(key)
    old_count = len(old_items)
    if n_train > len(session.train_ids):
        arrays = {}
        hashes = {}
        for model in (*session.pair, session.target):
            file = new_folder / 'logits' / f'model_{model}.npy'
            meta = json.loads(file.with_suffix('.json').read_text())
            if not meta['complete'] or meta['completed'] != len(new_items) or sha(file) != meta['sha256']:
                raise ValueError('New development bank incomplete or changed')
            new = np.load(file, mmap_mode='r')
            if new.shape != (len(new_items), 151936) or new.dtype != np.float32:
                raise ValueError('Unexpected additional bank shape')
            arrays[model] = np.concatenate([session.cache.logits('fit', model), new], axis=0)
            arrays[model].flags.writeable = False
            hashes[str(model)] = meta['sha256']
        extra_peers = tuple(gpu_bank(arrays[model][old_count:], session.device) for model in session.pair)
        for key, (basis, features, meta) in list(session.contexts.items()):
            if key == 'none':
                extra_features = torch.empty((len(new_items), 0), device=session.device)
            else:
                with np.load(Path(meta['manifest_path']).parent / 'features.npz') as norm:
                    mean = torch.as_tensor(norm['mean'], device=session.device)
                    scale = torch.as_tensor(norm['scale'], device=session.device)
                if key == 'stats':
                    raw = response_statistics(extra_peers)
                else:
                    raw = torch.cat([p @ basis for p in extra_peers], dim=1)
                extra_features = (raw - mean) / scale
            features['fit'] = torch.cat([features['fit'], extra_features], dim=0)
        session.peers['fit'] = tuple(torch.cat([old, extra], dim=0) for old, extra in zip(session.peers['fit'], extra_peers))
        session.targets['fit'] = torch.cat([session.targets['fit'], gpu_bank(arrays[session.target][old_count:], session.device)], dim=0)
        session.cache = ExtendedCache(session.cache, arrays)
        items = pd.concat([old_items, new_items], ignore_index=True)
        candidates = np.r_[session.train_ids, np.arange(old_count, len(items))]
    else:
        hashes = {}; items = old_items; candidates = session.train_ids
    session.train_ids = train_indices(items, candidates, n_train, session.seed)
    dump(session.folder / 'data_roles.json', {
        'fit_cluster_ids': items.cluster_id.tolist(),
        'validation_cluster_ids': session.cache.items('validation').cluster_id.tolist(),
        'train_indices': session.train_ids.tolist(), 'inner_stop_indices': session.stop_ids.tolist(),
        'training_documents': len(session.train_ids), 'inner_stop_documents': len(session.stop_ids),
        'training_source_counts': items.iloc[session.train_ids].groupby('source').size().to_dict(),
        'seed': session.seed, 'fold': session.fold, 'target': session.target, 'pair': list(session.pair),
        'new_development_cohort_sha256': lock['cohort_sha256'], 'new_response_sha256': hashes,
        'public_basis_and_normalizers_fit_only_on_original_selected_peer_responses': True,
        'no_new_confirmation_access': True,
    })


def fixed_recipes(target, pair):
    folder = ROOT / 'runs/aurum/development/expanded_rich_v1' / f't{target}_p{pair[0]}-{pair[1]}'
    if not (folder / 'complete.json').exists(): raise ValueError('Expanded recipe screen incomplete')
    ledger = pd.read_csv(folder / 'candidate_ledger.csv')
    result = []
    for family, context in [('linear', 'logit32'), ('linear', 'logit128'), ('gate', 'stats')]:
        rows = ledger[(ledger.stage == 'screen') & (ledger.architecture == family) & (ledger.context == context)]
        row = rows.sort_values(['inner_stop_tv', 'actual_bytes', 'candidate_id']).iloc[0].to_dict()
        result.append({'spec': json.loads(row['architecture_spec']), 'lr': row['learning_rate'],
                       'weight_decay': row['weight_decay'], 'prior_screen_candidate': row['candidate_id'],
                       'prior_screen_ledger_sha256': sha(folder / 'candidate_ledger.csv')})
    return result


def run_one(args, n, seed):
    folder = Path(args.output) / f'n{n}_seed{seed}'
    if (folder / 'complete.json').exists(): return
    source_names = ['learning_curve_rich.py', 'discover_rich.py', 'session.py', 'robust_calibration.py',
                    'data.py', 'models.py', 'codec.py', 'inference.py', 'train.py', 'losses.py']
    sources = {name: sha(ROOT / 'aurum' / name) for name in source_names}
    recipes = fixed_recipes(args.target, args.peers)
    specification = {'kind': 'controlled_data_scaling', 'n_train': n, 'seed': seed,
                     'recipes_fixed_from_original_development': recipes, 'source_sha256': sources,
                     'epochs': 100, 'tv_epochs': 30, 'inner_stop_patience': 20,
                     'primary_sizes': [512, 1024, 1364, 3412],
                     'sensitivity_seeds': [20261110, 20261111, 20261112],
                     'same_three_architectures_and_cheap_menu_for_all_pairs': True}
    session = DevelopmentSession(folder, args.target, args.peers, seed=seed, device=args.device)
    dump(folder / 'run_spec.json', specification)
    try:
        extend_session(session, args.new_data, n)
        anchor, anchor_row = strengthened_cheap_menu(session)
        for recipe in recipes:
            spec = recipe['spec']; context = spec['context']
            label = spec['family'] + '_' + context
            model = make_model(session, spec, anchor)
            training = session.optimize(model, label + '_KL', context, loss='kl', epochs=100,
                lr=recipe['lr'], weight_decay=recipe['weight_decay'], patience=20, batch_size=32)
            extra = {'objective': 'kl', 'n_train': n, 'recipe': json.dumps(recipe, sort_keys=True),
                     'anchor_candidate_id': anchor_row['candidate_id'], 'chosen_epoch': training['chosen_epoch']}
            offer_codecs(session, model, label + '_KL', context, extra)
            refinement = session.optimize(model, label + '_TV', context, loss='tv', epochs=30,
                lr=.0001, weight_decay=recipe['weight_decay'], patience=20, batch_size=32)
            offer_codecs(session, model, label + '_KL_TV', context, {**extra, 'objective': 'kl_then_tv',
                'chosen_epoch': refinement['chosen_epoch']})
        assert all(sha(ROOT / 'aurum' / name) == digest for name, digest in sources.items())
        session.finish(specification)
    except Exception as exc:
        dump(folder / 'failure.json', {'error': repr(exc), 'candidate_rows': len(session.rows), 'outputs_retained': True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=int, required=True); p.add_argument('--peers', type=int, nargs=2, required=True)
    p.add_argument('--new-data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--primary-only', action='store_true')
    args = p.parse_args()
    for n in (512, 1024, 1364, 3412): run_one(args, n, 20261110)
    if not args.primary_only:
        for seed in (20261111, 20261112): run_one(args, 3412, seed)


if __name__ == '__main__': main()
