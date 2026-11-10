"""Six-cell development experiment; no historical or fresh test responses.

The first factorial changes the legacy anchor and fitting objective while
retaining the exact old peer-only output basis. Every candidate is decoded
before scoring and enters the same catalog for fixed and joint selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import RichDevelopmentCache, fit_legacy_weights
from .models import FrozenAnchor, AnchoredResidual, TrainableCalibration
from .codec import export_model, decode
from .inference import DecodedPredictor
from .train import ArrayData, TrainConfig, fit, evaluate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numpy(value):
    return value.detach().cpu().numpy()


def gpu_bank(array, device):
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)


def public_context(cache, pair, rank, peer_fit, peer_validation, device):
    """Only selected-peer fit outputs define this reusable public context."""
    folder = ROOT / 'runs/aurum/public_contexts' / f'pair_{pair[0]}_{pair[1]}' / f'rank_{rank}'
    basis = gpu_bank(cache.basis(pair, rank), device)
    with torch.no_grad():
        train_raw = torch.cat([p @ basis for p in peer_fit], dim=1)
        val_raw = torch.cat([p @ basis for p in peer_validation], dim=1)
        mean = train_raw.mean(0)
        scale = train_raw.std(0, unbiased=False)
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        train_features = (train_raw - mean) / scale
        val_features = (val_raw - mean) / scale
    expected = {
        'pair': list(pair), 'basis_rank': rank, 'feature_dim': 2 * rank,
        'peer_fit_hashes': {str(p): cache.metadata('fit', p)['sha256'] for p in pair},
        'legacy_basis_sha256': cache.basis_metadata(pair)['sha256'],
        'definition': 'selectedpeer centered-logit projections; selectedpeerfit mean/std; no target data',
        'target_outputs_used': False,
        'basis_id': 100 + 10 * pair[0] + pair[1],
    }
    marker = folder / 'context.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        for key, value in expected.items():
            assert saved[key] == value, ('Public context drift', key)
        for name, digest in saved['files'].items():
            assert sha(folder / name) == digest
        with np.load(folder / 'features.npz') as prior:
            assert np.array_equal(prior['mean'], numpy(mean))
            assert np.array_equal(prior['scale'], numpy(scale))
            assert np.array_equal(prior['fit'], numpy(train_features))
    else:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'basis.npy', numpy(basis))
        np.savez(folder / 'features.npz', mean=numpy(mean), scale=numpy(scale),
                 fit=numpy(train_features), validation=numpy(val_features))
        saved = {**expected,
                 'deployed_public_numeric_bytes': basis.numel() * 4 + mean.numel() * 4 + scale.numel() * 4,
                 'experiment_feature_rows_are_not_deployment_state': True,
                 'files': {n: sha(folder / n) for n in ('basis.npy', 'features.npz')}}
        dump(marker, saved)
    return basis, train_features, val_features, {**saved, 'manifest_path': str(marker), 'manifest_sha256': sha(marker)}


def run_cell(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    pair = tuple(args.peers)
    assert pair in ((0, 1), (0, 2), (1, 2)) and args.target in (3, 4)
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Preserve existing run; choose a new output directory: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    dependencies = ('run.py', 'sprint.py', 'data.py', 'models.py', 'losses.py',
                    'codec.py', 'inference.py', 'train.py', 'calibration.py')
    source = {f'aurum/{name}': sha(ROOT / 'aurum' / name) for name in dependencies}
    source['experiments/rich_codec.py'] = sha(ROOT / 'experiments/rich_codec.py')
    start = time.time()
    config = vars(args).copy()
    config['output'] = str(folder)
    dump(folder / 'run_spec.json', {
        'arguments': config, 'source_sha256': source, 'started_unix': start,
        'scope': 'exploratory original rich fit/validation only',
        'old_campaign_files_modified': False,
        'target_may_only_enter_fitting_and_scoring': True,
    })
    ledger = []
    candidate_ids = set()
    cache = RichDevelopmentCache()
    try:
        peer_fit = tuple(gpu_bank(cache.logits('fit', p), device) for p in pair)
        peer_val = tuple(gpu_bank(cache.logits('validation', p), device) for p in pair)
        contexts = {rank: public_context(cache, pair, rank, peer_fit, peer_val, device) for rank in args.ranks}
        target_fit = gpu_bank(cache.logits('fit', args.target), device)
        target_val = gpu_bank(cache.logits('validation', args.target), device)
        fold = cache.grouped_folds(seed=args.seed, n_folds=3)[args.inner_fold]
        train_ids, stop_ids = fold['train_indices'], fold['stop_indices']
        dump(folder / 'data_roles.json', {
            'train_indices': train_ids.tolist(), 'inner_stop_indices': stop_ids.tolist(),
            'inner_fold': args.inner_fold, 'seed': args.seed,
            'fit_cluster_ids': cache.items('fit').cluster_id.tolist(),
            'validation_cluster_ids': cache.items('validation').cluster_id.tolist(),
            'fit_document_count': len(target_fit), 'validation_document_count': len(target_val),
            'input_hashes': {f'{role}/model_{i}': cache.metadata(role, i)['sha256']
                             for role in ('fit', 'validation') for i in (*pair, args.target)},
            'historical_test_responses_opened': False,
        })
        blank_fit = torch.empty((len(target_fit), 0), device=device)
        blank_val = torch.empty((len(target_val), 0), device=device)

        def datasets(rank=0):
            ff, vf = (contexts[rank][1:3] if rank else (blank_fit, blank_val))
            return (
                ArrayData(peer_fit, ff, target_fit, train_ids, role='inner_train'),
                ArrayData(peer_fit, ff, target_fit, stop_ids, role='inner_stop'),
                ArrayData(peer_val, vf, target_val, role='outer_validation'),
            )

        def offer(model, label, *, rank=0, objective='legacy', anchor_name='', training=None, codec='fp32', extra=None):
            context = contexts[rank][3] if rank else None
            blob = export_model(model, pair, codec=codec,
                                basis_id=context['basis_id'] if context else 0,
                                vocab=target_fit.shape[1], retired_target=args.target)
            digest = hashlib.sha256(blob).hexdigest()
            state = decode(blob)
            predictor = DecodedPredictor(state, contexts[rank][0] if rank else None, device=device)
            train_data, stop_data, validation_data = datasets(rank)
            scores = {role: evaluate(predictor, data, batch_size=64)
                      for role, data in [('fit', train_data), ('inner_stop', stop_data), ('validation', validation_data)]}
            state_path = folder / 'states' / (digest + '.aur1')
            state_path.parent.mkdir(exist_ok=True)
            if digest not in candidate_ids:
                state_path.write_bytes(blob)
                arrays = {}
                for role, result in scores.items():
                    arrays.update({role + '_' + k: v for k, v in result['per_item'].items()})
                    arrays[role + '_source_rows'] = result['source_row_indices']
                np.savez_compressed(state_path.with_suffix('.losses.npz'), **arrays)
                candidate_ids.add(digest)
            row = {
                'target_index': args.target, 'target_id': cache.registry[args.target]['model_id'],
                'pair': ','.join(map(str, pair)), 'label': label, 'family': state['family'],
                'anchor': anchor_name, 'objective': objective, 'basis_rank': rank,
                'feature_dim': state['feature_dim'], 'codec': codec,
                'actual_bytes': len(blob), 'actual_bits': 8 * len(blob),
                'resident_parameter_bytes': state['resident_parameter_bytes'],
                'candidate_id': digest, 'state_path': str(state_path.relative_to(folder)),
                'public_context_manifest': None if context is None else context['manifest_path'],
                'public_context_digest': None if context is None else context['manifest_sha256'],
                'public_numeric_bytes': 0 if context is None else context['deployed_public_numeric_bytes'],
                'fit_clusters': len(train_ids), 'inner_stop_clusters': len(stop_ids),
                'validation_clusters': len(target_val), 'seed': args.seed,
                'checkpoint_selected_on': 'inner_stop_TV', 'data_role': 'historical_fit_validation_development',
                'cheap': state['family'].startswith('anchor_'), 'status': 'scored_decoded',
                **{role + '_' + name: value for role, result in scores.items() for name, value in result['means'].items()},
            }
            if training:
                row.update(chosen_epoch=training.get('chosen_epoch'), completed_epochs=training.get('completed_epochs'),
                           training_seconds=training.get('seconds'),
                           last_fit_tv=training.get('last_fit_means', {}).get('tv'))
            if extra:
                row.update(extra)
            ledger.append(row)
            pd.DataFrame(ledger).to_csv(folder / 'candidate_ledger.csv', index=False)
            print(json.dumps({k: row[k] for k in ['target_index', 'pair', 'label', 'basis_rank', 'actual_bytes', 'fit_tv', 'inner_stop_tv', 'validation_tv']}), flush=True)
            dump(folder / 'progress.json', {'stage': label, 'candidates': len(ledger), 'unique_states': len(candidate_ids), 'last': row, 'updated_unix': time.time()})
            return row

        # Legacy cheap coefficients use only inner training labels. Their choice
        # for extension uses inner stopping TV, never outer validation.
        weights = fit_legacy_weights(tuple(cache.logits('fit', p) for p in pair),
                                     cache.logits('fit', args.target), train_ids,
                                     device=str(device), batch_size=32)
        dump(folder / 'legacy_inner_weights.json', {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in weights.items()})
        anchors = []
        cheap_specs = [('nnls', weights['nnls'], 'logit'), ('convex', weights['convex'], 'logit'),
                       ('equal_logit', np.array([.5, .5], np.float32), 'logit'),
                       ('equal_probability', np.array([.5, .5], np.float32), 'probability')]
        cheap_specs += [(f'nnls_temperature_{temperature}', weights['nnls'] / temperature, 'logit')
                        for temperature in (.5, .75, 1.25, 1.5, 2.)]
        cheap_specs += [('peer_' + str(pair[j]), np.eye(2, dtype=np.float32)[j], 'logit') for j in range(2)]
        for name, coefficient, kind in cheap_specs:
            model = FrozenAnchor(torch.as_tensor(coefficient, device=device), kind)
            row = offer(model, 'legacy_inner_' + name, anchor_name=name)
            anchors.append((name, model, row))
        competitive_name, competitive, competitive_row = min(anchors, key=lambda x: (x[2]['inner_stop_tv'], x[2]['actual_bytes'], x[0]))
        legacy_nnls = next(model for name, model, row in anchors if name == 'nnls')
        dump(folder / 'ablation_anchor_choice.json', {
            'selected_on': 'inner_stop_only', 'competitive_name': competitive_name,
            'competitive_candidate': competitive_row['candidate_id'], 'competitive_inner_stop_TV': competitive_row['inner_stop_tv'],
            'legacy_nnls_weights': numpy(legacy_nnls.weights).tolist(),
        })

        # Keep the genuinely historical decoded cheap states as strong fallback
        # predictions, re-encoded under the same sufficient AUR1 contract.
        legacy_dir = cache.root / 'fits' / f'target_{args.target}' / f'pair_{pair[0]}_{pair[1]}'
        from .data import _legacy_codec
        old_codec = _legacy_codec()
        seen = set()
        for record in json.loads((legacy_dir / 'envelope.json').read_text()):
            if record['family'] not in ('convex', 'nnls', 'temperature', 'logit_mean', 'probability_mean'):
                continue
            if record['candidate_id'] in seen:
                continue
            seen.add(record['candidate_id'])
            historical = old_codec.decode((cache.root / record['state_path']).read_bytes())
            coefficient = historical['arrays'][0] if historical['arrays'] else np.array([.5, .5], np.float32)
            kind = 'probability' if historical['family'] == 'probability_mean' else 'logit'
            model = FrozenAnchor(torch.as_tensor(coefficient, device=device), kind)
            offer(model, 'historical_fullfit_' + historical['family'], anchor_name=historical['family'],
                  extra={'anchor_fit_scope': 'historical_all2048fit; fallback_only_not_primary_factorial_anchor',
                         'legacy_state_digest': record['candidate_id'], 'legacy_serialized_bytes': record['actual_bytes']})

        def optimize(model, label, rank, loss, *, epochs=None, lr=None):
            train_data, stop_data, _ = datasets(rank)
            training_config = TrainConfig(loss=loss, epochs=epochs or args.epochs, lr=lr or args.lr,
                                         batch_size=args.batch_size, patience=args.patience, seed=args.seed,
                                         weight_decay=args.weight_decay)
            def callback(epoch, row, fitted, is_best):
                dump(folder / 'training_progress.json', {'label': label, 'rank': rank, **row})
                if epoch % 5 == 0:
                    print(json.dumps({'training': label, 'rank': rank, **row}), flush=True)
            info = fit(model, train_data, stop_data, training_config, callback=callback)
            dump(folder / 'training' / (label + f'_rank{rank}.json'), info)
            return info

        for rank in args.ranks:
            basis, ff, _, context = contexts[rank]
            train_data, stop_data, _ = datasets(rank)
            indices = torch.as_tensor(train_ids, device=device)
            for label, anchor_name, anchor in [('L0', 'nnls', legacy_nnls), ('L1', competitive_name, competitive)]:
                with torch.no_grad():
                    x = torch.cat([ff[indices], torch.ones((len(indices), 1), device=device)], dim=1).double()
                    pieces = []
                    for start_at in range(0, len(train_ids), 64):
                        ids = indices[start_at:start_at + 64]
                        peers = torch.stack([p[ids] for p in peer_fit], dim=1)
                        pieces.append((target_fit[ids] - anchor.logits(peers)) @ basis)
                    y = torch.cat(pieces).double()
                    gram = x.T @ x / len(x)
                    cross = x.T @ y / len(x)
                    penalty = torch.eye(x.shape[1], dtype=torch.float64, device=device)
                    penalty[-1, -1] = 0
                    for ridge in (1e-4, .01, 1., 100.):
                        coef = torch.linalg.solve(gram + ridge * penalty + 1e-10 * torch.eye(len(gram), device=device, dtype=torch.float64), cross).float()
                        model = AnchoredResidual(copy.deepcopy(anchor), basis, ff.shape[1])
                        model.linear.copy_(coef[:-1]); model.bias.copy_(coef[-1])
                        offer(model, label + '_ridge_' + str(ridge), rank=rank, objective='projected_logit_mse', anchor_name=anchor_name,
                              extra={'ablation_variant': label, 'ridge': ridge})
            for label, anchor_name, anchor in [('L2', 'nnls', legacy_nnls), ('L3', competitive_name, competitive)]:
                model = AnchoredResidual(copy.deepcopy(anchor), basis, ff.shape[1])
                check_peers, check_features, _ = train_data.batch(np.arange(min(16, len(train_data))), device)
                with torch.no_grad():
                    assert torch.equal(model(check_peers, check_features), anchor(check_peers)), 'Nested zero differs from anchor'
                info = optimize(model, label, rank, 'kl')
                assert any(v and v > 0 for v in info['first_step_gradient_norms'].values()), 'All initial gradients are zero'
                offer(model, label, rank=rank, objective='full_support_forward_kl', anchor_name=anchor_name, training=info,
                      extra={'ablation_variant': label})
                if label == 'L3':
                    for loss in ('tv', 'tv_kl'):
                        refined = copy.deepcopy(model)
                        refinement = optimize(refined, 'L4_' + loss, rank, loss, epochs=args.tv_epochs, lr=.0001)
                        offer(refined, 'L4_' + loss, rank=rank, objective='kl_then_' + loss,
                              anchor_name=anchor_name, training=refinement, extra={'ablation_variant': 'L4'})

        # This strengthened cheap menu is separate from the legacy anchor/loss
        # factorial. Every pair receives it and every policy can select it.
        for kind in ('logit', 'probability'):
            initial = competitive.weights if competitive.kind == kind else torch.full((2,), .5, device=device)
            calibrated = TrainableCalibration(initial, kind=kind, temperature=True)
            info = optimize(calibrated, 'calibration_KL_' + kind, 0, 'kl')
            offer(calibrated, 'calibration_KL_' + kind, objective='kl', anchor_name=kind, training=info)
            for loss in ('tv', 'tv_kl'):
                refined = copy.deepcopy(calibrated)
                refinement = optimize(refined, 'calibration_KL_then_' + loss + '_' + kind, 0, loss, epochs=args.tv_epochs, lr=.0001)
                offer(refined, 'calibration_KL_then_' + loss + '_' + kind, objective='kl_then_' + loss,
                      anchor_name=kind, training=refinement)
        final = pd.DataFrame(ledger)
        final.to_csv(folder / 'candidate_ledger.csv', index=False)
        # A factorial checkpoint/ridge is chosen on inner stopping, then its
        # untouched outer-development loss is reported alongside all candidates.
        ablation = final[final.get('ablation_variant', pd.Series('', index=final.index)).notna()].copy()
        selected = []
        for (_, _), group in ablation.groupby(['ablation_variant', 'basis_rank']):
            selected.append(group.sort_values(['inner_stop_tv', 'actual_bytes', 'candidate_id']).iloc[0].to_dict())
        pd.DataFrame(selected).to_csv(folder / 'anchor_loss_ablation.csv', index=False)
        assert all(sha(ROOT / name) == digest for name, digest in source.items()), 'Source changed during the cell; preserve outputs and inspect'
        dump(folder / 'complete.json', {'complete': True, 'target_index': args.target, 'pair': list(pair),
             'started_unix': start, 'finished_unix': time.time(), 'seconds': time.time() - start,
             'candidate_rows': len(ledger), 'unique_states': len(candidate_ids),
             'source_sha256': source, 'historical_test_opened': False,
             'outputs': {name: sha(folder / name) for name in ('candidate_ledger.csv', 'anchor_loss_ablation.csv', 'ablation_anchor_choice.json', 'data_roles.json')},
             'maximum_allocated_GPU_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
    except Exception as error:
        dump(folder / 'failure.json', {'error': repr(error), 'started_unix': start, 'failed_unix': time.time(), 'completed_candidates': len(ledger), 'outputs_retained': True})
        raise
