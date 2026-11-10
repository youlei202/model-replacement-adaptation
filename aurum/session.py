"""Reusable development context and literal-state catalog writer."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import RichDevelopmentCache
from .sprint import ROOT, dump, sha, gpu_bank, numpy, public_context
from .train import ArrayData, TrainConfig, fit, evaluate
from .codec import export_model, decode
from .inference import DecodedPredictor


def response_statistics(peers):
    """Twelve response-only features; no target, token text, or source labels."""
    values = []
    for offset in range(0, len(peers[0]), 64):
        p = torch.stack([x[offset:offset + 64] for x in peers], dim=1)
        logp = torch.log_softmax(p, -1); prob = logp.exp()
        entropy = -(prob * logp).sum(-1)
        top = prob.topk(2, dim=-1).values
        topscore = p.topk(2, dim=-1).values
        agreement = (p[:, 0].argmax(-1) == p[:, 1].argmax(-1)).float()[:, None]
        tv = (.5 * (prob[:, 0] - prob[:, 1]).abs().sum(-1))[:, None]
        kl01 = (prob[:, 0] * (logp[:, 0] - logp[:, 1])).sum(-1)[:, None]
        kl10 = (prob[:, 1] * (logp[:, 1] - logp[:, 0])).sum(-1)[:, None]
        values.append(torch.cat([entropy, top[:, :, 0], top[:, :, 0] - top[:, :, 1],
                                 topscore[:, :, 0] - topscore[:, :, 1], agreement, tv, kl01, kl10], dim=1))
    return torch.cat(values)


class DevelopmentSession:
    def __init__(self, output, target, pair, *, device='cuda:0', seed=20261110, fold=0):
        self.folder = Path(output).resolve()
        if self.folder.exists() and any(self.folder.iterdir()):
            raise ValueError('Preserve outputs; use a fresh output directory')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.target = target; self.pair = tuple(pair); self.device = torch.device(device)
        self.seed = seed; self.fold = fold; self.started = time.time()
        torch.set_num_threads(4); torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.cache = RichDevelopmentCache()
        self.peers = {role: tuple(gpu_bank(self.cache.logits(role, p), self.device) for p in pair)
                      for role in ('fit', 'validation')}
        self.targets = {role: gpu_bank(self.cache.logits(role, target), self.device) for role in self.peers}
        split = self.cache.grouped_folds(seed=seed)[fold]
        self.train_ids = split['train_indices']; self.stop_ids = split['stop_indices']
        self.contexts = {}; self.rows = []; self.ids = set()
        dump(self.folder / 'data_roles.json', {
            'fit_cluster_ids': self.cache.items('fit').cluster_id.tolist(),
            'validation_cluster_ids': self.cache.items('validation').cluster_id.tolist(),
            'train_indices': self.train_ids.tolist(), 'inner_stop_indices': self.stop_ids.tolist(),
            'seed': seed, 'fold': fold, 'target': target, 'pair': list(pair),
            'historical_test_opened': False,
        })

    def context(self, key):
        if key in self.contexts:
            return self.contexts[key]
        if key == 'none':
            result = (None, {role: torch.empty((len(t), 0), device=self.device) for role, t in self.targets.items()}, None)
        elif key.startswith('logit'):
            rank = int(key[5:])
            basis, ff, vf, meta = public_context(self.cache, self.pair, rank, self.peers['fit'], self.peers['validation'], self.device)
            result = (basis, {'fit': ff, 'validation': vf}, meta)
        elif key == 'stats':
            raw = {role: response_statistics(peers) for role, peers in self.peers.items()}
            mean = raw['fit'].mean(0); scale = raw['fit'].std(0, unbiased=False)
            scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            features = {role: (x - mean) / scale for role, x in raw.items()}
            path = ROOT / 'runs/aurum/public_contexts' / f'pair_{self.pair[0]}_{self.pair[1]}' / 'response_stats'
            path.mkdir(parents=True, exist_ok=True)
            if not (path / 'context.json').exists():
                np.savez(path / 'features.npz', mean=numpy(mean), scale=numpy(scale),
                         fit=numpy(features['fit']), validation=numpy(features['validation']))
                dump(path / 'context.json', {
                    'pair': list(self.pair), 'basis_rank': 0, 'feature_dim': 12,
                    'basis_id': 300 + 10 * self.pair[0] + self.pair[1],
                    'kind': 'response_statistics_12', 'target_outputs_used': False,
                    'definition': 'per-peer entropy, maxprob, probmargin, logitmargin; top1agreement, TV, KL01, KL10; selectedpeerfit standardization',
                    'deployed_public_numeric_bytes': mean.numel() * 8,
                    'files': {'features.npz': sha(path / 'features.npz')},
                })
            meta = json.loads((path / 'context.json').read_text())
            with np.load(path / 'features.npz') as saved:
                assert np.array_equal(saved['mean'], numpy(mean))
                assert np.array_equal(saved['scale'], numpy(scale))
            meta.update(manifest_path=str(path / 'context.json'), manifest_sha256=sha(path / 'context.json'))
            result = (None, features, meta)
        else:
            raise ValueError('Unknown public context')
        self.contexts[key] = result
        return result

    def datasets(self, context='none'):
        _, f, _ = self.context(context)
        return (ArrayData(self.peers['fit'], f['fit'], self.targets['fit'], self.train_ids, role='inner_train'),
                ArrayData(self.peers['fit'], f['fit'], self.targets['fit'], self.stop_ids, role='inner_stop'),
                ArrayData(self.peers['validation'], f['validation'], self.targets['validation'], role='outer_validation'))

    def offer(self, model, label, *, context='none', codec='fp32', extra=None):
        basis, features, meta = self.context(context)
        blob = export_model(model, self.pair, codec, basis_id=meta['basis_id'] if meta else 0,
                            vocab=self.targets['fit'].shape[1], retired_target=self.target)
        digest = hashlib.sha256(blob).hexdigest(); state = decode(blob)
        candidate = DecodedPredictor(state, basis, device=self.device)
        scores = {role: evaluate(candidate, data, batch_size=64) for role, data in
                  zip(('fit', 'inner_stop', 'validation'), self.datasets(context))}
        path = self.folder / 'states' / (digest + '.aur1'); path.parent.mkdir(exist_ok=True)
        if digest not in self.ids:
            path.write_bytes(blob)
            arrays = {role + '_' + key: value for role, score in scores.items() for key, value in score['per_item'].items()}
            arrays.update({role + '_source_rows': score['source_row_indices'] for role, score in scores.items()})
            np.savez_compressed(path.with_suffix('.losses.npz'), **arrays); self.ids.add(digest)
        row = {'target_index': self.target, 'target_id': self.cache.registry[self.target]['model_id'],
               'pair': ','.join(map(str, self.pair)), 'label': label, 'family': state['family'],
               'context': context, 'codec': codec, 'basis_rank': state['basis_rank'], 'feature_dim': state['feature_dim'],
               'actual_bytes': len(blob), 'actual_bits': len(blob) * 8,
               'resident_parameter_bytes': state['resident_parameter_bytes'], 'candidate_id': digest,
               'state_path': str(path), 'public_context_manifest': meta['manifest_path'] if meta else None,
               'public_context_digest': meta['manifest_sha256'] if meta else None,
               'public_numeric_bytes': meta['deployed_public_numeric_bytes'] if meta else 0,
               'cheap': state['family'].startswith('anchor_'), 'seed': self.seed, 'inner_fold': self.fold,
               'selection_data': 'historical_fit_validation_development', 'scored_decoded': True,
               **{role + '_' + k: v for role, score in scores.items() for k, v in score['means'].items()},
               **(extra or {})}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.folder / 'candidate_ledger.csv', index=False)
        dump(self.folder / 'progress.json', {'stage': label, 'candidates': len(self.rows), 'last': row, 'updated_unix': time.time()})
        print(json.dumps({k: row[k] for k in ('label', 'context', 'codec', 'actual_bytes', 'fit_tv', 'inner_stop_tv', 'validation_tv')}), flush=True)
        return row

    def optimize(self, model, label, context, **config):
        train, stop, _ = self.datasets(context)
        def progress(epoch, row, model, best):
            dump(self.folder / 'training_progress.json', {'label': label, **row})
        result = fit(model, train, stop, TrainConfig(seed=self.seed, **config), callback=progress)
        dump(self.folder / 'training' / (label + '.json'), result)
        return result

    def finish(self, specification):
        dump(self.folder / 'complete.json', {'complete': True, 'specification': specification,
             'seconds': time.time() - self.started, 'candidate_rows': len(self.rows), 'unique_states': len(self.ids),
             'candidate_ledger_sha256': sha(self.folder / 'candidate_ledger.csv'),
             'maximum_allocated_GPU_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None})
