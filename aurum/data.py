"""Read-only bridge to the original rich fit/validation response banks.

No model weights or historical test responses are loaded by this module. All row
indices refer to the original collector order within the named legacy role.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / 'experiments/runs/v2/rich_behavior'
PAIRS = ((0, 1), (0, 2), (1, 2))
TARGETS = (3, 4)
DEVELOPMENT_ROLES = ('fit', 'validation')
VOCAB = 151936


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _legacy_codec():
    spec = importlib.util.spec_from_file_location('aurum_readonly_legacy_codec', ROOT / 'experiments/rich_codec.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RichDevelopmentCache:
    """Memory map only fit/validation data, rejecting every other role."""

    def __init__(self, legacy_root: Path | str = LEGACY_ROOT):
        self.root = Path(legacy_root).resolve()
        self.registry = {int(x['global_id']): x for x in json.loads((self.root / 'model_registry.json').read_text())}
        columns = ['source', 'source_key', 'document_sha256', 'context_ids', 'cluster_id', 'item_id', 'role', 'position', 'context_tokens']
        self.documents = pd.read_parquet(self.root / 'documents.parquet', columns=columns,
                                         filters=[('role', 'in', list(DEVELOPMENT_ROLES))])
        self._items = {role: self.documents[self.documents.role.eq(role)].reset_index(drop=True)
                       for role in DEVELOPMENT_ROLES}
        self._arrays: dict[tuple[str, int], np.memmap] = {}
        self._metadata: dict[tuple[str, int], dict] = {}
        self._bases: dict[tuple[int, int], np.memmap] = {}
        for role, count in [('fit', 2048), ('validation', 1024)]:
            frame = self._items[role]
            if len(frame) != count or frame.item_id.duplicated().any() or frame.cluster_id.duplicated().any():
                raise ValueError(f'Invalid {role} cohort dimensions or duplicate semantic keys')
            if not frame.groupby('source').size().eq(count // 2).all():
                raise ValueError(f'Unexpected {role} source mixture')
        if set(self._items['fit'].cluster_id) & set(self._items['validation'].cluster_id):
            raise ValueError('Fit/validation document clusters overlap')
        self.cohort_digest = json.loads((self.root / 'data_lock.json').read_text())['documents_sha256']

    @staticmethod
    def _role(role: str) -> None:
        if role not in DEVELOPMENT_ROLES:
            raise ValueError('AURUM development bridge permits fit and validation only; historical test outcomes are not development inputs')

    def items(self, role: str) -> pd.DataFrame:
        self._role(role)
        return self._items[role].copy()

    def metadata(self, role: str, model_id: int) -> dict:
        self._role(role)
        key = (role, int(model_id))
        if key not in self._metadata:
            if model_id not in self.registry:
                raise ValueError('Unknown pinned model ID')
            path = self.root / 'logits' / role / f'model_{model_id}.json'
            meta = json.loads(path.read_text())
            expected = self.registry[model_id]
            assertions = [meta['complete'], meta['role'] == role,
                          meta['model_id'] == expected['model_id'],
                          meta['model_revision'] == expected['revision'],
                          meta['cohort_sha256'] == self.cohort_digest,
                          meta['vocab'] == VOCAB,
                          meta['completed'] == meta['expected'] == len(self._items[role]),
                          meta['storage_dtype'] == 'float32_centered_logits',
                          meta['inference_microbatch'] == 1,
                          meta['output_weight_shape'][0] == VOCAB]
            if not all(assertions):
                raise ValueError(f'Incompatible bank metadata: {path}')
            self._metadata[key] = meta
        return self._metadata[key].copy()

    def logits(self, role: str, model_id: int) -> np.memmap:
        self.metadata(role, int(model_id))
        key = (role, int(model_id))
        if key not in self._arrays:
            path = self.root / 'logits' / role / f'model_{model_id}.npy'
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (len(self._items[role]), VOCAB):
                raise ValueError(f'Unexpected bank shape/dtype: {path}')
            if array.flags.writeable:
                raise ValueError('The development bank must be read-only')
            self._arrays[key] = array
        return self._arrays[key]

    def basis(self, pair: tuple[int, int] | list[int], rank: int = 128) -> np.memmap:
        pair = tuple(map(int, pair))
        if pair not in PAIRS or not 1 <= rank <= 128:
            raise ValueError('Unknown original pair or basis rank')
        if pair not in self._bases:
            path = self.root / 'public_contexts' / f'pair_{pair[0]}_{pair[1]}.npy'
            meta = json.loads(path.with_suffix('.json').read_text())
            if meta['peers'] != list(pair) or meta['target_outputs_used'] or meta['basis_rank'] != 128:
                raise ValueError('Invalid peer-only public basis provenance')
            if meta['fit_peer_logits_sha256'] != {str(i): self.metadata('fit', i)['sha256'] for i in pair}:
                raise ValueError('Public basis refers to different peer fit responses')
            basis = np.load(path, mmap_mode='r', allow_pickle=False)
            if basis.shape != (VOCAB, 128) or basis.dtype != np.float32 or basis.flags.writeable:
                raise ValueError('Invalid public basis array')
            self._bases[pair] = basis
        return self._bases[pair][:, :rank]

    def basis_metadata(self, pair) -> dict:
        self.basis(pair)
        pair = tuple(pair)
        return json.loads((self.root / 'public_contexts' / f'pair_{pair[0]}_{pair[1]}.json').read_text())

    def grouped_folds(self, seed: int = 20261110, n_folds: int = 3) -> list[dict]:
        """Source-stratified whole-cluster folds using metadata, never outcomes.

        Each fit document appears in stopping exactly once; validation stays
        outside every fold. A basis computed from all fit peers is permitted
        as the unchanged target-independent representation in this ablation.
        """
        if n_folds < 2:
            raise ValueError('At least two folds required')
        frame = self._items['fit']
        assignment = np.full(len(frame), -1, dtype=np.int64)
        for source, group in frame.groupby('source', sort=True):
            if len(group) < n_folds:
                raise ValueError('Source stratum has too few clusters')
            ordered = sorted(group.index, key=lambda i: hashlib.sha256(
                f'aurum-inner-v1|{seed}|{source}|{frame.at[i, "cluster_id"]}'.encode()).hexdigest())
            for position, row in enumerate(ordered):
                assignment[row] = position % n_folds
        return [{'fold': fold, 'seed': seed, 'train_indices': np.flatnonzero(assignment != fold),
                 'stop_indices': np.flatnonzero(assignment == fold)} for fold in range(n_folds)]

    def legacy_anchor(self, target: int, pair, family: str) -> dict:
        """Best stored decoded family anchor, for legacy replay/fallback only.

        Its coefficients used all legacy fit rows. Refit on inner-training rows
        with fit_legacy_weights before using an inner stopping loss for training.
        """
        pair = tuple(pair)
        if target not in TARGETS or pair not in PAIRS or family not in ('nnls', 'convex'):
            raise ValueError('Unsupported legacy anchor cell')
        folder = self.root / 'fits' / f'target_{target}' / f'pair_{pair[0]}_{pair[1]}'
        ledger = pd.read_parquet(folder / 'validation_candidates.parquet')
        row = ledger[ledger.family.eq(family)].sort_values(['validation_tv', 'actual_bytes', 'candidate_id']).iloc[0].to_dict()
        path = folder / 'states' / (row['candidate_id'] + '.bin')
        if not path.exists():
            raise FileNotFoundError(f'Legacy best family state not retained: {path}')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['candidate_id']:
            raise ValueError('Legacy state digest mismatch')
        state = _legacy_codec().decode(blob)
        if state['peers'] != list(pair) or state['family'] != family or len(blob) != row['actual_bytes']:
            raise ValueError('Legacy state registry mismatch')
        return {'state': state, 'blob': blob, 'path': str(path), 'ledger': row,
                'role': 'legacy_all_fit_fallback_or_replay_only'}


def fit_legacy_weights(peers, target_logits, indices=None, *, device='cpu', batch_size: int = 32) -> dict:
    """Fit the unchanged squared-logit NNLS/convex anchors on supplied rows only.

    Accumulates Gram and cross-products in float64 using batches so large banks
    need not be copied to a device. The legacy ridge floor and two-peer closed
    form are preserved; no outer validation labels enter this helper.
    """
    import scipy.optimize
    import torch

    peers = tuple(peers)
    if len(peers) != 2 or any(x.shape != target_logits.shape for x in peers):
        raise ValueError('The initial AURUM legacy-weight helper expects two aligned peers')
    indices = np.arange(len(target_logits)) if indices is None else np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(target_logits):
        raise ValueError('Invalid training indices')
    gram = torch.zeros((2, 2), dtype=torch.float64, device=device)
    cross = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            x = torch.stack([torch.as_tensor(np.asarray(p[chosen]).copy(), dtype=torch.float64, device=device).reshape(-1)
                             for p in peers])
            y = torch.as_tensor(np.asarray(target_logits[chosen]).copy(), dtype=torch.float64, device=device).reshape(-1)
            gram += x @ x.T
            cross += x @ y
    G = gram.cpu().numpy(); c = cross.cpu().numpy()
    regularizer = max(float(np.trace(G)) * 1e-12, 1e-12)
    chol = np.linalg.cholesky(G + regularizer * np.eye(2))
    nnls = scipy.optimize.nnls(chol.T, np.linalg.solve(chol, c))[0].astype(np.float32)
    denominator = G[0, 0] + G[1, 1] - 2 * G[0, 1]
    first = np.clip((c[0] - c[1] - G[0, 1] + G[1, 1]) / max(denominator, 1e-30), 0, 1)
    convex = np.asarray([first, 1 - first], dtype=np.float32)
    return {'nnls': nnls, 'convex': convex, 'gram': G, 'cross': c,
            'training_count': len(indices), 'regularizer': regularizer,
            'training_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            'accumulation_dtype': 'float64', 'legacy_math_preserved': True}


def project_peer_features(peers, basis, indices=None, *, device='cpu', batch_size: int = 32) -> np.ndarray:
    """Unstandardized legacy peer projection features, without a constant column.

    AURUM models have their own charged bias. If standardization is used, fit it
    only from the selected peers' permitted fit responses and record it as public.
    """
    import torch
    indices = np.arange(len(peers[0])) if indices is None else np.asarray(indices, dtype=np.int64)
    u = torch.as_tensor(np.asarray(basis).copy(), dtype=torch.float32, device=device)
    result = np.empty((len(indices), len(peers) * u.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            values = [torch.as_tensor(np.asarray(p[chosen]).copy(), dtype=torch.float32, device=device) @ u for p in peers]
            result[start:start + len(chosen)] = torch.cat(values, dim=-1).cpu().numpy()
    return result
