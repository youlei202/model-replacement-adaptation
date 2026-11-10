"""Replay selected AUR1 and original RBV2 states using only their selected peers."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from .codec import decode
from .data import _legacy_codec
from .sprint import ROOT, sha
from .inference import DecodedPredictor
from .session import response_statistics


class DeployedPredictor(torch.nn.Module):
    def __init__(self, row, device='cpu'):
        super().__init__()
        self.row = row
        path = Path(row['state_path'])
        if sha(path) != row['candidate_id'] or path.stat().st_size != row['actual_bytes']:
            raise ValueError('Deployed state hash or byte count mismatch')
        self.legacy = row['state_format'] == 'RBV2'
        self.state = (_legacy_codec().decode if self.legacy else decode)(path.read_bytes())
        self.register_buffer('device_reference', torch.empty(0, device=device))
        self.register_buffer('basis', None); self.register_buffer('mean', None); self.register_buffer('scale', None)
        self.context_kind = 'none'
        manifest = row.get('public_context_manifest')
        if manifest and isinstance(manifest, str):
            path = Path(manifest)
            if sha(path) != row['public_context_digest']:
                raise ValueError('Public context changed')
            meta = json.loads(path.read_text())
            if self.legacy:
                basispath = path.with_suffix('.npy')
                if sha(basispath) != meta['sha256']:
                    raise ValueError('Legacy public basis changed')
                self.basis = torch.tensor(np.load(basispath)[:, :self.state['basis_rank']], device=device)
                self.context_kind = 'legacy'
            else:
                for name, digest in meta['files'].items():
                    if sha(path.parent / name) != digest:
                        raise ValueError('Public context payload changed')
                with np.load(path.parent / 'features.npz') as arrays:
                    self.mean = torch.tensor(arrays['mean'], device=device)
                    self.scale = torch.tensor(arrays['scale'], device=device)
                if meta.get('kind') == 'response_statistics_12':
                    self.context_kind = 'stats'
                else:
                    self.context_kind = 'logit'
                    self.basis = torch.tensor(np.load(path.parent / 'basis.npy'), device=device)
        if self.legacy:
            spec = importlib.util.spec_from_file_location('aurum_legacy_prediction', ROOT / 'experiments/rich_inference.py')
            self.legacy_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.legacy_module)
        else:
            self.predictor = DecodedPredictor(self.state, self.basis, device=device)

    def forward(self, peers, features=None):
        if self.legacy:
            result, is_probability = self.legacy_module.predict(self.state, list(peers.unbind(1)), self.basis)
            return result.clamp_min(torch.finfo(result.dtype).tiny).log() if is_probability else torch.log_softmax(result, -1)
        if self.context_kind == 'stats':
            raw = response_statistics(tuple(peers.unbind(1)))
        elif self.context_kind == 'logit':
            raw = torch.cat([peer @ self.basis for peer in peers.unbind(1)], 1)
        else:
            raw = None
        transformed = None if raw is None else (raw - self.mean) / self.scale
        return self.predictor(peers, transformed)
