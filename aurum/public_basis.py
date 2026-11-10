"""Optional larger peer-only public output bases; no target data are loaded.

Prepared for the diagnostic stage, not executed by the initial six-cell ablation.
Every new basis is separate from the exact legacy128 basis. Joint and fixed-peer
policies must receive the same representation menu and its full public-resource
cost must be measured. Importing this module performs no computation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import ROOT, PAIRS, RichDevelopmentCache, file_sha256


def build(pair, rank, output, *, device='cuda:0', seed=20261110, iterations=2):
    import torch
    pair = tuple(pair); output = Path(output).resolve()
    if pair not in PAIRS or rank not in (128, 256, 512):
        raise ValueError('Use an original admissible pair and rank128,256,or512')
    allowed = ROOT / 'runs/aurum/development/public_contexts_extended'
    if not output.is_relative_to(allowed):
        raise ValueError('Extended bases must use their separate new AURUM output directory')
    output.mkdir(parents=True, exist_ok=True)
    stem = f'pair_{pair[0]}_{pair[1]}_rank{rank}_seed{seed}_niter{iterations}'
    path = output / f'{stem}.npy'; meta_path = output / f'{stem}.json'
    if path.exists() or meta_path.exists():
        raise FileExistsError('Existing public basis is preserved; choose a new declared representation ID')
    data = RichDevelopmentCache(); source_hashes = {}
    for peer in pair:
        source = data.root / 'logits/fit' / f'model_{peer}.npy'
        source_hashes[str(peer)] = file_sha256(source)
        if source_hashes[str(peer)] != data.metadata('fit', peer)['sha256']:
            raise ValueError('Peer fit response bank drift')
    start = time.perf_counter(); code_hash = file_sha256(Path(__file__))
    torch.set_num_threads(4); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()
    # Target-independent public representation: only the two callable peers' fit outputs.
    values = torch.cat([torch.tensor(np.asarray(data.logits('fit', p)), device=device) for p in pair])
    pair_seed = int(seed) + 10 * pair[0] + pair[1]
    torch.manual_seed(pair_seed)
    with torch.inference_mode():
        _, singular, right = torch.svd_lowrank(values, q=rank + 16, niter=iterations)
        basis = right[:, :rank].contiguous().cpu().numpy()
    if not np.isfinite(basis).all() or file_sha256(Path(__file__)) != code_hash:
        raise ValueError('Invalid public basis or source changed during construction')
    with path.open('xb') as handle:
        np.save(handle, basis, allow_pickle=False)
    record = {'representation_id': stem, 'peers': list(pair), 'rank': rank, 'vocab': basis.shape[0],
              'target_outputs_used': False, 'fit_peer_logits_sha256': source_hashes,
              'public_basis_sha256': file_sha256(path), 'serialized_bytes': path.stat().st_size,
              'resident_float32_bytes': basis.nbytes, 'basis_seed': pair_seed,
              'algorithm': f'torch.svd_lowrank q={rank+16},niter={iterations}; stacked selected-peer fit centered logits',
              'fit_rows_per_peer': 2048, 'legacy_basis_replaced': False,
              'elapsed_seconds': time.perf_counter() - start, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'code_sha256': code_hash, 'singular_values': singular[:rank].cpu().tolist(),
              'strong_control_requirement': 'Expose this representation and adapter menu equally to every pair/policy; record full public state in resource measurement.'}
    with meta_path.open('x') as handle:
        json.dump(record, handle, indent=2); handle.write('\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=int, nargs=2, required=True)
    parser.add_argument('--rank', type=int, choices=[128, 256, 512], required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/aurum/development/public_contexts_extended')
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--iterations', type=int, default=2)
    args = parser.parse_args(); result = build(args.pair, args.rank, args.out, device=args.device,
                                              seed=args.seed, iterations=args.iterations)
    print(json.dumps({k: v for k, v in result.items() if k != 'singular_values'}, indent=2))


if __name__ == '__main__':
    main()
