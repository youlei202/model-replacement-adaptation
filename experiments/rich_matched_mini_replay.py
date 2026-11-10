#!/usr/bin/env python3
"""A four-document arithmetic replay slice of the completed supplemental cohort."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rich_setup import OUT, MODELS, digest, dump
from rich_codec import decode
SUP = OUT / 'matched_resource_supplement'


def main():
    assert json.load(open(SUP / 'quality_complete.json'))['complete']
    destination = SUP / 'mini_replay'
    if destination.exists():
        raise FileExistsError('Preserve immutable replay output')
    destination.mkdir()
    lock = json.load(open(SUP / 'prospective_lock.json')); ids = lock['test_item_ids'][:4]
    cohort = pd.read_parquet(SUP / 'cohort.parquet').set_index('item_id').loc[ids].reset_index()
    dump(destination / 'items.json', [{'item_id': r.item_id, 'cluster_id': r.cluster_id, 'source': r.source,
                                      'canonical_token_ids': r.context_ids.tolist(), 'context_tokens': int(r.context_tokens)} for r in cohort.itertuples()])
    models = []
    for i, (model, revision, role) in enumerate(MODELS):
        src = SUP / 'logits' / f'model_{i}.npy'; meta = json.load(open(src.with_suffix('.json')))
        dest = destination / src.name; np.save(dest, np.asarray(np.load(src, mmap_mode='r')[:4]).copy())
        models.append({'global_id': i, 'model_id': model, 'revision': revision, 'role': role, 'file': dest.name,
                       'full_bank_sha256': meta['sha256'], 'full_bank_shape': [2048, 151936],
                       'slice_shape': [4, 151936], 'slice_sha256': digest(dest), 'dtype': 'float32_centered_logits'})
    states = {}
    for row in lock['decisions']:
        if not row['feasible_on_development']:
            continue
        sid = row['candidate_id']; blob = (OUT / row['state_path']).read_bytes(); state = decode(blob)
        assert state['basis_rank'] == 0
        if sid not in states:
            dest = destination / 'states' / f'{sid}.bin'; dest.parent.mkdir(exist_ok=True); dest.write_bytes(blob)
            states[sid] = {'state_digest': sid, 'file': str(dest.relative_to(destination)), 'family': state['family'],
                           'peer_indices': state['peers'], 'target_indices': [], 'basis_rank': 0,
                           'serialized_bytes': len(blob), 'source_state_path': row['state_path']}
        if row['target_index'] not in states[sid]['target_indices']:
            states[sid]['target_indices'].append(row['target_index'])
    loss = pd.read_parquet(SUP / 'item_losses.parquet'); loss[loss.item_id.isin(ids)].to_csv(destination / 'recorded_losses.csv', index=False)
    shutil.copy2(ROOT / 'experiments/rich_codec.py', destination / 'rich_codec.py')
    template = OUT / 'reproduce/mini_replay/replay_demo.py'
    code = template.read_text().replace('Existing8documents', 'Existing4supplement_documents')
    (destination / 'replay_demo.py').write_text(code)
    (destination / 'README.md').write_text('''# Four-document supplemental numerical replay

Run `python replay_demo.py` with NumPy. The predictor uses peer arrays0/1/2;
target arrays3/4 are read separately for loss verification. No weights, network,
GPU or refitting is needed. Copy this directory to a temporary location first:
the demo writes `demo_result.json`, whose original is retained and hashed here.

These are the first4 already-evaluated documents from the separately locked
2048-document actual-resource-cap supplement, not another independent test or
a substitute for full-cohort inference. All8 selected scalar states and12
target-state combinations are replayed over the full151936 output slots.

The example compares canonical NumPy FP64 probabilities with recorded B200
FP32 metric arithmetic at a reported strict1e-6 tolerance; argmax is also checked.
The original experiment's separate Torch CPU precision failure remains in the
original eight-document fixture and is not replaced by this new example.
All8 new same-B200 live/cache bridge checks gave full-support TV difference0.

Full6.2GB supplemental model-response banks remain in the complete workspace;
hashes and model revisions are recorded. This slice cannot reproduce all2048
test losses or the original fitting/development search and is labeled accordingly.
''')
    manifest = {'scope': 'Four already-evaluated supplemental documents for arithmetic replay only',
                'selection_lock_sha256': digest(SUP / 'prospective_lock.json'), 'test_slice_indices': list(range(4)),
                'models': models, 'states': list(states.values()), 'full_banks_excluded': True,
                'template_replay_source_sha256': digest(template), 'hashes': {}}
    dump(destination / 'manifest.json', manifest)
    process = subprocess.run([sys.executable, str(destination / 'replay_demo.py')], capture_output=True, text=True)
    (destination / 'demo_run.log').write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError('Replay failed; outputs preserved')
    result = json.load(open(destination / 'demo_result.json'))
    assert len(result['results']) == 12
    # No hidden tolerance changes. A failure remains explicitly recorded.
    manifest['strict_1e_6_all_passed'] = all(r['strict_1e_6_check_passed'] for r in result['results'])
    manifest['argmax_all_match'] = all(r['predicted_argmax_matches'] for r in result['results'])
    cache = destination / '__pycache__'
    if cache.exists():
        shutil.rmtree(cache)
    manifest['hashes'] = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    dump(destination / 'manifest.json', manifest)
    print(json.dumps({'directory': str(destination), 'bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
                      'max_TV_difference': max(r['max_abs_TV_difference_CPU_FP64_vs_recorded_GPU_FP32'] for r in result['results']),
                      'strict_1e_6_all_passed': manifest['strict_1e_6_all_passed']}, indent=2))


if __name__ == '__main__':
    main()
