"""Check collected development-bank integrity and recorded role separation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .confirmation_data import exposure_inventory, file_sha256, _write_json


def audit(folder):
    folder = Path(folder); lock = json.loads((folder / 'cohort_lock.json').read_text())
    if lock['purpose'] != 'additional_development':
        raise ValueError('This audit only opens explicitly exposed development data')
    rows = pd.read_parquet(folder / 'cohort.parquet')
    assert file_sha256(folder / 'cohort.parquet') == lock['cohort_sha256']
    assert rows.item_id.tolist() == lock['item_ids']
    banks = []
    for model in lock['model_indices']:
        path = folder / 'logits' / f'model_{model}.npy'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['complete'] and meta['completed'] == meta['expected'] == len(rows)
        assert meta['cohort_sha256'] == lock['cohort_sha256'] and meta['role'] == 'aurum_new_development'
        assert meta['inference_microbatch'] == 1 and meta['runtime']['dtype'] == 'bfloat16'
        digest = file_sha256(path); assert digest == meta['sha256']
        bank = np.load(path, mmap_mode='r'); assert bank.shape == (len(rows), 151936) and bank.dtype == np.float32
        max_center = 0.
        for offset in range(0, len(bank), 128):
            block = bank[offset:offset + 128]
            assert np.isfinite(block).all()
            max_center = max(max_center, float(np.abs(block.mean(-1, dtype=np.float64)).max()))
        assert max_center < 1e-5
        banks.append({'model_index': model, 'sha256': digest, 'shape': list(bank.shape),
                      'file_bytes': path.stat().st_size, 'maximum_absolute_row_mean': max_center,
                      'inference_seconds': sum(t['seconds'] for t in meta['batch_timings']),
                      'seconds_including_load': meta['seconds_including_load'],
                      'memory_peak_allocated': meta['memory_peak_allocated'],
                      'runtime': meta['runtime']})
    remaining, exposure = exposure_inventory()
    assert not set(rows.item_id) & set(remaining.item_id)
    assert min(exposure['remaining_source_counts'].values()) >= 2048
    result = {'complete': True, 'purpose': 'additional_development', 'not_confirmation': True,
              'documents': len(rows), 'cohort_sha256': lock['cohort_sha256'],
              'source_counts': rows.groupby('source').size().to_dict(), 'banks': banks,
              'remaining_unexposed_documents': len(remaining),
              'remaining_unexposed_source_counts': exposure['remaining_source_counts'],
              'historical_and_confirmation_outcomes_opened': False}
    _write_json(folder / 'collection_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--folder', required=True, type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.folder), indent=2))
