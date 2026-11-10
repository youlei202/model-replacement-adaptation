"""Synthetic tests only: never select or collect actual confirmation rows."""
import json
from pathlib import Path

import pandas as pd
import pytest

from aurum import confirmation_data as module
from aurum.data import file_sha256


def fixture_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'AURUM', tmp_path / 'runs/aurum')
    monkeypatch.setattr(module, 'ACCESS_LOG', tmp_path / 'runs/aurum/data_access_registry.jsonl')
    legacy = tmp_path / 'legacy'; legacy.mkdir()
    monkeypatch.setattr(module, 'LEGACY_ROOT', legacy)
    (legacy / 'model_registry.json').write_text('[]')
    artifacts = []
    for relative in ['aurum/confirmation_data.py', 'aurum/codec.py', 'aurum/inference.py', 'state.bin']:
        path = tmp_path / relative; path.parent.mkdir(exist_ok=True, parents=True); path.write_text('immutable synthetic content')
        artifacts.append({'path': relative, 'sha256': file_sha256(path)})
    spec = {'campaign': 'AURUM', 'population': 'synthetic test population', 'targets': [3, 4],
            'admissible_pairs': [[0, 1], [0, 2], [1, 2]], 'budgets_bytes': [512], 'budget_weights': [1.0],
            'primary_endpoint': 'TV', 'contrast_family': ['fixed vs joint'],
            'historical_access_statement': 'Synthetic fixture only, no real data access',
            'confirmation_design': {'n_documents': 4, 'selection_seed': 123, 'inference_microbatch': 1,
                                    'inference_dtype': 'bfloat16', 'vocab': 151936}}
    lock = tmp_path / 'method_lock.json'
    lock.write_text(json.dumps({'locked_at_utc': '2026-11-10T00:00:00Z', 'specification': spec, 'artifacts': artifacts}))
    return lock, legacy


def test_changed_artifact_prevents_confirmation_preparation(tmp_path, monkeypatch):
    lock, _ = fixture_lock(tmp_path, monkeypatch)
    (tmp_path / 'state.bin').write_text('changed after freeze')
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        module.prepare(lock, tmp_path / 'runs/aurum/confirmation/synthetic')
    assert not (tmp_path / 'runs/aurum/confirmation').exists()


def test_incomplete_strong_control_family_is_rejected(tmp_path, monkeypatch):
    lock, _ = fixture_lock(tmp_path, monkeypatch)
    value = json.loads(lock.read_text()); value['specification']['admissible_pairs'] = [[1, 2]]
    lock.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='incomplete strong-control'):
        module._check_freeze(lock)


def test_aur1_only_state_family_is_accepted(tmp_path, monkeypatch):
    lock, _ = fixture_lock(tmp_path, monkeypatch)
    state = tmp_path / 'state.bin'
    state.rename(tmp_path / 'state.aur1')
    value = json.loads(lock.read_text())
    for record in value['artifacts']:
        if record['path'] == 'state.bin':
            record['path'] = 'state.aur1'
    lock.write_text(json.dumps(value))
    _, spec, _, _ = module._check_freeze(lock)
    assert spec['campaign'] == 'AURUM'


def test_synthetic_metadata_selection_balanced_and_write_once(tmp_path, monkeypatch):
    lock, legacy = fixture_lock(tmp_path, monkeypatch)
    rows = []
    for source in module.SOURCES:
        for index in range(5):
            key = f'{source}:{index}'
            rows.append({'source': source, 'source_key': key, 'document_sha256': key,
                         'cluster_id': key, 'item_id': key, 'role': 'test_pool',
                         'context_ids': [1, 2, 3], 'position': 3, 'context_tokens': 3, 'document_token_count': 10})
    frame = pd.DataFrame(rows); frame.to_parquet(legacy / 'documents.parquet', index=False)
    monkeypatch.setattr(module, 'exposure_inventory', lambda **kwargs: (frame.copy(), {'synthetic': True}))
    output = tmp_path / 'runs/aurum/confirmation/synthetic'
    result = module.prepare(lock, output)
    cohort = pd.read_parquet(output / 'cohort.parquet')
    assert len(cohort) == 4 and cohort.groupby('source').size().to_dict() == {s: 2 for s in module.SOURCES}
    assert result['selected_without_response_access']
    assert result['cohort_item_ids_sha256'] == module._ids_hash(cohort.item_id)
    assert len(module.ACCESS_LOG.read_text().splitlines()) == 1
    with pytest.raises(FileExistsError, match='already exists'):
        module.prepare(lock, output)
    assert len(module.ACCESS_LOG.read_text().splitlines()) == 1
