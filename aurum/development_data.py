"""Register and collect new development documents, never confirmation outcomes.

The first tranche tests learning with more documents after tiny-fit diagnostics
showed representation capacity but weak held-out improvement. Membership enters
the shared exposure registry before any model is run.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import pandas as pd
from . import confirmation_data as cd


def choose(remaining, n_documents, seed, reserve_per_source=2048):
    if n_documents < 2 or n_documents % 2:
        raise ValueError('An even positive balanced development cohort is required')
    groups = []
    for source in cd.SOURCES:
        part = remaining[remaining.source.eq(source)].copy()
        if len(part) - n_documents // 2 < reserve_per_source:
            raise ValueError('Development request would exhaust the confirmation reserve')
        part['selection_hash'] = [hashlib.sha256(f'aurum-new-development-v1|{seed}|{source}|{x}'.encode()).hexdigest()
                                  for x in part.cluster_id]
        groups.append(part.sort_values(['selection_hash', 'item_id']).head(n_documents // 2))
    chosen = pd.concat(groups).sort_values(['selection_hash', 'item_id']).reset_index(drop=True)
    if chosen.cluster_id.duplicated().any() or chosen.item_id.duplicated().any():
        raise ValueError('Duplicate development cluster')
    return chosen


def prepare(output, n_documents=2048, seed=20261110):
    output = Path(output).resolve()
    if not output.is_relative_to(cd.AURUM / 'development/new_data'):
        raise ValueError('Use runs/aurum/development/new_data/<new_name>')
    cd.AURUM.mkdir(parents=True, exist_ok=True)
    with (cd.AURUM / '.data_access_registry.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError('Preserve the existing cohort; output already exists')
        remaining, exposure = cd.exposure_inventory()
        chosen = choose(remaining, n_documents, seed)
        columns = ['source', 'source_key', 'document_sha256', 'cluster_id', 'item_id', 'context_ids',
                   'position', 'context_tokens', 'document_token_count']
        original = pd.read_parquet(cd.LEGACY_ROOT / 'documents.parquet', columns=columns)
        cohort = original.set_index('item_id').loc[chosen.item_id].reset_index()
        cohort['legacy_role'] = 'test_pool'; cohort['role'] = 'aurum_new_development'
        output.mkdir(parents=True)
        path = output / 'cohort.parquet'; cohort.to_parquet(path, index=False)
        record = {'campaign': 'AURUM', 'purpose': 'additional_development', 'selected_at_utc': cd._utc(),
                  'scientific_reason': 'Rank128 tiny-fit can memorize; first-six outer improvements are small. Test data scaling with equally strengthened controls.',
                  'cohort_sha256': cd.file_sha256(path), 'cohort_item_ids_sha256': cd._ids_hash(cohort.item_id),
                  'item_ids': cohort.item_id.tolist(), 'n_documents': len(cohort), 'selection_seed': seed,
                  'source_counts': cohort.groupby('source').size().to_dict(), 'model_indices': list(range(5)),
                  'model_registry_sha256': cd.file_sha256(cd.LEGACY_ROOT / 'model_registry.json'),
                  'collector_sha256': cd.file_sha256(Path(__file__)), 'inference_helper_sha256': cd.file_sha256(cd.__file__),
                  'inference_microbatch': 1, 'inference_dtype': 'bfloat16', 'vocab': cd.VOCAB,
                  'exposure_inventory_at_selection': exposure, 'selected_without_response_access': True,
                  'minimum_untouched_confirmation_reserve_per_source': 2048,
                  'not_independent_confirmation': True}
        cd._write_json(output / 'cohort_lock.json', record, exclusive=True)
        with cd.ACCESS_LOG.open('a') as handle:
            handle.write(json.dumps({'time_utc': cd._utc(), 'purpose': 'additional_development', 'stage': 'metadata_reserved',
                                    'cohort_path': str(path.relative_to(cd.ROOT)), 'cohort_sha256': record['cohort_sha256']}) + '\n')
    return record


def collect(cohort_lock, model_index, *, device='cuda:0', cache_root='/tmp/model-replacement-adaptation/hf_cache'):
    path = Path(cohort_lock).resolve(); out = path.parent
    lock = json.loads(path.read_text())
    if not out.is_relative_to(cd.AURUM / 'development/new_data') or lock.get('purpose') != 'additional_development':
        raise ValueError('An explicitly registered development cohort is required')
    cohort_path = out / 'cohort.parquet'
    for artifact, expected in [(cohort_path, lock['cohort_sha256']), (Path(__file__), lock['collector_sha256']),
                               (Path(cd.__file__), lock['inference_helper_sha256']),
                               (cd.LEGACY_ROOT / 'model_registry.json', lock['model_registry_sha256'])]:
        if cd.file_sha256(artifact) != expected:
            raise ValueError('Registered data, inference source or model registry drift')
    rows = pd.read_parquet(cohort_path)
    remaining, _ = cd.exposure_inventory(ignore_aurum_cohort=cohort_path)
    if rows.item_id.tolist() != lock['item_ids'] or not set(rows.item_id) <= set(remaining.item_id):
        raise ValueError('New development has overlap or changed row alignment')
    if model_index not in lock['model_indices']:
        raise ValueError('Unregistered model')
    result = cd.collect_bank(lock, out, rows, model_index, device=device, cache_root=cache_root,
                             role='aurum_new_development', collector_path=Path(__file__))
    if cd.file_sha256(cd.__file__) != lock['inference_helper_sha256']:
        raise ValueError('Inference helper changed during collection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--n-documents', type=int, default=2048); p.add_argument('--seed', type=int, default=20261110)
    p = sub.add_parser('collect'); p.add_argument('--cohort-lock', required=True, type=Path)
    p.add_argument('--model-index', required=True, type=int); p.add_argument('--device', default='cuda:0')
    p.add_argument('--cache-root', default='/tmp/model-replacement-adaptation/hf_cache')
    args = parser.parse_args()
    result = prepare(args.out, args.n_documents, args.seed) if args.action == 'prepare' else collect(
        args.cohort_lock, args.model_index, device=args.device, cache_root=args.cache_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ('item_ids', 'batch_timings')}, indent=2))


if __name__ == '__main__':
    main()
