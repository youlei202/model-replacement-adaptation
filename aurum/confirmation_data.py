"""Fresh rich-response collection, gated by a frozen scientific specification.

Nothing is selected or collected on import. ``inspect`` only reads membership,
cache timing metadata, and source provenance. ``prepare`` requires a kit-format
freeze manifest and reserves previously unexposed document clusters. ``collect``
requires that manifest plus its matching cohort lock, and retains all151936 logits.

The freeze specification must include ``confirmation_design`` with n_documents,
selection_seed, inference_microbatch=1, inference_dtype='bfloat16', vocab=151936,
and optionally explicit equal source_counts. It must hash this collector, codec,
inference code, deployed binary states, selection/analysis code, and scientific
choices. A lock records declared access history; it does not prove outcome secrecy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import LEGACY_ROOT, PAIRS, ROOT, file_sha256

SOURCES = ('arxiv_train_shard0', 'wikitext103')
VOCAB = 151936
AURUM = ROOT / 'runs/aurum'
ACCESS_LOG = AURUM / 'data_access_registry.jsonl'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _ids_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def _write_json(path, value, *, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2); handle.write('\n')
    else:
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def _under_root(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Missing or outside-workspace artifact: {relative}')
    return path


def _check_freeze(method_lock):
    method_lock = Path(method_lock).resolve()
    lock = json.loads(method_lock.read_text()); spec = lock.get('specification', {})
    required = ('population', 'targets', 'admissible_pairs', 'budgets_bytes', 'budget_weights',
                'primary_endpoint', 'contrast_family', 'historical_access_statement', 'confirmation_design')
    if spec.get('campaign') != 'AURUM' or not lock.get('locked_at_utc') or any(not spec.get(k) for k in required):
        raise ValueError('A complete kit-format frozen scientific specification is required')
    if not set(spec['targets']) <= {3, 4} or sorted(map(tuple, spec['admissible_pairs'])) != list(PAIRS):
        raise ValueError('Unknown target IDs or incomplete strong-control pair family')
    if (len(spec['budgets_bytes']) != len(spec['budget_weights']) or sum(spec['budget_weights']) <= 0
            or any(not math.isfinite(w) or w < 0 for w in spec['budget_weights'])
            or any(not isinstance(b, int) or b < 0 for b in spec['budgets_bytes'])):
        raise ValueError('Invalid frozen budget weighting')
    records = lock.get('artifacts', [])
    if not records or len({x['path'] for x in records}) != len(records):
        raise ValueError('Nonempty unique frozen artifact records required')
    names = {x['path'] for x in records}
    required_code = {'aurum/confirmation_data.py', 'aurum/codec.py', 'aurum/inference.py'}
    if not required_code <= names or not any(x.endswith(('.bin', '.aur1')) for x in names):
        raise ValueError('Freeze collector, decoder, inference code and deployed binary states before new collection')
    for record in records:
        if file_sha256(_under_root(record['path'])) != record['sha256']:
            raise ValueError(f'Frozen artifact changed: {record["path"]}')
    design = spec['confirmation_design']
    n = design.get('n_documents')
    if not isinstance(n, int) or n < 2 or n % 2 or 'selection_seed' not in design:
        raise ValueError('Freeze an even positive cohort size and explicit metadata-selection seed')
    if (design.get('inference_microbatch') != 1 or design.get('inference_dtype') != 'bfloat16'
            or design.get('vocab') != VOCAB):
        raise ValueError('Fresh data must retain canonical BF16 microbatch1 and full151936 output support')
    counts = design.get('source_counts', {s: n // 2 for s in SOURCES})
    if counts != {s: n // 2 for s in SOURCES}:
        raise ValueError('This bridge implements the declared equal WikiText/arXiv mixture only')
    return lock, spec, design, counts


def exposure_inventory(*, ignore_aurum_cohort=None):
    """Read cluster membership and provenance, never historical outcome arrays."""
    lock = json.loads((LEGACY_ROOT / 'data_lock.json').read_text())
    path = LEGACY_ROOT / 'documents.parquet'
    if file_sha256(path) != lock['documents_sha256']:
        raise ValueError('Historical canonical-document universe drift')
    columns = ['source', 'source_key', 'document_sha256', 'cluster_id', 'item_id', 'role']
    documents = pd.read_parquet(path, columns=columns)
    original_test = json.loads((LEGACY_ROOT / 'selection_lock.json').read_text())['test_item_ids']
    supplement = pd.read_parquet(LEGACY_ROOT / 'matched_resource_supplement/cohort.parquet', columns=columns)
    groups = {role: set(documents.loc[documents.role.eq(role), 'item_id']) for role in ('pilot', 'fit', 'validation')}
    groups['original_final_test'] = set(original_test); groups['matched_supplement_final_test'] = set(supplement.item_id)
    exposed = set().union(*groups.values()); aurum_reservations = []
    if len(exposed) != sum(map(len, groups.values())):
        raise ValueError('Historical cohort membership unexpectedly overlaps')
    if ACCESS_LOG.exists():
        seen_cohorts = set()
        for line in ACCESS_LOG.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line); cohort = _under_root(event['cohort_path'])
            if cohort == ignore_aurum_cohort or str(cohort) in seen_cohorts:
                continue
            seen_cohorts.add(str(cohort))
            if file_sha256(cohort) != event['cohort_sha256']:
                raise ValueError('Previously registered AURUM cohort was modified')
            used = pd.read_parquet(cohort, columns=['item_id', 'cluster_id'])
            exposed.update(used.item_id)
            aurum_reservations.append({'cohort_path': str(cohort), 'unique_documents': len(used),
                                       'purpose': event['purpose'], 'first_recorded_stage': event['stage']})
    remaining = documents.loc[~documents.item_id.isin(exposed)].copy()
    if remaining.cluster_id.duplicated().any() or not remaining.role.eq('test_pool').all():
        raise ValueError('Unexpected remaining population or duplicate cluster')
    cross_audits = []
    for cross_path in [LEGACY_ROOT / 'concurrent_mcq_cross_audit.json',
                       LEGACY_ROOT / 'matched_resource_supplement/concurrent_new_mcq_cross_audit.json']:
        audit = json.loads(cross_path.read_text())
        if not audit['passed'] or audit['cross_collision_count']:
            raise ValueError('Historical cross-population deduplication was not clear')
        cross_audits.append({'path': str(cross_path.relative_to(ROOT)), 'sha256': file_sha256(cross_path)})
    report = {'known_population_documents': len(documents), 'historical_membership_counts': {k: len(v) for k, v in groups.items()},
              'aurum_reservations': aurum_reservations, 'remaining_documents': len(remaining),
              'remaining_source_counts': remaining.groupby('source').size().to_dict(),
              'documents_sha256': lock['documents_sha256'], 'cross_population_audits': cross_audits,
              'historical_test_outcomes_read': False,
              'access_scope_limit': 'Known locked repository cohorts and registered AURUM cohorts; external access cannot be certified.'}
    return remaining, report


def inspect(output):
    remaining, report = exposure_inventory()
    estimates = []
    for model in range(5):
        meta = json.loads((LEGACY_ROOT / 'logits/fit' / f'model_{model}.json').read_text())
        inference_seconds = sum(x['seconds'] for x in meta['batch_timings'])
        estimates.append({'model_index': model, 'model_id': meta['model_id'], 'model_revision': meta['model_revision'],
                          'reference_documents': meta['expected'], 'reference_inference_seconds': inference_seconds,
                          'seconds_per_document': inference_seconds / meta['expected'],
                          'reference_load_io_overhead_seconds': meta['seconds_including_load'] - inference_seconds,
                          'observed_peak_allocated_GiB': meta['memory_peak_allocated'] / 2**30})
    report.update({'collection_estimates_from_historical_fit_metadata': estimates,
                   'float32_response_bytes_per_document_per_model': 4 * VOCAB,
                   'estimated_all5_model_bytes_per2048_documents': 2048 * 5 * 4 * VOCAB,
                   'source_requirements': 'Same pinned source documents, canonical stored context IDs, full151936 output support, BF16 microbatch1.',
                   'intrinsic_population_blocker': None,
                   'external_requirements': ['Frozen scientific choices and encoded states', 'Available CUDA device',
                                             'Pinned local model weights for requested model IDs', 'Sufficient new-output disk space'],
                   'no_confirmation_rows_selected': True, 'no_new_responses_collected': True})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'confirmation_readiness.json', report)
    remaining.to_parquet(output / 'available_confirmation_metadata.parquet', index=False)
    return report


def prepare(method_lock, output):
    lock, spec, design, counts = _check_freeze(method_lock)
    output = Path(output).resolve()
    if not output.is_relative_to(AURUM / 'confirmation'):
        raise ValueError('Use a new output directory under runs/aurum/confirmation/')
    AURUM.mkdir(parents=True, exist_ok=True)
    with (AURUM / '.data_access_registry.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        remaining, exposure = exposure_inventory()
        selected = []
        for source, count in counts.items():
            frame = remaining[remaining.source.eq(source)].copy()
            if len(frame) < count:
                raise ValueError(f'Only {len(frame)} fresh {source} clusters remain, requested {count}')
            frame['selection_hash'] = [hashlib.sha256(f'aurum-confirmation-v1|{design["selection_seed"]}|{source}|{cluster}'.encode()).hexdigest()
                                       for cluster in frame.cluster_id]
            selected.append(frame.sort_values(['selection_hash', 'item_id']).head(count))
        chosen = pd.concat(selected).sort_values(['selection_hash', 'item_id']).reset_index(drop=True)
        columns = ['source', 'source_key', 'document_sha256', 'cluster_id', 'item_id', 'context_ids',
                   'position', 'context_tokens', 'document_token_count']
        original = pd.read_parquet(LEGACY_ROOT / 'documents.parquet', columns=columns)
        cohort = original.set_index('item_id').loc[chosen.item_id].reset_index()
        cohort['legacy_role'] = 'test_pool'; cohort['role'] = 'aurum_independent_confirmation'
        if output.exists():
            raise FileExistsError('Confirmation output directory already exists; preserve it and use a new directory')
        output.mkdir(parents=True)
        cohort_path = output / 'cohort.parquet'; cohort.to_parquet(cohort_path, index=False)
        cohort_hash = file_sha256(cohort_path)
        record = {'campaign': 'AURUM', 'purpose': 'independent_confirmation', 'selected_at_utc': _utc(),
                  'method_lock_path': str(Path(method_lock).resolve().relative_to(ROOT)),
                  'method_lock_sha256': file_sha256(method_lock), 'cohort_sha256': cohort_hash,
                  'cohort_item_ids_sha256': _ids_hash(cohort.item_id), 'item_ids': cohort.item_id.tolist(),
                  'source_counts': counts, 'n_documents': len(cohort), 'selection_seed': design['selection_seed'],
                  'selection_rule': 'SHA256 order within each frozen source stratum, whole previously unexposed document clusters',
                  'model_indices': sorted(set(spec['targets']) | {p for pair in spec['admissible_pairs'] for p in pair}),
                  'model_registry_sha256': file_sha256(LEGACY_ROOT / 'model_registry.json'),
                  'exposure_inventory_at_selection': exposure, 'selected_without_response_access': True,
                  'collector_sha256': file_sha256(Path(__file__))}
        _write_json(output / 'cohort_lock.json', record, exclusive=True)
        with ACCESS_LOG.open('a') as handle:
            handle.write(json.dumps({'time_utc': _utc(), 'purpose': 'independent_confirmation', 'stage': 'metadata_reserved',
                                    'cohort_path': str(cohort_path.relative_to(ROOT)), 'cohort_sha256': cohort_hash}) + '\n')
    return record


def collect(cohort_lock, model_index, *, device='cuda:0', cache_root='/tmp/model-replacement-adaptation/hf_cache'):
    """Collect one pinned model after both method and fresh-cohort locks exist."""
    cohort_lock = Path(cohort_lock).resolve(); out = cohort_lock.parent
    lock = json.loads(cohort_lock.read_text())
    if not out.is_relative_to(AURUM / 'confirmation') or lock.get('purpose') != 'independent_confirmation':
        raise ValueError('A new AURUM independent-confirmation cohort lock is required')
    method_path = _under_root(lock['method_lock_path'])
    if file_sha256(method_path) != lock['method_lock_sha256']:
        raise ValueError('Method lock changed after cohort selection')
    _check_freeze(method_path)
    cohort_path = out / 'cohort.parquet'
    if file_sha256(cohort_path) != lock['cohort_sha256'] or file_sha256(Path(__file__)) != lock['collector_sha256']:
        raise ValueError('Frozen cohort or collector changed')
    if file_sha256(LEGACY_ROOT / 'model_registry.json') != lock['model_registry_sha256']:
        raise ValueError('Pinned model registry changed')
    remaining, _ = exposure_inventory(ignore_aurum_cohort=cohort_path)
    rows = pd.read_parquet(cohort_path)
    if not set(rows.item_id) <= set(remaining.item_id) or rows.item_id.tolist() != lock['item_ids']:
        raise ValueError('Confirmation cohort has acquired another exposure or row alignment changed')
    if model_index not in lock['model_indices']:
        raise ValueError('Model was not frozen in this confirmation design')
    return collect_bank(lock, out, rows, model_index, device=device, cache_root=cache_root,
                        role='aurum_independent_confirmation', collector_path=Path(__file__))


def collect_bank(lock, out, rows, model_index, *, device, cache_root, role, collector_path):
    """Canonical inference shared by separately gated DEV/confirmation entrypoints."""
    registry = json.loads((LEGACY_ROOT / 'model_registry.json').read_text())
    model_record = next(x for x in registry if x['global_id'] == model_index)
    snapshot = Path(cache_root) / ('models--' + model_record['model_id'].replace('/', '--')) / 'snapshots' / model_record['revision']
    if not snapshot.is_dir():
        raise FileNotFoundError(f'Pinned local model snapshot is unavailable: {snapshot}')
    folder = out / 'logits'; folder.mkdir(exist_ok=True)
    path = folder / f'model_{model_index}.npy'; meta_path = path.with_suffix('.json')
    if meta_path.exists() and json.loads(meta_path.read_text()).get('complete'):
        meta = json.loads(meta_path.read_text())
        if meta['cohort_sha256'] != lock['cohort_sha256'] or meta['sha256'] != file_sha256(path):
            raise ValueError('Existing complete response bank does not match this lock')
        return meta
    partial = path.with_suffix('.partial.npy')
    if path.exists() or partial.exists() or meta_path.exists():
        raise FileExistsError('An incomplete collection exists and is preserved; explicitly recover it before retrying')
    canonical_tokenizer = LEGACY_ROOT / 'tokenizers/Qwen--Qwen3-0.6B/tokenizer.json'
    if file_sha256(snapshot / 'tokenizer.json') != file_sha256(canonical_tokenizer):
        raise ValueError('Pinned model tokenizer differs from canonical shared tokenizer')
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    if {p.dtype for p in model.parameters() if p.is_floating_point()} != {torch.bfloat16}:
        raise ValueError('Collected model must use BF16 inference exactly')
    if model.get_output_embeddings().weight.shape[0] != VOCAB:
        raise ValueError('Pinned output vocabulary changed')
    array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(rows), VOCAB))
    timings = []; base = {'complete': False, 'expected': len(rows), 'model_index': model_index,
        'model_id': model_record['model_id'], 'model_revision': model_record['revision'],
        'cohort_sha256': lock['cohort_sha256'], 'cohort_item_ids_sha256': lock['cohort_item_ids_sha256'],
        'method_lock_sha256': lock.get('method_lock_sha256'), 'role': role,
        'vocab': VOCAB, 'inference_microbatch': 1, 'storage_dtype': 'float32_centered_logits',
        'collector_sha256': lock['collector_sha256']}
    for index, sequence in enumerate(rows.context_ids):
        ids = torch.tensor(np.asarray(sequence, dtype=np.int64)[None, :], device=device)
        if len(sequence) > 512 or len(sequence) == 0:
            raise ValueError('Canonical context length changed')
        mask = torch.ones_like(ids); torch.cuda.synchronize(); tic = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        torch.cuda.synchronize(); elapsed = time.perf_counter() - tic
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite full-support response')
        array[index] = (logits - logits.mean(dim=1, keepdim=True)).cpu().numpy()[0]
        timings.append({'start': index, 'batch': 1, 'tokens': len(sequence), 'seconds': elapsed})
        if (index + 1) % 32 == 0 or index + 1 == len(rows):
            array.flush(); _write_json(meta_path, {**base, 'completed': index + 1})
            print(f'{model_record["model_id"]} {role} {index+1}/{len(rows)}', flush=True)
    del array
    if file_sha256(collector_path) != lock['collector_sha256']:
        raise ValueError('Collector source changed during inference; partial artifact is preserved')
    partial.replace(path)
    meta = {**base, 'complete': True, 'completed': len(rows), 'sha256': file_sha256(path),
        'seconds_including_load': time.perf_counter() - start, 'batch_timings': timings,
        'memory_peak_allocated': torch.cuda.max_memory_allocated(), 'memory_peak_reserved': torch.cuda.max_memory_reserved(),
        'output_weight_shape': list(model.get_output_embeddings().weight.shape),
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'transformers': __import__('transformers').__version__, 'device': torch.cuda.get_device_name(), 'dtype': 'bfloat16'},
        'known_unused_output_token_ids': 'All151936 output slots retained, including267 reserved IDs151669..151935.'}
    _write_json(meta_path, meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    audit = sub.add_parser('inspect'); audit.add_argument('--out', type=Path, default=AURUM / 'audit')
    prepare_parser = sub.add_parser('prepare'); prepare_parser.add_argument('--method-lock', type=Path, required=True)
    prepare_parser.add_argument('--out', type=Path, required=True)
    collect_parser = sub.add_parser('collect'); collect_parser.add_argument('--cohort-lock', type=Path, required=True)
    collect_parser.add_argument('--model-index', type=int, required=True); collect_parser.add_argument('--device', default='cuda:0')
    collect_parser.add_argument('--cache-root', default='/tmp/model-replacement-adaptation/hf_cache')
    args = parser.parse_args()
    if args.action == 'inspect':
        result = inspect(args.out)
    elif args.action == 'prepare':
        result = prepare(args.method_lock, args.out)
    else:
        result = collect(args.cohort_lock, args.model_index, device=args.device, cache_root=args.cache_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ('batch_timings', 'item_ids')}, indent=2))


if __name__ == '__main__':
    main()
