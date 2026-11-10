#!/usr/bin/env python3
"""New-cohort collection and live serving for the registered resource-cap supplement."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rich_setup import OUT, MODELS, digest, dump
from rich_matched_resource import SUP, SEED, validate_lock
from rich_codec import decode
from rich_inference import predict
from rich_deploy import load_pair, forward, timed, state_account, row_active, minimal_basis, tensor


def ready():
    lock = validate_lock()
    audit = json.load(open(SUP / 'concurrent_new_mcq_cross_audit.json'))
    assert audit['passed'] and audit['rich_documents_sha256'] == digest(OUT / 'documents.parquet')
    cohort = pd.read_parquet(SUP / 'cohort.parquet')
    assert cohort.item_id.tolist() == lock['test_item_ids']
    return lock, cohort


def collect(index):
    lock, cohort = ready()
    directory = SUP / 'logits'; directory.mkdir(exist_ok=True)
    path = directory / f'model_{index}.npy'; meta = path.with_suffix('.json')
    if meta.exists():
        old = json.load(open(meta))
        if old.get('complete'):
            assert old['sha256'] == digest(path)
            assert old['prospective_lock_sha256'] == digest(SUP / 'prospective_lock.json')
            print('Already collected', index, flush=True); return
    start = time.perf_counter()
    mid, revision, role = MODELS[index]
    snapshot = Path('/tmp/model-replacement-adaptation/hf_cache') / ('models--' + mid.replace('/', '--')) / 'snapshots' / revision
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True).to('cuda:0').eval()
    load_seconds = time.perf_counter() - start
    assert model.get_output_embeddings().weight.shape[0] == 151936
    assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.bfloat16}
    mmap = np.lib.format.open_memmap(path.with_suffix('.partial.npy'), mode='w+', dtype=np.float32, shape=(len(cohort), 151936))
    timings = []
    for j, seq in enumerate(cohort.context_ids):
        ids = torch.as_tensor(np.asarray(seq).copy(), dtype=torch.long, device='cuda:0').reshape(1, -1)
        torch.cuda.synchronize(); begin = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        torch.cuda.synchronize()
        timings.append({'index': j, 'context_tokens': len(seq), 'seconds': time.perf_counter() - begin})
        assert torch.isfinite(logits).all()
        mmap[j] = (logits - logits.mean(-1, keepdim=True)).cpu().numpy()[0]
        if (j + 1) % 128 == 0 or j + 1 == len(cohort):
            mmap.flush()
            dump(meta, {'complete': False, 'completed': j + 1, 'expected': len(cohort), 'model_index': index})
            print('new_resource_cohort', index, j + 1, '/', len(cohort), flush=True)
    del mmap
    path.with_suffix('.partial.npy').replace(path)
    dump(meta, {'complete': True, 'completed': len(cohort), 'expected': len(cohort), 'model_index': index,
                'model_id': mid, 'model_revision': revision, 'role': role, 'retired_targets_used_for_scoring_only': role == 'target',
                'cohort_sha256': digest(SUP / 'cohort.parquet'), 'prospective_lock_sha256': digest(SUP / 'prospective_lock.json'),
                'crossaudit_sha256': digest(SUP / 'concurrent_new_mcq_cross_audit.json'),
                'sha256': digest(path), 'vocab': 151936, 'inference_microbatch': 1,
                'storage_dtype': 'FP32 centered logits; all output slots retained',
                'model_load_seconds': load_seconds, 'seconds_including_load': time.perf_counter() - start,
                'batch_timings': timings, 'memory_peak_allocated': torch.cuda.max_memory_allocated(),
                'memory_peak_reserved': torch.cuda.max_memory_reserved(),
                'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                            'transformers': __import__('transformers').__version__, 'device': torch.cuda.get_device_name()},
                'collector_code_sha256': digest(__file__)})


def telemetry(block, setting):
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    proc = subprocess.run(['nvidia-smi', '-i', physical, '--query-gpu=timestamp,uuid,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active', '--format=csv,noheader,nounits'], text=True, capture_output=True)
    with (SUP / 'gpu_telemetry_block_boundary.jsonl').open('a') as stream:
        stream.write(json.dumps({'time_unix': time.time(), 'block': block, **setting, 'exit_code': proc.returncode,
                                 'csv_fields': 'timestamp,uuid,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active',
                                 'raw': proc.stdout.strip(), 'error': proc.stderr.strip(),
                                 'scope': 'Outside timed requests; boundary samples cannot establish absence of throttling between samples'}) + '\n')


def benchmark():
    lock, cohort = ready()
    cohort_hash = digest(SUP / 'cohort.parquet')
    unique = {r['candidate_id']: r for r in lock['decisions'] if r['feasible_on_development']}
    path = SUP / 'timing_raw.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    done = {(r['system_digest'], r['batch_size'], r['context_tokens'], r['timing_block'], r['request']) for r in rows}
    assert len(rows) == len(done)
    rng = np.random.default_rng(SEED + 3)
    replay = []
    replay_seen = set()
    if (SUP / 'deployment_replay.json').exists():
        replay = json.load(open(SUP / 'deployment_replay.json'))['records']
        replay_seen = {r['candidate_id'] for r in replay}
    with path.open('a') as stream:
        for setting in lock['benchmark_settings']:
            batch, length = setting['batch_size'], setting['context_tokens']
            selected_docs = cohort[cohort.context_tokens.ge(length)].sort_values('supplement_order_hash').iloc[:256]
            assert len(selected_docs) >= 200 + batch
            contexts = [np.asarray(x[-length:] if length else x, dtype=np.int64) for x in selected_docs.context_ids]
            for block in range(10):
                if block % 2 == 0:
                    base_pairs = sorted({row_active(r) for r in unique.values()}); rng.shuffle(base_pairs)
                pairs = base_pairs if block % 2 == 0 else list(reversed(base_pairs))
                for pair in pairs:
                    systems = [r for r in unique.values() if row_active(r) == pair]; rng.shuffle(systems)
                    if all((r['candidate_id'], batch, length, block, j) in done for r in systems for j in range(20)):
                        continue
                    models, cold = load_pair(pair)
                    for order, row in enumerate(systems):
                        if all((row['candidate_id'], batch, length, block, j) in done for j in range(20)):
                            continue
                        blob = (OUT / row['state_path']).read_bytes(); state = decode(blob)
                        basis = tensor(np.load(minimal_basis(tuple(state['peers']), state['basis_rank']))) if state['basis_rank'] else None
                        account = state_account(row, models, basis)
                        if row['candidate_id'] not in replay_seen:
                            errors = []
                            for index in range(2):
                                live = forward(blob, models, basis, [cohort.context_ids.iloc[index]])
                                peer = [tensor(np.load(SUP / 'logits' / f'model_{p}.npy', mmap_mode='r')[index:index+1]) for p in state['peers']]
                                expected, isp = predict(state, peer, basis)
                                expected = expected if isp else torch.softmax(expected, dim=-1)
                                errors.append(float((.5 * (live - expected).abs().sum()).cpu()))
                                del live, peer, expected
                            assert max(errors) <= 1e-5, 'Fresh live deployment differs from the canonical frozen peer bank'
                            replay.append({'candidate_id': row['candidate_id'], 'max_full_support_TV': max(errors), 'contexts': 2, 'target_loaded': False})
                            replay_seen.add(row['candidate_id'])
                            dump(SUP / 'deployment_replay.json', {'complete': len(replay_seen) == len(unique), 'records': replay})
                        for j in range(20):
                            forward(blob, models, basis, [contexts[(j+k) % len(contexts)] for k in range(batch)])
                        for j in range(20):
                            key = (row['candidate_id'], batch, length, block, j)
                            if key in done:
                                continue
                            record = timed(blob, models, basis, [contexts[(block*20+j+k) % len(contexts)] for k in range(batch)])
                            record.update({'system_digest': row['candidate_id'], **setting, 'timing_block': block,
                                           'request': j, 'execution_order': order, 'cold_model_load_seconds': cold,
                                           'fresh_timing_cohort_sha256': cohort_hash,
                                           'failure_status': None, **account})
                            stream.write(json.dumps(record) + '\n'); stream.flush(); rows.append(record); done.add(key)
                        del basis; torch.cuda.empty_cache()
                        print('supplement_benchmark', batch, length, block, pair, row['candidate_id'][:10], flush=True)
                    del models; gc.collect(); torch.cuda.empty_cache()
                telemetry(block, setting)
                dump(SUP / 'benchmark_progress.json', {'complete': False, 'rows': len(rows), 'block': block, **setting})
    expected = len(unique) * len(lock['benchmark_settings']) * 200
    assert len(rows) == expected
    dump(SUP / 'benchmark_complete.json', {'complete': True, 'rows': len(rows), 'unique_systems': len(unique),
         'settings': len(lock['benchmark_settings']), 'timed_per_setting': 200, 'independent_blocks': 10,
         'raw_sha256': digest(path), 'prospective_lock_sha256': digest(SUP / 'prospective_lock.json'),
         'target_absent': all(r['target_absent'] for r in rows), 'fresh_inputs': True,
         'timing_scope': 'Complete live predictor; new cohort canonical batch1 primary; other settings descriptive only'})
    dump(SUP / 'benchmark_progress.json', {'complete': True, 'rows': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['collect', 'benchmark']); parser.add_argument('--model-index', type=int)
    args = parser.parse_args(); torch.set_num_threads(4)
    try:
        if args.action == 'collect':
            if args.model_index not in range(5):
                raise ValueError('Valid model index required')
            collect(args.model_index)
        else:
            benchmark()
    except Exception as error:
        SUP.mkdir(exist_ok=True)
        with (SUP / 'execution_failures.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_unix': time.time(), 'action': args.action, 'model_index': args.model_index,
                                     'exception_type': type(error).__name__, 'message': str(error), 'results_preserved': True}) + '\n')
        raise
