#!/usr/bin/env python3
"""Place the remaining whole context512 queue on the first released study GPU.

The assignment depends only on completed work and zero device memory, never on
timing or quality values. Frozen benchmark cases and statistical methods stay
unchanged. The primary natural-context queue remains on GPU0.
"""
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/runs/v2'
BASE = RUN / 'resource_cap_confirmation'
OUT = BASE / 'resources'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    assignment = OUT / 'queue_context512_assignment.json'
    plan = OUT / 'remaining_context512_execution_plan_03.json'
    if plan.exists() or assignment.exists():
        raise RuntimeError('Refusing a second context512 launcher')
    amendment = BASE / 'execution_prompt_schema_amendment.json'
    cache_amendment = BASE / 'execution_worker_cache_amendment.json'
    if not amendment.is_file() or not cache_amendment.is_file():
        raise RuntimeError('Reviewed execution amendments are required')
    specs = [BASE / 'benchmark_execution_specs_cache5' / f'batch{b}_context512.json' for b in (1, 8)]
    hashes = {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in specs}
    save(plan, {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'Execute remaining frozen secondary context512 cases without waiting on an already-busy assigned device.',
        'eligible_devices_in_tie_order': [1, 2, 0],
        'release_rule': 'Prior assigned whole queue complete and nvidia-smi memory.used == 0 MiB.',
        'selection_inputs': 'Queue completion booleans and device availability only; no quality or cost values.',
        'unchanged': '13 systems, frozen specs, batch1 then batch8, 20 warmups/20 timed requests x10 blocks, all systems on one GPU per entire setting.',
        'primary_natural_context_device_unchanged': 0,
        'spec_sha256': hashes,
        'prompt_interface_amendment_sha256': hashlib.sha256(amendment.read_bytes()).hexdigest(),
        'inactive_worker_cache_amendment_sha256': hashlib.sha256(cache_amendment.read_bytes()).hexdigest(),
        'previous_waiters_stopped_before_any_case': ['remaining_queue_interface_pause.json', 'remaining_queue_cache_pause.json'],
    })
    while True:
        statuses = {
            1: read(RUN / 'rich_behavior/matched_resource_supplement/run_status.json'),
            2: read(OUT / 'queue_context128_status.json'),
            0: read(OUT / 'queue_context0_status.json'),
        }
        ready = {1: statuses[1].get('complete') is True,
                 2: statuses[2].get('stage') == 'complete_gpu_released',
                 0: statuses[0].get('stage') == 'complete_gpu_released'}
        selected = None
        for gpu in (1, 2, 0):
            if not ready[gpu]:
                continue
            r = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            uuid, used = (v.strip() for v in r.stdout.strip().split(','))
            if int(used) == 0:
                selected = gpu
                save(OUT / 'remaining_context512_release_evidence.json', {
                    'created_unix': time.time(), 'selected_gpu': gpu, 'GPU_uuid': uuid,
                    'memory_MiB': int(used), 'queue_completion_status': statuses[gpu],
                    'execution_plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
                })
                break
        if selected is not None:
            break
        time.sleep(10)
    for path in specs:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(RUN))]:
            raise RuntimeError('Frozen secondary benchmark spec changed')
        alias = BASE / 'benchmark_execution_specs' / path.name
        if hashlib.sha256(alias.read_bytes()).hexdigest() != hashes[str(path.relative_to(RUN))]:
            raise RuntimeError('Active execution spec differs from reviewed immutable copy')
    cmd = [sys.executable, '-u', str(ROOT / 'experiments/mcq_resource_cap_benchmark_queue.py'),
           '--gpu', str(selected), '--context', '512']
    print(json.dumps({'launching': cmd, 'selected_gpu': selected}), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
