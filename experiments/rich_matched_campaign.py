#!/usr/bin/env python3
"""Durable sequential launcher; waits for the explicitly assigned GPU release."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/runs/v2/rich_behavior/matched_resource_supplement'


def write(name, value):
    tmp = OUT / (name + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(OUT / name)


def main():
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior_progress = ROOT / 'experiments/runs/v2/resource_benchmark/confirmation_catalog/final_measurement/progress.json'
    while True:
        progress = json.load(open(prior_progress))
        old = subprocess.run(['ps', '-p', '932111,932118', '-o', 'pid=,stat=,cmd='], capture_output=True, text=True)
        gpu = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        memory = int(gpu.stdout.split(',')[0].strip())
        alive = [line for line in old.stdout.splitlines() if line.strip() and not line.split()[1].startswith('Z')]
        evidence = {'time_unix': time.time(), 'prior_completed_blocks': progress['completed_blocks'],
                    'prior_planned_blocks': progress['planned_blocks'], 'old_process_rows': old.stdout,
                    'GPU1_memory_MiB': memory, 'GPU1_raw_status': gpu.stdout.strip(), 'host': os.uname().nodename}
        if progress['completed_blocks'] == progress['planned_blocks'] == 180 and not alive and memory == 0:
            write('gpu_release_evidence.json', evidence); break
        write('run_status.json', {'stage': 'waiting_for_original_GPU1_release', 'complete': False, **evidence})
        time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    python = str(ROOT / '.venv-inference/bin/python')
    stages = [(f'collect_model_{i}', [python, '-u', str(ROOT / 'experiments/rich_matched_resource_live.py'), 'collect', '--model-index', str(i)]) for i in range(5)]
    stages += [('quality', [python, '-u', str(ROOT / 'experiments/rich_matched_resource.py'), 'evaluate']),
               ('live_resource_benchmark', [python, '-u', str(ROOT / 'experiments/rich_matched_resource_live.py'), 'benchmark']),
               ('report', [python, '-u', str(ROOT / 'experiments/rich_matched_resource.py'), 'report'])]
    for stage, command in stages:
        start = time.time()
        write('run_status.json', {'stage': stage, 'complete': False, 'start_unix': start, 'gpu_physical_index': 1,
                                  'launcher_code_sha256': source_hash})
        with (OUT / f'{stage}.log').open('a') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {'stage': stage, 'command': command, 'start_unix': start, 'end_unix': time.time(),
                  'returncode': process.returncode, 'log': f'{stage}.log', 'gpu_physical_index': 1}
        with (OUT / 'execution_commands.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if process.returncode:
            write('run_status.json', {'stage': 'failed', 'failed_stage': stage, 'complete': False, **record})
            raise RuntimeError(f'{stage} failed; all records preserved')
    write('run_status.json', {'stage': 'complete', 'complete': True, 'time_unix': time.time(), 'gpu_physical_index': 1})


if __name__ == '__main__':
    main()
