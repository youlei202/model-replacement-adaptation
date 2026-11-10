#!/usr/bin/env python3
"""Verify execution partitions, then invoke the unchanged frozen analyzer."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/runs/v2/resource_benchmark'
CAT = OUT / 'confirmation_catalog'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def case(row):
    return tuple(row[k] for k in ('system_id', 'batch_size', 'context_tokens', 'timing_block'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true')
    a = p.parse_args()
    counts = {'final_measurement': 180, 'direct_measurement': 180,
              'direct_batch8_measurement': 60, 'direct_batch8_controlled_measurement': 120}
    while True:
        current = {name: read(CAT / name / 'progress.json')['completed_blocks']
                   if (CAT / name / 'progress.json').exists() else 0 for name in counts}
        if any(current[k] > counts[k] for k in counts):
            raise RuntimeError(f'Unexpected case boundary: {current}')
        if current == counts:
            break
        if not a.wait:
            raise RuntimeError(f'Measurements still incomplete: {current}')
        print(json.dumps(current), flush=True)
        time.sleep(30)
    handoffs = [CAT / name / 'case_partition_handoff.json'
                for name in ('direct_measurement', 'direct_batch8_measurement')]
    for path, count in zip(handoffs, (180, 60)):
        if read(path)['completed_blocks_at_handoff'] != count:
            raise RuntimeError('Unexpected handoff boundary')
    combined = CAT / 'combined_direct_case_handoffs.json'
    save(combined, {'scope': 'Execution placement only; every original case is retained.',
                    'records': [{'path': str(f.relative_to(OUT)), 'sha256': sha(f), 'record': read(f)}
                                for f in handoffs],
                    'plans': {str(f.relative_to(OUT)): sha(f) for f in
                              (CAT / 'direct_reference_case_sharding_plan.json',
                               CAT / 'direct_reference_case_sharding_plan_02.json')}})
    spec = read(CAT / 'direct_measurement/measurement_lock.json')['spec']
    expected = [{'system_id': s['system_id'], 'batch_size': b, 'context_tokens': c, 'timing_block': k}
                for s in spec['systems'] for b in spec['batch_sizes']
                for c in spec['context_lengths'] for k in range(spec['blocks'])]
    shards, seen = [], set()
    for name in ('direct_measurement', 'direct_batch8_measurement', 'direct_batch8_controlled_measurement'):
        raw = CAT / name / 'raw_blocks.jsonl'
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        assigned = [{k: row[k] for k in ('system_id', 'batch_size', 'context_tokens', 'timing_block')}
                    for row in rows]
        keys = {case(row) for row in rows}
        if len(keys) != len(rows) or len(rows) != counts[name] or seen & keys:
            raise RuntimeError('Missing or duplicate direct-reference cases')
        if name == 'direct_measurement' and any(r['batch_size'] != 1 for r in rows):
            raise RuntimeError('Batch1 case partition drift')
        if name == 'direct_batch8_measurement' and any(r['batch_size'] != 8 or r['context_tokens'] != 0 for r in rows):
            raise RuntimeError('Natural batch8 case partition drift')
        if name == 'direct_batch8_controlled_measurement' and any(r['batch_size'] != 8 or r['context_tokens'] not in (128, 512) for r in rows):
            raise RuntimeError('Controlled batch8 case partition drift')
        seen |= keys
        shards.append({'raw_path': str(raw.relative_to(OUT)), 'raw_sha256': sha(raw),
                       'measurement_lock_sha256': sha(raw.parent / 'measurement_lock.json'),
                       'assigned_cases': assigned})
    if seen != {case(row) for row in expected}:
        raise RuntimeError('Execution partitions differ from original registered schedule')
    save(OUT / 'execution_case_shards.json', {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'expected_cases': expected, 'shards': shards,
        'handoff_record': {'path': str(combined.relative_to(OUT)), 'sha256': sha(combined)},
        'scientific_methods_unchanged': True,
    })
    cells = ROOT / 'experiments/runs/v2/confirmation/fits/campaign/cells/seed_81724031/academic_mcq4'
    raw = [CAT / name / 'raw_blocks.jsonl' for name in counts]
    cmd = [sys.executable, str(ROOT / 'experiments/mcq_resource_analyze.py'),
           '--catalog', str(CAT), '--cells', str(cells / 'target_1/q_2'), str(cells / 'target_3/q_2'),
           '--raw', *map(str, raw), '--output', str(OUT / 'analysis')]
    save(OUT / 'final_analysis_command.json', {'argv': cmd,
         'analyzer_sha256': sha(ROOT / 'experiments/mcq_resource_analyze.py'),
         'execution_manifest_sha256': sha(OUT / 'execution_case_shards.json')})
    subprocess.run(cmd, cwd=ROOT, check=True)
    marker = OUT / 'analysis/resource_complete.json'
    save(OUT / 'resource_complete.json', {'deployment_measured': True,
         'scope': 'Original validation-frozen matched-fidelity route; later matched-resource supplement has a separate completion marker.',
         'artifacts_sha256': {str(marker.relative_to(OUT)): sha(marker),
                              'execution_case_shards.json': sha(OUT / 'execution_case_shards.json')}})
    print('Original matched-fidelity MCQ resources complete and analyzed.', flush=True)


if __name__ == '__main__':
    main()
