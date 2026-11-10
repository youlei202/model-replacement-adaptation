#!/usr/bin/env python3
"""Posthoc cost accounting for the independently locked resource-cap supplement."""
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rich_setup import OUT, digest, dump
SUP = OUT / 'matched_resource_supplement'


def main():
    complete = json.load(open(SUP / 'supplement_complete.json'))
    assert complete['complete'], 'Only export final costs after all results finish'
    rows = []
    for path in sorted((SUP / 'logits').glob('*.json')):
        d = json.load(open(path)); assert d['complete']
        rows.append({'stage': 'new_response_collection', 'model_id': d['model_id'],
                     'model_context_evaluations': d['completed'], 'model_load_seconds': d['model_load_seconds'],
                     'measured_stage_wall_seconds': d['seconds_including_load'],
                     'timed_forward_seconds': sum(r['seconds'] for r in d['batch_timings']),
                     'bytes_outputs': path.with_suffix('.npy').stat().st_size,
                     'status': 'complete', 'artifact': str(path.relative_to(OUT)),
                     'measurement_scope': 'Model load plus collection loop. Forward-time sum is contained in stage wall, not additional. Process import/cohort validation excluded.'})
    for line in (SUP / 'execution_commands.jsonl').read_text().splitlines():
        d = json.loads(line)
        rows.append({'stage': d['stage'], 'measured_stage_wall_seconds': d['end_unix'] - d['start_unix'],
                     'status': 'complete' if d['returncode'] == 0 else 'failed',
                     'artifact': 'matched_resource_supplement/execution_commands.jsonl',
                     'record_kind': 'overlapping_inclusive_process_wall',
                     'measurement_scope': 'Includes process startup, input verification, loading, computation, output writing and shutdown; collection-stage rows above overlap this inclusive interval. Quality/report/benchmark stages are separate sequential processes.'})
    raw = [json.loads(line) for line in (SUP / 'timing_raw.jsonl').read_text().splitlines()]
    timed_contexts = sum(r['batch_size'] for r in raw)
    timed_model_contexts = sum(r['batch_size'] * r['actual_call_q'] for r in raw)
    rows.append({'stage': 'live_serving_timed_requests', 'model_context_evaluations': timed_model_contexts,
                 'requests': len(raw), 'logical_contexts': timed_contexts,
                 'summed_request_wall_seconds': sum(r['end_to_end_ms'] for r in raw) / 1000,
                 'summed_CUDA_event_intervals_seconds': sum(r['GPU_ms'] for r in raw) / 1000,
                 'status': 'complete', 'artifact': 'matched_resource_supplement/timing_raw.jsonl',
                 'measurement_scope': 'Complete predictor request intervals only; repeated timing inputs are not extra independent scientific samples. CUDA intervals are not GPU busy time. Both overlap inclusive benchmark process duration.'})
    # Twenty warmups and twenty timed requests share each system/setting/block.
    rows.append({'stage': 'live_serving_registered_warmups', 'model_context_evaluations': timed_model_contexts,
                 'requests': len(raw), 'logical_contexts': timed_contexts, 'status': 'complete',
                 'artifact': 'matched_resource_supplement/prospective_lock.json',
                 'measurement_scope': 'Known forward count from20warmups per20timed block. Warmup elapsed time was not separately isolated; included in benchmark process duration.'})
    replay = json.load(open(SUP / 'deployment_replay.json'))
    q = {r['system_digest']: r['actual_call_q'] for r in raw}
    rows.append({'stage': 'live_cached_peer_numerical_bridge',
                 'model_context_evaluations': sum(r['contexts'] * q[r['candidate_id']] for r in replay['records']),
                 'status': 'complete', 'artifact': 'matched_resource_supplement/deployment_replay.json',
                 'measurement_scope': 'Two live peer-only bridge contexts per unique state. Cached mathematical evaluation does not issue target calls. Isolated elapsed time unavailable.'})
    pd.DataFrame(rows).to_csv(SUP / 'measured_cost_accounting.csv', index=False)
    dump(SUP / 'cost_accounting_scope.json', {
        'posthoc_accounting_only_no_reselection': True,
        'scientific_new_model_contexts': sum(r.get('model_context_evaluations', 0) for r in rows if r['stage'] == 'new_response_collection'),
        'timed_serving_model_contexts': timed_model_contexts, 'registered_warmup_model_contexts': timed_model_contexts,
        'new_fit_or_development_profile_cost': 0,
        'reuse_cost_note': 'All fitted states/development profiles were reused from previously accounted work; zero here means no new fitting, not that historical fitting was free.',
        'overlap_rule': 'Do not add inclusive process wall to contained stage/request intervals. Do not add CUDA event intervals to wall time or label either GPU busy seconds. Cold model-load values repeated on request rows are not summed.',
        'unmeasured_separate_items': ['supplement selection/cohort CPU preparation', 'crosscohort dedup CPU elapsed', 'initial global source/checkpoint download cost', 'energy and monetary charges'],
        'no_fabricated_amortization': 'No break-even extrapolation from small serialized states; serving savings require a separately confirmed positive resource contrast.',
        'original_completion_marker_sha256': digest(SUP / 'supplement_complete.json'),
        'cost_csv_sha256': digest(SUP / 'measured_cost_accounting.csv'), 'code_sha256': digest(__file__)})


if __name__ == '__main__':
    main()
