#!/usr/bin/env python3
"""Describe actual timing endpoints without changing any frozen measurement."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/runs/v2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_root;out=run/'reports';out.mkdir(parents=True,exist_ok=True)
    rows=[];execution=[]
    def add(path,route,fields,probability,decode,kind='measured endpoint interpretation'):
        if not path.exists():return
        rows.append({'artifact':str(path.relative_to(run)),'artifact_sha256':sha(path),'route':route,
            'original_field_names':fields,'correct_endpoint':'canonical probability materialization' if probability else 'predicted Helmert coordinates',
            'final_probability_conversion_inside_interval':probability,
            'adapter_decode_inside_warm_interval':decode,
            'network_serialization_inside_interval':False,
            'scope':'Warm serving interval; model loading and CPU-to-GPU activation are separate. Final probability conversion means inverse Helmert plus stable softmax for MCQ, full-vocabulary normalization for rich.',
            'host_RSS_scope':'Active predictor process only; excludes the sum of other inactive model groups cached by the benchmark coordinator.',
            'allocated_VRAM_scope':'Observed active process CUDA allocated peak on measured inputs; excludes a future-OOM guarantee and is not total device/context memory.',
            'orchestration_cache_scope':'Read the per-measurement execution_scope_map; host-side cache residency is not retained target-state memory.',
            'interpretation_kind':kind,'original_bytes_changed':False})
    old=run/'resource_benchmark'
    for name,fields in [
        ('analysis/resource_summary.csv','mean_end_to_end_ms_per1000;warm_p50_latency_ms;warm_p95_latency_ms;GPU_uuid;peak_observed_host_RSS'),
        ('analysis/timing_blocks.csv','mean_end_to_end_ms_per1000;mean_peer_GPU_ms_per1000;timed_requests'),
        ('analysis/paired_resource_contrasts.csv','joint_ms_per1000;fixed_ms_per1000;resource_reduction;registered_operating_point_evidence_pass'),
        ('analysis/resource_complete.json','deployment_measured;quality_evaluated_on_fresh_clusters'),
        ('confirmation_catalog/development_profiles_complete.jsonl','traces.end_to_end_ms;GPU_stream_elapsed_ms')]:
        add(old/name,'original MCQ development/matched-fidelity coordinate endpoint',fields,False,False)
    cap=run/'resource_cap_confirmation'
    for name,fields in [
        ('resource_summary.csv','mean_ms_per_context;latency_familywise_upper_one_sided;peak_host_RSS'),
        ('coverage_and_cap_compliance.csv','final_measured_point_cap_compliant;final_latency_bound_and_observed_memory_compliant'),
        ('guarded_confirmation_results.csv','same_resource_advantage_confirmed;both_policies_all_registered_cells_confidence_compliant'),
        ('complete.json','deployment_measured;complete')]:
        add(cap/name,'new MCQ cap route coordinate endpoint',fields,False,False,
            'Original cap flags remain coordinate-endpoint flags; final operational claims additionally require reports/resource_cap_probability_guard.')
    for subtree,route,prob in [(old,'original MCQ profiling/final coordinate endpoint',False),
            (cap/'resources','new MCQ cap route coordinate endpoint',False),
            (cap/'probability_endpoint_audit/measurement','later MCQ probability audit',True)]:
        for path in sorted(subtree.rglob('raw_blocks.jsonl')):
            add(path,route,'traces.end_to_end_ms;traces.GPU_stream_elapsed_ms;traces.peer_GPU_elapsed_ms;traces.host_RSS',prob,False)
            lockpath=path.parent/'measurement_lock.json'
            if lockpath.exists():
                lock=json.loads(lockpath.read_text());spec=lock['spec']
                execution.append({'raw_artifact':str(path.relative_to(run)),
                    'measurement_lock':str(lockpath.relative_to(run)),'measurement_lock_sha256':sha(lockpath),
                    'spec_sha256':lock['spec_sha256'],'measurement_role':spec.get('measurement_role'),
                    'batch_sizes':spec.get('batch_sizes'),'context_lengths':spec.get('context_lengths'),
                    'configured_maximum_worker_processes':spec.get('maximum_worker_processes'),
                    'configuration_missing_means':'Not explicitly recorded; do not infer a limit from a later source default.',
                    'memory_semantics':'Inactive groups retain CPU model weights; only the scheduled group occupies the active GPU. Per-worker RSS is not the aggregate controller cache.'})
    for name,fields in [('resource_summary.csv','mean_ms_per_context;latency_upper_one_sided_alpha_div16;peak_host_RSS'),
                        ('cap_compliance_additional_guard.csv','probability_endpoint_point_compliant;probability_endpoint_bound_compliant'),
                        ('complete.json','endpoint;post_outcome_additional_guard_not_preregistration')]:
        add(cap/'probability_endpoint_audit'/name,'later MCQ probability audit',fields,True,False,
            'Additional post-outcome audit of unchanged systems/caps/cohort; cannot rescue earlier failures or retroactively change original timing endpoints.')
    rich=run/'rich_behavior'
    for name,fields,decode in [
        ('resources/timing_raw.jsonl','end_to_end_ms;GPU_ms;host_RSS;state_decoded_every_request',True),
        ('resources/resource_summary.csv','mean_ms;mean_GPU_ms;resident_state_bytes;host_RSS',True),
        ('reports/resource_reduction.csv','mean_relative_resource_reduction;simultaneous_low;simultaneous_high',True),
        ('resources/direct_reference_timings.csv','end_to_end_ms;GPU_ms;host_RSS',False),
        ('resources/direct_reference_summary.csv','mean_end_to_end_ms;mean_GPU_ms',False),
        ('matched_resource_supplement/timing_raw.jsonl','end_to_end_ms;GPU_ms;host_RSS;state_decoded_every_request',True),
        ('matched_resource_supplement/resource_summary.csv','mean_wall_ms;mean_wall_upper_simultaneous;peak_host_RSS_bytes',True)]:
        add(rich/name,'rich full-vocabulary probability endpoint',fields,True,decode)
    paths=['resource_cap_confirmation/execution_prompt_schema_amendment.json',
           'resource_cap_confirmation/execution_worker_cache_amendment.json',
           'resource_cap_confirmation/probability_endpoint_audit/protocol.json',
           'resource_cap_confirmation/probability_endpoint_audit/memory_plan.json',
           'resource_cap_confirmation/probability_endpoint_audit/worker_cache_realization.json',
           'protocol/global_resource_cap_report_guard.json','protocol/probability_endpoint_report_guard.json']
    notes={
        'MCQ_coordinate_output':'Original labels containing end_to_end or complete_predictor refer to the timed request ending at coordinates. Peer neural/scoring computations and adapter prediction are inside; the final predicted-choice inverse-transform/softmax is outside. Do not relabel these old measurements as full probability-return latency.',
        'MCQ_warm_state':'The adapter is decoded and public arrays loaded before warm requests; this is a resident-state service measurement, not per-request binary decoding.',
        'rich_probability_output':'Original and supplemental rich replacement timings include full-support probability normalization and per-request state decoding. Direct contextual model references have no adapter to decode.',
        'MCQ_worker_cache_versions':'New coordinate batch1 natural and batch1/context128 retain limit4. Four wholly unstarted secondary settings use separately recorded limit5; all prior specs remain preserved. The later probability audit uses its separately locked limit9 for nine actual peer groups. These are orchestration caches, not additional deployed target state.',
        'cache_realization_snapshot':'probability_endpoint_audit/worker_cache_realization.json records the actually loaded nine-group weight sum, summed worker RSS and host anonymous memory at one instant. It is not a peak, physically disjoint memory accounting, or the RAM of one deployed predictor; summed process RSS can double-count shared pages.',
        'state_memory':'MCQ resident_target_state_total_bytes is encoded bytes plus decoded numeric payload; rich resident_state_bytes is numeric payload. Python objects, shared arrays, model weights, process RSS and CUDA allocation are separate. Resident-state supplements measure additional overhead.',
        'partial_vs_final':'This metadata is a snapshot. Final report/package generation must refresh it after all timing markers exist. It never edits original fields or numerical values.',
        'historical_gap_audit':'T6_RESOURCE_CAP_AUDIT.md records the earlier missing-route finding and is superseded by the completion amendment and two later independent cap experiments.'}
    complete=(cap/'complete.json').exists() and (cap/'probability_endpoint_audit/complete.json').exists()
    pd.DataFrame(rows).to_csv(out/'TIMING_SCOPE_ERRATA.csv',index=False)
    data={'generated_unix':time.time(),'code_sha256':sha(Path(__file__)),
        'all_MCQ_timing_routes_complete':complete,'original_artifacts_modified':False,
        'interpretation_notes':notes,'artifact_field_map':rows,'execution_scope_map':execution,
        'supporting_metadata_sha256':{name:sha(run/name) for name in paths if (run/name).exists()},
        'field_map_csv_sha256':sha(out/'TIMING_SCOPE_ERRATA.csv')}
    (out/'TIMING_SCOPE_ERRATA.json').write_text(json.dumps(data,indent=2)+'\n')
    text=['# Timing endpoint and memory scope errata','','Original measurements and frozen labels remain unchanged. The machine-readable JSON/CSV maps each artifact and field to the actual endpoint.','']
    text.extend(f'- **{key}**: {value}' for key,value in notes.items())
    text.extend(['',f'Current snapshot: {len(rows)} artifact mappings, {len(execution)} physical execution scopes. All MCQ timing routes complete: {complete}.'])
    (out/'TIMING_SCOPE_ERRATA.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'artifacts':len(rows),'execution_scopes':len(execution),'all_MCQ_timing_routes_complete':complete}))


if __name__=='__main__':main()
