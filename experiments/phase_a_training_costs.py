#!/usr/bin/env python3
"""T7 measured one-time/collection accounting, without equating CPU and GPU time."""
import json,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2'

def mcq_timing_costs(rows):
 """Count physical timing traces, never concatenated duplicate summary files."""
 for subtree,phase in [('resource_benchmark','mcq_resource_measurements'),('resource_cap_confirmation/resources','mcq_resource_caps_measurements'),('resource_cap_confirmation/probability_endpoint_audit/measurement','mcq_probability_endpoint_measurements')]:
  for path in sorted((RUN/subtree).rglob('raw_blocks.jsonl')):
   blocks=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
   if not blocks:continue
   good=[b for b in blocks if not b.get('failure_status')];traces=[t for b in good for t in b['traces']]
   models=sum(t['context_count']*b.get('actual_call_q',len(b['models'])) for b in good for t in b['traces'])
   warm=sum(b['warmup_requests']*b['batch_size']*b.get('actual_call_q',len(b['models'])) for b in good)
   note='Observed coordinate-output request intervals only; inverse transform/probability normalization is excluded and adapter decode/load is before warm timing. Timing repetitions are not independent scientific samples. Includes archived preliminary profiles where present; their pre-pruning active-call count is the recorded model-list length. Warmup/activation/loading and request intervals are different scopes; do not add them to inclusive process times. CUDA stream elapsed includes idle intervals and is not GPU busy time.'
   if 'probability' in phase:note='Observed warm resident-adapter intervals through canonical probability materialization; setup/decode/activation and network serialization excluded. Repeated timing inputs are not extra scientific samples; CUDA intervals include idle gaps and are not GPU busy time.'
   rows.append({'phase':phase,'operation':'measured probability-materialization timing requests' if 'probability' in phase else 'measured coordinate-output timing requests','query_model_evaluations':models,'timed_requests':len(traces),'failed_blocks_retained':len(blocks)-len(good),'summed_timed_request_wall_seconds':sum(t['end_to_end_ms'] for t in traces)/1000,'summed_CUDA_event_intervals_seconds':sum(t['GPU_stream_elapsed_ms'] for t in traces)/1000,'summed_peer_GPU_intervals_seconds':sum(t['peer_GPU_elapsed_ms'] for t in traces)/1000,'untimed_model_activation_seconds':sum(b.get('activation_seconds',0) for b in good),'record_kind':'serving_timing_model_contexts','status':'measured_completed_blocks','source':str(path.relative_to(RUN)),'note':note})
   rows.append({'phase':phase,'operation':'registered warmups for recorded timing blocks','query_model_evaluations':warm,'record_kind':'serving_warmup_model_contexts','status':'measured_completed_blocks','source':str(path.relative_to(RUN)),'note':'Forward count from actual per-block warmup count, batch size and active callable peers. Warmup time was not isolated; partial failed blocks can contain unrecorded calls and are not assumed free.'})
   startup=path.parent/'startup.json'
   if startup.exists():
    starts=json.loads(startup.read_text())
    rows.append({'phase':phase,'operation':'summed worker cold/model loading intervals','model_load_seconds':sum(s.get('cold_model_load_seconds',0) for s in starts),'worker_startups':len(starts),'record_kind':'summed_worker_load_intervals','status':'measured_completed_startups','source':str(startup.relative_to(RUN)),'note':'Actual worker initialization intervals; LRU reloads retained. Not campaign elapsed or GPU busy time. Replay checks/startup verification are additional operations; overwritten replay files do not establish a complete extra-call total.'})
   if 'probability' in phase:
    active={b['system_id']:b['actual_call_q'] for b in good};calls=0;checked=0
    for replaypath in path.parent.glob('*.replay.json'):
     replay=json.loads(replaypath.read_text())
     if replay.get('failure_status') or replay['system_id'] not in active:continue
     calls+=sum(c['query_count'] for c in replay['replay_checks'])*active[replay['system_id']];checked+=1
    rows.append({'phase':phase,'operation':'one replay per audit system, canonical and diagnostic batch8','query_model_evaluations':calls,'audited_systems':checked,'record_kind':'additional_diagnostic_calls','status':'measured_completed_replays','source':str(path.parent.relative_to(RUN)),'note':'The new wrapper records one replay per system, not one per model group. Coordinate and probability checks reuse the same neural outputs; they are counted once. Batch8 is diagnostic only. Isolated elapsed time was not logged.'})

def main():
 rows=[];p=RUN/'phase_a';profile=json.loads((p/'training_resource_profile.json').read_text())
 rows.append({'phase':'phase_a','operation':'complete context construction + adapter search + freezing','model_id':'cached six-model family','query_model_evaluations':0,'wall_seconds':profile['fit_and_freeze_wall_seconds'],'summed_worker_wall_seconds':profile['sum_context_worker_wall_seconds'],'peak_worker_RSS_bytes':profile['max_worker_peak_RSS_bytes'],'public_context_disk_bytes':profile['public_context_disk_bytes'],'workers':profile['workers'],'BLAS_threads_per_worker':profile['blas_threads'],'hardware_scope':'CPU cgroup48cores; no neural forward calls','status':'complete','source':'phase_a/training_resource_profile.json','note':'615 shared contexts,2250interface fits,5,715,000candidate evaluations. Context-worker wall includes construction and all targets; not CUDA elapsed or CPU core-seconds.'})
 lines=(RUN/'phase_a_evaluate.log').read_text().splitlines();first=time.mktime(time.strptime(lines[0][:19],'%Y-%m-%d %H:%M:%S'));last=next(line for line in reversed(lines) if ' COMPLETE ' in line);finish=time.mktime(time.strptime(last[:19],'%Y-%m-%d %H:%M:%S'))
 rows.append({'phase':'phase_a','operation':'frozen test evaluation + paired bootstrap + aggregation','query_model_evaluations':0,'wall_seconds':finish-first,'workers':6,'BLAS_threads_per_worker':4,'hardware_scope':'CPU cgroup48cores; cached responses','status':'complete','source':'phase_a_evaluate.log','note':'Log timestamps have1second precision; includes artifact hash validation and final aggregation.'})
 for path in sorted(list((RUN/'confirmation').glob('response_cache_*/*.provenance.json'))+list((RUN/'resource_cap_confirmation').glob('response_cache_*/*.provenance.json'))):
  phase='resource_cap_confirmation' if 'resource_cap_confirmation' in path.parts else 'confirmation'
  d=json.loads(path.read_text());rows.append({'phase':phase,'operation':f"fresh {d['stage']} response collection",'model_id':d['model_id'],'query_model_evaluations':d.get('completed_rows',0),'wall_seconds':d.get('collection_seconds'),'model_load_seconds':d.get('model_load_seconds'),'hardware_scope':d['runtime'].get('gpu'),'status':'complete' if d.get('completed') else 'running','source':str(path.relative_to(RUN)),'record_kind':'scientific_collection','note':'Collection-loop walltime excludes model loading and diagnostic checks; the inclusive process-duration rows overlap this timing and must not be summed with it.'})
  official=len((d.get('official_forward_check') or {}).get('rows',[]));bridge=len((d.get('historical_runtime_bridge') or {}).get('rows',[]))
  # The one-token codebook has one decision branch. Gemma obtains hidden and
  # official logits in one forward; other models compare two forwards.
  diagnostic_calls=official*(1 if 'gemma' in d['model_id'].lower() else 2)+bridge
  rows.append({'phase':phase,'operation':f"{d['stage']} official scorer and historical bridge checks",'model_id':d['model_id'],'query_model_evaluations':diagnostic_calls,'diagnostic_check_rows':official+bridge,'status':'complete' if d.get('completed') else 'running','source':str(path.relative_to(RUN)),'record_kind':'additional_diagnostic_calls','hardware_scope':d['runtime'].get('gpu'),'note':'Forward count reconstructed from recorded check rows and unchanged scorer. Diagnostic elapsed time was not isolated; blank time is unknown, not zero. Historical bridge is diagnostic only and does not enter fresh scientific data.'})
 commands=RUN/'confirmation/collection_commands.jsonl'
 if commands.exists():
  for line in commands.read_text().splitlines():
   d=json.loads(line);rows.append({'phase':'confirmation','operation':f"inclusive {d['stage']} collection process model {d['model_global_id']}",'model_id':d['model_global_id'],'wall_seconds':d['end_unix']-d['start_unix'],'hardware_scope':f"B200 logical device {d['gpu']}",'status':'complete' if d['returncode']==0 else 'failed','source':str(commands.relative_to(RUN)),'record_kind':'overlapping_inclusive_process_wall','note':'Includes process startup, model loading, checks, collection and shutdown. Overlaps corresponding collection/diagnostic rows; do not add these wall durations together.'})
 commands=RUN/'confirmation/fit_evaluation_commands.jsonl'
 if commands.exists():
  for line in commands.read_text().splitlines():
   d=json.loads(line);rows.append({'phase':'confirmation','operation':'fresh '+d['stage']+' adapter runner','wall_seconds':d['end_unix']-d['start_unix'],'query_model_evaluations':0,'hardware_scope':'CPU cached fresh bank','status':'complete' if d['returncode']==0 else 'failed','source':str(commands.relative_to(RUN)),'note':'All targets share pair contexts; repeated training seeds reuse one sealed test population, not extra independent queries.'})
 commands=RUN/'confirmation/analysis_commands.jsonl'
 if commands.exists():
  for line in commands.read_text().splitlines():
   d=json.loads(line);scripts=[Path(x).name for x in d['command'] if x.endswith('.py')]
   # confirmation_fit.py is an inclusive wrapper around the already counted
   # phase_a_run.py evaluation command, not a second scientific computation.
   if 'confirmation_fit.py' in scripts:continue
   rows.append({'phase':'confirmation','operation':'post-collection analysis '+','.join(scripts),'wall_seconds':d['end_unix']-d['start_unix'],'query_model_evaluations':0,'hardware_scope':'CPU cached fresh bank','status':'complete' if d['returncode']==0 else 'failed','source':str(commands.relative_to(RUN)),'record_kind':'postcollection_analysis','note':'Separate primary paired analysis or cheap-control evaluation process. The overlapping confirmation_fit.py wrapper is excluded because its inner runner is already counted.'})
 rich_cost=RUN/'rich_behavior/reports/measured_training_collection_costs.csv'
 if rich_cost.exists():
  for d in pd.read_csv(rich_cost).to_dict('records'):
   rows.append({'phase':'rich_behavior','operation':d['stage']+' '+str(d.get('role','')),'model_id':d.get('model_id'),'query_model_evaluations':d.get('model_context_evaluations'),'wall_seconds':d.get('measured_stage_wall_seconds'),'timed_forward_batch_seconds':d.get('timed_forward_seconds'),'candidate_count':d.get('candidate_count'),'bytes_outputs':d.get('bytes_outputs'),'hardware_scope':'B200; stage-scoped timing, not GPU busy time','status':d['status'],'source':'rich_behavior/'+d['artifact'],'record_kind':d['stage'],'note':d.get('measurement_scope')})
  scope=json.loads(rich_cost.with_name('cost_accounting_scope.json').read_text())
  rows.append({'phase':'rich_behavior','operation':'collection/deployment bridge and official full-forward checks','query_model_evaluations':scope['known_bridge_overhead_contexts']+scope['official_forward_check_extra_calls'],'status':'complete','source':'rich_behavior/reports/cost_accounting_scope.json','record_kind':'additional_diagnostic_calls','note':'Recorded additional diagnostics; isolated walltime unavailable. Preserved failed batch8 response collection is counted in its own rows above.'})
 else:
  rows.append({'phase':'rich_behavior','operation':'complete rich cost accounting','status':'running','source':'rich_behavior/reports/measured_training_collection_costs.csv','note':'Required measured collection/fit accounting has not been exported yet.'})
 # Later cap-route collection has its own fresh banks; it does not refit models.
 seen_commands=set()
 for filename in ['collection_commands.jsonl','parallel_collection_commands.jsonl']:
  commands=RUN/'resource_cap_confirmation'/filename
  if commands.exists():
   for line in commands.read_text().splitlines():
    d=json.loads(line);key=(d['model_global_id'],d['start_unix'])
    if key in seen_commands:continue
    seen_commands.add(key)
    rows.append({'phase':'resource_cap_confirmation','operation':f"inclusive second-cohort collection process model {d['model_global_id']}",'model_id':d['model_global_id'],'wall_seconds':d['end_unix']-d['start_unix'],'hardware_scope':f"B200 logical device {d['gpu']}",'status':'complete' if d['returncode']==0 else 'failed','source':str(commands.relative_to(RUN)),'record_kind':'overlapping_inclusive_process_wall','note':'Overlaps the corresponding new-cohort collection/diagnostic rows; no additional model fitting. Short resumed commands can verify an already-completed bank without new response collection; scientific counts come only from the unique bank provenance.'})
 cap_cost=RUN/'rich_behavior/matched_resource_supplement/measured_cost_accounting.csv'
 if cap_cost.exists():
  for d in pd.read_csv(cap_cost).to_dict('records'):
   rows.append({'phase':'rich_resource_caps','operation':d['stage'],'model_id':d.get('model_id'),'query_model_evaluations':d.get('model_context_evaluations'),'wall_seconds':d.get('measured_stage_wall_seconds'),'model_load_seconds':d.get('model_load_seconds'),'timed_forward_batch_seconds':d.get('timed_forward_seconds'),'summed_timed_request_wall_seconds':d.get('summed_request_wall_seconds'),'summed_CUDA_event_intervals_seconds':d.get('summed_CUDA_event_intervals_seconds'),'bytes_outputs':d.get('bytes_outputs'),'status':d['status'],'source':'rich_behavior/'+d['artifact'],'record_kind':d.get('record_kind') if pd.notna(d.get('record_kind')) else d['stage'],'note':d['measurement_scope']+' Existing fit/development costs remain in the original ledger; no second charge.'})
 else:
  rows.append({'phase':'rich_resource_caps','operation':'supplemental collection and live measurement accounting','status':'running','source':'rich_behavior/matched_resource_supplement/measured_cost_accounting.csv','note':'Final cost export is pending coordinator completion; existing fits are not charged a second time.'})
 execution_cost=RUN/'resource_cap_confirmation/final_compute_accounting/measured_resource_execution_costs.csv'
 if execution_cost.exists():
  for d in pd.read_csv(execution_cost).to_dict('records'):
   # The probability audit process is accounted directly from its journal below.
   if d['operation']=='probability_materialization_endpoint_audit':continue
   rows.append({'phase':'mcq_resource_caps_measurements','operation':'inclusive '+d['operation'],'wall_seconds':d['inclusive_command_wall_seconds'],'status':'complete' if d['returncode']==0 else 'preserved_startup_failure','hardware_scope':f"B200 logical device {d['GPU']}",'record_kind':'overlapping_inclusive_process_wall','nested_timed_requests':d['timed_requests'],'source':'resource_cap_confirmation/'+d['source'],'note':d['accounting_scope']+' Physical timed/warm model-context calls are separately counted from raw traces; these nested request counts are not additional calls.'})
  record=json.loads(execution_cost.with_name('manifest.json').read_text())
  for d in record['per_GPU_process_interval_unions']:
   rows.append({'phase':'mcq_resource_caps_measurements','operation':'union of recorded measurement-controller intervals','controller_interval_union_seconds':d['observed_controller_interval_union_seconds'],'hardware_scope':f"B200 logical device {d['GPU']}",'record_kind':'overlapping_process_interval_union','status':'complete','source':'resource_cap_confirmation/final_compute_accounting/manifest.json','note':'Posthoc per-device union of inclusive coordinate/probability measurement subprocess intervals, including preserved startup failures. Not hardware-active GPU time, energy, or extra work to add to the component/inclusive rows.'})
 probbase=RUN/'resource_cap_confirmation/probability_endpoint_audit'
 for name in ['execution_command.json','analysis_command.json']:
  path=probbase/name
  if path.exists():
   d=json.loads(path.read_text())
   rows.append({'phase':'mcq_probability_endpoint_measurements','operation':'inclusive '+name.removesuffix('.json'),'wall_seconds':d['end_unix']-d['start_unix'],'status':'complete' if d['returncode']==0 else 'failed','record_kind':'overlapping_inclusive_process_wall' if name=='execution_command.json' else 'postcollection_analysis','source':str(path.relative_to(RUN)),'note':'New audit process only. Waiting for an idle GPU is excluded; benchmark wall includes loading, warmups, replay and timed requests and must not be added to their component intervals.'})
 mcq_timing_costs(rows)
 out=RUN/'reports';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'T7_MEASURED_COMPUTE_ACCOUNTING.csv',index=False)
 text=['# Measured computation and amortization accounting','','This table reports measured elapsed times and memory in their original units. CPU elapsed time, summed worker time, CUDA-event time, model loading, and response collection are separate quantities. They are not added into a fictitious GPU-hour total or assigned an undocumented dollar price. Inclusive process-duration rows overlap collection/diagnostic rows; they must not be added. Blank durations mean unmeasured, not zero.','',f"Phase A used no new model queries: {profile['context_jobs']} public contexts and {profile['target_interface_fits']} fitted interfaces were completed and frozen in {profile['fit_and_freeze_wall_seconds']:.1f} wall seconds on 6 workers × 4 BLAS threads. Summed worker wall time was {profile['sum_context_worker_wall_seconds']:.1f} seconds; peak recorded RSS per worker was {profile['max_worker_peak_RSS_bytes']/2**20:.1f} MiB. Context files occupy {profile['public_context_disk_bytes']/1e9:.2f} GB including experiment-only arrays. The deployed memory accounting exports only necessary arrays and counts them explicitly.",'', 'Fresh response collection and rich full-vocabulary collection are listed per model and role. Official scorer/collection bridges are counted separately. Preserved failed rich batch8 banks, pilot collection, superseded fits, and public basis construction are included. Rich cost_accounting_scope.json records unmeasured dataset download, CPU tokenization/deduplication, process startup and incomplete interrupted work; none is silently treated as free. Resource-search and serving timing traces remain in the resource tables and must not be counted again as scientific collection. Active-worker RSS excludes the sum of inactive CPU peer groups retained by the benchmark coordinator; execution cache limits and cold/setup costs are separate infrastructure accounting. Rows labeled running are partial observations, not completion claims. Refresh this table after all collection and final measurements finish.','', 'Inference amortization requires an observed positive serving difference in a matched resource unit. No monetary or independent-tenant break-even is inferred from logical call counts or from target-state compression. A nonpositive measured serving gain has no finite positive break-even under the same cost definition.','', 'See `T7_MEASURED_COMPUTE_ACCOUNTING.csv` for all measured rows and their exact provenance filenames.']
 (out/'T7_MEASURED_COMPUTE_ACCOUNTING.md').write_text('\n'.join(text)+'\n');print(f'wrote{len(rows)} measured accounting rows')
if __name__=='__main__':main()
