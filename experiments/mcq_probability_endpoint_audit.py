#!/usr/bin/env python3
"""Separately locked, post-outcome audit of the full probability-return endpoint.

This adds a conservative operational guard; it does not reselect any methods,
change registered quality contrasts, or turn the earlier coordinate endpoint
into a prospectively registered probability endpoint.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from mcq_resource_benchmark import execution_records
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/runs/v2'
CAP=RUN/'resource_cap_confirmation'
BASE=CAP/'probability_endpoint_audit'
WRAPPER=ROOT/'experiments/mcq_probability_endpoint_benchmark.py'
ORIGINAL=ROOT/'experiments/mcq_resource_benchmark.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def request_ast(p):
 tree=ast.parse(Path(p).read_text())
 return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='request'),include_attributes=False)
def prepare():
 if (BASE/'audit_lock.json').exists():raise RuntimeError('Audit already frozen')
 old=json.loads((CAP/'benchmark_execution_specs/batch1_context0.json').read_text())
 refs=json.loads((RUN/'resource_benchmark/confirmation_catalog/direct_model_resource_spec.json').read_text())
 assert request_ast(WRAPPER)==request_ast(ORIGINAL),'Underlying request computation changed'
 spec=old|{'measurement_role':'post_outcome_conservative_probability_endpoint_audit','systems':old['systems']+refs['systems'],'maximum_worker_processes':9,'seed':81724053,
  'returned_endpoint':'canonical_four_option_probabilities','scope':'Natural untruncated batch1, all19 systems on one GPU and same10 randomized paired blocks. No quality or cap reselection. Same2768 fresh cohort; all choices already frozen. Warm decoded resident-state endpoint including canonical softmax; excludes cold/load/decode/activation.',
  'post_outcome_audit_not_retroactive_preregistration':True}
 assert len(spec['systems'])==19 and spec['batch_sizes']==[1] and spec['context_lengths']==[0]
 weights={}
 catalog=RUN/'resource_benchmark/confirmation_catalog'
 for pattern in ['profile_minimal_gpu*/startup.json','direct_measurement/startup.json','direct_batch8_measurement/startup.json','direct_batch8_controlled_measurement/startup.json']:
  for path in catalog.glob(pattern):
   for row in json.loads(path.read_text()):
    for k,v in row['weight_bytes'].items():
     if int(k) in weights:assert weights[int(k)]==v
     weights[int(k)]=v
 assert set(weights)==set(range(6)),weights
 groups={}
 for s in spec['systems']:
  records=execution_records(s);key=tuple(r['index'] for r in records)
  groups.setdefault(key,[]).append(s['system_id'])
 assert len(groups)==9
 group_rows=[{'peers':list(k),'systems':v,'weight_bytes':sum(weights[i] for i in k)} for k,v in groups.items()]
 total=sum(r['weight_bytes'] for r in group_rows);oldcache=json.loads((CAP/'execution_worker_cache_amendment.json').read_text())['total_peer_weights_one_full_worker_cache']
 dump(BASE/'benchmark_spec.json',spec)
 dump(BASE/'memory_plan.json',{'created_unix':time.time(),'worker_groups':group_rows,'model_weight_bytes':weights,'audit_full_cache_weights_bytes':total,'other_two_full_coordinate_cache_weights_bytes':2*oldcache,'three_controller_full_weight_bound_bytes':total+2*oldcache,'host_cgroup_limit_bytes':864000000000,'weights_only_bound_not_total_host_memory':True,'safety':'Actual memory/anonymous/reclaimable-file snapshot and idleGPU check before launch; reserve overhead beyond measured weights. Only one modelgroup active onGPU atonce. All9 cachedgroups retain CPUweights between blocks. Existingqueues unchanged.'})
 dump(BASE/'protocol.json',{'created_unix':time.time(),'audit_type':'post_outcome_endpoint_completion_no_reselection','old_endpoint':'prediction coordinates; probability conversion omitted from original timer','new_endpoint':'canonical four-option probabilities; core.probabilities inside synchronized wall interval','warm_state_definition':'Adapter decoded/resident before requests; cold model load, adapter construction and worker activation excluded and separately recorded','cohort_size':2768,'replacements':13,'contextual_direct_references':6,'batch_size':1,'context_tokens':0,'blocks':10,'timed_per_block':20,'warmup_per_block':20,'selection_changes':False,'registered_quality_and_cap_choices_unchanged':True,'direct_target_serving_reference_is_retirement_infeasible':True,'replay':'All19 systems compared to cached canonical coordinates and probabilities on8 fixed keys; batch1 gate1e-8 coord/prob,1e-12sum; batch8 numerical diagnostic only','future_claim_guard':'New full-probability cap compliance is an additional conservative condition. An earlier coordinate cap failure must never be rescued. No reaveraging or dropping of failed cap cells. Global guard also combines rich route and family16 error accounting.','latency_family_one_sided_divisor':16,'interval_bootstrap_draws':10000,'interval_seed':81724053,'gpu_assignment_rule':'First whole existing coordinate queue to complete and release its GPU; verify no process/GPU allocation before claiming. All19 systems on that one GPU. Existingqueues never interrupted.','no_summing_isolated_cpu_microbench_as_live_latency':True})
 print(json.dumps({'systems':19,'groups':len(groups),'cache_weight_GiB':total/2**30,'three_queue_weight_GiB':(total+2*oldcache)/2**30}))
def freeze():
 if (BASE/'audit_lock.json').exists():raise RuntimeError('Preserve existing audit lock')
 review=json.loads((BASE/'independent_review.json').read_text());assert review['approved_before_launch']
 previous=json.loads((CAP/'confirmation_lock.json').read_text())
 for rel,h in previous['frozen_artifact_hashes'].items():assert sha(CAP/rel)==h,rel
 assert request_ast(WRAPPER)==request_ast(ORIGINAL)
 paths=[WRAPPER,ORIGINAL,Path(__file__),ROOT/'experiments/mcq_probability_endpoint_run.py',BASE/'benchmark_diff.patch',BASE/'benchmark_spec.json',BASE/'protocol.json',BASE/'memory_plan.json',BASE/'independent_review.json',CAP/'confirmation_lock.json',CAP/'selection_lock.json',CAP/'required_items_test.parquet',CAP/'benchmark_prompts_preserving_test_role.parquet',CAP/'execution_prompt_schema_amendment.json',CAP/'execution_worker_cache_amendment.json']
 spec=json.loads((BASE/'benchmark_spec.json').read_text())
 for s in spec['systems']:
  if s.get('bundle'):paths.extend(p for p in Path(s['bundle']).iterdir() if p.is_file())
 paths.extend((CAP/'response_cache_test').glob('*.parquet'))
 dump(BASE/'audit_lock.json',{'created_unix':time.time(),'audit_frozen_before_new_live_measurements':True,'post_outcome_audit_not_retroactive_preregistration':True,'underlying_request_ast_unchanged':True,'no_fitted_state_cap_cohort_or_quality_selection_changes':True,'original_confirmation_lock_sha256':sha(CAP/'confirmation_lock.json'),'frozen_hashes':{str(p.resolve().relative_to(ROOT)):sha(p) for p in sorted(set(paths))}})
 print('AUDIT_FROZEN')
def verify_lock():
 lock=json.loads((BASE/'audit_lock.json').read_text())
 for rel,h in lock['frozen_hashes'].items():assert sha(ROOT/rel)==h,rel
 return lock

def analyze():
 lock=verify_lock()
 if (BASE/'complete.json').exists():raise RuntimeError('Preserve completed endpoint audit')
 spec=json.loads((BASE/'benchmark_spec.json').read_text());expected={s['system_id'] for s in spec['systems']};raw=BASE/'measurement/raw_blocks.jsonl';allrows=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
 assert len(allrows)==190 and {r['system_id'] for r in allrows}==expected
 uuids={r['runtime']['GPU_uuid'] for r in allrows};assert len(uuids)==1 and next(iter(uuids)),'Paired setting crossed GPUs or lacks UUID'
 rows=[];traces=[];failures=[];primary_replays=[]
 for sid in sorted(expected):
  blocks=[r for r in allrows if r['system_id']==sid];assert len(blocks)==10 and {r['timing_block'] for r in blocks}==set(range(10))
  valid=[r for r in blocks if not r.get('failure_status')];failures.extend(r for r in blocks if r.get('failure_status'))
  if valid:
   assert all(r['batch_size']==1 and r['context_tokens']==0 and r['returned_endpoint']=='canonical_four_option_probabilities' and r['warmup_requests']==20 and r['timed_requests']==20 and len(r['traces'])==20 for r in valid)
   means=np.array([np.mean([t['end_to_end_ms'] for t in r['traces']]) for r in valid]);flat=[t for r in valid for t in r['traces']]
   assert all(t['context_count']==1 and t['truncated_peer_prompts']==0 and t['probability_finite'] and t['probability_sum_max_error']<1e-12 and t['probability_output_bytes']==32 and len(t['probability_sha256'])==64 for t in flat)
   rng=np.random.default_rng(81724053);boot=means[rng.integers(len(means),size=(10000,len(means)))].mean(1)
   rec={'system_id':sid,'system_role':valid[0]['system_role'],'batch_size':1,'context_tokens':0,'completed_blocks':len(valid),'failed_blocks':len(blocks)-len(valid),'timed_requests':len(flat),'mean_ms_per_context':float(means.mean()),'latency_upper_one_sided_alpha_div16':float(np.quantile(boot,1-.05/16)),'peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in flat),'peak_reserved_bytes':max(t['peak_reserved_bytes'] for t in flat),'peak_host_RSS':max(t['host_RSS'] for t in flat),'maximum_probability_sum_error':max(t['probability_sum_max_error'] for t in flat),'returned_endpoint':'canonical_four_option_probabilities','timing_scope':valid[0]['timing_scope']}
   for r in valid:
    for t in r['traces']:traces.append({'system_id':sid,'timing_block':r['timing_block'],'system_role':r['system_role'],**t})
  else:rec={'system_id':sid,'completed_blocks':0,'failed_blocks':10,'timed_requests':0}
  rows.append(rec)
  replay=json.loads((BASE/f'measurement/{sid}.replay.json').read_text());assert not replay['failure_status'];primary=next(c for c in replay['replay_checks'] if c['batch_size']==1);assert primary['max_probability_difference']<=1e-8 and primary['max_prediction_coordinate_difference']<=1e-8 and primary['probability_finite'] and primary['probability_sum_max_error']<=1e-12;primary_replays.append({'system_id':sid,**primary})
 assert len(traces)==20*(190-len(failures))
 pd.DataFrame(rows).to_csv(BASE/'resource_summary.csv',index=False);pd.DataFrame(traces).to_parquet(BASE/'timing_raw.parquet',index=False);pd.DataFrame(primary_replays).to_csv(BASE/'canonical_replay_checks.csv',index=False);dump(BASE/'failed_blocks.json',failures)
 lookup={r['system_id']:r for r in rows};coverage=[]
 for r in pd.read_csv(CAP/'cap_quality_results.csv').to_dict('records'):
  x=lookup.get(r.get('system_id'),{});complete=x.get('completed_blocks')==10
  point=complete and x['mean_ms_per_context']<=r['latency_cap_ms_per_context'] and x['peak_allocated_bytes']<=r['allocated_VRAM_cap_bytes'];bound=point and x['latency_upper_one_sided_alpha_div16']<=r['latency_cap_ms_per_context']
  coverage.append({**r,'probability_endpoint_point_compliant':bool(point),'probability_endpoint_bound_compliant':bool(bound),'probability_mean_ms_per_context':x.get('mean_ms_per_context'),'probability_latency_upper_alpha_div16':x.get('latency_upper_one_sided_alpha_div16'),'probability_peak_allocated_bytes':x.get('peak_allocated_bytes'),'probability_peak_reserved_bytes':x.get('peak_reserved_bytes'),'probability_peak_host_RSS':x.get('peak_host_RSS'),'post_outcome_additional_guard':True})
 pd.DataFrame(coverage).to_csv(BASE/'cap_compliance_additional_guard.csv',index=False)
 spec_lock=json.loads((BASE/'measurement/measurement_lock.json').read_text());assert spec_lock['spec_sha256']==sha(BASE/'benchmark_spec.json') and spec_lock['benchmark_code_sha256']==sha(WRAPPER);assert lock['created_unix']<__import__('datetime').datetime.fromisoformat(spec_lock['created_at']).timestamp()
 (BASE/'ENDPOINT_SCOPE.md').write_text('The original six configuration measurements end at prediction coordinates. This separate post-outcome audit returns canonical four-option probabilities and times their conversion inside the synchronized wall interval. All states, caps, cohort keys and quality choices remain unchanged. It is an additional conservative operational guard, not retroactive preregistration. Earlier failures cannot be rescued; both policies must satisfy every registered common-feasible cap. The warm resident-state scope excludes model cold loading, adapter decoding/construction and GPU activation. Direct target-serving references are context only and retirement-infeasible. The returned NumPy probability array is materialized, without network serialization. GPU stream elapsed includes CPU-induced idle gaps and is not hardware utilization.\n')
 names=['resource_summary.csv','timing_raw.parquet','canonical_replay_checks.csv','failed_blocks.json','cap_compliance_additional_guard.csv','ENDPOINT_SCOPE.md'];files={str((BASE/n).relative_to(BASE)):sha(BASE/n) for n in names}
 for p in (BASE/'measurement').glob('*.json'):files[str(p.relative_to(BASE))]=sha(p)
 files['measurement/raw_blocks.jsonl']=sha(raw)
 dump(BASE/'complete.json',{'complete':True,'created_unix':time.time(),'endpoint':'canonical_four_option_probabilities','post_outcome_additional_guard_not_preregistration':True,'all19_systems_sameGPU':len({r['runtime']['GPU_uuid'] for r in allrows if not r.get('failure_status')})==1,'replacement_systems':13,'contextual_direct_references':6,'completed_blocks':len(allrows)-len(failures),'failed_blocks':len(failures),'planned_blocks':190,'timed_requests':len(traces),'cohort_size':2768,'audit_lock_sha256':sha(BASE/'audit_lock.json'),'selection_lock_sha256':sha(CAP/'selection_lock.json'),'outputs':files,'global_guard_must_require_old_and_new_compliance':True})
 print('PROBABILITY_ENDPOINT_AUDIT_COMPLETE',len(traces))
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','analyze']);a=p.parse_args();globals()[a.action]()
if __name__=='__main__':main()
