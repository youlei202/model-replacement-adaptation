#!/usr/bin/env python3
"""Evaluate a locked resource-cap policy on a second unused cohort, then guard it
with final live latency/memory measurements. No test-dependent reselection.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from mcq_deployment import DeployedAdapter
from adapters_v2 import core
from mcq_resource_analyze import intervals
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2';BASE=RUN/'resource_cap_confirmation'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,a):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def verify_lock():
 lock=json.loads((BASE/'confirmation_lock.json').read_text())
 for rel,digest in lock['frozen_artifact_hashes'].items():
  if sha(BASE/rel)!=digest:raise RuntimeError('Frozen input changed '+rel)
 assert sha(BASE/'data_lock.json')==lock['data_lock_sha256'];return json.loads((BASE/'selection_lock.json').read_text())
def quality():
 p=verify_lock();dump(BASE/'run_status.json',{'stage':'evaluating_frozen_policies','updated_at':time.time()})
 if (BASE/'quality_complete.json').exists():raise RuntimeError('Preserve existing completed quality outputs')
 rows=pd.read_parquet(BASE/'required_items_test.parquet').sort_values('item_id').reset_index(drop=True);reg=json.loads((ROOT/'experiments/experiment_config/configs/campaign.json').read_text())['model_registry'];arrays={};bankhash={};runtime=[]
 for model in reg:
  name=model['model_id'].replace('/','--');bank=BASE/f'response_cache_test/{name}.parquet';prov=BASE/f'response_cache_test/{name}.provenance.json';record=json.loads(prov.read_text());old=json.loads((RUN/f'confirmation/response_cache_fitval/{name}.provenance.json').read_text())
  assert record['completed'] and record['required_rows']==len(rows) and record['response_bank_sha256']==sha(bank) and record['official_forward_check']['passed'];assert record['runtime']==old['runtime'] and record['code_hashes']==old['code_hashes'],'Runtime/code mismatch against frozen methods'
  f=pd.read_parquet(bank);assert len(f)==len(rows) and not f.duplicated(['task_id','item_id','prompt_hash']).any();joined=rows[['item_id','prompt_hash']].merge(f,on=['item_id','prompt_hash'],validate='one_to_one',how='left',sort=False,indicator=True);assert joined._merge.eq('both').all();a=joined[[f'coord_{i}' for i in range(3)]].to_numpy(float);assert np.isfinite(a).all();arrays[model['index']]=a;bankhash[str(bank.relative_to(BASE))]=sha(bank);bankhash[str(prov.relative_to(BASE))]=sha(prov);runtime.append({'model_id':model['model_id'],'runtime_equal':True,'code_equal':True})
 states={r['system_id']:r['candidate'] for r in p['choices'] if r['system_id']};lossframes=[];summaries=[]
 for sid,c in states.items():
  adapter=DeployedAdapter(c['bundle']);pred=adapter.predict({i:arrays[i] for i in adapter.pair});target=arrays[c['target_index']];prob=core.probabilities(pred);truth=core.probabilities(target);frame=rows[['item_id','item_cluster','prompt_hash']].copy();frame['target_index']=c['target_index'];frame['system_id']=sid;frame['state_digest']=c['state_digest'];frame['tv']=np.abs(truth-prob).sum(1)/2;frame['kl']=(truth*(np.log(np.maximum(truth,1e-300))-np.log(np.maximum(prob,1e-300)))).sum(1);frame['argmax_disagreement']=(np.argmax(truth,1)!=np.argmax(prob,1)).astype(int)
  for j in range(3):frame[f'prediction_coord_{j}']=pred[:,j]
  lossframes.append(frame);summaries.append({'system_id':sid,'target_index':c['target_index'],'pair':c['pair'],'family':c['family'],'actual_bits':c['actual_bits'],'test_tv':float(frame.tv.mean()),'test_kl':float(frame.kl.mean()),'test_argmax_disagreement':float(frame.argmax_disagreement.mean()),'test_clusters':len(rows)})
 all_loss=pd.concat(lossframes,ignore_index=True);all_loss.to_parquet(BASE/'item_predictions_losses.parquet',index=False);summ=pd.DataFrame(summaries);summ.to_csv(BASE/'state_quality_results.csv',index=False);lookup={r['system_id']:r for r in summaries};caprows=[]
 for r in p['choices']:
  entry={k:v for k,v in r.items() if k!='candidate'};entry['selected_before_test']=True;entry['data_role']='second_untouched_external_population';entry['protocol_id']='replacement_v2_matched_resource_cap_supplement';entry['measured_resource_axis']='complete predictor natural batch1 mean wall latency AND observed peak allocatedVRAM'
  if r['system_id']:entry.update(lookup[r['system_id']])
  caprows.append(entry)
 pd.DataFrame(caprows).to_csv(BASE/'cap_quality_results.csv',index=False)
 vectors=[];argmax_vectors=[];meta=[];paired=[];choice={(r['target_index'],r['cap_id'],r['policy']):r for r in p['choices']};wide=all_loss.pivot(index=['item_id','item_cluster'],columns='system_id',values='tv').sort_index();argwide=all_loss.pivot(index=['item_id','item_cluster'],columns='system_id',values='argmax_disagreement').sort_index()
 for power in p['power']:
  target,comparator=power['target_index'],power['comparator'];caps=power['common_feasible_cap_ids'];delta=[];argdelta=[];fixed=[];joint=[]
  for cap in caps:
   fs=choice[target,cap,comparator]['system_id'];js=choice[target,cap,'budget_aware']['system_id'];difference=wide[fs]-wide[js];argdifference=argwide[js].astype(float)-argwide[fs].astype(float);argdelta.append(argdifference.to_numpy());delta.append(difference.to_numpy());fixed.append(wide[fs].to_numpy());joint.append(wide[js].to_numpy());frame=wide.index.to_frame(index=False);frame['target_index']=target;frame['comparator']=comparator;frame['cap_id']=cap;frame['difference']=difference.to_numpy();frame['fixed_loss']=wide[fs].to_numpy();frame['joint_loss']=wide[js].to_numpy();frame['joint_minus_fixed_argmax_disagreement']=argdifference.to_numpy();frame['fixed_system']=fs;frame['joint_system']=js;paired.append(frame)
  if not caps:continue
  vectors.append(np.mean(np.stack(delta,1),1));argmax_vectors.append(np.mean(np.stack(argdelta,1),1));meta.append({'target_index':target,'target_id':reg[target]['model_id'],'comparator':comparator,'primary':power['primary'],'number_common_caps':len(caps),'independent_test_clusters':len(rows),'common_feasible_cap_count':len(caps),'registered_total_caps':16,'test_clusters':len(rows),'fixed_tv':float(np.mean(fixed)),'joint_tv':float(np.mean(joint)),'gain_tv':float(np.mean(delta)),'familywise_primary_comparisons':16,'bootstrap_draws':10000,'selection_scope':'pretest cap/choices on frozen development information; delayed execution after original-cohort results existed','endpoint':'within-query mean over prospectively common-feasible cells; final cap failures never dropped','inference_scope':'family16 includes the other rich route; additional fixed andNNLS contrasts are descriptive','adequate_power':power['adequate_planned_n']})
 ci=intervals(np.stack(vectors,1),comparisons=16,seed=81724052,draws=10000);arg_ci=intervals(np.stack(argmax_vectors,1),comparisons=16,seed=81724052,draws=10000)
 for i,r in enumerate(meta):
  r.update(simultaneous_ci_low=float(ci['simultaneous_low'][i]),simultaneous_ci_high=float(ci['simultaneous_high'][i]),unadjusted_ci_low=float(ci['low'][i]),unadjusted_ci_high=float(ci['high'][i]));r['statistical_effect_criterion_met']=r['gain_tv']>=.005 and r['simultaneous_ci_low']>0;r['resource_cap_guard_pending']=True;r['joint_minus_fixed_argmax_disagreement']=float(arg_ci['mean'][i]);r['argmax_simultaneous_low']=float(arg_ci['simultaneous_low'][i]);r['argmax_simultaneous_high']=float(arg_ci['simultaneous_high'][i]);r['argmax_guard_margin']=.01;r['argmax_guard_passed']=r['argmax_simultaneous_high']<=.01;r['TV_and_argmax_statistical_criterion_met']=r['statistical_effect_criterion_met'] and r['argmax_guard_passed']
 summary=pd.DataFrame(meta);summary[summary.primary].to_csv(BASE/'primary_confirmation_results.csv',index=False);summary.to_csv(BASE/'all_comparator_results.csv',index=False);pd.concat(paired,ignore_index=True).to_parquet(BASE/'paired_query_differences.parquet',index=False)
 dump(BASE/'runtime_bridge.json',{'passed':True,'checks':runtime,'canonical_batch1':True,'unchanged_original_scorer':True});names=['item_predictions_losses.parquet','state_quality_results.csv','cap_quality_results.csv','primary_confirmation_results.csv','all_comparator_results.csv','paired_query_differences.parquet','runtime_bridge.json'];dump(BASE/'quality_complete.json',{'quality_completed':True,'created_unix':time.time(),'selection_lock_sha256':sha(BASE/'selection_lock.json'),'confirmation_lock_sha256':sha(BASE/'confirmation_lock.json'),'fresh_clusters':len(rows),'state_target_count':len(states),'primary_contrasts':8,'familywise_primary_contrasts_across_routes':16,'outputs':{n:sha(BASE/n) for n in names},'response_hashes':bankhash});dump(BASE/'run_status.json',{'stage':'quality_complete_awaiting_final_live_resources','updated_at':time.time(),'quality_complete':True});print(summary.to_string(index=False))
def resources(raw_paths):
 p=verify_lock();q=json.loads((BASE/'quality_complete.json').read_text());
 for name,h in q['outputs'].items():assert sha(BASE/name)==h
 if (BASE/'complete.json').exists():raise RuntimeError('Completed supplement already exists')
 expected=set(p['unique_final_systems']);blocks=[];fail=[];rawhash={}
 for raw in raw_paths:
  rawhash[str(raw.resolve())]=sha(raw)
  for line in raw.read_text().splitlines():
   if not line.strip():continue
   b=json.loads(line);b['_raw_source']=str(raw.resolve())
   if b.get('failure_status'):fail.append(b)
   else:blocks.append(b)
 for batch in [1,8]:
  for context in [0,128,512]:
   setting=[b for b in blocks+fail if b['batch_size']==batch and b['context_tokens']==context]
   assert len({b['_raw_source'] for b in setting})==1,'Whole setting must remain on one controller/GPU; do not split paired systems across GPUs'
 observed={};stats=[]
 for sid in sorted(expected):
  for batch in [1,8]:
   for context in [0,128,512]:
    bs=[b for b in blocks if b['system_id']==sid and b['batch_size']==batch and b['context_tokens']==context];failed=[b for b in fail if b.get('system_id')==sid and b.get('batch_size')==batch and b.get('context_tokens')==context]
    assert len(bs)+len(failed)==10 and len({b['timing_block'] for b in bs+failed})==10,'Resource blocks incomplete/duplicate'
    if bs:
     means=np.array([np.mean([t['end_to_end_ms']/t['context_count'] for t in b['traces']]) for b in bs]);tr=[t for b in bs for t in b['traces']];assert all(b['warmup_requests']>=20 and len(b['traces'])==20 for b in bs)
     if len(bs)>=2:
      rng=np.random.default_rng(81724053);boot=means[rng.integers(len(means),size=(10000,len(means)))].mean(1);low=float(np.quantile(boot,.05/(2*len(expected))));high=float(np.quantile(boot,1-.05/len(expected)))
     else:low=high=None
     row={'system_id':sid,'batch_size':batch,'context_tokens':context,'completed_blocks':len(bs),'failed_blocks':len(failed),'timed_requests':len(tr),'mean_ms_per_context':float(means.mean()),'latency_familywise_lower':low,'latency_familywise_upper_one_sided':high,'latency_interval_family_systems':len(expected),'peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in tr),'peak_reserved_bytes':max(t['peak_reserved_bytes'] for t in tr),'peak_host_RSS':max(t['host_RSS'] for t in tr),'quality_link':'primary canonical natural' if batch==1 and context==0 else 'secondary resource profile only; no transfer of primary quality','resource_scope':'complete predictor; shared public and target state separately in bundle manifest; conditional/standalone weight accounting both reported','model_parameter_and_buffer_bytes':max(b['model_parameter_and_buffer_bytes'] for b in bs),'actual_call_q':bs[0]['actual_call_q'],**bs[0]['adapter_accounting']}
    else:row={'system_id':sid,'batch_size':batch,'context_tokens':context,'completed_blocks':0,'failed_blocks':len(failed),'timed_requests':0,'mean_ms_per_context':None,'latency_familywise_upper_one_sided':None,'peak_allocated_bytes':None}
    stats.append(row)
    if batch==1 and context==0:observed[sid]=row
 pd.DataFrame(stats).to_csv(BASE/'resource_summary.csv',index=False);dump(BASE/'timing_failures.json',fail);capquality=pd.read_csv(BASE/'cap_quality_results.csv');coverage=[]
 for row in capquality.to_dict('records'):
  sid=row.get('system_id');exists=isinstance(sid,str) and sid in observed;record=observed.get(sid,{})
  point=exists and record['completed_blocks']==10 and record['mean_ms_per_context']<=row['latency_cap_ms_per_context'] and record['peak_allocated_bytes']<=row['allocated_VRAM_cap_bytes']
  bound=point and record['latency_familywise_upper_one_sided']<=row['latency_cap_ms_per_context'];coverage.append(row|{'final_measured_point_cap_compliant':bool(point),'final_latency_bound_and_observed_memory_compliant':bool(bound),'final_mean_ms_per_context':record.get('mean_ms_per_context'),'final_latency_upper':record.get('latency_familywise_upper_one_sided'),'final_peak_allocated_bytes':record.get('peak_allocated_bytes'),'final_peak_reserved_bytes':record.get('peak_reserved_bytes'),'final_peak_host_RSS':record.get('peak_host_RSS'),'no_postmeasurement_reselection':True})
 cov=pd.DataFrame(coverage);cov.to_csv(BASE/'coverage_and_cap_compliance.csv',index=False);allcompar=pd.read_csv(BASE/'all_comparator_results.csv');final=[]
 for row in allcompar.to_dict('records'):
  power=next(x for x in p['power'] if x['target_index']==row['target_index'] and x['comparator']==row['comparator']);sub=cov[(cov.target_index==row['target_index'])&cov.cap_id.isin(power['common_feasible_cap_ids'])&cov.policy.isin(['budget_aware',row['comparator']])];assert len(sub)==2*len(power['common_feasible_cap_ids']);point=bool(sub.final_measured_point_cap_compliant.all());bounds=bool(sub.final_latency_bound_and_observed_memory_compliant.all());row.update(resource_cap_guard_pending=False,all_common_caps_fresh_confirmed=point,same_resource_advantage_confirmed=bool(row['TV_and_argmax_statistical_criterion_met'] and bounds),both_policies_all_registered_cells_point_compliant=point,both_policies_all_registered_cells_confidence_compliant=bounds,quality_and_observed_resource_claim_supported=bool(row['TV_and_argmax_statistical_criterion_met'] and point),quality_and_confidence_resource_claim_supported=bool(row['TV_and_argmax_statistical_criterion_met'] and bounds));final.append(row)
 pd.DataFrame(final).to_csv(BASE/'guarded_confirmation_results.csv',index=False)
 names=['resource_summary.csv','timing_failures.json','coverage_and_cap_compliance.csv','guarded_confirmation_results.csv'];dump(BASE/'complete.json',{'complete':True,'completed':True,'quality_completed':True,'deployment_measured':True,'data_freshness_verified':True,'adequate_power':not p['underpowered'],'created_unix':time.time(),'fresh_clusters':q['fresh_clusters'],'primary_contrasts':8,'familywise_primary_contrasts_across_routes':16,'quality_outputs':q['outputs'],'outputs':{n:sha(BASE/n) for n in names},'raw_measurements':rawhash,'selection_lock_sha256':sha(BASE/'selection_lock.json'),'scientific_outcome':'See guarded_confirmation_results; generic latency/VRAM peer choice is not evidence of retained-state-cap consequence','failed_blocks':len(fail),'final_cap_failures_retained_no_reaveraging':True});dump(BASE/'run_status.json',{'stage':'complete','updated_at':time.time(),'complete':True,'quality_complete':True,'deployment_measured':True});print(pd.DataFrame(final).to_string(index=False))
def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['quality','resources']);a.add_argument('--raw',type=Path,nargs='*');args=a.parse_args();quality() if args.action=='quality' else resources(args.raw or [])
if __name__=='__main__':main()
