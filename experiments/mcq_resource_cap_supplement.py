#!/usr/bin/env python3
"""Bounded prospective measured-resource-cap supplement; existing methods fixed.

Plans use original frozen validation predictions and development costs only.
A second unused academic population is locked before any additional outcomes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time,os,shutil
from pathlib import Path
from statistics import NormalDist
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2';BASE=RUN/'resource_cap_confirmation';PREV=RUN/'confirmation';CAT=RUN/'resource_benchmark/confirmation_catalog'
FIXED=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret');Q=(.25,.5,.75,1.);SEED=81724051

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def plan():
 BASE.mkdir(exist_ok=True)
 if (BASE/'selection_lock.json').exists():raise RuntimeError('Already frozen; no planning overwrite')
 inputs={}
 def read(p):inputs[str(p.relative_to(RUN))]=sha(p);return json.loads(p.read_text())
 amendment=read(RUN/'protocol/matched_resource_completion_amendment.json');op=read(CAT/'operating_points_lock.json');catalog=read(CAT/'candidate_catalog.json');bfile=CAT/'development_profiles_complete.jsonl';inputs[str(bfile.relative_to(RUN))]=sha(bfile);blocks=[json.loads(s) for s in bfile.read_text().splitlines() if s]
 valfile=CAT/'validation_losses.npz';inputs[str(valfile.relative_to(RUN))]=sha(valfile)
 with np.load(valfile,allow_pickle=False) as a:loss=a['losses'];systems=a['systems'].tolist();cluster=a['cluster_id'];items=a['item_id']
 assert len(cluster)==len(set(cluster))==512 and loss.shape==(512,139)
 ix={s:i for i,s in enumerate(systems)};records=[]
 for c in op['eligibility']:
  b=[b for b in blocks if b['system_id']==c['system_id'] and not b.get('failure_status')];tr=[t for b in b for t in b['traces']];assert len(b)==2 and len(tr)==40
  assert abs(loss[:,ix[c['system_id']]].mean()-c['validation_tv'])<1e-12
  records.append(c|{'dev_ms_per_context':c['development_end_to_end_ms_per1000']/1000,'dev_peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in tr),'dev_peak_reserved_bytes':max(t['peak_reserved_bytes'] for t in tr),'dev_peak_host_RSS':max(t['host_RSS'] for t in tr),'actual_call_q':b[0]['actual_call_q']})
 choices=[];caps=[];power=[];fixedmeta=[]
 def best(pool):return min(pool,key=lambda x:(x['validation_tv'],x['actual_bits'],x['dev_ms_per_context'],x['system_id'])) if pool else None
 for target in [1,3]:
  pool=[r for r in records if r['target_index']==target]
  lat=[math.ceil(float(np.quantile([r['dev_ms_per_context'] for r in pool],q))/5)*5 for q in Q]
  mem=[math.ceil(float(np.quantile([r['dev_peak_allocated_bytes'] for r in pool],q))/(2**30))*2**30 for q in Q]
  assert len(set(lat))==4 and len(set(mem))==4,'Quartile grid degeneracy requires documented design amendment before outcomes'
  f=read(PREV/f'fits/campaign/cells/seed_81724031/academic_mcq4/target_{target}/q_2/selection_freeze.json');pairs={}
  for policy in FIXED:
   pset={tuple(s['pair']) for s in f['selections'] if s.get('selection_kind')=='policy' and s.get('scope')=='strengthened' and s.get('policy')==policy};assert len(pset)==1;pairs[policy]=next(iter(pset))
  grids=[]
  for i,l in enumerate(lat):
   for j,m in enumerate(mem):
    cap={'target_index':target,'cap_id':f'lat_q{int(Q[i]*100):03d}_vram_q{int(Q[j]*100):03d}','latency_cap_ms_per_context':l,'allocated_VRAM_cap_bytes':m,'latency_development_quantile':Q[i],'VRAM_development_quantile':Q[j]};caps.append(cap)
    eligible=[r for r in pool if r['dev_ms_per_context']<=l and r['dev_peak_allocated_bytes']<=m];grids.append((cap,eligible,best(eligible)))
  diag=[]
  for pair in sorted({tuple(r['pair']) for r in pool}):
   selections=[(joint,best([r for r in eligible if tuple(r['pair'])==pair])) for cap,eligible,joint in grids]
   covered=[(j,f) for j,f in selections if j is not None and f is not None]
   diag.append({'pair':list(pair),'feasible_cap_count':len(covered),'mean_validation_regret_over_covered_caps':float(np.mean([f['validation_tv']-j['validation_tv'] for j,f in covered])) if covered else None})
  ordered=sorted(diag,key=lambda x:(-x['feasible_cap_count'],x['mean_validation_regret_over_covered_caps'] if x['mean_validation_regret_over_covered_caps'] is not None else float('inf'),x['pair']));pairs['fixed_cost_aware_mean_cap']=tuple(ordered[0]['pair']);fixedmeta.append({'target_index':target,'fixed_pairs':{p:list(v) for p,v in pairs.items()},'cost_aware_pair_diagnostics':ordered})
  local=[]
  for cap,eligible,joint in grids:
   for policy in ('budget_aware',*FIXED,'fixed_cost_aware_mean_cap','budget_aware_nnls'):
    selected=joint if policy=='budget_aware' else best([r for r in eligible if r['family']=='nnls']) if policy=='budget_aware_nnls' else best([r for r in eligible if tuple(r['pair'])==pairs[policy]])
    entry=cap|{'policy':policy,'status':'eligible' if selected else 'infeasible','system_id':selected['system_id'] if selected else None,'candidate':selected,'selected_by':'minimum validation TV among same catalog within BOTH development-measured caps; then bytes/cost/systemID tie rule','selected_before_second_test':True};local.append(entry);choices.append(entry)
  lookup={(r['cap_id'],r['policy']):r for r in local}
  for policy in (*FIXED,'fixed_cost_aware_mean_cap','budget_aware_nnls'):
   common=[cap['cap_id'] for cap,eligible,joint in grids if joint is not None and lookup[cap['cap_id'],policy]['status']=='eligible']
   if common:
    delta=np.mean(np.stack([loss[:,ix[lookup[c,policy]['system_id']]]-loss[:,ix[lookup[c,'budget_aware']['system_id']]] for c in common],axis=1),axis=1);sd=float(np.std(delta,ddof=1));planning_sd=max(.01,sd);n=math.ceil(((NormalDist().inv_cdf(1-.05/(2*16))+NormalDist().inv_cdf(.9))*planning_sd/.005)**2)
   else:sd=None;planning_sd=None;n=None
   power.append({'target_index':target,'comparator':policy,'primary':policy in FIXED,'common_feasible_cap_ids':common,'common_feasible_cap_count':len(common),'total_cap_count':16,'validation_paired_cluster_sd':sd,'planning_sd':planning_sd,'calculated_n':n,'planned_test_clusters':max(1024,n) if n is not None else None,'endpoint':'equal average over pretest-common-feasible cap cells WITHIN each cluster, then cluster mean; no infeasible-loss imputation'})
 regfile=PREV/'global_duplicate_registry.parquet';inputs[str(regfile.relative_to(RUN))]=sha(regfile);reg=pd.read_parquet(regfile);eligiblefile=PREV/'academic_mcq4/eligible_population.parquet';inputs[str(eligiblefile.relative_to(RUN))]=sha(eligiblefile);pool=pd.read_parquet(eligiblefile).drop(columns=['item_cluster','historical_cluster_overlap','union_cluster_index'],errors='ignore');pool=pool.merge(reg[reg.task_id=='academic_mcq4'][['item_id','item_cluster','historical_cluster_overlap']],on='item_id',validate='one_to_one');pool=pool[~pool.historical_cluster_overlap].sort_values('sample_order').drop_duplicates('item_cluster')
 used=[]
 for role in ['fitval','test']:
  p=PREV/f'required_items_{role}.parquet';inputs[str(p.relative_to(RUN))]=sha(p);used.append(pd.read_parquet(p))
 used=pd.concat(used,ignore_index=True);pool=pool[~pool.item_cluster.isin(used.item_cluster)&~pool.item_id.isin(used.item_id)&~pool.prompt_hash.isin(used.prompt_hash)].copy()
 pool['partition_order']=[hashlib.sha256(('resource_cap_confirmation81724051|'+c).encode()).hexdigest() for c in pool.item_cluster];pool=pool.sort_values(['partition_order','item_id']).reset_index(drop=True)
 n=max(p['planned_test_clusters'] for p in power if p['primary'] and p['planned_test_clusters']);actual=min(n,len(pool));selected=pool.head(actual).copy();selected['data_role']='second_untouched_external_resource_cap_confirmation';selected['split']='iid_test';selected['seed']=SEED
 pool.to_parquet(BASE/'unused_candidate_reservoir.parquet',index=False);selected.to_parquet(BASE/'proposed_required_items_test.parquet',index=False)
 for p in power:
  p['actual_shared_second_test_clusters']=actual;p['adequate_planned_n']=p['planned_test_clusters'] is not None and actual>=p['planned_test_clusters'];sd=p['planning_sd'];z=NormalDist().inv_cdf(1-.05/(2*16));nc=.005*math.sqrt(actual)/sd if sd else None;p['normal_approx_expected_power']=NormalDist().cdf(nc-z)+NormalDist().cdf(-nc-z) if nc else None
  selected_choice={(c['target_index'],c['cap_id'],c['policy']):c for c in choices}
  p['deterministic_policy_identity_on_all_common_caps']=all(selected_choice[p['target_index'],c,p['comparator']]['system_id']==selected_choice[p['target_index'],c,'budget_aware']['system_id'] for c in p['common_feasible_cap_ids'])
  if p['deterministic_policy_identity_on_all_common_caps']:
   p['hypothetical_power_with_planning_sd_floor']=p['normal_approx_expected_power'];p['normal_approx_expected_power']=None;p['power_interpretation']='Identical frozen predictors imply no positive difference; planning floor/minimum retained, powerfornonzeroeffect not established by zeroSD.'
 selectedsystems=sorted({c['system_id'] for c in choices if c['system_id']});proposed={'status':'proposed_not_yet_test_authorized','created_unix':time.time(),'namespace':str(BASE),'previous_confirmation_unchanged':True,'supplement_reason':'Complete separately requested matched measured latency/VRAM-cap route; not a search for positive outcomes','targets':[1,3],'task_id':'academic_mcq4','seed':SEED,'source_candidate_count':139,'catalog_restriction':'Only frozen q2 family/pair/serialized-cap/unbounded validation champions already measured; not exhaustive timing optimization of the full encoded grid. Same restricted catalog for all policies; cheap raw/q1 context remains separate.','cap_rule':'Each target development quartiles .25,.50,.75,1 of synchronized natural batch1 latency and peak allocated VRAM; round up 5ms/1GiB respectively; full4x4cross product','caps':caps,'choices':choices,'fixed_pair_rules':fixedmeta,'power':power,'familywise_primary_comparisons':16,'supplement_family_alpha':.05,'effect_design_TV':.005,'power_design':.9,'minimum_sample_size':1024,'sd_floor':.01,'primary_common_feasible_sets_frozen_before_test':True,'final_cap_guard':'Final live mean latency and maximum measured allocated VRAM must both satisfy each cap for BOTH paired systems; additionally report bootstrap latency bounds, reservedVRAM and hostRSS. If failed, retain original primary loss endpoint and mark resource-cap claim not established; never drop/reselect based on final compliance.','full_grid_coverage_report_required':True,'unique_final_systems':selectedsystems,'unused_fresh_reservoir_clusters':len(pool),'requested_clusters':n,'proposed_test_clusters':actual,'underpowered':actual<n,'input_hashes':inputs,'new_response_banks_exist':False,'cohort_sha256':sha(BASE/'proposed_required_items_test.parquet'),'remaining_reservoir_sha256':sha(BASE/'unused_candidate_reservoir.parquet'),'new_rich_cross_audit_required_before_freeze':True,'analysis_code_sha256':sha(__file__)}
 dump(BASE/'proposal.json',proposed);pd.DataFrame(power).to_csv(BASE/'power_plan.csv',index=False);pd.DataFrame([{k:v for k,v in c.items() if k!='candidate'} for c in choices]).to_csv(BASE/'proposed_choices.csv',index=False)
 print(json.dumps({k:proposed[k] for k in ['namespace','unused_fresh_reservoir_clusters','requested_clusters','proposed_test_clusters','underpowered','unique_final_systems']},indent=2));print(pd.DataFrame(power)[['target_index','comparator','common_feasible_cap_count','validation_paired_cluster_sd','planned_test_clusters','normal_approx_expected_power']].to_string(index=False))

def freeze():
 if (BASE/'confirmation_lock.json').exists():raise RuntimeError('Already frozen')
 p=json.loads((BASE/'proposal.json').read_text())
 for rel,d in p['input_hashes'].items():
  if sha(RUN/rel)!=d:raise RuntimeError('Planning input drift '+rel)
 assert sha(BASE/'proposed_required_items_test.parquet')==p['cohort_sha256']
 cross=RUN/'rich_behavior/matched_resource_supplement/concurrent_new_mcq_cross_audit.json';audit=json.loads(cross.read_text())
 # Cross-audit schema remains supplied by the independent rich-data workflow.
 assert audit.get('passed') is True and audit.get('cross_collision_count')==0,audit
 reservoir=BASE/'unused_candidate_reservoir.parquet'
 assert sha(reservoir)==p['remaining_reservoir_sha256']
 assert sha(reservoir) in json.dumps(audit),'Cross-audit does not bind the full unused MCQ reservoir'
 final=BASE/'required_items_test.parquet';shutil.copyfile(BASE/'proposed_required_items_test.parquet',final)
 shutil.copyfile(PREV/'required_items_fitval.parquet',BASE/'required_items_fitval.parquet')
 p.update(status='frozen_before_second_test',created_unix=time.time(),analysis_code_sha256=sha(__file__),cross_audit_path=str(cross.relative_to(RUN)),cross_audit_sha256=sha(cross),fresh_test_cohort_sha256=sha(final),final_endpoint='Gain fixed-minus-joint TV equal-averaged WITHIN query over pre-frozen common-feasible16-grid subset. Primaryfamily16 across bothsupplemental routes. Secondary cost-awarefixed/NNLS explicitlydescriptive.',bootstrap_draws=10000,bootstrap_seed=81724052,argmax_guard={'margin':.01,'contrast':'joint minus fixed decision disagreement averaged over the SAME common-feasible caps within each query','rule':'upper simultaneous bound <=.01; intersection-union with positive TV gain and resourcecap guard','primary_family':16,'power_scope':'sample size designed for TV endpoint; no separate joint-power guarantee'},latency_definition='mean synchronized complete-predictor wall milliseconds per context; not individualrequest SLO orp95',memory_definition='maximum observed allocated VRAM; reserved VRAM and hostRSS separatelyreported',latency_uncertainty='10000 paired-block/bootstrap mean samples; one-sided Bonferroni upperbound across13 systems. Mean/peak compliance and conservative-bound compliance reported separately.',no_final_resource_cap_reselection=True)
 dump(BASE/'selection_lock.json',p)
 data={'created_unix':time.time(),'protocol_id':'replacement_v2_second_untouched_resource_cap_confirmation','task_id':'academic_mcq4','test_clusters':p['proposed_test_clusters'],'source_population':'Same response-blind-acquired external academic MCQ4 population; disjoint from all prior projectMCQ and concurrentrich exposures under recorded duplicate detection','available_before_selection':p['unused_fresh_reservoir_clusters'],'deterministic_partition_seed':SEED,'input_hashes':{'required_items_test.parquet':sha(final),'required_items_fitval.parquet':sha(BASE/'required_items_fitval.parquet')},'fitval_prompt_file_role':'Previously exposed development prompts ONLY for unchangedofficial-scorer numerical bridge; no new fitting and not counted as independentdata','cross_audit_sha256':sha(cross),'selection_lock_sha256':sha(BASE/'selection_lock.json'),'all_second_test_outputs_uncollected':not (BASE/'response_cache_test').exists()}
 assert data['all_second_test_outputs_uncollected'];dump(BASE/'data_lock.json',data)
 systems={r['system_id']:r['candidate'] for r in p['choices'] if r['system_id']}
 common={'blocks':10,'dimension':3,'measurement_role':'final_live_second_cohort_resource_cap_confirmation','peer_response_dir':str(BASE/'response_cache_test'),'prompts':str(final),'seed':81724053,'selected_on_validation_only':True,'systems':[{'bundle':systems[s]['bundle'],'system_id':s} for s in sorted(systems)],'timed_requests_per_block':20,'warmup_requests':20,'maximum_worker_processes':4,'scope':'One complete batch/context setting per GPU/controller; allsystems interleaved in same10pairedblocks, newcohort canonicalprompts. Fixedschema/randomseed beforecollection.'}
 specs=[]
 for batch in [1,8]:
  for context in [0,128,512]:
   spec=BASE/'benchmark_specs'/f'batch{batch}_context{context}.json';dump(spec,common|{'batch_sizes':[batch],'context_lengths':[context]});specs.append(spec)
 artifacts=[BASE/'selection_lock.json',BASE/'data_lock.json',final,BASE/'required_items_fitval.parquet',BASE/'power_plan.csv',BASE/'pretest_replay_and_freshness_verification.json',reservoir,cross,ROOT/'experiments/mcq_resource_cap_supplement.py',ROOT/'experiments/mcq_resource_cap_analyze.py',ROOT/'experiments/mcq_resource_cap_collect.py',ROOT/'experiments/confirmation_collect.py',ROOT/'experiments/mcq_resource_benchmark.py',ROOT/'experiments/mcq_resource_analyze.py',ROOT/'experiments/mcq_deployment.py',ROOT/'experiments/adapters_v2.py',ROOT/'benchmarks/selection_core.py',ROOT/'benchmarks/collect_responses.py',ROOT/'reference_implementation/kernel_isqed/llm.py',ROOT/'reference_implementation/kernel_isqed/response.py',*specs]
 artifacts += [RUN/rel for rel in p['input_hashes']]
 for state in systems.values():artifacts.extend(x for x in Path(state['bundle']).iterdir() if x.is_file())
 freeze={'created_unix':time.time(),'all_fits_and_policies_frozen':True,'resource_operating_points_frozen':True,'data_lock_sha256':sha(BASE/'data_lock.json'),'selection_lock_sha256':sha(BASE/'selection_lock.json'),'frozen_artifact_hashes':{os.path.relpath(x,BASE):sha(x) for x in sorted(set(artifacts))},'test_outputs_allowed_only_after_this_lock':True,'new_method_fit':False,'second_untouched_cohort':True,'primary_familywise_comparisons':16,'resource_claim_requires_new_live_cap_compliance':True}
 dump(BASE/'confirmation_lock.json',freeze)
 dump(BASE/'run_status.json',{'stage':'frozen_awaiting_collection','updated_at':time.time(),'implementation_complete':True,'selection_and_data_frozen':True,'new_test_collection_complete':False,'quality_complete':False,'deployment_measured':False,'complete':False,'scientific_outcome':'pending second untouched cohort and live resource-cap guard'})
 print(json.dumps({'frozen':True,'test_clusters':p['proposed_test_clusters'],'systems':len(systems),'method_lock_sha256':sha(BASE/'confirmation_lock.json'),'frozen_artifacts':len(freeze['frozen_artifact_hashes']),'benchmark_specs':[str(x) for x in specs]},indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','freeze']);a=ap.parse_args();plan() if a.action=='plan' else freeze()
if __name__=='__main__':main()
