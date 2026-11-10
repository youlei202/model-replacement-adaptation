#!/usr/bin/env python3
"""Posthoc presentation of PRETEST development measurements; no test selection.

Measured resource thresholds below are descriptive breakpoints, not retroactive
prospective latency/VRAM-cap experiments. Only validation quality is read.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2'
FIXED=['fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=RUN/'reports';inputs={};rows=[];pairs={};mcq=RUN/'resource_benchmark/confirmation_catalog';rich=RUN/'rich_behavior'
 def read(path):inputs[str(path.relative_to(RUN))]=sha(path);return json.loads(path.read_text())
 ml=read(mcq/'operating_points_lock.json');raw=mcq/'development_profiles_complete.jsonl';inputs[str(raw.relative_to(RUN))]=sha(raw)
 blocks=[json.loads(line) for line in raw.read_text().splitlines() if line]
 for candidate in ml['eligibility']:
  bs=[b for b in blocks if b['system_id']==candidate['system_id'] and not b.get('failure_status')];traces=[t for b in bs for t in b['traces']]
  assert len(bs)==2 and len(traces)==40
  rows.append({'route':'MCQ','target':candidate['target_index'],'system_id':candidate['system_id'],'pair':','.join(map(str,candidate['pair'])),'family':candidate['family'],'validation_tv':candidate['validation_tv'],'development_ms_per_context':candidate['development_end_to_end_ms_per1000']/1000,'development_peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in traces),'shared_deployment_resident_bytes':candidate['shared_deployment_resident_bytes'],'resident_target_numeric_bytes':candidate['resident_target_coefficients_bytes'],'serialized_state_bytes':candidate['state_binary_bytes'],'timed_development_requests':len(traces),'development_blocks':len(bs),'actual_call_q':bs[0]['actual_call_q'],'quality_role':'validation_only','cost_role':'pretest_development_profiling','resource_metric':'synchronized_wall_end_to_end_ms','is_final_test_inference':False})
 for target in [1,3]:
  p=RUN/f'confirmation/fits/campaign/cells/seed_81724031/academic_mcq4/target_{target}/q_2/selection_freeze.json';f=read(p)
  for policy in FIXED:
   pairset={tuple(r['pair']) for r in f['selections'] if r.get('selection_kind')=='policy' and r.get('scope')=='strengthened' and r.get('policy')==policy};assert len(pairset)==1
   pairs['MCQ',target,policy]=','.join(map(str,next(iter(pairset))))
  pairs['MCQ',target,'fixed_cost_aware_mean_cap']=','.join(map(str,ml['cost_aware_fixed_diagnostic'][str(target)]['chosen_pair']))
 profiles=read(rich/'resources/development_profiles.json');profile={r['candidate_id']:r for r in profiles};rlock=read(rich/'selection_lock.json');rop=read(rich/'resources/operating_points_lock.json')
 candidates={}
 for path in sorted((rich/'fits').glob('target_*/pair_*/validation_candidates.parquet')):
  inputs[str(path.relative_to(RUN))]=sha(path)
  for r in pd.read_parquet(path).to_dict('records'):candidates[r['target_index'],r['candidate_id']]=r
 for (target,digest),c in candidates.items():
  if digest not in profile:continue
  p=profile[digest];traces=p['raw_timings']
  rows.append({'route':'rich','target':target,'system_id':digest,'pair':c['peers'],'family':c['family'],'validation_tv':c['validation_tv'],'development_ms_per_context':p['mean_GPU_ms'],'development_peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in traces),'shared_deployment_resident_bytes':p['shared_resident_basis_bytes'],'resident_target_numeric_bytes':p['resident_state_bytes'],'serialized_state_bytes':p['serialized_state_bytes'],'timed_development_requests':len(traces),'development_blocks':1,'actual_call_q':p['actual_call_q'],'quality_role':'validation_only','cost_role':'pretest_development_profiling','resource_metric':'synchronized_cuda_complete_pipeline_interval_ms','is_final_test_inference':False})
 for target in [3,4]:
  for policy in FIXED:
   pairset={r['peers'] for r in rlock['choices'] if r['target_index']==target and r['policy']==policy};assert len(pairset)==1;pairs['rich',target,policy]=next(iter(pairset))
  diagnostic=next(r for r in rop['cost_aware_fixed_pair_diagnostic'] if r['target_index']==target);pairs['rich',target,'fixed_cost_aware_mean_cap']=','.join(map(str,diagnostic['selected']['pair']))
 table=pd.DataFrame(rows);envelopes=[]
 for (route,target),g in table.groupby(['route','target']):
  for axis in ['development_ms_per_context','development_peak_allocated_bytes']:
   for ix,row in g.iterrows():
    dominated=((g[axis]<=row[axis])&(g.validation_tv<=row.validation_tv)&((g[axis]<row[axis])|(g.validation_tv<row.validation_tv))).any()
    table.loc[ix,'validation_pareto_'+axis]=not dominated
   for cap in sorted(g[axis].unique()):
    eligible=g[g[axis]<=cap]
    for policy in ['budget_aware',*FIXED,'fixed_cost_aware_mean_cap']:
     pool=eligible if policy=='budget_aware' else eligible[eligible.pair==pairs[route,target,policy]]
     rec={'route':route,'target':target,'measured_resource_axis':axis,'descriptive_cap':cap,'policy':policy,'feasible_candidates':len(pool),'inferential_role':'posthoc_visualization_of_pretest_development_data_only','selected_before_test_as_resource_cap_policy':False}
     if len(pool):
      selected=pool.sort_values(['validation_tv',axis,'serialized_state_bytes','system_id']).iloc[0]
      rec.update(best_validation_tv=selected.validation_tv,descriptive_system_id=selected.system_id,descriptive_pair=selected.pair)
     envelopes.append(rec)
 table.to_csv(out/'T6_DEVELOPMENT_RESOURCE_FRONTIER.csv',index=False);pd.DataFrame(envelopes).to_csv(out/'T6_DESCRIPTIVE_MATCHED_RESOURCE_CAPS.csv',index=False)
 note='''# Resource-cap versus fidelity-requirement audit

The prospective resource experiment freezes a high-fidelity replacement reference, the TV-margin grid (0, .0025, .005, .01), candidate eligibility and the cheapest measured eligible system for joint and each fixed policy before new test outputs. The primary .005 results use natural batch-one inference; final complete-predictor timing and fresh TV/argmax checks assess those frozen systems. This is the executed **same-fidelity, lower-resource** experiment.

`T6_DEVELOPMENT_RESOURCE_FRONTIER.csv` exposes every measured development catalog entry with validation TV, synchronized complete-predictor timing, observed peak allocated GPU memory, target numerical state, shared context, serialized state and actual called peers. Cost axes differ explicitly between routes: MCQ uses synchronized wall latency; rich selection uses synchronized complete-pipeline CUDA intervals. They cannot be directly pooled as one metric. MCQ development uses two 20-request blocks per state; rich uses 10 development requests. These select operating points and are not the final 200-request/10-block measurement replications.

`T6_DESCRIPTIVE_MATCHED_RESOURCE_CAPS.csv` expresses those same development measurements as best validation behavior at each observed latency or allocated-VRAM threshold, within the measured catalog and each frozen fixed-pair restriction. Every feasible and infeasible row is kept. This is a posthoc descriptive view of already-frozen development data, with **no final-test values read and no new deployed selection**. Its measured thresholds were not prospectively registered as a distinct latency/VRAM-cap experiment, so it must not be described as an independently confirmed same-resource-cap quality gain. A serialized adapter bit cap is also not a measured total-RAM or latency cap.

The full final measured points and matched-fidelity comparisons are reported separately. At the primary fidelity margin, identical final joint/fixed/cost-aware systems must be identified as identical predictors; any independent timing variation cannot create an algorithmic improvement. Conditional existing-peer accounting still lists its shared infrastructure; standalone accounting includes peer weights and actual deployment contexts. The small state changes do not alone demonstrate a binding capacity limit. No external deployment compatibility or production threshold was supplied.

The remaining output-contract limitation is explicit: this campaign does not provide a separately pretest-frozen latency/VRAM-cap-selected fresh quality experiment, or an observed simultaneous multi-target capacity workload. These are absent evidence, not missing rows repaired by test-based reselection. All bounded executed routes and null/negative outcomes remain reportable.
'''
 (out/'T6_RESOURCE_CAP_AUDIT.md').write_text(note)
 files=['T6_DEVELOPMENT_RESOURCE_FRONTIER.csv','T6_DESCRIPTIVE_MATCHED_RESOURCE_CAPS.csv','T6_RESOURCE_CAP_AUDIT.md']
 (out/'T6_RESOURCE_CAP_AUDIT_MANIFEST.json').write_text(json.dumps({'source_script_sha256':sha(__file__),'input_sha256':inputs,'outputs':{f:sha(out/f) for f in files},'development_candidate_rows':len(table),'cap_comparison_rows':len(envelopes),'reads_finaltest_quality':False,'changes_pretest_choices':False,'distinct_resource_cap_confirmatory_experiment_executed':False},indent=2)+'\n')
 print(json.dumps({'development_candidate_rows':len(table),'cap_comparison_rows':len(envelopes)},indent=2))
if __name__=='__main__':main()
