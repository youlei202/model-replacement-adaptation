"""Preregistered external confirmation inference from one-shot frozen evaluation.

Primary: first training seed, equal average of the four primary budget differences
within each query cluster; eight simultaneous Bonferroni marginal bootstrap CIs.
Resampling is shared across all target/comparator vectors on the same task.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BUDGETS=(1024,2048,8192,16384)
COMPARATORS=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap_joint(matrix,draws,seed,alpha,comparisons):
 if matrix.ndim!=2 or not np.isfinite(matrix).all() or len(matrix)<2:raise ValueError('Finite cluster x comparison matrix required')
 rng=np.random.default_rng(seed);out=np.empty((draws,matrix.shape[1]))
 for start in range(0,draws,64):
  indices=rng.integers(len(matrix),size=(min(64,draws-start),len(matrix)));out[start:start+len(indices)]=matrix[indices].mean(1)
 return np.quantile(out,[alpha/(2*comparisons),1-alpha/(2*comparisons),alpha/2,1-alpha/2],axis=0)
def cell_location(base,setting,seed):return base/'fits'/'campaign'/'cells'/f'seed_{seed}'/setting['task_id']/f'target_{setting["target_global_id"]}'/'q_2'
def cell_differences(base,setting,seed):
 d=cell_location(base,setting,seed);f=json.loads((d/'selection_freeze.json').read_text());done=json.loads((d/'evaluation_complete.json').read_text())
 if done['freeze_sha256']!=sha(d/'selection_freeze.json'):raise ValueError('Freeze mismatch')
 for name,h in done['outputs'].items():
  if sha(d/name)!=h:raise ValueError('Changed evaluated artifact')
 dec={(r['policy'],r['budget_bits']):r for r in f['selections'] if r['selection_kind']=='policy' and r['scope']=='strengthened' and r['budget_bits'] in BUDGETS and r['policy'] in COMPARATORS+('budget_aware',)}
 states={r['state_digest'] for r in dec.values()};loss=pd.read_parquet(d/'item_predictions_losses.parquet',columns=['item_id','item_cluster','state_digest','tv_loss'],filters=[('state_digest','in',list(states))]);wide=loss.pivot(index=['item_id','item_cluster'],columns='state_digest',values='tv_loss');rows=[]
 for comp in COMPARATORS:
  for b in BUDGETS:
   fixed,joint=dec[comp,b],dec['budget_aware',b]
   part=wide.index.to_frame(index=False);part['difference']=wide[fixed['state_digest']].to_numpy()-wide[joint['state_digest']].to_numpy();part['fixed_loss']=wide[fixed['state_digest']].to_numpy();part['joint_loss']=wide[joint['state_digest']].to_numpy();part['comparator']=comp;part['budget_bits']=b;part['target_id']=setting['target_id'];part['task_id']=setting['task_id'];part['seed']=seed;part['fixed_state_digest']=fixed['state_digest'];part['joint_state_digest']=joint['state_digest'];rows.append(part)
 return pd.concat(rows,ignore_index=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT/'experiments/runs/v2/confirmation');args=a.parse_args();base=args.root;lock=json.loads((base/'data_lock.json').read_text());method=json.loads((base/'confirmation_lock.json').read_text())
 if method['analysis_code_sha256']!=sha(__file__):raise ValueError('Analysis script not same frozen version')
 frames=[cell_differences(base,s,seed) for s in lock['settings'] for seed in lock['training_seeds']];differences=pd.concat(frames,ignore_index=True);differences.to_parquet(base/'paired_query_differences.parquet',index=False)
 primary=differences[differences.seed==lock['primary_training_seed']];averages=primary.groupby(['task_id','target_id','comparator','item_id','item_cluster'],as_index=False)[['difference','fixed_loss','joint_loss']].mean();primaryrows=[]
 for task,sub in averages.groupby('task_id'):
  matrix=sub.pivot(index='item_cluster',columns=['target_id','comparator'],values='difference').sort_index();assert not matrix.isna().any().any();seed=lock['bootstrap_seed']+int(hashlib.sha256(task.encode()).hexdigest()[:8],16)
  ci=bootstrap_joint(matrix.to_numpy(),lock['bootstrap_draws'],seed,lock['familywise_alpha'],lock['familywise_comparisons'])
  for i,(target,comp) in enumerate(matrix.columns):
   d=sub[(sub.target_id==target)&(sub.comparator==comp)];gain=float(d.difference.mean());fixed=float(d.fixed_loss.mean());joint=float(d.joint_loss.mean())
   primaryrows.append({'task_id':task,'target_id':target,'comparator':comp,'primary_training_seed':lock['primary_training_seed'],'budgets_averaged':json.dumps(BUDGETS),'independent_test_clusters':len(matrix),'fixed_tv':fixed,'joint_tv':joint,'gain_tv':gain,'relative_gain':gain/fixed if fixed else 0.,'simultaneous_ci_low':float(ci[0,i]),'simultaneous_ci_high':float(ci[1,i]),'unadjusted_ci_low':float(ci[2,i]),'unadjusted_ci_high':float(ci[3,i]),'familywise_comparisons':lock['familywise_comparisons'],'bootstrap_draws':lock['bootstrap_draws'],'passes_registered_effect_and_positive_lower_bound':gain>=lock['design_gain_tv'] and ci[0,i]>0,'data_role':'external_confirmation_new_population','conditional_uncertainty':'given frozen trained methods; public finite candidate population; foundation-model pretraining exposure unknown'})
 pd.DataFrame(primaryrows).to_csv(base/'primary_confirmation_results.csv',index=False)
 # Secondary per-budget and training sensitivity: same queries, never pooled as iid.
 summaries=differences.groupby(['task_id','target_id','seed','comparator','budget_bits'],as_index=False).agg(gain_tv=('difference','mean'),fixed_tv=('fixed_loss','mean'),joint_tv=('joint_loss','mean'),test_clusters=('item_cluster','nunique'));summaries['inference_role']='descriptive secondary; budgets and repeated training seeds are not independent test samples';summaries.to_csv(base/'budget_and_training_sensitivity.csv',index=False)
 sensitivity=differences.groupby(['task_id','target_id','seed','comparator','item_cluster'],as_index=False).difference.mean().groupby(['task_id','target_id','seed','comparator'],as_index=False).agg(gain_tv=('difference','mean'),clusters=('item_cluster','nunique'));sensitivity.to_csv(base/'average_budget_training_sensitivity.csv',index=False)
 gains=pd.read_csv(base/'fits/campaign/policy_gains.csv')
 cheap=gains[(gains['q']==2)&(gains['scope']=='strengthened')&(gains['comparator']=='budget_aware_nnls')&gains['budget_bits'].isin(BUDGETS)].copy()
 cheap['inference_role']='secondary cheap-calibration control; per-budget/seed intervals are descriptive and not additional primary confirmations'
 cheap.to_csv(base/'cheap_calibration_controls.csv',index=False)
 verdict=[]
 for setting in lock['settings']:
  rr=[r for r in primaryrows if r['target_id']==setting['target_id'] and r['task_id']==setting['task_id']]
  verdict.append({'target_id':setting['target_id'],'task_id':setting['task_id'],'minimum_mean_gain_across_fixed_comparators':min(r['gain_tv'] for r in rr),'all_four_mean_gains_at_least_design_effect':all(r['gain_tv']>=lock['design_gain_tv'] for r in rr),'all_four_simultaneous_lower_bounds_positive':all(r['simultaneous_ci_low']>0 for r in rr),'sample_size_satisfied':setting['sample_size_satisfied'],'independent_confirmation_complete':True,'level1_confirmed_decision_effect':all(r['passes_registered_effect_and_positive_lower_bound'] for r in rr) and setting['sample_size_satisfied']})
 lines=['# Independent external confirmation','',f"The registered two-target follow-up uses {lock['settings'][0]['actual_test_clusters']:,} genuinely new query clusters from a four-choice academic/reading population. It is an external confirmation population, not an IID resplit of the old ten-choice MMLU-Pro data. The six model revisions and collector runtime are fixed.",'',"Each target has all ten eligible two-peer sets and the same convex, NNLS, KRR, linear-ridge and NNLS-plus-low-rank-residual menu for every policy. Fixed peers are retuned at every budget. All candidate fits, policy decisions, operational operating points and analysis code were frozen before any final-test response was collected.",'',"Primary inference uses the first training seed and averages the four registered budget differences within each query cluster. The table uses 10,000 shared paired-cluster bootstrap draws with eight-comparison Bonferroni marginal intervals. Repeated budgets/targets/training seeds do not multiply the independent query count.",'', '| Target | Fixed comparator | Fixed mean TV | Joint mean TV | Gain (fixed minus joint) | Simultaneous interval |','|---|---|---:|---:|---:|---|']
 for r in primaryrows:lines.append(f"| {r['target_id']} | {r['comparator']} | {r['fixed_tv']:.8f} | {r['joint_tv']:.8f} | {r['gain_tv']:.8f} | [{r['simultaneous_ci_low']:.8f}, {r['simultaneous_ci_high']:.8f}] |")
 lines += ['', 'The fixed comparator with the smallest observed gain is a retrospective evidence summary, not a test-selected deployable method. The predeclared large-decision-effect rule requires mean gain at least 0.005 and positive simultaneous lower bounds against all claimed fixed comparators. The dataset is public; foundation-model pretraining exposure and undisclosed author experiments outside the available project cannot be certified absent.', '', 'All individual budgets and all three training seeds are retained in the sensitivity CSVs. Per-query paired differences, predictions, state binaries, candidate ledgers, fit/public contexts and freeze hashes are preserved. The final resource and rich-behavior verdicts are reported separately; this table alone does not establish practical resource savings.', '']
 for r in verdict:lines.append(f"- {r['target_id']}: minimum registered-comparator mean gain {r['minimum_mean_gain_across_fixed_comparators']:.8f}; confirmed large decision effect under the registered rule: {r['level1_confirmed_decision_effect']}.")
 (base/'CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
 files=['primary_confirmation_results.csv','budget_and_training_sensitivity.csv','average_budget_training_sensitivity.csv','paired_query_differences.parquet','cheap_calibration_controls.csv','CONFIRMATION_REPORT.md'];result={'completed':True,'primary_rows':len(primaryrows),'data_lock_sha256':sha(base/'data_lock.json'),'method_lock_sha256':sha(base/'confirmation_lock.json'),'primary_endpoint':'equal-budget average within query cluster, first training seed','settings':verdict,'outputs':{p:sha(base/p) for p in files}}
 (base/'confirmation_complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
