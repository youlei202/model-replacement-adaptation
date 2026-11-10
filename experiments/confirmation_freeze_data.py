"""Choose registered follow-ups and lock fresh cohorts before new inference."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from statistics import NormalDist
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BUDGETS=(1024,2048,8192,16384)
COMPARATORS=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret')
SEEDS=(81724031,81724032,81724033)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expected_power(sd,n,alpha,comparisons,effect=.005):
 if sd==0:return None
 z=NormalDist().inv_cdf(1-alpha/(2*comparisons));nc=effect*math.sqrt(n)/sd
 return NormalDist().cdf(nc-z)+NormalDist().cdf(-nc-z)
def discovery_sd(phase,setting,modelindex):
 rows=[]
 for seed in (20261109,20261110,20261111,20261112,20261113):
  d=phase/'cells'/f'seed_{seed}'/setting.task_id/f'target_{modelindex}'/'q_2'
  f=json.loads((d/'selection_freeze.json').read_text());dec={(r['policy'],r['budget_bits']):r for r in f['selections'] if r['selection_kind']=='policy' and r['scope']=='strengthened' and r['budget_bits'] in BUDGETS and r['policy'] in COMPARATORS+('budget_aware',)}
  states={r['state_digest'] for r in dec.values()}
  loss=pd.read_parquet(d/'item_predictions_losses.parquet',columns=['item_id','item_cluster','state_digest','tv_loss'],filters=[('state_digest','in',list(states))]);wide=loss.pivot(index=['item_id','item_cluster'],columns='state_digest',values='tv_loss')
  for comp in COMPARATORS:
   delta=np.mean(np.stack([(wide[dec[comp,b]['state_digest']]-wide[dec['budget_aware',b]['state_digest']]).to_numpy() for b in BUDGETS],axis=1),axis=1)
   frame=pd.DataFrame({'cluster':wide.index.get_level_values('item_cluster'),'delta':delta}).groupby('cluster').delta.mean()
   rows.append({'discovery_target_id':setting.target_id,'discovery_task_id':setting.task_id,'seed':seed,'comparator':comp,'paired_cluster_sd':float(frame.std(ddof=1)),'clusters':len(frame),'endpoint':'same-query equal average over four primary caps; SD not mean used for prospective planning'})
 return rows

def main():
 a=argparse.ArgumentParser();a.add_argument('--phase-a',type=Path,default=ROOT/'experiments/runs/v2/phase_a');a.add_argument('--output',type=Path,default=ROOT/'experiments/runs/v2/confirmation');args=a.parse_args();phase=args.phase_a;out=args.output
 if (out/'data_lock.json').exists():raise RuntimeError('Data already frozen; no overwrite')
 if not (phase/'complete.json').exists():raise RuntimeError('All Phase A cells must complete before follow-up ranking')
 ranking=pd.read_csv(phase/'followup_validation_ranking.csv');chosen=ranking[ranking.selected_for_confirmation].copy()
 assert 1<=len(chosen)<=2 and chosen.target_id.nunique()==len(chosen)
 # Independently reproduce selection rule using validation fields only.
 allg=pd.read_csv(phase/'policy_gains.csv',usecols=['q','scope','comparator','budget_bits','target_id','task_id','validation_gain'])
 chk=allg[(allg.q==2)&(allg.scope=='strengthened')&(allg.comparator=='fixed_mean_budget')&allg.budget_bits.isin(BUDGETS)].groupby(['target_id','task_id'],as_index=False).validation_gain.mean().sort_values(['validation_gain','target_id','task_id'],ascending=[False,True,True]).drop_duplicates('target_id').head(2)
 assert list(zip(chk.target_id,chk.task_id))==list(zip(chosen.target_id,chosen.task_id))
 config=json.loads((ROOT/'experiments/experiment_config/configs/campaign.json').read_text());ids={r['model_id']:r['index'] for r in config['model_registry']};poplock=json.loads((out/'population_acquisition_preregistration.json').read_text());mapping=poplock['task_mapping'];comparisons=len(chosen)*4;alpha=.05;effect=.005;power=.9
 sdrows=[];settings=[]
 for row in chosen.itertuples(index=False):
  rr=discovery_sd(phase,row,ids[row.target_id]);sdrows.extend(rr);sd=max(r['paired_cluster_sd'] for r in rr);planning_sd=max(.01,sd)
  n=math.ceil(((NormalDist().inv_cdf(1-alpha/(2*comparisons))+NormalDist().inv_cdf(power))*planning_sd/effect)**2)
  settings.append({'original_task_id':row.task_id,'task_id':mapping[row.task_id]['external_task_id'],'target_id':row.target_id,'target_global_id':ids[row.target_id],'discovery_validation_gain':row.validation_gain,'maximum_discovery_sd_across_seeds_and_fixed_comparators':sd,'planning_sd':planning_sd,'planned_test_clusters':max(1024,n),'calculated_n':n})
 globalreg=pd.read_parquet(out/'global_duplicate_registry.parquet');selectedtasks={s['task_id'] for s in settings};newreg=globalreg[(~globalreg.historical)&globalreg.task_id.isin(selectedtasks)]
 cross=set(newreg.groupby('item_cluster').task_id.nunique().loc[lambda s:s>1].index)
 manifests={seed:[] for seed in SEEDS};fitval=[];test=[];counts=[]
 for task in sorted(selectedtasks):
  pool=pd.read_parquet(out/task/'eligible_population.parquet').drop(columns=['item_cluster','historical_cluster_overlap','union_cluster_index'],errors='ignore')
  pool=pool.merge(globalreg[globalreg.task_id==task][['item_id','item_cluster','historical_cluster_overlap']],on='item_id',validate='one_to_one')
  pool=pool[~pool.historical_cluster_overlap & ~pool.item_cluster.isin(cross)].sort_values('sample_order').drop_duplicates('item_cluster').copy()
  pool['partition_order']=[hashlib.sha256(('fresh_partition81724031|'+x).encode()).hexdigest() for x in pool.item_cluster];pool=pool.sort_values(['partition_order','item_id']).reset_index(drop=True)
  nfit,nval=(1024,256) if task=='preference_external' else (2048,512);nrequested=max(s['planned_test_clusters'] for s in settings if s['task_id']==task);navailable=len(pool)-nfit-nval;ntest=min(navailable,nrequested)
  if ntest<2:raise RuntimeError(f'Insufficient fresh clusters {task}')
  dev=pool.head(nfit+nval).copy();sealed=pool.iloc[nfit+nval:nfit+nval+ntest].copy();dev['data_role']='external_confirmation_development';sealed['data_role']='external_confirmation_sealed_test';dev['split']='fitval';sealed['split']='iid_test';fitval.append(dev);test.append(sealed)
  for seed in SEEDS:
   d=dev.copy();order=np.random.Generator(np.random.PCG64(seed)).permutation(len(d));d['split']='validation';d.iloc[order[:nfit],d.columns.get_loc('split')]='fit';d['seed']=seed;t=sealed.copy();t['seed']=seed;manifests[seed].append(pd.concat([d,t],ignore_index=True))
  counts.append({'task_id':task,'fresh_eligible_clusters':len(pool),'fit':nfit,'validation':nval,'test':ntest,'requested_test':nrequested,'underpowered_by_planned_sample_size':ntest<nrequested,'new_population_not_same_query_law':True,'cross_selected_population_clusters_excluded':len(cross)})
  for s in settings:
   if s['task_id']==task:s.update(actual_test_clusters=ntest,expected_normal_approx_power=expected_power(s['planning_sd'],ntest,alpha,comparisons,effect),sample_size_satisfied=ntest>=s['planned_test_clusters'])
 mfdir=out/'manifests';mfdir.mkdir(exist_ok=True);input_hashes={}
 for seed,parts in manifests.items():
  mf=pd.concat(parts,ignore_index=True);assert not mf.duplicated(['task_id','item_id']).any();assert mf.groupby('item_cluster').split.nunique().max()==1
  path=mfdir/f'split_seed_{seed}.parquet';mf.to_parquet(path,index=False);input_hashes[str(path.relative_to(out))]=sha(path)
 for role,parts in [('fitval',fitval),('test',test)]:
  path=out/f'required_items_{role}.parquet';df=pd.concat(parts,ignore_index=True);df.to_parquet(path,index=False);input_hashes[path.name]=sha(path)
 assert not(set(pd.concat(fitval).item_cluster)&set(pd.concat(test).item_cluster))
 (out/'dimensions.json').write_text(json.dumps({s['task_id']:(1 if s['task_id']=='preference_external' else 3) for s in settings},indent=2)+'\n');input_hashes['dimensions.json']=sha(out/'dimensions.json')
 pd.DataFrame(sdrows).to_csv(out/'power_discovery_sd.csv',index=False);pd.DataFrame(counts).to_csv(out/'fresh_cohort_counts.csv',index=False)
 lock={'protocol_id':'replacement_v2_external_confirmation_v1','created_unix':time.time(),'selection_rule':'top two distinct targets by mean Phase A validation gain vs fixed_mean_budget, primary q2 strengthened menu, equal weights over 5 seeds and4caps','settings':settings,'comparators':COMPARATORS,'primary_budget_bits':BUDGETS,'primary_endpoint':'Within each selected setting, mean per-cluster fixed-minus-budget-aware TV difference averaged equally across four primary budgets. First training seed primary.','training_seeds':SEEDS,'primary_training_seed':SEEDS[0],'familywise_alpha':alpha,'familywise_comparisons':comparisons,'bootstrap_draws':10000,'bootstrap_seed':81724032,'design_gain_tv':effect,'design_power':power,'minimum_test_clusters_design_choice':1024,'sd_floor_design_choice':.01,'data_role':'genuinely_new_external_population; no old response or item reuse; public-model-pretraining overlap not known','input_hashes':input_hashes,'historical_registry_sha256':sha(ROOT/'experiments/runs/v2/audit/exposure_registry.parquet'),'historical_all_candidate_registry_sha256':sha(ROOT/'experiments/runs/v2/audit/historical_text_registry.parquet'),'global_new_old_duplicate_registry_sha256':sha(out/'global_duplicate_registry.parquet'),'followup_ranking_sha256':sha(phase/'followup_validation_ranking.csv'),'discovery_complete_sha256':sha(phase/'complete.json'),'data_preparation_code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/confirmation_prepare.py',ROOT/'experiments/confirmation_global_clusters.py',ROOT/'experiments/confirmation_tokenizers.py']},'all_new_test_responses_uncollected':True,'test_optional_stopping':False,'population_counts':counts}
 (out/'data_lock.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(lock,indent=2))
if __name__=='__main__':main()
