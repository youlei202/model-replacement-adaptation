"""Check prospective planning against fresh validation paired variance.

This never opens final-test responses and never changes sample size or methods.
The precommitted discovery-based n is retained; any variance discrepancy is
reported to the root coordinator before the final method/resource lock.
"""
from pathlib import Path
import json,sys,math
from statistics import NormalDist
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import phase_a_run as runner
from adapters_v2 import core,predict
base=ROOT/'experiments/runs/v2/confirmation';lock=json.loads((base/'data_lock.json').read_text());responses,_=runner.old.read_responses([base/'response_cache_fitval'],runner.MODELS,runner.REVISIONS);output=base/'fits/campaign';rows=[]
for seed in lock['training_seeds']:
 mf=pd.read_parquet(base/'manifests'/f'split_seed_{seed}.parquet')
 for setting in lock['settings']:
  task,target=setting['task_id'],setting['target_global_id'];vr=mf[(mf.task_id==task)&(mf.split=='validation')].sort_values('item_id').reset_index(drop=True);peers=runner.join_models(vr,responses,3);y=peers[:,target,:];folder=runner.cell_dir(output,seed,task,target,2);freeze=json.loads((folder/'selection_freeze.json').read_text());dec={(r['policy'],r['budget_bits']):r for r in freeze['selections'] if r['selection_kind']=='policy' and r['scope']=='strengthened' and r['budget_bits'] in runner.PRIMARY};loss={};contexts={};projections={}
  for rec in dec.values():
   digest=rec['state_digest'];pair=tuple(rec['pair'])
   if digest in loss:continue
   if pair not in contexts:
    path=(folder/freeze['contexts'][runner.pname(pair)]['path']).resolve()
    with np.load(path) as c:contexts[pair]=core.Context(pair,c['mean'],c['scale'],float(c['sigma']),c['features'],c['eigenvalues'],c['eigenvectors'],c['validation_projection'])
   encoded,matrix=core.decode((folder/'pairs'/runner.pname(pair)/'states'/f'{digest}.bin').read_bytes())
   if encoded.family=='krr' and pair not in projections:projections[pair]=contexts[pair].project(peers)
   pred=predict(encoded,matrix,peers,contexts[pair],projections.get(pair));loss[digest]=core.tv(y,pred)
   if abs(loss[digest].mean()-rec['validation_tv'])>1e-10:raise ValueError('Validation replay disagrees with frozen score')
  for comp in lock['comparators']:
   diff=np.mean(np.stack([loss[dec[comp,b]['state_digest']]-loss[dec['budget_aware',b]['state_digest']] for b in runner.PRIMARY],axis=1),axis=1);sd=float(np.std(diff,ddof=1));z=NormalDist().inv_cdf(1-lock['familywise_alpha']/(2*lock['familywise_comparisons']));zp=NormalDist().inv_cdf(lock['design_power']);required=math.ceil(((z+zp)*max(.01,sd)/lock['design_gain_tv'])**2)
   rows.append({'task_id':task,'target_id':setting['target_id'],'training_seed':seed,'comparator':comp,'validation_clusters':len(vr),'paired_cluster_sd':sd,'validation_mean_gain':float(diff.mean()),'normal_approx_required_n_at_validation_sd':required,'locked_actual_test_n':setting['actual_test_clusters'],'locked_n_satisfies_validation_sd_plan':setting['actual_test_clusters']>=required,'scope':'development variance check; same validation data used for choices; no final-test outcomes'})
pd.DataFrame(rows).to_csv(base/'fresh_validation_power_check.csv',index=False);report={'used_test_responses':False,'changed_test_n':False,'all_locked_n_satisfy_fresh_validation_variance_plan':all(r['locked_n_satisfies_validation_sd_plan'] for r in rows),'max_fresh_validation_sd':max(r['paired_cluster_sd'] for r in rows),'max_required_n':max(r['normal_approx_required_n_at_validation_sd'] for r in rows),'locked_test_n':lock['settings'][0]['actual_test_clusters'],'checks':len(rows)};(base/'fresh_validation_power_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
