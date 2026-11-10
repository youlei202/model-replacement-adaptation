"""Freeze all rich-policy choices and prospective test size, then evaluate once."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'experiments/experiment_config'))
from rich_setup import OUT,MODELS,digest,dump
from rich_codec import decode
from rich_fit import PAIRS,TARGETS,CAPS,tensor,loadlog,score
from reference.statistics import paired_cluster_interval,paired_power_n
POLICIES=['fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret']

def envelopes(target):
 out={}
 for pair in PAIRS:
  f=OUT/'fits'/f'target_{target}'/f'pair_{pair[0]}_{pair[1]}'
  assert (f/'complete.json').exists()
  rows=json.load(open(f/'envelope.json'))
  for r in rows:
   r['validation_tv']=float(np.load(OUT/r['state_path'].replace('.bin','.validation.npz'))['tv'].mean())
  out[pair]=rows
 return out

def key(r):return (r['validation_tv'],r['actual_bytes'],r['peer_mask'],r['candidate_id'])
def choose(rows,scope,cap):return min([r for r in rows if r['scope']==scope and r['cap_bytes']==cap],key=key)
def val_losses(row):return np.load(OUT/row['state_path'].replace('.bin','.validation.npz'))['tv']

def freeze():
 from rich_deploy import profile,freeze_resource
 cross=json.load(open(OUT/'concurrent_mcq_cross_audit.json'));assert cross['passed']
 if not (OUT/'collection_deployment_bridge.json').exists():
  import subprocess
  subprocess.run([sys.executable,str(ROOT/'experiments/rich_runtime_bridge.py')],check=True)
 assert json.load(open(OUT/'collection_deployment_bridge.json'))['passed']
 profile()
 lockpath=OUT/'selection_lock.json'
 if lockpath.exists():
  freeze_resource();print('lock already exists; immutable',digest(lockpath));return
 choices=[];contrasts=[];frozenfiles={}
 for target in TARGETS:
  table=envelopes(target);curve={p:{cap:choose(rows,'all',cap) for cap in CAPS} for p,rows in table.items()};joint={cap:min([curve[p][cap] for p in PAIRS],key=key) for cap in CAPS}
  fixed={
   'fixed_native_full':min(PAIRS,key=lambda p:key(choose(table[p],'native_full',0))),
   'fixed_unbounded_best':min(PAIRS,key=lambda p:key(choose(table[p],'unbounded',0))),
   'fixed_mean_budget':min(PAIRS,key=lambda p:(np.mean([curve[p][cap]['validation_tv'] for cap in CAPS]),p)),
   'fixed_minimax_validation_regret':min(PAIRS,key=lambda p:(max(curve[p][cap]['validation_tv']-joint[cap]['validation_tv'] for cap in CAPS),p))}
  for cap in CAPS:
   selections={'budget_aware':joint[cap],**{name:curve[p][cap] for name,p in fixed.items()},'budget_aware_cheap':min([choose(rows,'cheap',cap) for rows in table.values()],key=key)}
   for name,row in selections.items():choices.append({'target_index':target,'policy':name,'cap_bytes':cap,**row})
  for name,pair in fixed.items():
   difference=np.mean([val_losses(curve[pair][cap])-val_losses(joint[cap]) for cap in CAPS],axis=0)
   contrasts.append({'target_index':target,'fixed_policy':name,'peer_pair':list(pair),'validation_mean_gain':float(difference.mean()),'validation_sd_paired_budget_averaged':float(difference.std(ddof=1))})
  for pair,rows in table.items():
   folder=OUT/'fits'/f'target_{target}'/f'pair_{pair[0]}_{pair[1]}'
   for p in [folder/'complete.json',folder/'envelope.json',folder/'validation_candidates.parquet']:frozenfiles[str(p.relative_to(OUT))]=digest(p)
   for row in rows:frozenfiles[row['state_path']]=digest(OUT/row['state_path'])
 maxsd=max(x['validation_sd_paired_budget_averaged'] for x in contrasts)
 powern=paired_power_n(max(maxsd,1e-12),.005,comparisons=8,power=.9)
 desired=max(2048,2*math.ceil(powern/2));docs=pd.read_parquet(OUT/'documents.parquet');pool=docs[docs.role.eq('test_pool')].sort_values('order_hash');per_source=min(pool.groupby('source').size());actual=min(desired,2*per_source)
 test=pool.groupby('source',group_keys=False).head(actual//2).sort_values('order_hash')
 assert len(test)==actual and test.cluster_id.nunique()==actual
 for role in ['fit','validation']:
  for i in range(5):
   bank=OUT/'logits'/role/f'model_{i}.npy';meta=bank.with_suffix('.json');record=json.load(open(meta))
   assert record['complete'] and record['inference_microbatch']==1 and record['role']==role and record['model_revision']==MODELS[i][1]
   assert record['cohort_sha256']==digest(OUT/'documents.parquet') and digest(bank)==record['sha256']
   assert np.load(bank,mmap_mode='r').shape==((2048 if role=='fit' else 1024),151936)
   frozenfiles[str(bank.relative_to(OUT))]=record['sha256'];frozenfiles[str(meta.relative_to(OUT))]=digest(meta)
 for p in ['protocol.json','data_lock.json','documents.parquet','model_registry.json','compatible_tokenizer_audit.json','concurrent_mcq_cross_audit.json','collection_deployment_bridge.json','feasibility_amendment_01.json','resource_design_freeze.json']:frozenfiles[p]=digest(OUT/p)
 for pair in PAIRS:
  p=OUT/'public_contexts'/f'pair_{pair[0]}_{pair[1]}.npy';frozenfiles[str(p.relative_to(OUT))]=digest(p)
 decisionfile=OUT/'validation_policy_decisions.json';dump(decisionfile,choices)
 frozenfiles['validation_policy_decisions.json']=digest(decisionfile)
 lock={'all_choices_frozen':True,'time_unix':time.time(),'choices':choices,'fixed_validation_contrasts':contrasts,'primary_metric':'equalweightedbudgetaveragepairedTVgainperdocument','familywise_comparisons':8,'perbudget_secondary_comparisons':64,'bootstrap_draws':10000,'bootstrap_seed':81724032,'design_gain_tv':.005,'power':.9,'paired_sd_from_validation_max':maxsd,'normal_approximation_power_n':powern,'desired_test_clusters':desired,'actual_test_clusters':actual,'available_balanced_test_clusters':2*per_source,'adequate_nominal_sample_size':actual>=powern,'minimum2048_rule':'extra precision for absolutequality/resourcefrontier whenpolicygainSDdegenerate','test_item_ids':test.item_id.tolist(),'frozen_artifacts_sha256':frozenfiles,'code_sha256':{p.name:digest(p) for p in (ROOT/'experiments').glob('rich_*.py')},'no_test_outputs_opened':True,'target_exclusion_verified':all(t not in p for t in TARGETS for p in PAIRS),'data_scope':'WikiText103wholearticles and arxivfirsttrainshard; fixedcanonicalcontext externalconfirmation; knownlocalhistoryexcluded'}
 dump(lockpath,lock);dump(OUT/'power_plan.json',{k:v for k,v in lock.items() if k in ['normal_approximation_power_n','desired_test_clusters','actual_test_clusters','available_balanced_test_clusters','adequate_nominal_sample_size','paired_sd_from_validation_max','design_gain_tv','power','familywise_comparisons']});freeze_resource();print('FROZEN',digest(lockpath),'test_n',actual,'desired',desired,flush=True)

def evaluate():
 lockpath=OUT/'selection_lock.json';lock=json.load(open(lockpath));assert lock['all_choices_frozen']
 for p,h in lock['frozen_artifacts_sha256'].items():assert digest(OUT/p)==h,(p,'modified afterfreeze')
 result=OUT/'results';result.mkdir(exist_ok=True)
 if (result/'evaluation_complete.json').exists():print('evaluation alreadycomplete');return
 docs=pd.read_parquet(OUT/'documents.parquet').set_index('item_id').loc[lock['test_item_ids']].reset_index();score_rows=[];all_losses=[];mapping={}
 for pair in PAIRS:
  peers=[tensor(loadlog('test',i)) for i in pair];basis=tensor(np.load(OUT/'public_contexts'/f'pair_{pair[0]}_{pair[1]}.npy'))
  for target in TARGETS:
   y=tensor(loadlog('test',target));unique={r['candidate_id']:r for r in envelopes(target)[pair]}
   for sid,row in unique.items():
    state=decode((OUT/row['state_path']).read_bytes());assert state['peers']==list(pair) and target not in pair
    loss=score(state,peers,y,basis);rec={**row,'test_tv':float(loss['tv'].mean()),'test_kl':float(loss['kl'].mean()),'test_argmax_disagreement':float(loss['argmax_disagreement'].mean()),'test_cluster_count':len(docs),'state_sha256':digest(OUT/row['state_path'])};score_rows.append(rec)
    df=docs[['item_id','cluster_id','source']].copy();df['target_index']=target;df['candidate_id']=sid
    for k,v in loss.items():df[k]=v
    all_losses.append(df);mapping[(target,sid)]=loss['tv']
   del y
  del peers,basis;torch.cuda.empty_cache()
 scores=pd.DataFrame(score_rows);lossframe=pd.concat(all_losses,ignore_index=True);assert not lossframe.duplicated(['target_index','candidate_id','item_id']).any()
 scores.to_csv(result/'all_frozen_states.csv',index=False);lossframe.to_parquet(result/'item_losses.parquet',index=False)
 choices=lock['choices'];policyrows=[];policy_losses=[]
 for choice in choices:
  s=scores[(scores.target_index==choice['target_index'])&(scores.candidate_id==choice['candidate_id'])].iloc[0].to_dict();policyrows.append({**choice,**{k:v for k,v in s.items() if k.startswith('test_')}})
  df=lossframe[(lossframe.target_index==choice['target_index'])&(lossframe.candidate_id==choice['candidate_id'])].copy();df['policy']=choice['policy'];df['cap_bytes']=choice['cap_bytes'];policy_losses.append(df)
 pd.DataFrame(policyrows).to_csv(result/'policy_scores.csv',index=False);pd.concat(policy_losses,ignore_index=True).to_parquet(result/'policy_item_losses.parquet',index=False)
 contrasts=[];primary=[]
 for target in TARGETS:
  lookup={(r['policy'],r['cap_bytes']):r for r in choices if r['target_index']==target}
  for pi,policy in enumerate(POLICIES+['budget_aware_cheap']):
   differences=[]
   for cap in CAPS:
    a,b=lookup[(policy,cap)],lookup[('budget_aware',cap)];d=mapping[(target,a['candidate_id'])]-mapping[(target,b['candidate_id'])];differences.append(d)
    ci=paired_cluster_interval(d,docs.cluster_id,draws=10000,comparisons=64 if policy in POLICIES else 16,seed=81724032+target*100+pi)
    contrasts.append({'target_index':target,'fixed_policy':policy,'cap_bytes':cap,'gain_tv':ci.estimate,'simultaneous_low':ci.low,'simultaneous_high':ci.high,'clusters':ci.independent_clusters,'comparison_count':ci.comparisons,'definition':'fixedTV-budgetawareTV','inferential_role':'secondary_per_budget'})
   d=np.mean(differences,axis=0);ci=paired_cluster_interval(d,docs.cluster_id,draws=10000,comparisons=8 if policy in POLICIES else 2,seed=81724032+target*100+pi)
   primary.append({'target_index':target,'fixed_policy':policy,'budget_averaged_gain_tv':ci.estimate,'simultaneous_low':ci.low,'simultaneous_high':ci.high,'clusters':ci.independent_clusters,'comparison_count':ci.comparisons,'predeclared_gain_threshold':.005,'supported_positive_gain':bool(ci.low>0),'design_effect_reached':bool(ci.estimate>=.005),'inferential_role':'primary' if policy in POLICIES else 'cheap_baseline_secondary'})
 pd.DataFrame(contrasts).to_csv(result/'policy_gains_by_budget.csv',index=False);pd.DataFrame(primary).to_csv(result/'primary_confirmation.csv',index=False)
 # Actual selected pair curves, never a test-optimal oracle masquerading as a deployable method.
 allpairs=[]
 for target in TARGETS:
  for pair,rows in envelopes(target).items():
   for r in rows:
    s=scores[(scores.target_index==target)&(scores.candidate_id==r['candidate_id'])].iloc[0]
    allpairs.append({**r,'test_tv':s.test_tv,'test_kl':s.test_kl,'test_argmax_disagreement':s.test_argmax_disagreement})
 pd.DataFrame(allpairs).to_csv(result/'all_pairs_by_budget.csv',index=False)
 dump(result/'evaluation_complete.json',{'complete':True,'selection_lock_sha256':digest(lockpath),'test_clusters':len(docs),'all_frozen_states':len(scores),'item_loss_rows':len(lossframe),'simultaneous_primary_comparisons':8,'prediction_reproduction':'Full probabilities reconstructed exactly by rich_fit.predict from savedstate/publicbasis andstoredfullpeerlogits; not all densepredictor matricespersisted','artifacts_sha256':{p.name:digest(p) for p in result.iterdir() if p.is_file()},'code_sha256':digest(Path(__file__))})
 print(pd.DataFrame(primary).to_string(index=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate']);a=p.parse_args();torch.set_num_threads(4)
 {'freeze':freeze,'evaluate':evaluate}[a.action]()
if __name__=='__main__':main()
