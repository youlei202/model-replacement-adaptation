"""Pre-test cheap single-peer controls, symmetric and separate from primary q2.

Direct identity uses one literal byte: high nibble 1 is the public identity-codec
version/operator, low nibble is the global model ID (0..5). All public model
weights/revisions are separately charged by the live resource experiment.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import phase_a_run as runner
from confirmation_analyze import bootstrap_joint
from mcq_deployment import DeployedAdapter
core=runner.core

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonwrite(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def metrics(target,pred):
 tp,pp=core.probabilities(target),core.probabilities(pred)
 return np.column_stack([np.abs(tp-pp).sum(1)/2,(tp*(np.log(np.maximum(tp,1e-300))-np.log(np.maximum(pp,1e-300)))).sum(1),(tp.argmax(1)!=pp.argmax(1)).astype(float)])
def inputs(base,role):
 lock=json.loads((base/'data_lock.json').read_text());seed=lock['primary_training_seed'];mf=pd.read_parquet(base/'manifests'/f'split_seed_{seed}.parquet');mf=mf[mf.split==role].sort_values('item_id').reset_index(drop=True);responses,_=runner.old.read_responses([base/('response_cache_test' if role=='iid_test' else 'response_cache_fitval')],runner.MODELS,runner.REVISIONS);return lock,mf,runner.join_models(mf,responses,3)
def predict_state(folder,row,x):
 data=(folder/'states'/row['state_file']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['state_digest']
 if row['family']=='direct_identity':
  assert len(data)==1 and data[0]>>4==1 and (data[0]&15)==row['peer'];return x[:,data[0]&15,:]
 encoded,params=core.decode(data);assert encoded.family=='nnls' and encoded.pair==(row['peer'],);return core.predict(encoded,params,x)
def prepare(base,folder):
 if (folder/'candidate_freeze.json').exists():raise RuntimeError('Cheap-control fit freeze already exists')
 lock,fit,fx=inputs(base,'fit');_,val,vx=inputs(base,'validation');folder.mkdir(exist_ok=True);(folder/'states').mkdir(exist_ok=True);ledger=[];selected=[]
 for setting in lock['settings']:
  target=setting['target_global_id'];y=fx[:,target,:];vy=vx[:,target,:]
  for peer in range(6):
   if peer==target:continue
   raw=bytes([0x10|peer]);digest=hashlib.sha256(raw).hexdigest();name=f'direct_{digest}.bin';(folder/'states'/name).write_bytes(raw);m=metrics(vy,vx[:,peer,:]);row={'target':target,'target_id':setting['target_id'],'peer':peer,'peer_id':runner.MODELS[peer],'family':'direct_identity','state_file':name,'state_digest':digest,'actual_bits':8,'meaningful_bits':8,'codec':'identity_v1_global_id_byte','validation_tv':float(m[:,0].mean()),'validation_kl':float(m[:,1].mean()),'validation_argmax_disagreement':float(m[:,2].mean()),'selection_scope':'all predetermined peers reported; target-selected identity charged8bits'};ledger.append(row);selected.append(row)
   weights=core.fit_weights(fx,y,(peer,),'nnls').reshape(-1,1);candidates=[]
   for codec in core.CODECS:
    enc=core.encode(weights,'nnls',(peer,),codec);decoded,params=core.decode(enc.blob);pred=core.predict(decoded,params,vx);m=metrics(vy,pred);name=f'nnls_{enc.digest}.bin';(folder/'states'/name).write_bytes(enc.blob);row={'target':target,'target_id':setting['target_id'],'peer':peer,'peer_id':runner.MODELS[peer],'family':'nnls','state_file':name,'state_digest':enc.digest,'actual_bits':enc.bits,'meaningful_bits':enc.meaningful_bits,'codec':codec,'validation_tv':float(m[:,0].mean()),'validation_kl':float(m[:,1].mean()),'validation_argmax_disagreement':float(m[:,2].mean()),'decoded_weight':float(params[0,0]),'fit_weight':float(weights[0,0]),'selection_scope':'best validation codec per fixed single peer, tiesactualbits/statehash; same12codecs allpeers'};ledger.append(row);candidates.append(row)
   best=min(candidates,key=lambda r:(r['validation_tv'],r['actual_bits'],r['state_digest']));selected.append(best)
   assert all(r['actual_bits']<=1024 for r in candidates),'q1 NNLS unexpectedly exceeds smallest primary cap'
 pd.DataFrame(ledger).to_csv(folder/'validation_candidates.csv',index=False);pd.DataFrame(selected).to_csv(folder/'frozen_candidates.csv',index=False)
 freeze={'created_unix':time.time(),'role':'secondary cheap-control study registered before finaltest; primaryq2unchanged','data_lock_sha256':sha(base/'data_lock.json'),'primary_training_seed':lock['primary_training_seed'],'code_sha256':sha(__file__),'candidate_count':len(selected),'full_codec_candidate_count':len(ledger),'codec_menu':list(core.CODECS),'target_free':True,'all_peers_per_target':5,'q1_nnls_primary_cap_envelope':'All12codecstates <=1024bits, so same beststate feasible atallfourprimarycaps','direct_identity_format':'Onebyte: highnibble1 identityoperator/version; lownibbleglobalpeerID0..5. Nofree target-selectedID. Publicregistry/modelweights counted separately.','candidates':selected,'files_sha256':{str(p.relative_to(folder)):sha(p) for p in [folder/'validation_candidates.csv',folder/'frozen_candidates.csv']+list((folder/'states').glob('*.bin'))}}
 jsonwrite(folder/'candidate_freeze.json',freeze);print('cheap candidates frozen',len(selected))
def evaluate_arrays(base,folder,resource_lock,role):
 lock,rows,x=inputs(base,role);freeze=json.loads((folder/'candidate_freeze.json').read_text());assert freeze['data_lock_sha256']==sha(base/'data_lock.json');assert freeze['code_sha256']==sha(__file__)
 for p,h in freeze['files_sha256'].items():assert sha(folder/p)==h
 resource=json.loads(resource_lock.read_text());refs={};results=[];lossrecords=[];vectors=[]
 for row in freeze['candidates']:
  target=row['target']
  if target not in refs:
   ref=resource['references'][str(target)];adapter=DeployedAdapter(Path(ref['bundle']));rp=adapter.predict({i:x[:,i,:] for i in adapter.pair});refs[target]=(ref,metrics(x[:,target,:],rp))
  ref,rm=refs[target];pred=predict_state(folder,row,x);m=metrics(x[:,target,:],pred);delta=m-rm;vectors.extend([delta[:,0],delta[:,2]])
  result={**row,'reference_system':ref['system_id'],'reference_state_digest':ref['state_digest'],'data_role':role,'clusters':len(rows),'tv':float(m[:,0].mean()),'kl':float(m[:,1].mean()),'argmax_disagreement':float(m[:,2].mean()),'reference_tv':float(rm[:,0].mean()),'reference_kl':float(rm[:,1].mean()),'reference_argmax_disagreement':float(rm[:,2].mean()),'tv_degradation':float(delta[:,0].mean()),'kl_degradation':float(delta[:,1].mean()),'argmax_degradation':float(delta[:,2].mean())};results.append(result)
  d=rows[['item_id','item_cluster','prompt_hash']].copy();d['target']=target;d['peer']=row['peer'];d['family']=row['family'];d['state_digest']=row['state_digest'];d['reference_state_digest']=ref['state_digest']
  for j,name in enumerate(['tv','kl','argmax_disagreement']):d[name]=m[:,j];d['reference_'+name]=rm[:,j];d['difference_'+name]=delta[:,j]
  lossrecords.append(d)
 comparisons=2*len(results);ci=bootstrap_joint(np.column_stack(vectors),10000,81724055,.05,comparisons)
 for i,r in enumerate(results):
  r.update(tv_ci_low=float(ci[0,2*i]),tv_ci_high=float(ci[1,2*i]),argmax_ci_low=float(ci[0,2*i+1]),argmax_ci_high=float(ci[1,2*i+1]),tv_unadjusted_ci_low=float(ci[2,2*i]),tv_unadjusted_ci_high=float(ci[3,2*i]),argmax_unadjusted_ci_low=float(ci[2,2*i+1]),argmax_unadjusted_ci_high=float(ci[3,2*i+1]),familywise_comparisons=comparisons,ni_tv_margin=.005,ni_argmax_margin=.01,meets_joint_ni=bool(ci[1,2*i]<=.005 and ci[1,2*i+1]<=.01))
 # KL is a secondary descriptive paired interval, not a third NI criterion.
 klmatrix=np.column_stack([d.difference_kl.to_numpy() for d in lossrecords]);klci=bootstrap_joint(klmatrix,10000,81724055,.05,1)
 for i,r in enumerate(results):r.update(kl_unadjusted_ci_low=float(klci[2,i]),kl_unadjusted_ci_high=float(klci[3,i]))
 return pd.DataFrame(results),pd.concat(lossrecords,ignore_index=True),comparisons

def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['fit','lock','evaluate'],required=True);a.add_argument('--resource-lock',type=Path,default=ROOT/'experiments/runs/v2/resource_benchmark/confirmation_catalog/operating_points_lock.json');args=a.parse_args();base=ROOT/'experiments/runs/v2/confirmation';folder=base/'cheap_single_peer_controls'
 if args.stage=='fit':prepare(base,folder);return
 if args.stage=='lock':
  if (folder/'selection_lock.json').exists():raise RuntimeError('Cheap-control NI lock already exists')
  if list((base/'response_cache_test').glob('*.parquet')):raise RuntimeError('Finaltest already collected before cheap-control selection')
  r,d,n=evaluate_arrays(base,folder,args.resource_lock,'validation');r.to_csv(folder/'validation_ni_results.csv',index=False);d.to_parquet(folder/'validation_losses.parquet',index=False)
  lock={'created_unix':time.time(),'selected_before_test':True,'candidate_freeze_sha256':sha(folder/'candidate_freeze.json'),'resource_lock_sha256':sha(args.resource_lock),'code_sha256':sha(__file__),'familywise_comparisons':n,'bootstrap_draws':10000,'ni_tv_margin':.005,'ni_argmax_margin':.01,'no_primary_q2_changes':True,'all_candidates_retained':True,'eligible_validation_candidates':r.loc[r.meets_joint_ni,['target','peer','family','state_digest','actual_bits']].to_dict('records'),'outputs_sha256':{name:sha(folder/name) for name in ['validation_ni_results.csv','validation_losses.parquet']}}
  jsonwrite(folder/'selection_lock.json',lock);print(json.dumps(lock,indent=2));return
 lock=json.loads((folder/'selection_lock.json').read_text());assert lock['code_sha256']==sha(__file__);assert lock['resource_lock_sha256']==sha(args.resource_lock)
 r,d,n=evaluate_arrays(base,folder,args.resource_lock,'iid_test');r['selected_for_ni_before_test']=[any(q['target']==x.target and q['state_digest']==x.state_digest for q in lock['eligible_validation_candidates']) for x in r.itertuples(index=False)];r['secondary_scope']='allprespecifieddirect/q1NNLScontrols; simultaneousqualityintervalsfamily40; notadditionalprimaryq2claims';r.to_csv(folder/'test_results.csv',index=False);d.to_parquet(folder/'test_losses.parquet',index=False)
 jsonwrite(folder/'complete.json',{'completed':True,'selection_lock_sha256':sha(folder/'selection_lock.json'),'candidate_count':len(r),'test_clusters':int(r.clusters.iloc[0]),'outputs_sha256':{name:sha(folder/name) for name in ['test_results.csv','test_losses.parquet']}});print(r[['target_id','peer_id','family','tv','kl','argmax_disagreement','tv_degradation','tv_ci_high','argmax_ci_high','meets_joint_ni']].to_string(index=False))
if __name__=='__main__':main()
