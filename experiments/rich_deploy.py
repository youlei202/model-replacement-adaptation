"""Target-free live full-vocabulary predictor; actual peer inference and literal-state replay.

Development profiling is performed before the immutable test lock. Final timing
uses randomized repeated blocks and records wall and CUDA-event durations distinctly.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np,pandas as pd,psutil,torch
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump
from rich_codec import decode
from rich_inference import predict
PAIRS=[(0,1),(0,2),(1,2)];TARGETS=[3,4];CAPS=[128,512,2048,8192,32768,131072,524288,2097152]
def tensor(a):return torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32,device="cuda:0")
CACHE=Path('/tmp/model-replacement-adaptation/hf_cache')
RESOURCE=OUT/'resources';RESOURCE.mkdir(exist_ok=True)

def load_pair(pair):
 models={};begin=time.perf_counter()
 assert set(pair)<=set(range(3)), 'Retired target cannot be a deployed peer'
 for i in pair:
  mid,rev,role=MODELS[i];assert role=='peer'
  path=CACHE/('models--'+mid.replace('/','--'))/'snapshots'/rev
  models[i]=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0').eval()
 return models,time.perf_counter()-begin

def get_contexts(length,batch):
 docs=pd.read_parquet(OUT/'documents.parquet');docs=docs[docs.role.eq('validation') & docs.context_tokens.ge(length)].sort_values('order_hash')
 if len(docs)<batch:raise RuntimeError('Insufficient natural contexts for length')
 # Same deterministic validation contexts for all systems, repeated requests are timing units only.
 return [np.asarray(x[-length:] if length else x,dtype=np.int64) for x in docs.context_ids.iloc[:max(batch,256)]]

def forward(blob,models,basis,contexts):
 state=decode(blob);inputs=[torch.as_tensor(np.asarray(c).copy(),dtype=torch.long,device='cuda:0').reshape(1,-1) for c in contexts];logits=[]
 with torch.inference_mode():
  live_logits={}
  for peer_index,model in models.items():
   x=torch.cat([model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[:,-1,:].float() for ids in inputs],dim=0);live_logits[peer_index]=x-x.mean(-1,keepdim=True)
  template=next(iter(live_logits.values()));logits=[live_logits[i] if i in live_logits else torch.zeros_like(template) for i in state['peers']]
  pred,prob=predict(state,logits,basis)
  output=pred if prob else torch.softmax(pred,dim=-1)
 return output

def timed(blob,models,basis,contexts):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);tic=time.perf_counter();start.record();output=forward(blob,models,basis,contexts);end.record();torch.cuda.synchronize();wall=(time.perf_counter()-tic)*1000;event=start.elapsed_time(end)
 rec={'end_to_end_ms':wall,'cuda_event_elapsed_ms':event,'GPU_ms':event,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_RSS':psutil.Process().memory_info().rss,'prediction_probability_sum_max_error':float((output.sum(-1)-1).abs().max().cpu()),'output_argmax':output.argmax(-1).cpu().tolist()}
 return rec

def minimal_basis(pair,rank):
 if not rank:return None
 folder=RESOURCE/'deployment_contexts';folder.mkdir(exist_ok=True);path=folder/f'pair_{pair[0]}_{pair[1]}_rank_{rank}.npy'
 if not path.exists():
  full=np.load(OUT/'public_contexts'/f'pair_{pair[0]}_{pair[1]}.npy',mmap_mode='r');np.save(path,np.asarray(full[:,:rank]).copy())
 return path

def state_account(row,models,basis):
 blob=(OUT/row['state_path']).read_bytes();state=decode(blob);weightbytes=sum(p.numel()*p.element_size() for model in models.values() for p in model.parameters());basispath=minimal_basis(tuple(state['peers']),state['basis_rank']);publicbytes=0 if basispath is None else basispath.stat().st_size;publicresident=0 if basis is None else basis.numel()*4
 return {'serialized_state_bytes':len(blob),'resident_state_bytes':state['resident_array_bytes'],'shared_deployment_bytes':publicbytes,'shared_resident_basis_bytes':publicresident,'deployment_basis_path':None if basispath is None else str(basispath.relative_to(OUT)),'deployment_basis_sha256':None if basispath is None else digest(basispath),'peer_weight_resident_bytes':weightbytes,'inference_source_bytes':sum((ROOT/'experiments'/p).stat().st_size for p in ['rich_codec.py','rich_inference.py','rich_deploy.py','rich_setup.py']),'peer_weight_serialized_bytes':sum(p.stat().st_size for i in models for p in (CACHE/('models--'+MODELS[i][0].replace('/','--'))/'snapshots'/MODELS[i][1]).glob('*.safetensors')),'standalone_accounted_bytes_weights_basis_state':weightbytes+publicresident+state['resident_array_bytes'],'conditional_accounted_bytes_target_state':state['resident_array_bytes'],'model_dtype':'bfloat16','real_target_count':1,'registered_distinct_targets':2,'hardware':torch.cuda.get_device_name(),'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':__import__('transformers').__version__},'target_absent':True,'state_digest':digest(OUT/row['state_path']),'peer_ids':list(models),'nominal_peer_ids':state['peers'],'actual_call_q':len(models),'nominal_q':len(state['peers']),'complete_pipeline':True,'GPU_metric_semantics':'synchronized CUDA event interval including allpeerforward/projection/softmax; not kernelbusytime or sumofmodelonlytimings','state_decoded_every_request':True,'exact_zero_scalar_dead_peer_elimination':True,'inference_microbatch':1,'logical_batch8_is_eight_canonical_single_request_forwards':True,'shared_basis_trimmed_to_selected_rank':True}

def allrows():
 rows=[]
 for target in TARGETS:
  for pair in PAIRS:
   folder=OUT/'fits'/f'target_{target}'/f'pair_{pair[0]}_{pair[1]}'
   assert (folder/'complete.json').exists()
   rows.extend(json.load(open(folder/'envelope.json')))
 return rows

def active_peers(state):
 # Exact zero scalar coefficients eliminate calls only when no residual input depends on that peer.
 if state['family'] in ('nnls','convex','temperature','vocabulary_bias'):
  active=tuple(p for p,w in zip(state['peers'],state['arrays'][0]) if w!=0)
  if active:return active
 return tuple(state['peers'])

def row_active(row):return active_peers(decode((OUT/row['state_path']).read_bytes()))

def profile():
 output=RESOURCE/'development_profiles.json'
 if output.exists():return
 rows=allrows();records=[];rng=np.random.default_rng(81724035);contexts=get_contexts(0,1)
 for pair in sorted(set(row_active(r) for r in rows)):
  models,cold=load_pair(pair);unique={r['candidate_id']:r for r in rows if row_active(r)==pair};keys=list(unique);rng.shuffle(keys)
  for sid in keys:
   row=unique[sid];blob=(OUT/row['state_path']).read_bytes();state=decode(blob);rank=state['basis_rank'];basis=tensor(np.load(minimal_basis(tuple(state['peers']),rank))) if rank else None
   replay_errors=[]
   for i in range(5):
    live=forward(blob,models,basis,[contexts[i]])
    if i<2:
     cached=[tensor(np.load(OUT/'logits/validation'/f'model_{peer}.npy',mmap_mode='r')[i:i+1]) for peer in state['peers']]
     reference,isp=predict(state,cached,basis);reference=reference if isp else torch.softmax(reference,dim=-1)
     replay_errors.append(float(.5*(live-reference).abs().sum().cpu()));del cached,reference
   if max(replay_errors)>1e-5:raise RuntimeError('Minimaldeploymentstate/basis replay differs fromfrozenpeerresponses')
   stats=[timed(blob,models,basis,[contexts[(i+5)%len(contexts)]]) for i in range(10)]
   rec={'candidate_id':sid,'state_path':row['state_path'],'development_only':True,'saved_state_replay_max_TV':max(replay_errors),'replay_peer_cache_only_no_target':True,'batch_size':1,'context_tokens':0,'context_scope':'canonical_natural_untruncated_max512','warmup_requests':5,'timed_requests':10,'mean_end_to_end_ms':float(np.mean([r['end_to_end_ms'] for r in stats])),'mean_GPU_ms':float(np.mean([r['GPU_ms'] for r in stats])),'cold_model_load_seconds':cold,**state_account(row,models,basis),'raw_timings':stats};records.append(rec);print('profile',pair,sid[:10],round(rec['mean_GPU_ms'],3),flush=True)
   del basis;torch.cuda.empty_cache()
  del models;gc.collect();torch.cuda.empty_cache()
 dump(output,records);dump(RESOURCE/'development_profiles_freeze.json',{'sha256':digest(output),'protocol':'batch1 canonicalnaturalmax512 5warmup10timedrequest perretainedstate, onlydevelopmentperformance usedforoperatingpointselection; finalbenchmark20warmup200timed10blocks','time_unix':time.time(),'no_final_test_outcomes_used':True})

def freeze_resource():
 output=RESOURCE/'operating_points_lock.json'
 if output.exists():return
 from rich_evaluate import envelopes,choose,key,val_losses,POLICIES
 profiles=json.load(open(RESOURCE/'development_profiles.json'));cost={r['candidate_id']:r['mean_GPU_ms'] for r in profiles};points=[];sweep=[];fixed_cost_decisions=[];margins=[0,.0025,.005,.01];policylock=json.load(open(OUT/'selection_lock.json'))
 for target in TARGETS:
  table=envelopes(target)
  # All family/rank/cap validationchampions plusdirect andnative controls remain available.
  allcandidates={r['candidate_id']:r for rows in table.values() for r in rows}
  reference=min(allcandidates.values(),key=key);ref_loss=val_losses(reference);upper={}
  for sid,r in allcandidates.items():
   d=val_losses(r)-ref_loss;upper[sid]=float(d.mean()+1.6448536269514722*d.std(ddof=1)/np.sqrt(len(d)))
  fixedpairs={p:next(r['peer_mask'] for r in policylock['choices'] if r['target_index']==target and r['policy']==p) for p in POLICIES}
  def best(cap,margin,mask=None):
   candidates=[r for sid,r in allcandidates.items() if r['actual_bytes']<=cap and upper[sid]<=margin+1e-12 and (mask is None or r['peer_mask']==mask)]
   return min(candidates,key=lambda r:(cost[r['candidate_id']],r['validation_tv'],r['actual_bytes'],r['candidate_id'])) if candidates else None
  # Extra strong deployment fixedpair diagnostic: onepair chosen acrossstatecaps onvalidationcost/quality.
  joint_caps={cap:best(cap,.005) for cap in CAPS};feasible_caps=[cap for cap,row in joint_caps.items() if row is not None];pair_scores=[]
  for pair in PAIRS:
   mask=sum(1<<p for p in pair);chosen={cap:best(cap,.005,mask) for cap in feasible_caps};coverage=sum(r is not None for r in chosen.values());regret=sum(cost[r['candidate_id']]/cost[joint_caps[cap]['candidate_id']]-1 for cap,r in chosen.items() if r is not None)/max(coverage,1)
   pair_scores.append({'peer_mask':mask,'pair':list(pair),'feasible_caps':coverage,'joint_feasible_caps':len(feasible_caps),'mean_relative_cost_regret_on_covered_caps':regret})
  selected=min(pair_scores,key=lambda r:(-r['feasible_caps'],r['mean_relative_cost_regret_on_covered_caps'],r['peer_mask']));fixedpairs['fixed_cost_aware_mean_frontier']=selected['peer_mask'];fixed_cost_decisions.append({'target_index':target,'selected':selected,'all_pairs':pair_scores,'selection_rule':'maximizecoverageofjoint-feasiblecapsatmargin.005thenminimizeequalcapmeanrelativecostregret; target-specificsinglefixedpair,no finaltest'})
  for cap in CAPS:
   for margin in margins:
    for policy in ['budget_aware']+list(fixedpairs):
     chosen=best(cap,margin,None if policy=='budget_aware' else fixedpairs[policy]);base={'target_index':target,'policy':policy,'margin_tv':margin,'resource_state_cap_bytes':cap,'quality_reference_state':reference['candidate_id'],'quality_reference_validation_tv':reference['validation_tv']}
     if chosen is None:point={**base,'feasible':False,'failure_status':'no_state_met_frozen_validation_margin_and_statecap'}
     else:point={**base,**chosen,'feasible':True,'validation_gap_upper':upper[chosen['candidate_id']],'development_GPU_ms':cost[chosen['candidate_id']]}
     sweep.append(point)
     if cap==max(CAPS):points.append(point)
 dump(output,{'all_resource_choices_frozen':True,'points':points,'state_cap_sweep_points':sweep,'cost_aware_fixed_pair_diagnostic':fixed_cost_decisions,'primary_margin':.005,'secondary_margin_grid':margins,'primary_cost_setting':{'batch':1,'context_tokens':0,'scope':'canonicalnaturaluntruncatedmax512 samequerylawasqualitytest'},'cost_metric':'synchronizedCUDAeventelapsed_ms percontext ofcompletepipeline; walllatencyseparate','resource_candidate_catalog':'allfamily_by_basisrank_by_cap validationchampions plusdirectpeer andnative/unbounded controls; sameforjoint/fixed','zero_weight_elimination':'exactzero scalarweights removepeerinferenceonlyifnoresidualdependsonthepeer; uniformlyallpolicies; actual_call_qrecorded','development_profiles_sha256':digest(RESOURCE/'development_profiles.json'),'policy_selection_lock_sha256':digest(OUT/'selection_lock.json'),'time_unix':time.time(),'final_test_outcomes_opened':False,'interpretation':'savingsagainstquality-selectedfixedsetsalone donotprovebudgetcoupling; additionalcost-awarefixedpairandallcapssweepmustbeexamined'})

def benchmark():
 if (RESOURCE/'benchmark_complete.json').exists():
  benchmark_direct_references()
  import subprocess
  subprocess.run([sys.executable,str(ROOT/'experiments/rich_report.py')],check=True)
  return
 lockpath=RESOURCE/'operating_points_lock.json';lock=json.load(open(lockpath));assert lock['all_resource_choices_frozen'];points=[p for p in lock['state_cap_sweep_points'] if p['feasible']];unique={r['candidate_id']:r for r in points};rows=[];rng=np.random.default_rng(81724036)
 statuspath=RESOURCE/'benchmark_progress.json';rawpath=RESOURCE/'timing_raw.jsonl'
 completed_requests=set()
 if rawpath.exists():
  rows=[json.loads(line) for line in rawpath.read_text().splitlines() if line.strip()]
  completed_requests={(r['system_digest'],r['batch_size'],r['context_tokens'],r['timing_block'],r['request']) for r in rows}
  if len(completed_requests)!=len(rows):raise RuntimeError('Duplicated timingrequests inexistingledger')
 # Interleave pair order and candidate order in10randomizedblocks. Eachsystem gets20warmups+20requests/block=200 timed.
 with rawpath.open('a') as stream:
  for batch in [1,8]:
   for length in [0,128,512]:
    contexts=get_contexts(length,batch)
    for block in range(10):
     if block%2==0:
      base_pairs=sorted(set(row_active(r) for r in unique.values()));rng.shuffle(base_pairs)
     pairs=base_pairs if block%2==0 else list(reversed(base_pairs))
     for pair in pairs:
      systems=[r for r in unique.values() if row_active(r)==pair]
      if not systems:continue
      rng.shuffle(systems);models,cold=load_pair(pair)
      for order,row in enumerate(systems):
       blob=(OUT/row['state_path']).read_bytes();state=decode(blob);basis=tensor(np.load(minimal_basis(tuple(state['peers']),state['basis_rank']))) if state['basis_rank'] else None;account=state_account(row,models,basis)
       for j in range(20):forward(blob,models,basis,[contexts[(j+k)%len(contexts)] for k in range(batch)])
       for j in range(20):
        if (row['candidate_id'],batch,length,block,j) in completed_requests:continue
        rec=timed(blob,models,basis,[contexts[(block*20+j+k)%len(contexts)] for k in range(batch)]);rec.update({'system_digest':row['candidate_id'],'batch_size':batch,'context_tokens':length,'phase':'warm','timing_block':block,'execution_order':order,'request':j,'cold_model_load_seconds':cold,'failure_status':None,**account});stream.write(json.dumps(rec)+'\n');stream.flush();rows.append(rec)
       del basis;torch.cuda.empty_cache();print('benchmark',batch,length,block,pair,row['candidate_id'][:8],flush=True)
      del models;gc.collect();torch.cuda.empty_cache()
     dump(statuspath,{'batch':batch,'length':length,'block':block,'rows':len(rows),'complete':False})
 frame=pd.DataFrame(rows);frame.drop(columns=['runtime','output_argmax']).to_csv(RESOURCE/'timing_raw.csv',index=False)
 summary=frame.groupby(['system_digest','batch_size','context_tokens']).agg(mean_ms=('end_to_end_ms','mean'),p50_ms=('end_to_end_ms','median'),p95_ms=('end_to_end_ms',lambda x:x.quantile(.95)),mean_GPU_ms=('GPU_ms','mean'),requests=('request','size'),blocks=('timing_block','nunique'),peak_allocated=('peak_allocated_bytes','max'),peak_reserved=('peak_reserved_bytes','max'),host_RSS=('host_RSS','max'),serialized_state_bytes=('serialized_state_bytes','first'),resident_state_bytes=('resident_state_bytes','first'),shared_deployment_bytes=('shared_deployment_bytes','first'),peer_weight_resident_bytes=('peer_weight_resident_bytes','first')).reset_index();summary['GPU_ms_per_1000_contexts']=summary.mean_GPU_ms*1000/summary.batch_size;summary['contexts_per_second']=summary.batch_size*1000/summary.mean_ms;summary.to_csv(RESOURCE/'resource_summary.csv',index=False)
 dump(RESOURCE/'benchmark_complete.json',{'complete':True,'unique_systems':len(unique),'rows':len(frame),'operating_points_lock_sha256':digest(lockpath),'timing_raw_sha256':digest(rawpath),'minimum_requests_per_setting':int(summary.requests.min()),'minimum_blocks':int(summary.blocks.min()),'note':'randomized10blockpair/systemorder,20warmupsbeforeeach20requestblock; modelreloadsseparatecold timing; fullpredictor onlyregisteredpeers, retiredtargetsneverloaded'})
 print(summary.to_string(index=False),flush=True)
 benchmark_direct_references()
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'experiments/rich_report.py')],check=True)

def benchmark_direct_references():
 path=RESOURCE/'direct_reference_timings.csv'
 if path.exists():return
 records=[];rng=np.random.default_rng(81724041)
 # Direct serving is an explicit contextual reference, never loaded by replacement forward().
 for model_index in range(5):
  mid,rev,role=MODELS[model_index];model=AutoModelForCausalLM.from_pretrained(str(CACHE/('models--'+mid.replace('/','--'))/'snapshots'/rev),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0').eval()
  for batch in [1,8]:
   for length in [0,128,512]:
    contexts=get_contexts(length,batch)
    def direct(indices):
     inputs=[torch.as_tensor(np.asarray(c).copy(),dtype=torch.long,device='cuda:0').reshape(1,-1) for c in indices]
     with torch.inference_mode():
      logits=torch.cat([model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[:,-1,:].float() for ids in inputs])
      return torch.softmax(logits,dim=-1)
    for j in range(20):direct([contexts[(j+k)%len(contexts)] for k in range(batch)])
    for block in range(10):
     order=rng.permutation(20)
     for j in order:
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);tic=time.perf_counter();start.record();output=direct([contexts[(block*20+int(j)+k)%len(contexts)] for k in range(batch)]);end.record();torch.cuda.synchronize();wall=(time.perf_counter()-tic)*1000
      records.append({'model_index':model_index,'model_id':mid,'model_role':role,'reference_role':'direct_target_context_only' if role=='target' else 'direct_small_peer','batch_size':batch,'context_tokens':length,'timing_block':block,'request':int(j),'end_to_end_ms':wall,'GPU_ms':start.elapsed_time(end),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_RSS':psutil.Process().memory_info().rss,'weight_resident_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'inference_microbatch':1,'retirement_feasible':role=='peer'})
    print('direct_reference',model_index,batch,length,flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 frame=pd.DataFrame(records);frame.to_csv(path,index=False);summary=frame.groupby(['model_index','model_id','model_role','reference_role','batch_size','context_tokens']).agg(mean_end_to_end_ms=('end_to_end_ms','mean'),mean_GPU_ms=('GPU_ms','mean'),p95_ms=('end_to_end_ms',lambda x:x.quantile(.95)),peak_allocated_bytes=('peak_allocated_bytes','max'),weight_resident_bytes=('weight_resident_bytes','first'),requests=('request','size'),blocks=('timing_block','nunique')).reset_index();summary['GPU_ms_per1000contexts']=summary.mean_GPU_ms*1000/summary.batch_size;summary.to_csv(RESOURCE/'direct_reference_summary.csv',index=False)
 dump(RESOURCE/'direct_references_complete.json',{'rows':len(frame),'models':5,'scope':'independentcontextualcostreferences; directtargetsareinfeasibleforretirement andnotqueriedbyreplacementpipeline','sha256':digest(path),'batch8_mode':'8canonicalmicrobatch1requests'})

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['profile','freeze','benchmark','direct']);a=p.parse_args();torch.set_num_threads(4);{'profile':profile,'freeze':freeze_resource,'benchmark':benchmark,'direct':benchmark_direct_references}[a.action]()
if __name__=='__main__':main()
