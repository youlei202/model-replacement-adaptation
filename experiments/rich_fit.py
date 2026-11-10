"""Strong target-free full-vocabulary replacement with peer-only public bases.

All menus and baselines use the identical finite candidate envelope. Only fit and
validation logits are opened by fit(); evaluation requires an immutable lock.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np,pandas as pd,scipy.optimize,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump
from rich_codec import encode,decode,CODECS
PAIRS=list(itertools.combinations(range(3),2));TARGETS=[3,4]
CAPS=[128,512,2048,8192,32768,131072,524288,2097152]
RANKS=[16,32,64,128];LAMBDAS=[1e-4,.01,1.,100.];FACTOR_RANKS=[1,2,4,8]
DEVICE='cuda:0'
EXECUTION_SOURCE_HASHES={name:digest(ROOT/'experiments'/name) for name in ['rich_fit.py','rich_codec.py','rich_inference.py']}
def loadlog(role,i):
 p=OUT/'logits'/role/f'model_{i}.npy';meta=json.load(open(p.with_suffix('.json')))
 assert meta['complete'] and meta['model_revision']==MODELS[i][1]
 return np.load(p,mmap_mode='r')
def tensor(a):return torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32,device=DEVICE)
def array(a):return a.detach().cpu().numpy()
def filehash(b):return hashlib.sha256(b).hexdigest()
from rich_inference import predict,metrics

def score(state,peers,target,basis,batch=64):
 vals={k:[] for k in ['tv','kl','argmax_disagreement','target_argmax','prediction_argmax','target_margin']}
 with torch.inference_mode():
  for s in range(0,len(target),batch):
   prediction,isp=predict(state,[x[s:s+batch] for x in peers],basis)
   m=metrics(target[s:s+batch],prediction,isp)
   for k,v in m.items():vals[k].append(array(v))
 return {k:np.concatenate(v) for k,v in vals.items()}

def build_basis(pair,fit):
 folder=OUT/'public_contexts';folder.mkdir(exist_ok=True);path=folder/f'pair_{pair[0]}_{pair[1]}.npy';meta=path.with_suffix('.json')
 input_hash={str(i):digest(OUT/'logits/fit'/f'model_{i}.npy') for i in pair}
 if path.exists():
  old=json.load(open(meta));assert old['fit_peer_logits_sha256']==input_hash and old['peers']==list(pair)
  return tensor(np.load(path)),old
 start=time.perf_counter();torch.manual_seed(81724033+pair[0]*10+pair[1])
 # Randomized SVD of peer-only centered fit logits. No target outputs can enter this function.
 X=torch.cat(fit,dim=0);u,s,v=torch.svd_lowrank(X,q=144,niter=2)
 basis=v[:,:128].contiguous();np.save(path,array(basis));del X,u,s,v
 rec={'peers':list(pair),'fit_peer_logits_sha256':input_hash,'target_outputs_used':False,'basis_rank':128,'vocab':basis.shape[0],'sha256':digest(path),'serialized_bytes':path.stat().st_size,'resident_float32_bytes':basis.numel()*4,'basis_algorithm':'torch.svd_lowrank q144 niter2 fixedpeer-dependentseed onstackedselectedpeerfitcenteredlogits','seconds':time.perf_counter()-start,'experiment_projections_excluded_from_deployment':True}
 dump(meta,rec);return basis,rec

def nnls_weights(fit,target):
 q=len(fit);G=np.zeros((q,q));c=np.zeros(q)
 for i in range(q):
  c[i]=float((fit[i].double()*target.double()).sum().cpu())
  for j in range(q):G[i,j]=float((fit[i].double()*fit[j].double()).sum().cpu())
 reg=max(np.trace(G)*1e-12,1e-12);L=np.linalg.cholesky(G+reg*np.eye(q));weights=scipy.optimize.nnls(L.T,np.linalg.solve(L,c))[0]
 if q==2:
  denominator=G[0,0]+G[1,1]-2*G[0,1]
  first=np.clip((c[0]-c[1]-G[0,1]+G[1,1])/max(denominator,1e-30),0,1)
  convex=np.asarray([first,1-first],dtype=np.float32)
 else:
  scale=max(np.max(np.abs(G)),1)
  opt=scipy.optimize.minimize(lambda w:(.5*w@G@w-c@w)/scale,np.full(q,1/q),jac=lambda w:(G@w-c)/scale,bounds=[(0,1)]*q,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],method='SLSQP',options={'maxiter':1000,'ftol':1e-12})
  if not opt.success:raise RuntimeError('Convex optimizer failed:'+opt.message)
  convex=opt.x.astype(np.float32)
 return weights.astype(np.float32),convex


def fit_cell(pair,target,fit,val,basis,ctx):
 assert target not in pair and set(pair)<=set(range(3)) and target in TARGETS
 folder=OUT/'fits'/f'target_{target}'/f'pair_{pair[0]}_{pair[1]}';folder.mkdir(parents=True,exist_ok=True)
 done=folder/'complete.json'
 if done.exists():return
 start=time.perf_counter();tfit=tensor(loadlog('fit',target));tval=tensor(loadlog('validation',target));w,convex=nnls_weights(fit,tfit);rows=[];best={};candidate_count=0;native_reference=None
 def offer(family,arrs,rank=0,lam=None,factor_rank=None):
  nonlocal candidate_count,native_reference
  for codec in (['fp32'] if family in ('logit_mean','probability_mean') else list(CODECS)):
   blob=encode(family,list(pair),rank,arrs,codec);sha=filehash(blob);state=decode(blob)
   stats=score(state,val,tval,basis);loss=float(stats['tv'].mean());n=len(blob)
   row={'candidate_id':sha,'target_index':target,'peer_mask':sum(1<<i for i in pair),'peers':','.join(map(str,pair)),'family':family,'codec':codec,'basis_rank':rank,'factor_rank':factor_rank,'lambda':lam,'actual_bytes':n,'actual_bits':8*n,'validation_tv':loss,'validation_kl':float(stats['kl'].mean()),'validation_disagreement':float(stats['argmax_disagreement'].mean()),'public_context_digest':ctx['sha256'] if rank else None,'selected_before_test':True}
   rows.append(row);candidate_count+=1
   # Preserve exactly those states needed by per-cap envelope, families and unbounded controls.
   keys=[('all',cap) for cap in CAPS if n<=cap]+[('unbounded',0)]+[(f'resource_{family}_rank{rank}',cap) for cap in CAPS if n<=cap]
   if family=='temperature' and arrs and np.count_nonzero(arrs[0])==1:
    keys.append((f'direct_peer_{pair[int(np.flatnonzero(arrs[0])[0])]}',0))
   if family in ('nnls','temperature','convex','logit_mean','probability_mean'):keys.extend([('cheap',cap) for cap in CAPS if n<=cap])
   for key in keys:
    old=best.get(key)
    if old is None or (loss,n,sha)<(old[0]['validation_tv'],old[0]['actual_bytes'],old[0]['candidate_id']):best[key]=(row,blob,stats)
   # Native-full residual reference: full output rank, unquantized; tune lambda fairly.
   if family=='ridge_residual' and rank==128 and codec=='fp32':
    if native_reference is None or (loss,n,sha)<(native_reference[0]['validation_tv'],native_reference[0]['actual_bytes'],native_reference[0]['candidate_id']):native_reference=(row,blob,stats)
  if candidate_count%40<4:print('fit',pair,target,candidate_count,round(time.perf_counter()-start),flush=True)
 offer('logit_mean',[]);offer('probability_mean',[]);offer('nnls',[w]);offer('convex',[convex])
 # Distillation-temperature candidates, common predeclared grid for every pair/policy.
 for temperature in [.5,.75,1.,1.25,1.5,2.]:offer('temperature',[w/temperature],lam=temperature)
 # Direct peers are contained by literal one-hot weights; model selector is counted.
 for j in range(len(pair)):
  direct=np.zeros(len(pair),dtype=np.float32);direct[j]=1;offer('temperature',[direct])
 base=sum(float(wi)*x for wi,x in zip(w,fit));residual=tfit-base;bias=array(residual.mean(0));offer('vocabulary_bias',[w,bias])
 for rank in RANKS:
  u=basis[:,:rank];features=torch.cat([x@u for x in fit]+[torch.ones((len(tfit),1),device=DEVICE)],1).double();y=(residual@u).double();gram=features.T@features/len(features);cross=features.T@y/len(features)
  diag=torch.eye(features.shape[1],dtype=torch.float64,device=DEVICE);diag[-1,-1]=0
  for lam in LAMBDAS:
   coef=torch.linalg.solve(gram+lam*diag+1e-10*torch.eye(len(diag),device=DEVICE,dtype=torch.float64),cross).T.float();A=array(coef)
   offer('ridge_residual',[w,A],rank,lam)
   left,s,right=torch.linalg.svd(coef,full_matrices=False)
   for k in FACTOR_RANKS:
    if k>=min(coef.shape):continue
    offer('factored_residual',[w,array(left[:,:k]*s[:k]),array(right[:k])],rank,lam,k)
 del tfit,tval,residual,base
 saved={};decisions=[]
 for key,value in list(best.items())+[ (('native_full',0),native_reference) ]:
  row,blob,stats=value;sha=row['candidate_id'];statepath=folder/'states'/f'{sha}.bin';statepath.parent.mkdir(exist_ok=True)
  if sha not in saved:
   statepath.write_bytes(blob);np.savez_compressed(folder/'states'/f'{sha}.validation.npz',**stats);saved[sha]=str(statepath.relative_to(OUT))
  decisions.append({'scope':key[0],'cap_bytes':key[1],**row,'state_path':saved[sha]})
 pd.DataFrame(rows).to_parquet(folder/'validation_candidates.parquet',index=False);pd.DataFrame(decisions).to_json(folder/'envelope.json',orient='records',indent=2)
 assert all(digest(ROOT/'experiments'/name)==h for name,h in EXECUTION_SOURCE_HASHES.items()),'Scoringcodechangedduringfit; preservefailedoutputsandrefit'
 dump(done,{'execution_source_sha256':EXECUTION_SOURCE_HASHES,'candidate_count':candidate_count,'retained_states':len(saved),'seconds':time.perf_counter()-start,'target_index':target,'peers':list(pair),'fit_document_count':len(fit[0]),'validation_document_count':len(val[0]),'fit_target_sha256':digest(OUT/'logits/fit'/f'model_{target}.npy'),'validation_target_sha256':digest(OUT/'logits/validation'/f'model_{target}.npy'),'candidate_ledger_sha256':digest(folder/'validation_candidates.parquet'),'public_context_digest':ctx['sha256'],'code_sha256':digest(Path(__file__)),'test_opened':False})
 print('completed',target,pair,flush=True)

def fit():
 if not (OUT/'collection_deployment_bridge.json').exists():
  import subprocess
  subprocess.run([sys.executable,str(ROOT/'experiments/rich_runtime_bridge.py')],check=True)
 assert json.load(open(OUT/'collection_deployment_bridge.json'))['passed']
 for pair in PAIRS:
  f=[tensor(loadlog('fit',i)) for i in pair];v=[tensor(loadlog('validation',i)) for i in pair];basis,meta=build_basis(pair,f)
  for target in TARGETS:fit_cell(pair,target,f,v,basis,meta)
  del f,v,basis;torch.cuda.empty_cache()

def main():
 global DEVICE
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fit']);p.add_argument('--device',default='cuda:0');a=p.parse_args();DEVICE=a.device;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 if a.action=='fit':fit()
if __name__=='__main__':main()
