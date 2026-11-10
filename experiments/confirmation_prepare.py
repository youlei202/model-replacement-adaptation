"""Prepare fresh external query populations, independently of response outcomes."""
from __future__ import annotations
import argparse,hashlib,json,re,time,sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference_implementation'))
from kernel_isqed.response import canonical_permutation
from experiments.exposure_audit import normalize_text,text_hash,digest

def shingles(text):
 words=normalize_text(text).split()
 return set(' '.join(words[i:i+5]) for i in range(len(words)-4)) if len(words)>=5 else {' '.join(words)}

def signature(text):
 ss=shingles(text)
 return [min(int.from_bytes(hashlib.blake2b((str(seed)+token).encode(),digest_size=8).digest(),'big') for token in ss) for seed in range(64)]

def signature_pair(pair):return signature(pair[0]),signature(pair[1])

class UnionFind:
 def __init__(self,n):self.parent=list(range(n))
 def find(self,x):
  while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def cluster_union(frame,cache,workers=8):
 """Original exact+LSH/Jaccard process plus stricter context-only duplicates.

 Signature cache stores uint64 bits exactly, not float64 approximations.
 """
 semantic=[c+'\n'+'\n'.join(json.loads(o)) for c,o in frame[['context','options']].itertuples(index=False,name=None)]
 contexts=frame.context.tolist();texts=[semantic,contexts]
 keys=[hashlib.sha256((a+'\0'+b).encode()).hexdigest() for a,b in zip(semantic,contexts)]
 fingerprint=hashlib.sha256(json.dumps(keys).encode()).hexdigest()
 if cache.exists():
  data=np.load(cache);assert str(data['fingerprint'])==fingerprint;sig=data['signatures']
 else:
  start=time.monotonic();sig=np.empty((len(frame),2,64),dtype=np.uint64)
  with ProcessPoolExecutor(max_workers=workers) as pool:
   for i,x in enumerate(pool.map(signature_pair,zip(semantic,contexts),chunksize=32)):
    sig[i]=x
    if i%2000==0:print(f'signatures {i}/{len(frame)} elapsed{time.monotonic()-start:.1f}s',flush=True)
  np.savez_compressed(cache,signatures=sig,fingerprint=fingerprint)
 uf=UnionFind(len(frame));edges=[]
 # Preserve all previously registered original clusters.
 oldgroups={}
 for i,row in frame.iterrows():
  if row.historical:
   key=(row.task_id,row.item_cluster)
   if key in oldgroups:uf.union(i,oldgroups[key])
   else:oldgroups[key]=i
 for mode,values in enumerate(texts):
  exact={};buckets=defaultdict(list)
  @lru_cache(maxsize=8192)
  def getshingle(i):return shingles(values[i])
  for i,text in enumerate(values):
   h=text_hash(text)
   if h in exact:
    other=exact[h];uf.union(i,other);edges.append((i,other,'exact_semantic' if mode==0 else 'exact_context',1.))
   else:exact[h]=i
   checked=set()
   for band in range(8):
    k=(band,sig[i,mode,band*8:(band+1)*8].tobytes())
    for other in buckets[k]:
     if other in checked:continue
     checked.add(other)
     a,b=getshingle(i),getshingle(other);j=len(a&b)/len(a|b) if a|b else 1.
     if j>=.85:uf.union(i,other);edges.append((i,other,'near_semantic' if mode==0 else 'near_context',j))
    buckets[k].append(i)
 frame=frame.copy();frame['union_cluster_index']=[uf.find(i) for i in range(len(frame))]
 blocked=set(frame.loc[frame.historical,'union_cluster_index'])
 frame['historical_cluster_overlap']=frame.union_cluster_index.isin(blocked)
 frame['item_cluster']=['external_cluster_'+hashlib.sha256(keys[root].encode()).hexdigest()[:24] for root in frame.union_cluster_index]
 return frame,pd.DataFrame(edges,columns=['left_row','right_row','match_rule','jaccard'])

def external_rows(task,src,revision):
 d=pd.read_parquet(src);rows=[]
 for i,r in enumerate(d.to_dict('records')):
  if task=='academic_mcq4':context=str(r['question']);options=list(r['choices']);gold=int(r['answer']);domain=str(r['subject'] or 'auxiliary_unspecified');source_id=str(i)
  elif task=='hellaswag_external':context=str(r['ctx']);options=list(r['endings']);gold=int(r['label']);domain=str(r['activity_label']);source_id=str(r['source_id'])
  else:
   context=str(r['prompt']);chosen=[m['content'] for m in r['chosen'] if m['role']=='assistant'][-1];rejected=[m['content'] for m in r['rejected'] if m['role']=='assistant'][-1]
   swap=int(hashlib.sha256(str(r['prompt_id']).encode()).hexdigest(),16)%2
   options=[chosen,rejected] if swap==0 else [rejected,chosen];gold=swap;domain='ultrafeedback';source_id=str(r['prompt_id'])
  options=[str(v) for v in options]
  count=2 if task=='preference_external' else 4
  if len(options)!=count or not context.strip() or any(not x.strip() for x in options):continue
  item=f'{task}:{i}';perm=canonical_permutation(item,count,'external_confirmation_v1')
  optiontext='\n'.join(f'{j}. {options[k]}' for j,k in enumerate(perm));system='Choose the single best option. Return only its numeric answer code.'
  if task=='academic_mcq4':user=f'Question:\n{context}\n\nOptions:\n{optiontext}\n\nReturn only one code from 0 to 3.'
  elif task=='hellaswag_external':user=f'Context:\n{context}\n\nPossible continuations:\n{optiontext}\n\nReturn only one code from 0 to 3.'
  else:user=f'Conversation:\n{context}\n\nCandidate responses:\n{optiontext}\n\nWhich response should a careful, helpful, and safe assistant prefer?\nReturn only 0 or 1.'
  prompt_hash=hashlib.sha256(json.dumps([system,user],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  rows.append(dict(task_id=task,item_id=item,context=context,options=json.dumps(options,ensure_ascii=False),gold=gold,domain=domain,source_id=source_id,item_cluster='',dataset_revision=revision,permutation=json.dumps(perm),system_prompt=system,user_prompt=user,prompt_hash=prompt_hash,contrast_dim=count-1,historical=False))
 return pd.DataFrame(rows)

def main():
 a=argparse.ArgumentParser();a.add_argument('--tasks',nargs='+',default=['academic_mcq4','hellaswag_external','preference_external']);a.add_argument('--workers',type=int,default=8);a.add_argument('--candidate-cap',type=int,default=16384);args=a.parse_args()
 out=ROOT/'experiments/runs/v2/confirmation';audit=ROOT/'experiments/runs/v2/audit';lock=json.loads((out/'population_acquisition_preregistration.json').read_text());hist=pd.read_parquet(audit/'historical_text_registry.parquet');hist['historical']=True
 for old,info in lock['task_mapping'].items():
  task=info['external_task_id']
  if task not in args.tasks:continue
  folder=out/task;folder.mkdir(exist_ok=True)
  fresh=external_rows(task,out/'data_sources'/task/info['file'],info['revision'])
  nraw=len(fresh)
  fresh['sample_order']=[hashlib.sha256(('external81724031|'+s).encode()).hexdigest() for s in fresh.item_id]
  # This predetermined response-blind subsampling bounds duplicate-audit compute.
  fresh=fresh.sort_values('sample_order').head(args.candidate_cap).reset_index(drop=True)
  combined=pd.concat([hist,fresh],ignore_index=True)
  clustered,edges=cluster_union(combined,folder/'duplicate_signatures.npz',args.workers)
  edges.to_parquet(folder/'duplicate_edges.parquet',index=False)
  clustered[['task_id','item_id','historical','item_cluster','union_cluster_index','historical_cluster_overlap']].to_parquet(folder/'union_duplicate_registry.parquet',index=False)
  external=clustered[~clustered.historical].copy();remaining=external[~external.historical_cluster_overlap].copy()
  remaining=remaining.sort_values('sample_order').drop_duplicates('item_cluster').reset_index(drop=True)
  remaining.to_parquet(folder/'eligible_before_tokenizers.parquet',index=False)
  report=dict(task_id=task,external_population=True,source_rows=nraw,response_blind_candidate_cap=args.candidate_cap,sampled_candidates=len(fresh),historical_texts=len(hist),historical_overlap_items=int(external.historical_cluster_overlap.sum()),remaining_unique_clusters=len(remaining),duplicate_rule='original normalized5word-shingle64BLAKE2b64MinHash8bands8hashesJaccard>=0.85 plus context-only same rule; all original clusters retained',historical_registry_sha256=digest(audit/'historical_text_registry.parquet'),candidate_file_sha256=digest(folder/'eligible_before_tokenizers.parquet'))
  (folder/'freshness_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
