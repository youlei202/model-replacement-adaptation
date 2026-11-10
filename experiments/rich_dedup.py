"""Freeze rich document clusters after original near-duplicate union with historical queries."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.confirmation_prepare import signature,shingles,UnionFind
from experiments.exposure_audit import normalize_text,text_hash
from rich_setup import OUT,digest,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 fresh=pd.read_parquet(OUT/'candidate_documents.parquet');hist=pd.read_parquet(ROOT/'experiments/runs/v2/audit/historical_text_registry.parquet')
 # Nodes represent full articles and presented contexts. A full article and its context are joined regardless of similarity.
 texts=[];source_owner=[];unique={};docnodes=[];histnodes=[]
 def node(text,owner):
  h=text_hash(text)
  if h not in unique:unique[h]=len(texts);texts.append(text);source_owner.append(owner)
  return unique[h]
 for i,r in hist.iterrows():
  histnodes.append((node(r.context,f'historical:{i}'),node(r.context+'\n'+'\n'.join(json.loads(r.options)),f'historical:{i}')))
 for i,r in fresh.iterrows():docnodes.append((node(r.text,f'fresh:{i}'),node(r.context_text,f'fresh:{i}')))
 keyhash=hashlib.sha256(json.dumps([text_hash(t) for t in texts]).encode()).hexdigest();cache=OUT/'rich_duplicate_signatures.npz'
 if cache.exists():
  obj=np.load(cache);assert str(obj['fingerprint'])==keyhash;sig=obj['signatures']
 else:
  sig=np.empty((len(texts),64),dtype=np.uint64);start=time.perf_counter()
  with ProcessPoolExecutor(a.workers) as pool:
   for i,v in enumerate(pool.map(signature,texts,chunksize=8)):
    sig[i]=v
    if i%1000==0:print('signatures',i,'/',len(texts),'elapsed',round(time.perf_counter()-start),flush=True)
  np.savez_compressed(cache,signatures=sig,fingerprint=keyhash)
 uf=UnionFind(len(texts));edges=[];oldgroups={}
 for i,(left,right) in enumerate(histnodes):
  uf.union(left,right);key=(hist.iloc[i].task_id,hist.iloc[i].item_cluster)
  if key in oldgroups:uf.union(left,oldgroups[key])
  else:oldgroups[key]=left
 for left,right in docnodes:uf.union(left,right)
 @lru_cache(maxsize=128)
 def ss(i):return shingles(texts[i])
 buckets=defaultdict(list)
 for i in range(len(texts)):
  checked=set()
  for band in range(8):
   key=(band,sig[i,band*8:(band+1)*8].tobytes())
   for j in buckets[key]:
    if j in checked:continue
    checked.add(j);x,y=ss(i),ss(j);jac=len(x&y)/len(x|y) if x|y else 1
    if jac>=.85:uf.union(i,j);edges.append((i,j,jac))
   buckets[key].append(i)
 blocked={uf.find(a) for a,b in histnodes};fresh['historical_overlap']=[uf.find(a) in blocked for a,b in docnodes];fresh['cluster_id']=['rich_cluster_'+hashlib.sha256(texts[uf.find(a)].encode()).hexdigest()[:24] for a,b in docnodes]
 kept=fresh[~fresh.historical_overlap].sort_values('order_hash').drop_duplicates('cluster_id').copy();kept['role']='test_pool'
 for source,group in kept.groupby('source'):
  if len(group)<64+1024+512+1024:raise RuntimeError('Insufficient sourceclusters')
  ix=group.index
  kept.loc[ix[:64],'role']='pilot';kept.loc[ix[64:1088],'role']='fit';kept.loc[ix[1088:1600],'role']='validation'
 kept=kept.sort_values(['role','order_hash']).reset_index(drop=True);kept.to_parquet(OUT/'documents.parquet',index=False)
 fresh[['item_id','source','cluster_id','historical_overlap']].to_parquet(OUT/'document_cluster_registry.parquet',index=False);pd.DataFrame(edges,columns=['left_node','right_node','jaccard']).to_parquet(OUT/'duplicate_edges.parquet',index=False)
 report={'documents_sha256':digest(OUT/'documents.parquet'),'historical_registry_sha256':digest(ROOT/'experiments/runs/v2/audit/historical_text_registry.parquet'),'candidate_document_sha256':digest(OUT/'candidate_documents.parquet'),'signature_sha256':digest(cache),'historical_queries':len(hist),'unique_text_nodes':len(texts),'fresh_document_candidates':len(fresh),'historical_overlap_documents':int(fresh.historical_overlap.sum()),'eligible_unique_document_clusters':len(kept),'counts':kept.groupby(['role','source']).size().to_dict(),'duplicate_rule':'original lower/punctuation/whitespace normalization; word5shingles,64BLAKE2b64 minhash,8bands8,verifiedJaccard>=0.85; both fullarticle and presentedcontext; preserve allhistorical clusters and join eacharticle to itscontext','no_model_outcomes_used':True,'test_pool_never_collected_before_selection_lock':True,'known_history_scope':'all23008 originalcandidatequeries plus subsequentknown caches; no claim about foundationmodel pretraining contamination'}
 report['counts']={str(k):int(v) for k,v in report['counts'].items()};dump(OUT/'data_lock.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
