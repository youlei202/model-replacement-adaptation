"""Cross-audit concurrently prepared MCQ and rich cohorts before either final test opens."""
import hashlib,json,sys,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.confirmation_prepare import signature,shingles
from experiments.exposure_audit import text_hash
from rich_setup import OUT,dump,digest

def main():
 hist=pd.read_parquet(ROOT/'experiments/runs/v2/audit/historical_text_registry.parquet');fresh=pd.read_parquet(OUT/'candidate_documents.parquet');kept=pd.read_parquet(OUT/'documents.parquet');roles=kept.set_index('item_id').role.to_dict();texts=[];unique={};richnodes={}
 def node(text):
  h=text_hash(text)
  if h not in unique:unique[h]=len(texts);texts.append(text)
  return unique[h]
 for r in hist.itertuples():node(r.context);node(r.context+'\n'+'\n'.join(json.loads(r.options)))
 for r in fresh.itertuples():
  a,b=node(r.text),node(r.context_text)
  if r.item_id in roles:
   richnodes.setdefault(a,[]).append((r.item_id,roles[r.item_id]));richnodes.setdefault(b,[]).append((r.item_id,roles[r.item_id]))
 raw=np.load(OUT/'rich_duplicate_signatures.npz');sig=raw['signatures'];assert len(texts)==len(sig)
 fingerprint=hashlib.sha256(json.dumps([text_hash(t) for t in texts]).encode()).hexdigest();assert str(raw['fingerprint'])==fingerprint
 buckets=defaultdict(list)
 for i in richnodes:
  for b in range(8):buckets[(b,sig[i,b*8:b*8+8].tobytes())].append(i)
 mcq=[];files=[]
 for role in ['fitval','test']:
  path=ROOT/'experiments/runs/v2/confirmation'/f'required_items_{role}.parquet';frame=pd.read_parquet(path);files.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path),'rows':len(frame)})
  for r in frame.itertuples():
   mcq.append((r.item_id,role,r.context));mcq.append((r.item_id,role,r.context+'\n'+'\n'.join(json.loads(r.options))))
 @lru_cache(256)
 def sh(i):return shingles(texts[i])
 edges=[]
 with ProcessPoolExecutor(8) as ex:
  for (item,role,text),s in zip(mcq,ex.map(signature,[r[2] for r in mcq],chunksize=32)):
   arr=np.asarray(s,dtype=np.uint64);matches=set();exact=unique.get(text_hash(text))
   if exact in richnodes:matches.add(exact)
   for b in range(8):matches.update(buckets.get((b,arr[b*8:b*8+8].tobytes()),[]))
   a=shingles(text)
   for j in matches:
    b=sh(j);jac=len(a&b)/len(a|b) if a|b else 1
    if jac>=.85:
     for richitem,richrole in richnodes[j]:edges.append({'mcq_item_id':item,'mcq_role':role,'rich_item_id':richitem,'rich_role':richrole,'jaccard':jac})
 output=OUT/'concurrent_mcq_cross_audit.json';report={'complete':True,'before_final_test_outputs':True,'mcq_inputs':files,'rich_documents_sha256':digest(OUT/'documents.parquet'),'rich_near_duplicate_cache_sha256':digest(OUT/'rich_duplicate_signatures.npz'),'rule':'original normalizedword5-shingles64BLAKE2b64minhash8bands8,verifiedJaccard>=.85; MCQcontextandsemantic versusrichfullarticleandactualcontext','cross_collisions':edges,'cross_collision_count':len(edges),'passed':len(edges)==0,'known_scope':'alllockedMCQfitvalidationsealedtestqueries vsallrichpilotfitvalidationandsealedpoolfullarticles/contexts'};dump(output,report);print(json.dumps(report),flush=True)
 if edges:raise RuntimeError('Concurrent crosscohortcollisionrequiresregisteredamendmentbeforetesting')
if __name__=='__main__':main()
