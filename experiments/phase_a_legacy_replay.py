#!/usr/bin/env python3
"""Validate every old frozen state on its ORIGINAL public context and validation data.

New eigendecompositions are not substituted for original context hashes. Their
near-null rotations can alter low-bit quantization and are a separate audit.
"""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from phase_a_run import ROOT,MODELS,REVISIONS,join_models,old,sha,atomic_json
from adapters_v2 import core,predict

def main():
 out=ROOT/'experiments/runs/v2/phase_a';response,_=old.read_responses([ROOT/'benchmarks/response_cache/recompute_all'],MODELS,REVISIONS)
 rows=[]
 with threadpool_limits(limits=4):
  for path in sorted((ROOT/'benchmarks/results_complete').glob('seed_*/*/*/selection_freeze.json')):
   freeze=json.loads(path.read_text());seed=freeze['seed'];task=freeze['task'];target=MODELS.index(freeze['target'])
   mf=pd.read_parquet(ROOT/f'benchmarks/splits/split_seed_{seed}.parquet');mf=mf[(mf.task_id==task)&(mf.split=='validation')].sort_values('item_id')
   dim={'mmlu_pro':9,'hellaswag':3,'rewardbench':1}[task];values=join_models(mf,response,dim);contexts={};losses={}
   for r in freeze['selections']:
    digest=r['state_digest'];pair=tuple(r['pair'])
    if digest in losses:continue
    if digest.startswith('public_mean_'):pred=values[:,pair,:].mean(1)
    else:
     enc,m=core.decode((path.parent/'states'/f'{digest}.bin').read_bytes())
     if enc.digest!=digest or enc.bits!=r['actual_bits']:raise ValueError('Old state digest/size changed')
     if enc.family=='krr':
      if pair not in contexts:
       rel=freeze['contexts']['-'.join(map(str,pair))];cp=(path.parent/rel).resolve()
       if sha(cp)!=freeze['artifacts'][rel]:raise ValueError('Old original context hash changed')
       with np.load(cp) as c:contexts[pair]=core.Context(pair,c['mean'],c['scale'],float(c['sigma']),c['features'],c['eigenvalues'],c['eigenvectors'],c['validation_projection'])
     context=contexts.get(pair);pred=predict(enc,m,values,context,context.validation_projection if context is not None else None)
    loss=float(core.tv(values[:,target,:],pred).mean());losses[digest]=loss
    rows.append({'seed':seed,'task':task,'target':target,'state_digest':digest,'stored_validation_tv':r['validation_tv'],'replayed_validation_tv':loss,'absolute_difference':abs(loss-r['validation_tv'])})
 pd.DataFrame(rows).to_csv(out/'legacy_original_context_replay.csv',index=False)
 report={'states':len(rows),'cells':45,'max_absolute_difference':max(r['absolute_difference'] for r in rows),'old_contexts_hash_checked':True,'new_eigensystems_substituted':False,'passed':all(r['absolute_difference']<1e-11 for r in rows)}
 atomic_json(out/'legacy_original_context_replay.json',report);print(json.dumps(report))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
