#!/usr/bin/env python3
"""Predetermined public equal-logit means/direct peers, never selected for free."""
import argparse,itertools,json,time
from pathlib import Path
import pandas as pd
from phase_a_run import ROOT,MODELS,REVISIONS,old,join_models,sha,atomic_json
from adapters_v2 import core

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','evaluate'],required=True);p.add_argument('--output',type=Path,default=ROOT/'experiments/runs/v2/phase_a');a=p.parse_args()
 plan=json.loads((a.output/'run_plan.json').read_text());protocol=plan['protocol'];response,_=old.read_responses([Path(f) for f in protocol['response_hashes']],MODELS,REVISIONS)
 lockpath=a.output/'predetermined_public_controls_lock.json';rows=[];lossframes=[]
 if a.stage=='freeze':
  spec=[{'seed':s,'task':t,'target':target,'q':q,'pair':list(pair),'policy':'predetermined_public_equal_logit_mean','selection_used_target':False,'actual_target_state_bits':0} for s in protocol['seeds'] for t in protocol['tasks'] for target in protocol['targets'] for q in protocol['q'] for pair in itertools.combinations([i for i in range(6) if i!=target],q)]
  atomic_json(lockpath,{'protocol_contract':plan['contract'],'script_sha256':sha(Path(__file__)),'controls':spec,'created_unix':time.time(),'caveat':'Each pair is independently predetermined and reported; choosing a target-specific winning pair from these would require charging the selector and is NOT done.'})
  print('frozen predetermined public controls',len(spec));return
 lock=json.loads(lockpath.read_text())
 if lock['protocol_contract']!=plan['contract'] or lock['script_sha256']!=sha(Path(__file__)):raise ValueError('Control lock drift')
 for s in protocol['seeds']:
  mf=pd.read_parquet(next(Path(f) for f in protocol['manifest_hashes'] if f.endswith(f'_{s}.parquet')))
  for t in protocol['tasks']:
   selection=[r for r in lock['controls'] if r['seed']==s and r['task']==t]
   vals={}
   for split in ('validation','iid_test'):
    meta=mf[(mf.task_id==t)&(mf.split==split)].sort_values('item_id');x=join_models(meta,response,protocol['dimensions'][t])
    for r in selection:
     pair=tuple(r['pair']);values=core.tv(x[:,r['target'],:],x[:,pair,:].mean(1));k=(r['target'],r['q'],pair)
     vals.setdefault(k,{})[split]=float(values.mean())
     if split=='iid_test':
      frame=meta[['item_id','item_cluster','prompt_hash']].copy();frame['seed']=s;frame['task_id']=t;frame['target_global_id']=r['target'];frame['q']=r['q'];frame['pair']='-'.join(map(str,pair));frame['tv_loss']=values;lossframes.append(frame)
   for r in selection:rows.append(r|{'target_id':MODELS[r['target']],'validation_tv':vals[(r['target'],r['q'],tuple(r['pair']))]['validation'],'test_tv':vals[(r['target'],r['q'],tuple(r['pair']))]['iid_test'],'diagnostic_only':True})
 pd.DataFrame(rows).to_csv(a.output/'predetermined_public_controls.csv',index=False)
 pd.concat(lossframes,ignore_index=True).to_parquet(a.output/'predetermined_public_control_losses.parquet',index=False)
 atomic_json(a.output/'predetermined_public_controls_complete.json',{'lock_sha256':sha(lockpath),'rows':len(rows),'evaluated_unix':time.time()})
 print('evaluated',len(rows))
if __name__=='__main__':main()
