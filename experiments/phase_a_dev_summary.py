#!/usr/bin/env python3
"""Validation-only follow-up ranking; reads frozen decisions, never test files."""
import argparse,hashlib,itertools,json
from pathlib import Path
import pandas as pd
from phase_a_run import PRIMARY,POLICIES,MODELS,sha,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs/v2/phase_a');a=p.parse_args()
 globalfreeze=json.loads((a.output/'all_selections_frozen.json').read_text());rows=[];contrasts=[]
 for rel,digest in globalfreeze['freeze_hashes'].items():
  path=a.output/rel
  if sha(path)!=digest:raise ValueError('Frozen decision drift')
  freeze=json.loads(path.read_text());meta={k:freeze[k] for k in ('seed','task','target','target_id','q')}
  for r in freeze['selections']:
   if r['selection_kind']=='policy':rows.append(meta|r)
 frame=pd.DataFrame(rows)
 frame.to_json(a.output/'validation_policy_decisions.jsonl',orient='records',lines=True)
 for key,g in frame[(frame.q==2)&(frame.scope=='strengthened')&frame.budget_bits.isin(PRIMARY)].groupby(['seed','task','target','target_id','budget_bits']):
  d={r['policy']:r for r in g.to_dict('records')};joint=d['budget_aware']
  for comp in POLICIES:
   r=d[comp];contrasts.append(dict(zip(['seed','task_id','target_global_id','target_id','budget_bits'],key))|{'comparator':comp,'validation_gain':r['validation_tv']-joint['validation_tv'],'joint_validation_tv':joint['validation_tv'],'fixed_validation_tv':r['validation_tv'],'joint_state_digest':joint['state_digest'],'fixed_state_digest':r['state_digest'],'joint_pair':joint['pair'],'fixed_pair':r['pair']})
 c=pd.DataFrame(contrasts);c.to_csv(a.output/'validation_policy_gains.csv',index=False)
 rank=c[c.comparator=='fixed_mean_budget'].groupby(['target_global_id','target_id','task_id'],as_index=False).validation_gain.mean().sort_values(['validation_gain','target_id','task_id'],ascending=[False,True,True])
 rank['selected_for_confirmation']=False;seen=set()
 for ix,r in rank.iterrows():
  if r.target_id not in seen and len(seen)<2:rank.loc[ix,'selected_for_confirmation']=True;seen.add(r.target_id)
 rank.to_csv(a.output/'followup_validation_ranking.csv',index=False)
 atomic_json(a.output/'followup_selection_lock.json',{'selection_rule':'Top two distinct target/task cells by mean validation gain vs fixed_mean_budget across all5seeds and4primarycaps; strengthened menu,q2','all_selections_frozen_sha256':sha(a.output/'all_selections_frozen.json'),'ranking_sha256':sha(a.output/'followup_validation_ranking.csv'),'selected':rank[rank.selected_for_confirmation].to_dict('records'),'test_used_for_selection':False})
 print(rank.to_string(index=False))
if __name__=='__main__':main()
