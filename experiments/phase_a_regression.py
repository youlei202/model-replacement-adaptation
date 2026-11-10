#!/usr/bin/env python3
"""Read-only regression against all first-benchmarks selected legacy candidates."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from phase_a_run import MODELS,ROOT,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/runs/v2/phase_a');p.add_argument('--allow-partial',action='store_true');a=p.parse_args();rows=[];missing=[]
 for oldpath in sorted((ROOT/'benchmarks/results_complete').glob('seed_*/*/*/selection_freeze.json')):
  old=json.loads(oldpath.read_text());target=MODELS.index(old['target'])
  for pair in [(0,1),(0,2),(1,2),(0,1,2)]:
   newpath=a.output/'cells'/f"seed_{old['seed']}"/old['task']/f'target_{target}'/f'q_{len(pair)}'/'pairs'/'-'.join(map(str,pair))/'pair_selections.json'
   if not newpath.exists():missing.append(str(newpath));continue
   new=json.loads(newpath.read_text());index={(r['family'],r['budget_bits']):r for r in new}
   for r in old['selections']:
    if r['selection_kind']!='pair_family' or tuple(r['pair'])!=pair or r['budget_bits'] not in(-1,1024,2048,8192,16384):continue
    n=index[r['family'],r['budget_bits']]
    rows.append({'seed':old['seed'],'task':old['task'],'target':target,'pair':str(pair),'family':r['family'],'budget_bits':r['budget_bits'],'old_validation_tv':r['validation_tv'],'new_validation_tv':n['validation_tv'],'absolute_difference':abs(r['validation_tv']-n['validation_tv']),'state_digest_equal':r['state_digest']==n['state_digest'],'actual_bits_equal':r['actual_bits']==n['actual_bits']})
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'legacy_regression.csv',index=False)
 report={'compared':len(rows),'missing_pairs':len(missing),'max_validation_difference':float(frame.absolute_difference.max()) if len(frame) else None,'matching_state_digests':int(frame.state_digest_equal.sum()) if len(frame) else 0,'matching_actual_bits':int(frame.actual_bits_equal.sum()) if len(frame) else 0,'passed':bool(len(frame) and frame.absolute_difference.max()<1e-10 and frame.actual_bits_equal.all() and (a.allow_partial or not missing)),'first_benchmarks_modified':False}
 atomic_json(a.output/'legacy_regression.json',report);print(json.dumps(report))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
