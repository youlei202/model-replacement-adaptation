#!/usr/bin/env python3
"""Explicit ledgers for every family/interface/cap, including infeasible rows."""
import json
from pathlib import Path
import pandas as pd
from phase_a_run import ROOT,FAMILIES,CAPS,MODELS,atomic_json

def main():
 out=ROOT/'experiments/runs/v2/phase_a';rows=[];status=[];total=0;fits=0
 for path in sorted((out/'cells').glob('seed_*/*/target_*/q_*/pairs/*/pair_selections.json')):
  fits+=1
  ledger=pd.read_parquet(path.with_name('validation_candidates.parquet'),columns=['family','status','actual_bits'])
  d=json.loads(path.read_text());parts=path.relative_to(out/'cells').parts
  meta={'seed':int(parts[0].removeprefix('seed_')),'task_id':parts[1],'target_global_id':int(parts[2].removeprefix('target_')),'q':int(parts[3].removeprefix('q_')),'pair':parts[5]}
  meta['target_id']=MODELS[meta['target_global_id']];lookup={(r['family'],r['budget_bits']):r for r in d}
  for family in FAMILIES:
   for budget in CAPS:
    r=lookup.get((family,budget));rows.append(meta|{'family':family,'budget_bits':budget,'status':'feasible' if r else 'infeasible_no_encoded_candidate_under_cap','minimum_possible_observed_state_bits':int(ledger.loc[(ledger.family==family)&(ledger.status=='eligible'),'actual_bits'].min()),'selected_state_digest':r['state_digest'] if r else None,'selected_actual_bits':r['actual_bits'] if r else None})
  total+=len(ledger)
  for (family,st),n in ledger.groupby(['family','status']).size().items():status.append(meta|{'family':family,'status':st,'candidate_count':int(n)})
 pd.DataFrame(rows).to_csv(out/'candidate_feasibility.csv',index=False);pd.DataFrame(status).to_csv(out/'candidate_status_counts.csv',index=False)
 report={'candidate_ledger_rows':total,'target_interface_fits':fits,'family_interface_cap_rows':len(rows),'infeasible_family_interface_caps':sum(r['status']!='feasible' for r in rows),'failed_candidate_encodings':sum(r['candidate_count'] for r in status if r['status']!='eligible'),'all_infeasible_rows_retained':True}
 atomic_json(out/'candidate_accounting.json',report);print(json.dumps(report))
if __name__=='__main__':main()
