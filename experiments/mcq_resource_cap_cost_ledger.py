#!/usr/bin/env python3
"""Report measured supplemental collection costs without adding overlapping time."""
import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];sources={}
 for p in sorted((BASE/'response_cache_test').glob('*.provenance.json')):
  a=json.loads(p.read_text());assert a['completed'];sources[str(p.relative_to(BASE))]=sha(p);rows.append({'phase':'second_MC Q_resource_cap_confirmation'.replace(' ',''),'operation':'new_unused_query_collection','model_id':a['model_id'],'query_model_evaluations':a['required_rows'],'wall_seconds':a['collection_seconds'],'model_load_seconds':a['model_load_seconds'],'record_kind':'measured_model_collection_loop','diagnostic_forward_checks':len(a['official_forward_check']['rows']),'diagnostic_elapsed_seconds':None,'time_scope':'Collection loop and model load separately; official diagnostic elapsed not logged; CPU/GPU overlap not summed into fictional GPU hours','source':str(p.relative_to(BASE)),'status':'complete'})
 assert len(rows)==6
 rows.append({'phase':'second_MC Q_resource_cap_confirmation'.replace(' ',''),'operation':'adapter_and_public_context_training','model_id':None,'query_model_evaluations':0,'wall_seconds':0,'model_load_seconds':0,'record_kind':'no_new_fit_existing_frozen_states_reused','diagnostic_forward_checks':0,'diagnostic_elapsed_seconds':0,'time_scope':'No refitting or new public basis; original training cost remains in original T7 account','source':'selection_lock.json','status':'complete'})
 pd.DataFrame(rows).to_csv(BASE/'measured_collection_training_costs.csv',index=False);(BASE/'compute_accounting_manifest.json').write_text(json.dumps({'source_sha256':sources,'script_sha256':sha(Path(__file__)),'query_model_evaluations':sum(r['query_model_evaluations'] for r in rows),'runtime_uniformity_guard':'quality_complete/runtime_bridge must pass','parallel_cases':'parallel_collection_case_assignment.json; sixmodel wall loops overlap; do not sum to campaign elapsed or GPUtime','extra_live_benchmark_costs':'raw_blocks.jsonl and resource_summary.csv include200timed+20warm perblock; raw timings cannot estimate all cold/load/controller overhead','official_checks_not_in_query_count':24,'output_sha256':sha(BASE/'measured_collection_training_costs.csv')},indent=2)+'\n');print('Measured collection rows',len(rows))
if __name__=='__main__':main()
