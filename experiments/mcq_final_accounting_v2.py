#!/usr/bin/env python3
"""Final accounting-only export; inclusive runtime is never added to nested time."""
import hashlib,json,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((BASE/'complete.json').read_text())['complete'];assert json.loads((BASE/'probability_endpoint_audit/complete.json').read_text())['complete']
 out=BASE/'final_compute_accounting';out.mkdir(exist_ok=True)
 if (out/'manifest.json').exists():raise RuntimeError('Preserve completed accounting export')
 rows=[];sources={}
 for subdir in ['resources_startup_failure_prompt_role_filter','resources']:
  for path in sorted((BASE/subdir).glob('queue_context*_commands.jsonl')):
   sources[str(path.relative_to(BASE))]=sha(path)
   for r in map(json.loads,path.read_text().splitlines()):
    raw=BASE/subdir/f"batch{r['batch_size']}_context{r['context_tokens']}/raw_blocks.jsonl";blocks=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()] if raw.exists() else [];good=[b for b in blocks if not b.get('failure_status')]
    if raw.exists():sources[str(raw.relative_to(BASE))]=sha(raw)
    rows.append({'operation':'coordinate_endpoint_measurement' if r['returncode']==0 else 'preserved_startup_failure','GPU':str(r['gpu']),'batch_size':r['batch_size'],'context_tokens':r['context_tokens'],'start_unix':r['start_unix'],'end_unix':r['end_unix'],'inclusive_command_wall_seconds':r['end_unix']-r['start_unix'],'returncode':r['returncode'],'timed_blocks':len(good),'failed_timed_blocks':len(blocks)-len(good),'timed_requests':sum(len(b['traces']) for b in good),'timed_contexts':sum(t['context_count'] for b in good for t in b['traces']),'planned_warm_contexts_for_completed_blocks':sum(b['warmup_requests']*b['batch_size'] for b in good),'planned_warm_requests_for_completed_blocks':sum(b['warmup_requests'] for b in good),'source':str(path.relative_to(BASE))})
 path=BASE/'probability_endpoint_audit/execution_command.json';r=json.loads(path.read_text());sources[str(path.relative_to(BASE))]=sha(path);raw=BASE/'probability_endpoint_audit/measurement/raw_blocks.jsonl';blocks=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()];good=[b for b in blocks if not b.get('failure_status')];sources[str(raw.relative_to(BASE))]=sha(raw)
 rows.append({'operation':'probability_materialization_endpoint_audit','GPU':str(r['GPU']),'batch_size':1,'context_tokens':0,'start_unix':r['start_unix'],'end_unix':r['end_unix'],'inclusive_command_wall_seconds':r['end_unix']-r['start_unix'],'returncode':r['returncode'],'timed_blocks':len(good),'failed_timed_blocks':len(blocks)-len(good),'timed_requests':sum(len(b['traces']) for b in good),'timed_contexts':sum(t['context_count'] for b in good for t in b['traces']),'planned_warm_contexts_for_completed_blocks':sum(b['warmup_requests']*b['batch_size'] for b in good),'planned_warm_requests_for_completed_blocks':sum(b['warmup_requests'] for b in good),'source':str(path.relative_to(BASE))})
 table=pd.DataFrame(rows);assert len(table)==9 and sum(table.returncode!=0)==2;assert table.timed_requests.sum()==19400 and table.timed_contexts.sum()==74000
 table['accounting_scope']='Inclusive process wall: model loads, startup replay, warm/timed requests, offload/activation and cleanup. Queue waiting before subprocess launch excluded. Nested trace durations must not be added to this value.'
 table.to_csv(out/'measured_resource_execution_costs.csv',index=False)
 overlaps=[]
 for gpu,part in table.groupby('GPU'):
  spans=sorted(zip(part.start_unix,part.end_unix));last=None;union=0.
  for start,end in spans:
   if last is None:last=[start,end]
   elif start<=last[1]:last[1]=max(last[1],end)
   else:union+=last[1]-last[0];last=[start,end]
  if last:union+=last[1]-last[0]
  overlaps.append({'GPU':gpu,'observed_controller_interval_union_seconds':union,'not_hardware_active_time':True})
 record={'created_unix':time.time(),'scope':'Posthoc accounting only; no method/output/selection changes','completed_timed_requests':19400,'completed_timed_context_deliveries':74000,'request_and_sample_scope':'A timed request can be a batch of8 contexts. Context deliveries repeat frozen cohort keys and are not additional independent quality samples.','coordinate_timed_requests':15600,'probability_timed_requests':3800,'failed_startups':2,'per_GPU_process_interval_unions':overlaps,'existing_metadata_correction':{'source':'compute_accounting_manifest.json:extra_live_benchmark_costs','old_text':'raw_blocks.jsonl and resource_summary.csv include200timed+20warm perblock','correct_text':'Each block has20timed+20warm;10blocks produce200timed+200warm requests per system/setting. The earlier wording was a metadata typo; all frozen specs/raw traces have always used20/20×10.','old_artifact_preserved':True},'collection_training':'measured_collection_training_costs.csv remains separate; sixcollection loops overlap and must not be added to campaign elapsed. No new adapter fits in this supplement.','unknown_or_nested_costs':'Hardware-active GPU hours and energy are not inferred from processwall. Request traces and load/activation spans are nested, not extra costs. Collection official-forward-check elapsed was not separately logged.','source_hashes':sources,'source_code_sha256':sha(Path(__file__)),'outputs':{'measured_resource_execution_costs.csv':sha(out/'measured_resource_execution_costs.csv')}}
 (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print('Final measured accounting:',len(rows),'commands,19400timedrequests,2startupfailures')
if __name__=='__main__':main()
