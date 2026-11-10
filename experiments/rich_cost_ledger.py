"""Measured one-time and serving work ledger, retaining failed numerical attempts."""
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,dump,digest

def main():
 rows=[]
 paths=list((OUT/'logits').glob('*/*.json'))+list((OUT/'diagnostics/batch8_numerical_bridge_failure').glob('*/*.json'))
 for p in paths:
  d=json.load(open(p))
  if 'batch_timings' not in d:continue
  rows.append({'stage':'response_collection','artifact':str(p.relative_to(OUT)),'status':'formal_complete' if 'diagnostics' not in p.parts and d['role']!='pilot' else ('development_pilot' if d['role']=='pilot' else 'excluded_failed_numerical_bridge'),'model_id':d['model_id'],'role':d['role'],'model_context_evaluations':d['completed'],'timed_forward_seconds':sum(x['seconds'] for x in d['batch_timings']),'measured_stage_wall_seconds':d['seconds_including_load'],'measurement_scope':'model_load_and_collectionloop; excludesprocessimportandcanonicalcohortreadingbeforetimer','bytes_outputs':p.with_suffix('.npy').stat().st_size})
 for p in (OUT/'public_contexts').glob('*.json'):
  d=json.load(open(p));rows.append({'stage':'peer_public_basis','artifact':str(p.relative_to(OUT)),'status':'complete','measured_stage_wall_seconds':d['seconds'],'measurement_scope':'randomizedSVDandbasiswrite; excludespeerbankread/copybeforefunction','bytes_outputs':d['serialized_bytes']})
 for p in (OUT/'fits').glob('target_*/pair_*/complete.json'):
  d=json.load(open(p));rows.append({'stage':'adapter_candidates','artifact':str(p.relative_to(OUT)),'status':'complete','measured_stage_wall_seconds':d['seconds'],'measurement_scope':'targetread/transfer+fitting/validation/encoding; nofinaltest','candidate_count':d['candidate_count'],'retained_states':d['retained_states']})
 for p in (OUT/'diagnostics/pretest_convex_decoder_bugfix_fits').glob('target_*/pair_*/complete.json'):
  d=json.load(open(p));rows.append({'stage':'adapter_candidates','artifact':str(p.relative_to(OUT)),'status':'superseded_implementation_fix','measured_stage_wall_seconds':d['seconds'],'measurement_scope':'completedearlierfitsretained; interruptedlastpartialfitwallnotrecorded','candidate_count':d['candidate_count'],'retained_states':d['retained_states']})
 frame=pd.DataFrame(rows);reports=OUT/'reports';reports.mkdir(exist_ok=True);frame.to_csv(reports/'measured_training_collection_costs.csv',index=False)
 summary={'measured_collected_model_contexts_including_failedbank':int(frame.model_context_evaluations.sum()),'measured_forward_seconds_collection_including_failedbank':float(frame.timed_forward_seconds.sum()),'sum_scoped_stage_wall_seconds':float(frame.measured_stage_wall_seconds.sum()),'formal_candidate_grid_rows':int(frame[(frame.stage=='adapter_candidates')&(frame.status=='complete')].candidate_count.sum()),'timed_inference_replay_development_profile_states':len(json.load(open(OUT/'resources/development_profiles.json'))),'scope':'stage-scopedmeasurements, notcompletewallruntimeorGPUbusyseconds; stagesmayincludeoverlappingCPU/GPUwork','explicit_unmeasured_items':['datasetdownloadnetworktime andinitialmodeldownloadelapsednotpersistedasnumericmetrics','documenttokenization andnear-duplicateauditCPUtimeexceptprogresslogs','Pythonprocessimport/cohortreadoutsidecollector timer','warmupforwardcallsnotincludedinrawtimedbenchmarkrows','interruptedpartialadapterfitwalltime','end-to-end energy/billing monetarycost'],'finite_amortization_of_joint_selection_savings':'none demonstrated:jointandstrongfixedchooseidenticalsystems atallprimarycaps,sozero measuredpositiveperquerysaving','known_bridge_overhead_contexts':40,'official_forward_check_extra_calls':10,'batch8_failed_bank_preserved':True,'posthoc_accounting_only':True};dump(reports/'cost_accounting_scope.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
