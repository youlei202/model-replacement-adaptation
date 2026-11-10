"""Export both accounting scenarios without treating shared infrastructure as free."""
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,digest,dump

def main():
 resource=OUT/'resources';assert (resource/'benchmark_complete.json').exists();timing=pd.read_csv(resource/'timing_raw.csv');supp=json.load(open(resource/'resident_state_supplement.json'));mem={r['state_digest']:r for r in supp['measurements']};rows=[]
 for scope in ['conditional_existing_peers','standalone_total_footprint']:
  frame=timing.copy();frame['resource_scope']=scope;frame['resident_target_state_numeric_payload_bytes']=frame.resident_state_bytes;frame['resident_target_decoded_python_deep_bytes_including_referenced_metadata']=frame.system_digest.map(lambda s:mem[s]['decoded_python_object_deep_bytes_including_referenced_metadata']);frame['resident_target_gpu_allocator_bytes']=frame.system_digest.map(lambda s:mem[s].get('gpu_state_tensor_allocator_bytes'))
  frame['peer_infrastructure_included_in_scope']=scope=='standalone_total_footprint';frame['conditional_shared_infrastructure_is_reported_not_free']=True;frame['complete_process_RSS_and_VRAM_measured_in_both_scopes']=True;frame['resource_state_cap_is_serialized_bytes_not_total_object_RAM']=True;frame['canonical_quality_population_link']=frame.context_tokens.eq(0);rows.append(frame)
 result=pd.concat(rows,ignore_index=True);result.to_parquet(resource/'timing_by_accounting_scope.parquet',index=False);dump(resource/'accounting_scope_manifest.json',{'source_timing_sha256':digest(resource/'timing_raw.csv'),'source_resident_audit_sha256':digest(resource/'resident_state_supplement.json'),'rows':len(result),'scopes':['conditional_existing_peers','standalone_total_footprint'],'same_measurements_reexpressed_not_twoindependentrepetitions':True,'numeric_resident_old_field':'resident_state_bytes isnumericarraypayload, not totalPython/CUDAallocation','basis_accounting':'actualminimalnpyserializedsize andresidentbasis separately; inactiveexperimentbasisfiles excludedwithproofruntimeusesnoneforprimarycheapstates','runtime_accounting':'wholeprocessRSS andCUDAallocated/reserved includeloadedruntime; OS/container/runtimeinstallation disk isnot includedinmodel/statebundlebytecolumns','posthoc_export_only_no_reselection':True});print('Exported',len(result),'rows across2scopes')
if __name__=='__main__':main()
