"""Supplemental measured resident-state accounting; no model/state selection changes."""
import argparse,gc,json,sys,tracemalloc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,digest,dump
from rich_codec import decode

def deep_size(obj,seen):
 if id(obj) in seen:return 0
 seen.add(id(obj));size=sys.getsizeof(obj)
 if isinstance(obj,dict):size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
 elif isinstance(obj,(tuple,list)):size+=sum(deep_size(v,seen) for v in obj)
 elif isinstance(obj,np.ndarray) and obj.base is not None:size+=deep_size(obj.base,seen)
 return size

def main():
 p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args();lock=json.load(open(OUT/'resources/operating_points_lock.json'));selected={r['candidate_id']:r for r in lock['state_cap_sweep_points'] if r['feasible']};records=[]
 for sid,row in selected.items():
  blob=(OUT/row['state_path']).read_bytes();samples=[];peaks=[]
  for _ in range(20):
   gc.collect();tracemalloc.start();before,_=tracemalloc.get_traced_memory();state=decode(blob);after,peak=tracemalloc.get_traced_memory();samples.append(after-before);peaks.append(peak-before);tracemalloc.stop()
  rec={'state_digest':sid,'serialized_literal_bytes':len(blob),'decoded_numeric_payload_bytes':sum(x.nbytes for x in state['arrays']),'decoded_python_object_deep_bytes_including_referenced_metadata':deep_size(state,set()),'serialized_python_bytes_object_bytes':sys.getsizeof(blob),'traced_cpu_decode_net_allocation_median_bytes':float(np.median(samples)),'traced_cpu_decode_peak_allocation_median_bytes':float(np.median(peaks)),'qualifier':'deep size includesreferencedsharedmetadata, trace excludesuntrackednativeallocations; neitherisRSS; realwholepredictorRSS/VRAM intimingledger'}
  if a.cuda:
   import torch
   torch.cuda.empty_cache();torch.cuda.synchronize();before=torch.cuda.memory_allocated();tensors=[torch.as_tensor(x,device='cuda:0') for x in state['arrays']];torch.cuda.synchronize();after=torch.cuda.memory_allocated();rec.update({'gpu_state_tensor_numeric_bytes':sum(t.numel()*t.element_size() for t in tensors),'gpu_state_tensor_allocator_bytes':after-before,'GPU':torch.cuda.get_device_name(),'torch':torch.__version__});del tensors;torch.cuda.empty_cache()
  records.append(rec)
 dump(OUT/'resources/resident_state_supplement.json',{'operating_points_lock_sha256':digest(OUT/'resources/operating_points_lock.json'),'measurements':records,'posthoc_accounting_only_no_reselection':True,'legacy_field_correction':'resourceledger resident_state_bytes equalsnumericpayloadonly; useexpandedmetricsforactualobject/allocatorfootprint; fullRSS/VRAM alreadymeasured','code_sha256':digest(Path(__file__))});print(json.dumps(records,indent=2))
if __name__=='__main__':main()
