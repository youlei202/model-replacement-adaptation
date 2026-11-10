"""Exact full-vocabulary last-position outputs on canonical common token sequences."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/runs/v2/rich_behavior'
sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import MODELS,digest,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-index',type=int,required=True);p.add_argument('--role',choices=['pilot','fit','validation','test'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--cache',default='/tmp/model-replacement-adaptation/hf_cache');p.add_argument('--batch-size',type=int,default=1);a=p.parse_args()
 torch.set_num_threads(4)
 collector_code_at_start=digest(Path(__file__))
 mid,rev,role=MODELS[a.model_index];snap=Path(a.cache)/('models--'+mid.replace('/','--'))/'snapshots'/rev
 cohort_path=OUT/'documents.parquet';cohort_hash=digest(cohort_path);cohort=pd.read_parquet(cohort_path)
 if a.role=='test':
  design=json.load(open(OUT/'selection_lock.json')); keep=design['test_item_ids']; rows=cohort.set_index('item_id').loc[keep].reset_index()
 else: rows=cohort[cohort.role.eq(a.role)].reset_index(drop=True)
 if not len(rows):raise ValueError('Empty cohort')
 lock=json.load(open(OUT/'data_lock.json'))
 assert lock['documents_sha256']==cohort_hash
 if a.role=='test':
  design=json.load(open(OUT/'selection_lock.json'));assert design['all_choices_frozen']
  resource_lock=json.load(open(OUT/'resources/operating_points_lock.json'));assert resource_lock['all_resource_choices_frozen']
  assert design['test_item_ids']==rows.item_id.tolist()
 path=OUT/'logits'/a.role/f'model_{a.model_index}.npy';path.parent.mkdir(parents=True,exist_ok=True)
 meta_path=path.with_suffix('.json');n=len(rows)
 if meta_path.exists():
  old=json.load(open(meta_path))
  if old.get('complete'):
   assert old['cohort_sha256']==cohort_hash and old['model_revision']==rev and old['sha256']==digest(path)
   print('already complete',path,flush=True);return
 start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True).to(a.device).eval();tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
 assert {x.dtype for x in model.parameters() if x.is_floating_point()}=={torch.bfloat16}
 vocab=model.get_output_embeddings().weight.shape[0]
 assert vocab==151936
 mmap=np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype=np.float32,shape=(n,vocab));errors=[];timings=[];official_checks=[]
 for s in range(0,n,a.batch_size):
  batch=rows.iloc[s:s+a.batch_size];lens=[len(x) for x in batch.context_ids];width=max(lens)
  ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long,device=a.device);mask=torch.zeros_like(ids)
  for i,seq in enumerate(batch.context_ids):ids[i,-len(seq):]=torch.tensor(seq.astype(np.int64),device=a.device);mask[i,-len(seq):]=1
  torch.cuda.synchronize();tic=time.perf_counter()
  with torch.inference_mode():
   logits=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
  torch.cuda.synchronize();elapsed=time.perf_counter()-tic;timings.append({'start':s,'batch':len(batch),'tokens':width,'seconds':elapsed})
  if not torch.isfinite(logits).all():raise ValueError('Nonfinite full logits')
  centered=logits-logits.mean(dim=1,keepdim=True)
  if a.role=='pilot':
   tv=.5*(torch.softmax(centered,dim=-1)-torch.softmax(centered.half().float(),dim=-1)).abs().sum(-1)
   errors.extend(tv.cpu().tolist())
   # Independent official no-logits-slicing check, at most one short item/model.
   if s==0:
    one=ids[:1,-min(width,64):]
    with torch.inference_mode():
     full=model(input_ids=one,use_cache=False).logits[:,-1,:].float()
     selected=model(input_ids=one,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
    official_checks.append(float((full-selected).abs().max().cpu()))
  mmap[s:s+len(batch)]=centered.cpu().numpy();mmap.flush()
  dump(meta_path,{'complete':False,'completed':s+len(batch),'expected':n,'model_id':mid,'model_revision':rev,'cohort_sha256':cohort_hash,'vocab':vocab,'role':a.role})
  print(mid,a.role,s+len(batch),'/',n,round(elapsed,3),flush=True)
 del mmap;path.with_suffix('.partial.npy').replace(path)
 dump(meta_path,{'complete':True,'completed':n,'expected':n,'model_id':mid,'model_revision':rev,'cohort_sha256':cohort_hash,'vocab':vocab,'role':a.role,'sha256':digest(path),'inference_microbatch':a.batch_size,'storage_dtype':'float32_centered_logits','collector_code_at_start_sha256':collector_code_at_start,'collector_file_sha256_at_finalization':digest(Path(__file__)),'fp16_storage_error_tv':{'mean':float(np.mean(errors)),'max':float(np.max(errors))} if errors else None,'official_last_position_max_difference':official_checks,'seconds_including_load':time.perf_counter()-start,'batch_timings':timings,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':__import__('transformers').__version__,'device':torch.cuda.get_device_name(),'dtype':'bfloat16'},'memory_peak_allocated':torch.cuda.max_memory_allocated(),'memory_peak_reserved':torch.cuda.max_memory_reserved(),'output_weight_shape':list(model.get_output_embeddings().weight.shape),'known_unused_output_token_ids':'151669..151935 are shared output slots; all151936 positions preserved, no intersection/renormalization'})
if __name__=='__main__':main()
