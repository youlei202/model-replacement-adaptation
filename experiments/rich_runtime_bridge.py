"""Numerical bridge from batched collection to natural single-context deployment."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump

def main():
 torch.set_num_threads(4);docs=pd.read_parquet(OUT/'documents.parquet');docs=docs[docs.role.eq('validation')].reset_index(drop=True);rows=[]
 for i,(mid,rev,role) in enumerate(MODELS):
  model=AutoModelForCausalLM.from_pretrained('/tmp/model-replacement-adaptation/hf_cache/models--'+mid.replace('/','--')+'/snapshots/'+rev,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0').eval();cached=np.load(OUT/'logits/validation'/f'model_{i}.npy',mmap_mode='r')
  for j in range(4):
   ids=torch.as_tensor(np.asarray(docs.iloc[j].context_ids).copy(),device='cuda:0').reshape(1,-1)
   with torch.inference_mode():x=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[:,-1,:].float();x=x-x.mean(-1,keepdim=True)
   old=torch.as_tensor(np.asarray(cached[j]).copy(),device='cuda:0');p=torch.softmax(x,dim=-1);q=torch.softmax(old,dim=-1);rows.append({'model_index':i,'item_id':docs.iloc[j].item_id,'full_vocabulary_TV':float(.5*(p-q).abs().sum().cpu()),'max_probability_difference':float((p-q).abs().max().cpu()),'max_centered_logit_difference':float((x-old).abs().max().cpu())})
  del model;torch.cuda.empty_cache()
 report={'passed':all(r['full_vocabulary_TV']<=.002 for r in rows),'tolerance_full_vocabulary_TV':.002,'comparison':'BF16 canonicalmicrobatch1 cached vsBF16 batch1 live onidenticalcanonicalcontext, sameB200/runtime/checkpoints','records':rows,'roles':'4validationdocuments permodel, nofinaltest','code_sha256':digest(Path(__file__))};dump(OUT/'collection_deployment_bridge.json',report);print(json.dumps(report),flush=True)
 if not report['passed']:raise RuntimeError('Batchshape numericalbridgefailed; donotmixcachedqualityandlivepredictorclaims')
if __name__=='__main__':main()
