"""Six-model tokenizer intersection using original frozen chat rendering."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'reference_implementation'))
from kernel_isqed.llm import SemanticOptionScorer,installed_transformers_commit
from kernel_isqed.contracts import TRANSFORMERS_COMMIT
from transformers import AutoConfig,AutoTokenizer,AutoProcessor
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--model-index',type=int);a.add_argument('--finalize',action='store_true');a.add_argument('--tasks',nargs='+');args=a.parse_args()
 out=ROOT/'experiments/runs/v2/confirmation';config=json.loads((ROOT/'experiments/experiment_config/configs/campaign.json').read_text());registry=config['model_registry']
 inputs=sorted(out.glob('*/eligible_before_tokenizers.parquet'))
 if args.tasks:inputs=[p for p in inputs if p.parent.name in args.tasks]
 if not inputs:raise RuntimeError('No fresh populations ready')
 rows=pd.concat([pd.read_parquet(p) for p in inputs],ignore_index=True)
 if args.finalize:
  pieces=[pd.read_parquet(out/f'tokenizers/model_{r["index"]}.parquet') for r in registry];allrows=pd.concat(pieces,ignore_index=True)
  for p in inputs:
   task=p.parent.name;d=pd.read_parquet(p);sub=allrows[allrows.task_id==task]
   counts=sub.groupby('item_id').model_id.nunique();assert len(counts)==len(d) and counts.eq(6).all()
   good=sub.groupby('item_id').within_limit.all();d['within_tokenizer_intersection']=d.item_id.map(good);d['maximum_rendered_tokens']=d.item_id.map(sub.groupby('item_id').rendered_tokens.max())
   d=d[d.within_tokenizer_intersection].copy();d.to_parquet(p.parent/'eligible_population.parquet',index=False)
   report={'task_id':task,'before_tokenizers':len(counts),'eligible_clusters':len(d),'max_rendered_tokens':int(d.maximum_rendered_tokens.max()),'models':registry,'eligibility_file_sha256':sha(p.parent/'eligible_population.parquet'),'source_freshness_audit_sha256':sha(p.parent/'freshness_audit.json')};(p.parent/'population_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
  return
 if args.model_index is None:raise ValueError('model-index required')
 assert installed_transformers_commit()==TRANSFORMERS_COMMIT
 r=registry[args.model_index];model=r['model_id'];revision=r['revision'];c=AutoConfig.from_pretrained(model,revision=revision,trust_remote_code=True)
 kwargs={'revision':revision,'trust_remote_code':True}
 if model.startswith('mistralai/'):kwargs['fix_mistral_regex']=True
 if c.model_type in {'gemma4','gemma4_unified','mistral3','qwen3_5'}:engine=AutoProcessor.from_pretrained(model,**kwargs).tokenizer
 else:engine=AutoTokenizer.from_pretrained(model,**kwargs)
 # Stub uses exactly the scorer's frozen rendering/candidate-ID implementation,
 # without loading model weights or producing model outputs.
 stub=SemanticOptionScorer.__new__(SemanticOptionScorer);stub.model_id=model;stub.model_type=c.model_type;stub.tokenizer=engine;stub.device=torch.device('cpu')
 checks=[]
 for task,d in rows.groupby('task_id'):
  for row in d.head(4).itertuples(index=False):
   ids=stub.render(row.system_prompt,row.user_prompt);codes=stub._candidate_ids(row.system_prompt,row.user_prompt,[str(i) for i in range(int(row.contrast_dim)+1)],ids)
   if {len(v) for v in codes}!={1}:raise RuntimeError(f'{model} {task} answer codes not single-token')
   checks.append({'task_id':task,'item_id':row.item_id,'code_ids':codes})
 results=[];started=time.monotonic()
 for i,row in enumerate(rows.itertuples(index=False),1):
  length=len(stub._template_ids([{'role':'system','content':row.system_prompt},{'role':'user','content':row.user_prompt}],add_generation_prompt=True,tensors=False))
  results.append({'model_id':model,'revision':revision,'task_id':row.task_id,'item_id':row.item_id,'prompt_hash':row.prompt_hash,'rendered_tokens':length,'within_limit':length<=4096})
  if i%2000==0:print(model,i,len(rows),time.monotonic()-started,flush=True)
 folder=out/'tokenizers';folder.mkdir(exist_ok=True);path=folder/f'model_{args.model_index}.parquet';pd.DataFrame(results).to_parquet(path,index=False)
 (folder/f'model_{args.model_index}.json').write_text(json.dumps({'model':r,'rows':len(rows),'source_files':{str(p.relative_to(out)):sha(p) for p in inputs},'elapsed_seconds':time.monotonic()-started,'official_template_code_checks':checks,'all_collected_rows_code_check':'Collector independently checks every scored row before retaining it','file_sha256':sha(path)},indent=2)+'\n')
if __name__=='__main__':main()
