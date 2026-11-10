"""Collect fresh external confirmation responses with sealed-test gating.

Uses the unchanged original revision-pinned scorer and coordinates. No historical
response values are reused. Each stage is a separate immutable-prompt bank.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'reference_implementation';sys.path.insert(0,str(BUNDLE));sys.path.insert(0,str(ROOT))
from kernel_isqed.response import scores_to_coordinates,coordinates_to_probabilities,helmert_matrix
from benchmarks.collect_responses import atomic_json,checkpoint

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(scorer,row):
 count=int(row.contrast_dim)+1;result=scorer.score(row.system_prompt,row.user_prompt,[str(i) for i in range(count)])
 if {len_ for len_ in result.code_token_lengths}!={1}:raise RuntimeError('Answer-code contract failed')
 semantic=np.empty(count);semantic[np.asarray(json.loads(row.permutation),int)]=result.log_scores
 return scores_to_coordinates(semantic)[0]

def numerical_bridge(scorer,model):
 """Predeclared old-fit rows; diagnostic only, never reused in fresh cohort."""
 old=pd.read_parquet(ROOT/'benchmarks/response_cache/recompute_all'/(model.replace('/','--')+'.parquet'))
 mf=pd.read_parquet(ROOT/'benchmarks/splits/split_seed_20261109.parquet');fit=mf[mf.split=='fit'][['task_id','item_id']]
 p=pd.read_parquet(BUNDLE/'data/canonical/prompt_manifest.parquet')
 d=fit.merge(old,on=['task_id','item_id']).merge(p[['task_id','item_id','system_prompt','user_prompt','permutation']],on=['task_id','item_id']);d['order']=d.item_id.map(lambda x:hashlib.sha256(('confirmation_bridge_v1|'+x).encode()).hexdigest());d=d.sort_values('order').groupby('task_id').head(4);d['contrast_dim']=d.task_id.map({'mmlu_pro':9,'hellaswag':3,'rewardbench':1})
 records=[]
 for row in d.itertuples(index=False):
  new=score(scorer,row);expected=np.array([getattr(row,f'coord_{i}') for i in range(int(row.contrast_dim))]);p1,p2=coordinates_to_probabilities(new)[0],coordinates_to_probabilities(expected)[0]
  records.append({'task_id':row.task_id,'item_id':row.item_id,'centered_logit_max_difference':float(np.max(np.abs((new-expected)@helmert_matrix(len(new)+1)))),'probability_max_difference':float(np.max(np.abs(p1-p2))),'TV':float(np.abs(p1-p2).sum()/2)})
 return {'passed':all(r['centered_logit_max_difference']<=.02 and r['probability_max_difference']<=.002 for r in records),'rows':records,'policy':'Diagnostic cross-phase bridge only; fresh cohort always uniformly recollected'}

def main():
 a=argparse.ArgumentParser();a.add_argument('--model-index',type=int,required=True);a.add_argument('--stage',choices=['fitval','test'],required=True);a.add_argument('--data-dir',type=Path,default=ROOT/'experiments/runs/v2/confirmation');a.add_argument('--device',default='cuda:0');a.add_argument('--max-items',type=int,default=0);args=a.parse_args()
 config=json.loads((ROOT/'experiments/experiment_config/configs/campaign.json').read_text());reg=config['model_registry'][args.model_index];model,revision=reg['model_id'],reg['revision'];name=model.replace('/','--');base=args.data_dir
 lock=base/'data_lock.json'
 if not lock.exists():raise RuntimeError('Frozen cohort data lock required')
 data_lock=json.loads(lock.read_text());manifest=base/f'required_items_{args.stage}.parquet'
 if digest(manifest)!=data_lock['input_hashes'][manifest.name]:raise RuntimeError('Input manifest differs from frozen data lock')
 if args.stage=='test':
  confirmation_lock=base/'confirmation_lock.json'
  if not confirmation_lock.exists():raise RuntimeError('Test collection is prohibited until all states and analysis choices frozen')
  final=json.loads(confirmation_lock.read_text())
  if not final.get('all_fits_and_policies_frozen') or not final.get('resource_operating_points_frozen') or final['data_lock_sha256']!=digest(lock):raise RuntimeError('Confirmation method/resource lock incomplete')
  for rel,expected in final['frozen_artifact_hashes'].items():
   if digest(base/rel)!=expected:raise RuntimeError(f'Frozen method artifact changed:{rel}')
 required=pd.read_parquet(manifest);fitval=pd.read_parquet(base/'required_items_fitval.parquet');keys=['task_id','item_id']
 if required.duplicated(keys).any():raise RuntimeError('Duplicate required keys')
 folder=base/f'response_cache_{args.stage}';folder.mkdir(exist_ok=True);path=folder/(name+'.parquet');provpath=folder/(name+'.provenance.json');lockpath=folder/(name+'.lock')
 code={'collector':digest(__file__),'scorer':digest(BUNDLE/'kernel_isqed/llm.py'),'coordinates':digest(BUNDLE/'kernel_isqed/response.py'),'checkpoint_helper':digest(ROOT/'benchmarks/collect_responses.py')}
 existing=pd.read_parquet(path) if path.exists() else pd.DataFrame();prior=json.loads(provpath.read_text()) if provpath.exists() else None
 if len(existing):
  if not prior or prior['code_hashes']!=code or prior['manifest_sha256']!=digest(manifest) or prior['revision']!=revision:raise RuntimeError('Checkpoint identity mismatch')
  if existing.duplicated(keys).any():raise RuntimeError('Duplicate checkpoint keys')
  joined=required[keys+['prompt_hash']].merge(existing[keys+['prompt_hash']],on=keys,suffixes=['_required','_cached'],how='inner')
  assert joined.prompt_hash_required.eq(joined.prompt_hash_cached).all()
  completed=set(map(tuple,existing[keys].to_numpy()))
 else:completed=set()
 pending=required[[x not in completed for x in required[keys].itertuples(index=False,name=None)]].sort_values(keys)
 if len(pending)==0:
  assert prior and prior['completed'];print('already complete',model,args.stage);return
 fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode())
 try:
  import torch
  from kernel_isqed.llm import SemanticOptionScorer
  torch.manual_seed(20260722);torch.set_num_threads(4)
  runtime={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','pandas','huggingface-hub','tokenizers']};runtime.update(python=sys.version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(torch.device(args.device)))
  if prior and prior['runtime']!=runtime:raise RuntimeError('Refusing mixed runtimes')
  start=time.monotonic();scorer=SemanticOptionScorer(model,revision,args.device);load_s=time.monotonic()-start
  checks=[]
  for row in fitval.sort_values('item_id').groupby('task_id').head(4).itertuples(index=False):
   check=scorer.compare_official_logits(row.system_prompt,row.user_prompt,[str(i) for i in range(int(row.contrast_dim)+1)]);checks.append({'task_id':row.task_id,'item_id':row.item_id,**check})
  passed=all(c['centered_logit_max_difference']<=.02 and c['probability_max_difference']<=.002 for c in checks)
  bridge=numerical_bridge(scorer,model) if args.stage=='fitval' and not prior else (prior.get('historical_runtime_bridge') if prior else None)
  provenance={'model_id':model,'revision':revision,'stage':args.stage,'manifest_sha256':digest(manifest),'data_lock_sha256':digest(lock),'code_hashes':code,'runtime':runtime,'required_rows':len(required),'completed':False,'model_load_seconds':load_s,'official_forward_check':{'passed':passed,'rows':checks},'historical_runtime_bridge':bridge}
  atomic_json(provpath,provenance)
  if not passed:raise RuntimeError('Official forward comparison failed')
  if args.max_items:pending=pending.head(args.max_items)
  rows=[];start=time.monotonic()
  for i,row in enumerate(pending.itertuples(index=False),1):
   y=score(scorer,row)
   if not np.isfinite(y).all():raise RuntimeError('Nonfinite coordinates')
   item={'model_id':model,'revision':revision,'task_id':row.task_id,'item_id':row.item_id,'prompt_hash':row.prompt_hash,'item_cluster':row.item_cluster,'domain':row.domain,'gold':int(row.gold),'split':args.stage,'contrast_dim':int(row.contrast_dim),'response_source':'genuinely_new_external_confirmation_v1'}
   item.update({f'coord_{j}':float(y[j]) if j<len(y) else np.nan for j in range(9)});rows.append(item)
   if len(rows)>=50:
    existing=checkpoint(path,existing,rows);rows=[];print(f'{model} {args.stage} {len(existing)}/{len(required)} elapsed{time.monotonic()-start:.1f}s',flush=True)
  if rows:existing=checkpoint(path,existing,rows)
  provenance.update(completed=len(existing)==len(required),completed_rows=len(existing),collection_seconds=time.monotonic()-start,response_bank_sha256=digest(path));atomic_json(provpath,provenance);print(json.dumps(provenance),flush=True)
 finally:os.close(fd);lockpath.unlink()
if __name__=='__main__':main()
