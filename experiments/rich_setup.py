"""Prepare pinned common-tokenizer family and natural document population, no model outcomes."""
import argparse,concurrent.futures,hashlib,json,re,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/runs/v2/rich_behavior'
MODELS=[('Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca','peer'),('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','peer'),('Qwen/Qwen3-4B','1cfa9a7208912126459214e8b04321603b3df60c','peer'),('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218','target'),('Qwen/Qwen3-14B','40c069824f4251a91eefaf281ebe4c544efd3e18','target')]
SOURCES=[{'id':'Salesforce/wikitext','revision':'b08601e04326c79dfdd32d625aee71d232d685c3','files':['wikitext-103-raw-v1/train-00000-of-00002.parquet','wikitext-103-raw-v1/train-00001-of-00002.parquet'],'license':'card YAML: CC-BY-SA-3.0 + GFDL; card body: CC-BY-SA-4.0; preserve original card; do not resolve discrepancy silently','cluster':'entire article delimited by top-level titles'}, {'id':'ccdv/arxiv-summarization','revision':'240aaf1a969b3f8cd0ade6986bfad0cd730ee288','files':['document/train-00000-of-00015.parquet'],'license':'dataset card does not declare blanket license; individual paper rights retained; private research use, do not redistribute raw article text automatically','cluster':'whole article'}]
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2)+'\n')
def download(url,path):
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');urllib.request.urlretrieve(url,tmp);tmp.replace(path)
 return {'path':str(path.relative_to(OUT)),'sha256':digest(path),'bytes':path.stat().st_size}
def setup_sources():
 rows=[]
 for s in SOURCES:
  root=OUT/'source_data'/s['id'].replace('/','--')
  files=s['files']+['README.md']
  with concurrent.futures.ThreadPoolExecutor(3) as ex:
   rec=list(ex.map(lambda fn:download('https://huggingface.co/datasets/'+s['id']+'/resolve/'+s['revision']+'/'+fn,root/fn),files))
  rows.append({**s,'downloaded':rec});print(s['id'],'downloaded',flush=True)
 dump(OUT/'sources.json',rows)
def setup_models(cache):
 from huggingface_hub import snapshot_download
 for mid,rev,role in MODELS:
  print('Download',mid,flush=True)
  p=snapshot_download(mid,revision=rev,cache_dir=cache,allow_patterns=['*.safetensors','*.json','*.txt','*.model','LICENSE','README.md'],max_workers=8)
  print('Ready',p,flush=True)
 dump(OUT/'model_registry.json',[{'global_id':i,'model_id':m,'revision':r,'role':role,'license':'Apache-2.0','cache_path':str(Path(cache)/('models--'+m.replace('/','--'))/'snapshots'/r)} for i,(m,r,role) in enumerate(MODELS)])
def prepare_documents():
 import numpy as np,pandas as pd
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(OUT/'tokenizers/Qwen--Qwen3-0.6B'),local_files_only=True)
 rows=[];wiki=[];title=None;parts=[]
 for fn in SOURCES[0]['files']:
  for value in pd.read_parquet(OUT/'source_data/Salesforce--wikitext'/fn)['text']:
   if re.match(r'^\s*= [^=].*[^=] =\s*$',value):
    if title and parts:wiki.append((title,'\n'.join(parts)))
    title=value.strip();parts=[value]
   elif title:parts.append(value)
 if title and parts:wiki.append((title,'\n'.join(parts)))
 for i,(title,txt) in enumerate(wiki):rows.append({'source':'wikitext103','source_key':title,'text':txt})
 frame=pd.read_parquet(OUT/'source_data/ccdv--arxiv-summarization'/SOURCES[1]['files'][0])
 for i,r in frame.iterrows():rows.append({'source':'arxiv_train_shard0','source_key':str(i),'text':r['article']})
 for r in rows:
  r['document_sha256']=hashlib.sha256(re.sub(r'\s+',' ',r['text'].lower()).encode()).hexdigest();r['order_hash']=hashlib.sha256(('rich-natural-docs-v1|'+r['document_sha256']).encode()).hexdigest()
 df=pd.DataFrame(rows).drop_duplicates('document_sha256').sort_values('order_hash')
 # Keep a larger population than final pilot/fit/validation/test requirements before outcomes.
 df=df.groupby('source',group_keys=False).head(9000).reset_index(drop=True)
 chosen=[]
 for r in df.to_dict('records'):
  ids=tok.encode(r['text'],add_special_tokens=False)
  if len(ids)<129:continue
  pos=128+int(r['document_sha256'][:16],16)%(len(ids)-128)
  context=ids[max(0,pos-512):pos]
  r.update({'context_ids':context,'position':pos,'context_tokens':len(context),'document_token_count':len(ids),'context_text':tok.decode(context),'cluster_id':'rich:'+r['document_sha256'],'item_id':'rich:'+r['document_sha256']})
  chosen.append(r)
 df=pd.DataFrame(chosen);df.to_parquet(OUT/'candidate_documents.parquet',index=False)
 dump(OUT/'document_preparation.json',{'candidate_counts':df.groupby('source').size().to_dict(),'position_rule':'128 + int(document_sha256[:16],16) mod (N_tokens-128); last <=512 prefix tokens','no_outcomes_used':True,'candidate_sha256':digest(OUT/'candidate_documents.parquet'),'historical_dedup_status':'pending; collection forbidden until finalized'})
 print(df.groupby('source').size(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['sources','models','documents']);p.add_argument('--cache',default='/tmp/model-replacement-adaptation/hf_cache');a=p.parse_args()
 {'sources':setup_sources,'models':lambda:setup_models(a.cache),'documents':prepare_documents}[a.action]()
