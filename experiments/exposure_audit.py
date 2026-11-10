"""Audit every available historical project response/preflight/loss key.

No model-response values or losses are read. Dataset candidates not scored are
retained in a separate conservative exclusion registry for external confirmation.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def normalize_text(value):
 value=re.sub(r'\s+',' ',str(value).strip().lower())
 return re.sub(r'[^\w\s]','',value)

def text_hash(s):return hashlib.sha256(normalize_text(s).encode()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'experiments/runs/v2/audit');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 can=ROOT/'reference_implementation/data/canonical'
 metadata=pd.read_parquet(can/'candidate_manifest.parquet');prompts=pd.read_parquet(can/'prompt_manifest.parquet',columns=['task_id','item_id','prompt_hash'])
 metadata=metadata.merge(prompts,on=['task_id','item_id'],validate='one_to_one')
 oldsplit=pd.read_parquet(can/'split_manifest.parquet',columns=['task_id','item_id','split']).rename(columns={'split':'original_split'})
 metadata=metadata.merge(oldsplit,on=['task_id','item_id'],how='left',validate='one_to_one')
 sources=[];edges=[];skip=[]
 for root in [can,ROOT/'reference_implementation/results',ROOT/'benchmarks']:
  for p in sorted(root.rglob('*.parquet')):
   schema=pq.read_schema(p);names=set(schema.names)
   is_observed=any(n.startswith('coord_') for n in names) or bool(names & {'tv_loss','loss','TV','tv','seconds_per_view','deterministic_max_difference'})
   if not {'item_id','task_id'}.issubset(names) or not is_observed:continue
   cols=[c for c in ['task_id','item_id','split','model_id','seed','prompt_hash'] if c in names]
   d=pd.read_parquet(p,columns=cols).drop_duplicates();rel=str(p.relative_to(ROOT))
   sources.append({'path':rel,'sha256':digest(p),'rows':pq.ParquetFile(p).metadata.num_rows,'distinct_metadata_rows':len(d),'read_columns':cols})
   d['source']=rel
   edges.append(d)
 if not edges:raise RuntimeError('No observed keys found')
 edge=pd.concat(edges,ignore_index=True).fillna('');edge.to_parquet(out/'exposure_source_edges.parquet',index=False)
 group=edge.groupby(['task_id','item_id'],sort=True)
 observed=group.agg(source_count=('source','nunique'),sources=('source',lambda s:json.dumps(sorted(set(s)))),observed_splits=('split',lambda s:json.dumps(sorted(set(s)-{''}))),observed_models=('model_id',lambda s:json.dumps(sorted(set(s)-{''})))).reset_index()
 registry=observed.merge(metadata,on=['task_id','item_id'],how='left',validate='one_to_one')
 if registry.context.isna().any():raise RuntimeError('Observed keys lack historical semantic text; freshness audit incomplete')
 for d in [metadata,registry]:
  d['normalized_context_sha256']=d.context.map(text_hash)
  d['normalized_semantic_sha256']=[text_hash(c+'\n'+'\n'.join(json.loads(o))) for c,o in d[['context','options']].itertuples(index=False,name=None)]
  d['normalized_option_order_invariant_sha256']=[text_hash(c+'\n'+'\n'.join(sorted(json.loads(o)))) for c,o in d[['context','options']].itertuples(index=False,name=None)]
 registry['exposure_scope']='all_available_project_response_preflight_loss_banks'
 registry.to_parquet(out/'exposure_registry.parquet',index=False)
 observed_key_set=set(map(tuple,observed[['task_id','item_id']].to_numpy()))
 metadata['has_observed_response']=[x in observed_key_set for x in metadata[['task_id','item_id']].itertuples(index=False,name=None)]
 metadata.to_parquet(out/'historical_text_registry.parquet',index=False)
 pop=pd.read_parquet(ROOT/'benchmarks/splits/population.parquet');rows=[]
 for task,p in pop.groupby('task_id'):
  r=registry[registry.task_id==task];exposedkeys=set(r.item_id);exposedclusters=set(r.item_cluster)
  rows.append({'task_id':task,'corrected_eligible_clusters':len(p),'exposed_item_keys_in_support':int(p.item_id.isin(exposedkeys).sum()),'exposed_clusters_in_support':int(p.item_cluster.isin(exposedclusters).sum()),'unscored_cluster_upper_bound':int((~p.item_cluster.isin(exposedclusters)).sum()),'historical_all_candidate_items':int(metadata.task_id.eq(task).sum()),'historical_observed_items':len(r),'used_as_new_confirmation':False})
 pd.DataFrame(rows).to_csv(out/'query_reservoir_audit.csv',index=False)
 summary={'scope':'All response, preflight, robustness and per-item loss artifacts within original canonical/results and benchmarks. Duplicate copies in deliverables excluded; outside-project undisclosed author experiments cannot be verified. External confirmation conservatively excludes ALL 23008 historical candidate texts plus every exposed cluster, not merely observed response keys.','observed_unique_keys':len(registry),'all_candidate_texts':len(metadata),'sources':sources,'registry_sha256':digest(out/'exposure_registry.parquet'),'all_historical_text_sha256':digest(out/'historical_text_registry.parquet'),'reservoir':rows}
 (out/'exposure_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
