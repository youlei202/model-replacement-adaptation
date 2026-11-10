"""Close near-duplicate clusters jointly across ALL new candidate populations."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.confirmation_prepare import external_rows,cluster_union
from experiments.exposure_audit import digest
out=ROOT/'experiments/runs/v2/confirmation';hist=pd.read_parquet(ROOT/'experiments/runs/v2/audit/historical_text_registry.parquet');hist['historical']=True;lock=json.loads((out/'population_acquisition_preregistration.json').read_text());frames=[hist];signatures=[]
for old,info in lock['task_mapping'].items():
 task=info['external_task_id'];fresh=external_rows(task,out/'data_sources'/task/info['file'],info['revision']);fresh['sample_order']=[hashlib.sha256(('external81724031|'+s).encode()).hexdigest() for s in fresh.item_id];fresh=fresh.sort_values('sample_order').head(16384).reset_index(drop=True);frames.append(fresh)
 with np.load(out/task/'duplicate_signatures.npz') as data:
  if not signatures:signatures.append(data['signatures'][:len(hist)].copy())
  signatures.append(data['signatures'][len(hist):].copy())
combined=pd.concat(frames,ignore_index=True);semantic=[c+'\n'+'\n'.join(json.loads(o)) for c,o in combined[['context','options']].itertuples(index=False,name=None)];keys=[hashlib.sha256((a+'\0'+b).encode()).hexdigest() for a,b in zip(semantic,combined.context)];fingerprint=hashlib.sha256(json.dumps(keys).encode()).hexdigest();cache=out/'global_duplicate_signatures.npz';np.savez_compressed(cache,signatures=np.concatenate(signatures),fingerprint=fingerprint)
clustered,edges=cluster_union(combined,cache,workers=1);edges.to_parquet(out/'global_duplicate_edges.parquet',index=False);clustered[['task_id','item_id','historical','item_cluster','union_cluster_index','historical_cluster_overlap']].to_parquet(out/'global_duplicate_registry.parquet',index=False)
report={'historical_items':len(hist),'candidate_new_items':len(combined)-len(hist),'new_overlap_by_task':clustered[~clustered.historical].groupby('task_id').historical_cluster_overlap.sum().to_dict(),'new_cluster_counts':clustered[~clustered.historical & ~clustered.historical_cluster_overlap].groupby('task_id').item_cluster.nunique().to_dict(),'registry_sha256':digest(out/'global_duplicate_registry.parquet'),'description':'Global transitive closure of original near-duplicate algorithm plus stricter context duplicates across historical and all three new populations; applied before sampling fit/validation/test. Tokenizer-qualified rows subsequently use these global cluster IDs.'};(out/'global_freshness_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
