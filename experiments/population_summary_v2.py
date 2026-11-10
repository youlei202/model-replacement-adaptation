#!/usr/bin/env python3
"""Read-only T1 supplement: role, label/domain counts and source-key provenance."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=RUN/'reports';out.mkdir(exist_ok=True)
 rows=[];details=[];inputs={}
 for scope,folder in [('exploratory_existing_cache',ROOT/'benchmarks/splits'),('external_confirmation_new_population',RUN/'confirmation/manifests')]:
  for path in sorted(folder.glob('split_seed_*.parquet')):
   inputs[str(path.relative_to(ROOT))]=sha(path);data=pd.read_parquet(path)
   data=data[data.split.isin(['fit','validation','iid_test'])]
   for (seed,task,split),g in data.groupby(['seed','task_id','split']):
    assert not g.duplicated(['item_id','prompt_hash']).any()
    rows.append({'scope':scope,'seed':int(seed),'task_or_source':task,'role':split,'items':len(g),'clusters':g.item_cluster.nunique(),'semantic_key':'task_id,item_id,prompt_hash','independence_note':'Seeds/targets reuse clusters; descriptive counts cannot be summed as independent replications.'})
    for column in ['domain','gold']:
     if column not in g:continue
     for value,count in g[column].fillna('<missing>').astype(str).value_counts(dropna=False).items():
      details.append({'scope':scope,'seed':int(seed),'task':task,'role':split,'field':column,'value':value,'count':int(count)})
 lock=RUN/'rich_behavior/selection_lock.json';data=RUN/'rich_behavior/documents.parquet';inputs[str(lock.relative_to(ROOT))]=sha(lock);inputs[str(data.relative_to(ROOT))]=sha(data)
 selected=set(json.loads(lock.read_text())['test_item_ids']);docs=pd.read_parquet(data,columns=['item_id','cluster_id','source','role'])
 docs.loc[docs.item_id.isin(selected),'role']='iid_test';docs=docs[docs.role.isin(['pilot','fit','validation','iid_test'])]
 for (source,role),g in docs.groupby(['source','role']):
  rows.append({'scope':'independent_rich_nexttoken','seed':'fixed_protocol','task_or_source':source,'role':role,'items':len(g),'clusters':g.cluster_id.nunique(),'semantic_key':'item_id,cluster_id,canonical_context_token_ids','independence_note':'One context per document cluster; both target models share this test population.'})
 pd.DataFrame(rows).to_csv(out/'T1_SCOPE_COUNTS.csv',index=False);pd.DataFrame(details).to_csv(out/'T1_MCQLABEL_DOMAIN_COUNTS.csv',index=False)
 (out/'T1_SCOPE_COUNTS.md').write_text('''# T1 population, domain and label supplement

`T1_SCOPE_COUNTS.csv` reports exact query/document counts by population, task/source, seed and data role. `T1_MCQLABEL_DOMAIN_COUNTS.csv` contains all domain and gold-label counts, including missing domains; these are descriptive input metadata, not new accuracy outcomes. Gold indices follow each source's semantic choices, not display permutations. The new academic source provides no usable subject annotation; its blank domain is not silently invented.

The five Phase A corrected splits reuse historical queries. The three fresh MCQ development seeds reuse the same independent 4891-query test population. Two targets sharing queries do not create twice as many independent samples. Rich fit/validation/pilot/test sets are balanced across the two declared sources; all target comparisons share one document test population. These counts supplement, rather than replace, the original/new overlap registries, response-key completeness audits, prospective power lock and full-vocabulary tokenizer audit.

Exact source hashes are recorded in `T1_SCOPE_COUNTS_MANIFEST.json`. All rows are generated from manifests/locked document IDs; no model outcomes influence them.
''')
 manifest={'input_sha256':inputs,'outputs':{name:sha(out/name) for name in ['T1_SCOPE_COUNTS.csv','T1_MCQLABEL_DOMAIN_COUNTS.csv','T1_SCOPE_COUNTS.md']},'generation_script_sha256':sha(Path(__file__))}
 (out/'T1_SCOPE_COUNTS_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'scope_rows':len(rows),'metadata_count_rows':len(details)},indent=2))
if __name__=='__main__':main()
