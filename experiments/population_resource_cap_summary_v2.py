#!/usr/bin/env python3
"""T1 supplement for the prospectively sealed second untouched cohorts."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 mcq=RUN/'resource_cap_confirmation/required_items_test.parquet';rich=RUN/'rich_behavior/matched_resource_supplement/cohort.parquet';a=pd.read_parquet(mcq);b=pd.read_parquet(rich,columns=['item_id','cluster_id','source']);old=pd.concat([pd.read_parquet(RUN/f'confirmation/required_items_{role}.parquet') for role in ['fitval','test']],ignore_index=True);oldrich=set(json.loads((RUN/'rich_behavior/selection_lock.json').read_text())['test_item_ids']);assert not set(a.item_cluster)&set(old.item_cluster);assert not set(b.item_id)&oldrich
 rows=[{'route':'MCQ','source':'cais/mmlu auxiliary_train','role':'second_untouched_test','items':len(a),'clusters':a.item_cluster.nunique(),'original_confirmation_overlap':0,'target_count':2,'supplement_primary_family':16,'note':'Same queries across targets/caps; no multiplication of independent sample count'}]
 rows += [{'route':'rich','source':source,'role':'second_untouched_test','items':len(g),'clusters':g.cluster_id.nunique(),'original_confirmation_overlap':0,'target_count':2,'supplement_primary_family':16,'note':'Canonical document contexts; source-balanced separatecohort; no free-running generation claim'} for source,g in b.groupby('source')]
 details=[]
 for field in ['gold','domain']:
  for value,count in a[field].fillna('<missing>').astype(str).value_counts().items():details.append({'route':'MCQ','field':field,'value':value,'count':int(count)})
 out=RUN/'reports';pd.DataFrame(rows).to_csv(out/'T1_RESOURCE_CAP_COHORT_COUNTS.csv',index=False);pd.DataFrame(details).to_csv(out/'T1_RESOURCE_CAP_LABEL_DOMAIN_COUNTS.csv',index=False);files=['T1_RESOURCE_CAP_COHORT_COUNTS.csv','T1_RESOURCE_CAP_LABEL_DOMAIN_COUNTS.csv'];(out/'T1_RESOURCE_CAP_COHORT_MANIFEST.json').write_text(json.dumps({'inputs':{str(p.relative_to(RUN)):sha(p) for p in [mcq,rich]},'source_sha256':sha(Path(__file__)),'outputs':{f:sha(out/f) for f in files},'scope':'Metadataonly counts from both separately frozen secondcohorts; main confirmation kept separate'},indent=2)+'\n');print(rows)
if __name__=='__main__':main()
