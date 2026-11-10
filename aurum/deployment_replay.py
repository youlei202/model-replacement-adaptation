"""Independent deployment-path validation replay using selected peers only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import RichDevelopmentCache
from .deployment import DeployedPredictor
from .train import ArrayData,evaluate
from .sprint import dump,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    folder=Path(args.catalog);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(folder/'ALL_CANDIDATES.csv')
    selected=pd.DataFrame(json.loads((folder/'POLICY_DECISIONS.json').read_text()))
    # Extend selected states to cover every observed decoder family/context.
    coverage=catalog.sort_values('validation_tv').drop_duplicates(['state_format','family','context'])
    rows=pd.concat([selected,coverage],ignore_index=True).drop_duplicates(['target_index','pair','candidate_id','state_format'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cache=RichDevelopmentCache();records=[]
    for row in rows.to_dict('records'):
        pair=tuple(map(int,row['pair'].split(',')));target=int(row['target_index'])
        model=DeployedPredictor(row,args.device)
        data=ArrayData(tuple(cache.logits('validation',p) for p in pair),None,cache.logits('validation',target),role='outer_validation')
        result=evaluate(model,data,batch_size=64)
        prefix='' if row['state_format']=='RBV2' else 'validation_'
        with np.load(row['losses_path']) as old:
            drift={name:float(np.max(np.abs(result['per_item'][name].astype(float)-old[prefix+name].astype(float)))) for name in ('tv','kl','argmax_disagreement')}
        records.append({'target_index':target,'pair':row['pair'],'candidate_id':row['candidate_id'],'family':row['family'],
                        'context':row.get('context'),'state_format':row['state_format'],'max_item_tv_drift':drift['tv'],
                        'max_item_kl_drift':drift['kl'],'argmax_mismatches':int(drift['argmax_disagreement']!=0),
                        'mean_tv_drift':abs(result['tv']-row['validation_tv'])})
        pd.DataFrame(records).to_csv(out/'DEPLOYMENT_REPLAY.csv',index=False)
        print(records[-1],flush=True);del model
    frame=pd.DataFrame(records)
    dump(out/'complete.json',{'states':len(frame),'max_item_TV_drift':float(frame.max_item_tv_drift.max()),
          'max_item_KL_drift':float(frame.max_item_kl_drift.max()),'argmax_states_with_disagreement':int(frame.argmax_mismatches.sum()),
          'source_sha256':sha(Path(__file__)),'policy_decisions_sha256':sha(folder/'POLICY_DECISIONS.json'),
          'scope':'old validation replay, never new confirmation; deployment features recomputed from selected peer responses'})


if __name__=='__main__':main()
