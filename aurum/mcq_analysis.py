"""All-pair MCQ development frontiers and paired loss replay, never confirmation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from .mcq import ROOT, LEGACY, SEED, CAPS, DIMS, cached_task, sha, dump
from .codec import decode
from .inference import DecodedPredictor
from .train import ArrayData, evaluate


def ordered(frame):
    return frame.sort_values(['validation_tv','actual_bytes','family','codec','candidate_id','pair'],kind='stable')


def policies(catalog,caps):
    decisions=[];summaries=[]
    for (task,target),frame in catalog.groupby(['task','target_index']):
        pairs=sorted(frame.pair.unique())
        if len(pairs)!=10:raise ValueError('Incomplete admissible pair family')
        env={};cheap={}
        for pair in pairs:
            group=frame[frame.pair.eq(pair)]
            for cap in caps:
                allowed=group[group.actual_bytes.le(cap)]
                if allowed.empty:raise ValueError('Infeasible pair at requested cap')
                env[pair,cap]=ordered(allowed).iloc[0]
                cheap[pair,cap]=ordered(allowed[allowed.cheap]).iloc[0]
        joint={cap:min((env[pair,cap] for pair in pairs),key=lambda r:(r.validation_tv,r.actual_bytes,r.candidate_id,r.pair)) for cap in caps}
        jointcheap={cap:min((cheap[pair,cap] for pair in pairs),key=lambda r:(r.validation_tv,r.actual_bytes,r.candidate_id,r.pair)) for cap in caps}
        native=frame[frame.codec.isin(['fp32','native_fp32','native_fp64'])]
        fixed={'fixed_native_full':ordered(native).iloc[0].pair,
               'fixed_unbounded_menu':ordered(frame).iloc[0].pair,
               'fixed_mean_budget':min(pairs,key=lambda p:(np.mean([env[p,b].validation_tv for b in caps]),p)),
               'fixed_minimax_regret':min(pairs,key=lambda p:(max(env[p,b].validation_tv-joint[b].validation_tv for b in caps),p)),
               **{'fixed_pair_'+p.replace(',','_'):p for p in pairs}}
        for cap in caps:
            for name,row in [('joint',joint[cap]),('joint_cheap',jointcheap[cap])]+[(name,env[p,cap]) for name,p in fixed.items()]:
                decisions.append({**row.to_dict(),'policy':name,'cap_bytes':cap,'budget_weight':1/len(caps)})
        deltas={p:np.mean([env[p,b].validation_tv-joint[b].validation_tv for b in caps]) for p in pairs}
        summaries.append({'task':task,'target_index':target,'Gamma_development':min(deltas.values()),
                          'best_fixed_pair':min(deltas,key=deltas.get),'joint_pairs':';'.join(sorted({r.pair for r in joint.values()})),
                          'joint_weighted_tv':np.mean([r.validation_tv for r in joint.values()]),
                          'joint_cheap_minus_joint':np.mean([jointcheap[b].validation_tv-joint[b].validation_tv for b in caps]),
                          'caps_bytes':','.join(map(str,caps)), 'scope':'exploratory development; selected on same validation cohort'})
    return pd.DataFrame(decisions),pd.DataFrame(summaries)


def replay_selected(output,decisions):
    """Replay all selected states on validation, including old literal RBTL states."""
    sys.path.insert(0,str(ROOT/'experiments'))
    import adapters_v2 as legacy
    records=[];losses={};max_drift=0.
    for task,group in decisions.groupby('task'):
        data=cached_task(output,task);coords=data['validation_coords'].astype(float)
        basis=torch.tensor(legacy.core.helmert(DIMS[task]).T,dtype=torch.float32)
        contexts={};projections={};newcontexts={}
        for row in group.drop_duplicates(['target_index','pair','candidate_id']).to_dict('records'):
            pair=tuple(map(int,row['pair'].split(',')));target=int(row['target_index'])
            if row['state_format']=='AUR1':
                path=output/row['state_path'];state=decode(path.read_bytes())
                if pair not in newcontexts:
                    with np.load(output/'public_contexts'/task/f'p{pair[0]}-{pair[1]}.npz') as c:
                        newcontexts[pair]=torch.tensor((data['validation_coords'][:,pair,:].reshape(len(coords),-1)-c['mean'])/c['scale'])
                bank=ArrayData(torch.tensor(data['validation_logits'][:,pair,:]),newcontexts[pair],
                               torch.tensor(data['validation_logits'][:,target,:]),role='validation')
                measured=evaluate(DecodedPredictor(state,basis),bank,batch_size=512)['per_item']
            else:
                path=ROOT/row['state_path'];encoded,matrix=legacy.core.decode(path.read_bytes())
                if pair not in contexts:
                    cp=LEGACY/'public_contexts'/f'seed_{SEED}'/task/f'peers_{pair[0]}-{pair[1]}.npz'
                    with np.load(cp) as c:
                        contexts[pair]=legacy.core.Context(pair,c['mean'],c['scale'],float(c['sigma']),c['features'],c['eigenvalues'],c['eigenvectors'],np.empty((0,0)))
                if encoded.family=='krr' and pair not in projections:projections[pair]=contexts[pair].project(coords)
                pred=legacy.predict(encoded,matrix,coords,contexts[pair],projections.get(pair))
                pp=legacy.core.probabilities(pred);tp=legacy.core.probabilities(coords[:,target,:])
                measured={'tv':.5*np.abs(tp-pp).sum(1),
                          'kl':(tp*(np.log(np.maximum(tp,1e-300))-np.log(np.maximum(pp,1e-300)))).sum(1),
                          'argmax_disagreement':(pp.argmax(1)!=tp.argmax(1)).astype(float)}
            drift=abs(float(measured['tv'].mean())-row['validation_tv']);max_drift=max(max_drift,drift)
            if drift>2e-6:raise ValueError(f'Decoded state replay drift {drift}: {path}')
            key=(task,target,row['candidate_id']);losses[key]=measured['tv'].astype(float)
            records.append(pd.DataFrame({'task':task,'target_index':target,'pair':row['pair'],'candidate_id':row['candidate_id'],
                              'item_id':data['validation_item_ids'],'cluster_id':data['validation_clusters'],
                              **{k:measured[k] for k in ('tv','kl','argmax_disagreement')}}))
    pd.concat(records,ignore_index=True).to_parquet(output/'DEVELOPMENT_PER_CLUSTER.parquet',index=False)
    contrasts=[]
    for (task,target),group in decisions.groupby(['task','target_index']):
        weighted={}
        for policy,part in group.groupby('policy'):
            weighted[policy]=sum(row['budget_weight']*losses[task,target,row['candidate_id']] for row in part.to_dict('records'))
        clusters=cached_task(output,task)['validation_clusters']
        for policy,values in weighted.items():
            delta=values-weighted['joint']
            cluster_delta=pd.DataFrame({'cluster':clusters,'delta':delta}).groupby('cluster').delta.mean().to_numpy()
            contrasts.append({'task':task,'target_index':target,'policy':policy,
                              'weighted_validation_tv':float(values.mean()),'fixed_minus_joint':float(delta.mean()),
                              'document_cluster_sd':float(cluster_delta.std(ddof=1)),'clusters':len(cluster_delta),
                              'scope':'same-validation selection; descriptive paired variation, not independent inference'})
    pd.DataFrame(contrasts).to_csv(output/'DEVELOPMENT_PAIRED_CONTRASTS.csv',index=False)
    dump(output/'decoded_replay.json',{'states_replayed':len(records),'max_validation_mean_drift':max_drift,
         'historical_test_responses_opened':False,'new_confirmation':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'runs/aurum/mcq');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);out=args.output
    catalog=pd.read_csv(out/'ALL_CANDIDATES.csv')
    decisions,summaries=policies(catalog,[b for b in CAPS if b>=128])
    decisions.to_csv(out/'POLICY_DECISIONS.csv',index=False)
    dump(out/'POLICY_DECISIONS.json',json.loads(decisions.to_json(orient='records')))
    summaries.to_csv(out/'FIXED_CHOICE_PRICE_DEVELOPMENT.csv',index=False)
    _,sensitivity=policies(catalog,CAPS);sensitivity.to_csv(out/'SMALL_CAP_SENSITIVITY.csv',index=False)
    if args.replay:replay_selected(out,decisions)
    print(summaries.sort_values('Gamma_development',ascending=False).to_string(index=False))


if __name__=='__main__':main()
