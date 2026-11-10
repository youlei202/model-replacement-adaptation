"""Heterogeneous six-model MCQ development extension; no test response arrays.

All pairs receive the same fitted/calibrated neural menu and literal AUR1 codecs.
Historical validation-selected strong states remain available as RBV2 fallbacks.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from .codec import export_model, decode
from .inference import DecodedPredictor
from .models import FrozenAnchor, TrainableCalibration, AnchoredResidual, AnchoredResponseGate, AnchoredMLP
from .train import ArrayData, TrainConfig, fit, evaluate
from .robust_calibration import optimize_coefficients

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from selection_core import helmert, fit_weights

SEED=20261109
DIMS={'mmlu_pro':9,'hellaswag':3,'rewardbench':1}
REGISTRY=ROOT/'experiments/experiment_config/configs/campaign.json'
LEGACY=ROOT/'experiments/runs/v2/phase_a'
CODECS=('fp32','fp16','int8','int4')
CAPS=(32,64,128,256,512,1024,2048,4096,8192)
_CACHE={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_hashes():
    names=('mcq.py','models.py','losses.py','train.py','codec.py','inference.py','robust_calibration.py')
    result={str((ROOT/'aurum'/n).relative_to(ROOT)):sha(ROOT/'aurum'/n) for n in names}
    result['benchmarks/selection_core.py']=sha(ROOT/'benchmarks/selection_core.py')
    return result


def prepare(output):
    folder=output/'development_cache';marker=folder/'manifest.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        for name,digest in value['cache_hashes'].items():assert sha(folder/name)==digest
        return value
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/f'benchmarks/splits/split_seed_{SEED}.parquet'
    # Predicate filtering is applied before any response coordinate arrays exist.
    manifest=pd.read_parquet(manifest_path,filters=[('split','in',['fit','validation'])])
    assert set(manifest.split)=={'fit','validation'}
    assert not manifest.duplicated(['task_id','item_id']).any()
    assert manifest.groupby(['task_id','item_cluster']).split.nunique().max()==1
    models=json.loads(REGISTRY.read_text())['model_registry']
    response_paths=sorted((ROOT/'benchmarks/response_cache/recompute_all').glob('*.parquet'))
    responses=[];input_hashes={str(manifest_path.relative_to(ROOT)):sha(manifest_path),str(REGISTRY.relative_to(ROOT)):sha(REGISTRY)}
    for path in response_paths:
        data=pd.read_parquet(path,filters=[('item_id','in',manifest.item_id.tolist())],
                             columns=['model_id','task_id','item_id','prompt_hash','revision']+[f'coord_{i}' for i in range(9)])
        data=data.merge(manifest[['task_id','item_id','prompt_hash']],on=['task_id','item_id','prompt_hash'],validate='one_to_one')
        responses.append(data);input_hashes[str(path.relative_to(ROOT))]=sha(path)
    responses=pd.concat(responses,ignore_index=True)
    assert len(responses)==len(manifest)*6
    assert set(responses.model_id)=={m['model_id'] for m in models}
    for model in models:
        assert set(responses[responses.model_id.eq(model['model_id'])].revision)=={model['revision']}
    cache_hashes={};counts={}
    for task,dim in DIMS.items():
        arrays={};counts[task]={}
        for role in ('fit','validation'):
            rows=manifest[manifest.task_id.eq(task)&manifest.split.eq(role)].sort_values('item_id').reset_index(drop=True)
            model_values=[]
            for model in models:
                joined=rows[['task_id','item_id','prompt_hash']].merge(responses[responses.model_id.eq(model['model_id'])],on=['task_id','item_id','prompt_hash'],how='left',validate='one_to_one',sort=False)
                coords=joined[[f'coord_{i}' for i in range(dim)]].to_numpy(np.float64)
                assert np.isfinite(coords).all()
                model_values.append(coords)
            coords=np.stack(model_values,axis=1)
            arrays[role+'_coords']=coords.astype(np.float32)
            arrays[role+'_logits']=(coords@helmert(dim)).astype(np.float32)
            arrays[role+'_item_ids']=rows.item_id.to_numpy(dtype=str)
            arrays[role+'_clusters']=rows.item_cluster.to_numpy(dtype=str)
            counts[task][role]=len(rows)
            if role=='fit':
                # Grouped deterministic inner split, independent of target outputs.
                cluster_order=sorted(set(rows.item_cluster.astype(str)),key=lambda x:hashlib.sha256(('aurum_mcq_inner_v1|'+x).encode()).hexdigest())
                stop_clusters=set(cluster_order[::3]);stop=rows.item_cluster.astype(str).isin(stop_clusters).to_numpy()
                arrays['train_indices']=np.flatnonzero(~stop);arrays['stop_indices']=np.flatnonzero(stop)
        path=folder/(task+'.npz');np.savez(path,**arrays);cache_hashes[path.name]=sha(path)
    value={'split_seed':SEED,'roles':['fit','validation'],'test_response_arrays_materialized':False,
           'response_filter':'Parquet item_id predicate restricted to fit/validation manifest keys, then exact task/item/prompt join',
           'model_registry':models,'input_hashes':input_hashes,'cache_hashes':cache_hashes,'counts':counts,
           'all_derived_arrays_use_existing_development_only':True}
    dump(marker,value);return value


def cached_task(output,task):
    key=(str(output),task)
    if key not in _CACHE:
        with np.load(output/'development_cache'/f'{task}.npz',allow_pickle=False) as data:
            _CACHE[key]={k:data[k] for k in data.files}
    return _CACHE[key]


def run_cell(job):
    output,task,target,pair,epochs,patience=job
    output=Path(output);pair=tuple(pair)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    folder=output/'cells'/task/f't{target}_p{pair[0]}-{pair[1]}'
    marker=folder/'complete.json';sources=source_hashes()
    if marker.exists():
        prior=json.loads(marker.read_text());assert prior['source_sha256']==sources
        assert sha(folder/'candidate_ledger.csv')==prior['ledger_sha256']
        return {'cell':str(folder.relative_to(output)),'cached':True,'seconds':prior['seconds']}
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('Incomplete prior cell retained; use another output root: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True);(folder/'states').mkdir()
    start=time.time();ledger=[];scored={};histories=[]
    try:
        data=cached_task(output,task);dim=DIMS[task];vocab=dim+1
        train_ids=data['train_indices'];stop_ids=data['stop_indices']
        fitcoords=data['fit_coords'];valcoords=data['validation_coords']
        raw=fitcoords[:,pair,:].reshape(len(fitcoords),-1)
        mean=raw.mean(0);scale=raw.std(0);scale=np.where(scale<1e-6,1,scale)
        features=(raw-mean)/scale;vfeatures=(valcoords[:,pair,:].reshape(len(valcoords),-1)-mean)/scale
        basis=torch.tensor(helmert(dim).T,dtype=torch.float32)
        context=output/'public_contexts'/task/f'p{pair[0]}-{pair[1]}.npz'
        # Each cell writes its own exact context first. Shared context creation
        # is performed by the parent before workers start to avoid races.
        assert context.exists()
        with np.load(context) as saved:
            assert np.array_equal(mean,saved['mean']) and np.array_equal(scale,saved['scale'])
        peers=torch.tensor(data['fit_logits'][:,pair,:]);vpeers=torch.tensor(data['validation_logits'][:,pair,:])
        target_logits=torch.tensor(data['fit_logits'][:,target,:]);vtarget=torch.tensor(data['validation_logits'][:,target,:])
        x=torch.tensor(features);vx=torch.tensor(vfeatures)
        whole=ArrayData(peers,x,target_logits,role='fit')
        training=whole.subset(train_ids,role='fit');inner=whole.subset(stop_ids,role='inner_stop')
        validation=ArrayData(vpeers,vx,vtarget,role='validation')
        dump(folder/'run_spec.json',{'task':task,'target':target,'pair':pair,'epochs':epochs,'patience':patience,
             'source_sha256':sources,'training_items':len(training),'inner_stop_items':len(inner),'validation_items':len(validation),
             'feature_context':str(context.relative_to(output)),'context_sha256':sha(context),
             'basis':'fixed mathematical Helmert basis; no target-fitted output subspace',
             'public_numeric_bytes':mean.nbytes+scale.nbytes,'test_response_arrays_opened':False})
        def offer(model,label,objective='none',extra=None):
            records=[]
            for codec in CODECS:
                blob=export_model(model,pair,codec=codec,basis_id=700+10*pair[0]+pair[1],vocab=vocab,retired_target=target)
                digest=hashlib.sha256(blob).hexdigest();state=decode(blob)
                if digest not in scored:
                    path=folder/'states'/(digest+'.aur1');path.write_bytes(blob)
                    predictor=DecodedPredictor(state,basis)
                    metrics={}
                    for role,bank in [('fit',training),('inner_stop',inner),('validation',validation)]:
                        result=evaluate(predictor,bank,batch_size=512,return_per_item=False)
                        metrics.update({role+'_'+name:result[name] for name in ('tv','kl','argmax_disagreement')})
                    scored[digest]=metrics
                row={'task':task,'target_index':target,'pair':','.join(map(str,pair)),'label':label,
                     'family':state['family'],'codec':codec,'objective':objective,'candidate_id':digest,
                     'state_format':'AUR1','actual_bytes':len(blob),'resident_parameter_bytes':state['resident_parameter_bytes'],
                     'public_numeric_bytes':mean.nbytes+scale.nbytes if not state['family'].startswith('anchor_') else 0,
                     'state_path':str((folder/'states'/(digest+'.aur1')).relative_to(output)),
                     'cheap':state['family'].startswith('anchor_'),'status':'scored_decoded',
                     'fit_scope':'inner_fit_only','selection_scope':'checkpoint_on_inner_stop; catalog_selection_on_outer_validation',
                     **scored[digest],**(extra or {})}
                ledger.append(row);records.append((row,state))
            return records
        def optimize(model,label,loss,lr,wd=0.,max_epochs=None):
            cfg=TrainConfig(loss=loss,epochs=max_epochs or epochs,lr=lr,batch_size=256,eval_batch_size=512,
                            patience=patience,seed=SEED,weight_decay=wd)
            info=fit(model,training,inner,cfg)
            histories.append({'label':label,'loss':loss,'lr':lr,'weight_decay':wd,**info})
            dump(folder/'progress.json',{'stage':label,'completed_candidates':len(ledger),'elapsed_seconds':time.time()-start})
            return info
        weights={name:fit_weights(fitcoords[train_ids],fitcoords[train_ids,target,:],pair,name).astype(np.float32)
                 for name in ('nnls','convex')}
        initial={'nnls':weights['nnls'],'convex':weights['convex'],'equal':np.array([.5,.5],np.float32),
                 'peer0':np.array([1.,0.],np.float32),'peer1':np.array([0.,1.],np.float32)}
        anchor_options=[]
        for name,weight in initial.items():
            for kind in (('logit','probability') if name=='equal' else ('logit',)):
                anchor=FrozenAnchor(torch.tensor(weight),kind);records=offer(anchor,'anchor_'+name+'_'+kind)
                anchor_options.append((records[0][0]['inner_stop_tv'],anchor))
        # Direct bounded coefficients avoid near-zero softmax weights trapping
        # the cheap comparator. Single-peer temperature fits are explicit.
        for kind, single in [('logit',None),('probability',None),('logit',0),('logit',1)]:
            start_weights=weights['convex'] if single is None else initial['peer'+str(single)]
            model, result=optimize_coefficients(training,start_weights,kind=kind,single=single,loss='kl',device='cpu')
            label=f'bounded_{kind}_single{single}'
            histories.append({'label':label+'_KL',**result})
            records=offer(model,label+'_KL','kl')
            anchor_options.append((records[0][0]['inner_stop_tv'],model))
            refined, result=optimize_coefficients(training,model.weights.numpy(),kind=kind,single=single,loss='tv',device='cpu')
            histories.append({'label':label+'_TV',**result})
            records=offer(refined,label+'_TV','tv')
            anchor_options.append((records[0][0]['inner_stop_tv'],refined))
        # Smoothed simplex start avoids effectively frozen zero coordinates.
        convex=weights['convex'];soft=.95*convex+.05*np.array([.5,.5],np.float32)
        for kind in ('logit','probability'):
            for start_name,start_weights in [('equal',np.array([.5,.5],np.float32)),('soft_convex',soft)]:
                for lr in (.01,.05):
                    label=f'calibration_{kind}_{start_name}_lr{lr}'
                    model=TrainableCalibration(torch.tensor(start_weights),kind,temperature=kind=='logit')
                    optimize(model,label+'_KL','kl',lr);records=offer(model,label+'_KL','kl')
                    anchor_options.append((records[0][0]['inner_stop_tv'],model.frozen()))
                    optimize(model,label+'_TV','tv_kl',lr/3,max_epochs=max(20,epochs//2));records=offer(model,label+'_TV','kl_then_tv_kl')
                    anchor_options.append((records[0][0]['inner_stop_tv'],model.frozen()))
        anchor=copy.deepcopy(min(anchor_options,key=lambda item:item[0])[1])
        dump(folder/'anchor_choice.json',{'selection':'inner_stop_only_from_unquantized_cheap_candidates','weights':anchor.weights.tolist(),'kind':anchor.kind})
        families=['linear','factor1','gate','mlp16']
        if dim>2:families.append('factor2')
        for family in families:
            for lr in (.003,.015):
                for wd in (0.,.001):
                    label=f'{family}_lr{lr}_wd{wd}'
                    if family=='linear':model=AnchoredResidual(copy.deepcopy(anchor),basis,2*dim)
                    elif family.startswith('factor'):model=AnchoredResidual(copy.deepcopy(anchor),basis,2*dim,factor_rank=int(family[-1]),seed=SEED)
                    elif family=='gate':model=AnchoredResponseGate(copy.deepcopy(anchor),2*dim)
                    else:model=AnchoredMLP(copy.deepcopy(anchor),basis,2*dim,hidden=16,seed=SEED)
                    optimize(model,label+'_KL','kl',lr,wd);offer(model,label+'_KL','kl',{'lr':lr,'weight_decay':wd})
                    optimize(model,label+'_TV','tv_kl',lr/3,wd,max_epochs=max(20,epochs//2))
                    offer(model,label+'_TV','kl_then_tv_kl',{'lr':lr/3,'weight_decay':wd})
        # Retain all old per-family/per-budget validation winners, not test scores.
        legacy_folder=LEGACY/'cells'/f'seed_{SEED}'/task/f'target_{target}'/'q_2'/'pairs'/f'{pair[0]}-{pair[1]}'
        legacy_path=legacy_folder/'pair_selections.json'
        old=json.loads(legacy_path.read_text());seen=set()
        for row in old:
            digest=row['state_digest']
            if digest in seen:continue
            seen.add(digest);state_path=legacy_folder/'states'/(digest+'.bin')
            blob=state_path.read_bytes();assert hashlib.sha256(blob).hexdigest()==digest
            assert len(blob)*8==row['actual_bits']
            ledger.append({'task':task,'target_index':target,'pair':','.join(map(str,pair)),
                           'label':'historical_'+row['family'],'family':row['family'],'codec':row['codec'],
                           'objective':'historical_family_fit','candidate_id':digest,'state_format':'RBTL',
                           'actual_bytes':len(blob),'state_path':str(state_path.relative_to(ROOT)),
                           'cheap':row['family'] in ('nnls','convex'),'status':'historical_decoded_validation_reused',
                           'fit_scope':'historical_all_fit','selection_scope':'historical_outer_validation_family_budget_winner',
                           'validation_tv':row['validation_tv'],'legacy_selection_sha256':sha(legacy_path)})
        pd.DataFrame(ledger).to_csv(folder/'candidate_ledger.csv',index=False)
        dump(folder/'training_history.json',histories)
        assert source_hashes()==sources,'Scientific source changed during cell'
        dump(marker,{'complete':True,'task':task,'target':target,'pair':pair,'seconds':time.time()-start,
                     'candidate_rows':len(ledger),'unique_states':len({r['candidate_id'] for r in ledger}),
                     'source_sha256':sources,'ledger_sha256':sha(folder/'candidate_ledger.csv'),
                     'historical_test_arrays_opened':False})
        return {'cell':str(folder.relative_to(output)),'seconds':time.time()-start,'candidate_rows':len(ledger)}
    except Exception as error:
        dump(folder/'failure.json',{'error':repr(error),'seconds':time.time()-start,'candidate_rows':len(ledger),'outputs_retained':True})
        if ledger:pd.DataFrame(ledger).to_csv(folder/'partial_candidate_ledger.csv',index=False)
        raise


def initialize_contexts(output,tasks):
    for task in tasks:
        data=cached_task(output,task)
        for pair in itertools.combinations(range(6),2):
            raw=data['fit_coords'][:,pair,:].reshape(len(data['fit_coords']),-1)
            mean=raw.mean(0);scale=raw.std(0);scale=np.where(scale<1e-6,1,scale)
            path=output/'public_contexts'/task/f'p{pair[0]}-{pair[1]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():np.savez(path,mean=mean,scale=scale)


def summarize(output):
    paths=sorted((output/'cells').glob('*/*/candidate_ledger.csv'))
    frames=[]
    for p in paths:
        marker=p.parent/'complete.json'
        if marker.exists():
            assert sha(p)==json.loads(marker.read_text())['ledger_sha256'];frames.append(pd.read_csv(p))
    if not frames:return
    catalog=pd.concat(frames,ignore_index=True);catalog.to_csv(output/'ALL_CANDIDATES.csv',index=False)
    rows=[]
    for (task,target,pair),group in catalog.groupby(['task','target_index','pair']):
        for cap in CAPS:
            feasible=group[group.actual_bytes.le(cap)]
            if len(feasible):
                best=feasible.sort_values(['validation_tv','actual_bytes','candidate_id']).iloc[0].to_dict()
                rows.append({**best,'cap_bytes':cap})
    frontier=pd.DataFrame(rows);frontier.to_csv(output/'PAIR_FRONTIERS.csv',index=False)
    gains=[]
    for (task,target),group in frontier.groupby(['task','target_index']):
        if group.pair.nunique()!=10:continue
        eligible=group.groupby('cap_bytes').pair.nunique();caps=eligible[eligible.eq(10)].index.tolist()
        group=group[group.cap_bytes.isin(caps)]
        table=group.pivot(index='cap_bytes',columns='pair',values='validation_tv')
        dynamic=table.min(axis=1);regrets=table.sub(dynamic,axis=0)
        fixed_mean=table.mean().sort_values().index[0];fixed_minimax=regrets.max().sort_values().index[0]
        fixed_unbounded=table.loc[max(caps)].sort_values().index[0]
        for cap in caps:
            for name,pair in [('fixed_mean',fixed_mean),('fixed_minimax',fixed_minimax),('fixed_highest_cap',fixed_unbounded)]:
                gains.append({'task':task,'target_index':target,'cap_bytes':cap,'fixed_policy':name,'fixed_pair':pair,
                              'joint_pair':table.loc[cap].sort_values().index[0],'joint_validation_tv':dynamic.loc[cap],
                              'fixed_validation_tv':table.loc[cap,pair],'validation_gain':table.loc[cap,pair]-dynamic.loc[cap],
                              'scope':'exploratory_development; all_fixed_pairs_reselect_full_menu_at_each_cap'})
    pd.DataFrame(gains).to_csv(output/'FIXED_PEER_COMPARISON.csv',index=False)
    dump(output/'summary.json',{'complete_cells':len(paths),'planned_cells':180,'candidate_rows':len(catalog),
                               'historical_test_arrays_opened':False,'confirmation':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/aurum/mcq')
    parser.add_argument('--tasks',nargs='+',choices=tuple(DIMS),default=list(DIMS))
    parser.add_argument('--targets',nargs='+',type=int,default=list(range(6)))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--max-cells',type=int)
    parser.add_argument('--stage',choices=['prepare','run','summarize'],default='run')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.stage=='summarize':summarize(output);return
    provenance=prepare(output)
    if args.stage=='prepare':print(json.dumps(provenance));return
    initialize_contexts(output,args.tasks)
    jobs=[(str(output),task,target,pair,args.epochs,args.patience) for task in args.tasks for target in args.targets
          for pair in itertools.combinations([i for i in range(6) if i!=target],2)]
    plan={'jobs':len(jobs),'tasks':args.tasks,'targets':args.targets,'epochs':args.epochs,'patience':args.patience,
          'codecs':CODECS,'state_caps_bytes':CAPS,'source_sha256':source_hashes(),'input_manifest_sha256':sha(output/'development_cache/manifest.json'),
          'scope':'existing_MCq_development_only; no independent confirmation claim'}
    plan_path=output/'run_spec.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==json.loads(json.dumps(plan)),'Run spec drift; use new output root'
    else:dump(plan_path,plan)
    selected=jobs[:args.max_cells] if args.max_cells else jobs
    results=[];failures=[];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(run_cell,job):job for job in selected}
        for future in concurrent.futures.as_completed(pending):
            try:result=future.result();results.append(result);print(json.dumps(result),flush=True)
            except Exception as error:
                failures.append({'job':pending[future],'error':repr(error)});print(json.dumps(failures[-1]),flush=True)
            dump(output/'progress.json',{'stage':'mcq_training','completed_candidates':len(results),'planned_candidates':len(selected),
                 'total_cells_planned':len(jobs),'failures':failures,'elapsed_seconds':time.time()-start})
    summarize(output)
    if failures:raise RuntimeError(f'{len(failures)} MCQ cells failed; artifacts retained')
    if len(selected)==len(jobs):dump(output/'complete.json',{'complete':True,'cells':len(results),'seconds':time.time()-start,'historical_test_arrays_opened':False})


if __name__=='__main__':main()
