#!/usr/bin/env python3
"""Freeze a symmetric resource candidate catalog using validation only.

The catalog contains every pair/family's frozen winner at four primary state
caps and without a state cap. Cost profiles are development measurements. The
same catalog is available to joint and all four fixed-peer policies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mcq_deployment import export_bundle,DeployedAdapter,sha,core

PRIMARY=(1024,2048,8192,16384)
FIXED=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret')

def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')

def prepare(args):
    if len(args.targets)!=len(args.cells) or len(set(args.targets))!=len(args.targets):raise ValueError('Targets and cells must match one-to-one')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'candidate_catalog.json').exists():raise ValueError('Existing catalog is immutable; select a new namespace')
    registry=json.loads(args.registry.read_text())['model_registry']
    manifest=pd.read_parquet(args.manifest)
    validation=manifest[(manifest.task_id==args.task)&(manifest.split=='validation')].sort_values('item_id').reset_index(drop=True)
    if not len(validation):raise ValueError('No validation cohort')
    validation.to_parquet(out/'development_prompts.parquet',index=False)
    models={}
    for row in registry:
        data=pd.read_parquet(args.response_dir/(row['model_id'].replace('/','--')+'.parquet'))
        aligned=validation[['task_id','item_id','prompt_hash']].merge(data,on=['task_id','item_id','prompt_hash'],validate='one_to_one')
        if len(aligned)!=len(validation):raise ValueError('Incomplete validation response keys')
        models[row['index']]=aligned[[f'coord_{j}' for j in range(args.dimension)]].to_numpy()
    catalog=[];systems=[];losses=[];freeze_hashes={};fixed_pairs={}
    for target,cell in zip(args.targets,args.cells):
        freeze_path=cell/'selection_freeze.json';f=json.loads(freeze_path.read_text())
        if f['target']!=target or f['task']!=args.task or f['q']!=2 or f['seed']!=args.seed:raise ValueError('Incorrect resource cell')
        freeze_hashes[str(freeze_path.resolve())]=sha(freeze_path)
        choices=[r for r in f['selections'] if r['selection_kind']=='pair_family' and r['budget_bits'] in (*PRIMARY,-2) and r['status']=='eligible']
        unique={}
        for row in choices:unique.setdefault(row['state_digest'],row)
        fp={}
        for policy in FIXED:
            row=next(r for r in f['selections'] if r['selection_kind']=='policy' and r['scope']=='strengthened' and r['budget_bits']==1024 and r['policy']==policy)
            fp[policy]=row['pair']
        fixed_pairs[str(target)]=fp
        for digest,row in sorted(unique.items()):
            pair=tuple(row['pair']);pair_name='-'.join(map(str,pair));state=cell/'pairs'/pair_name/'states'/f'{digest}.bin'
            ctx=(cell/f['contexts'][pair_name]['path']).resolve()
            if sha(state)!=digest or sha(ctx)!=f['contexts'][pair_name]['sha256']:raise ValueError('Frozen artifact changed')
            sid=f't{target}_{digest[:16]}';bundle=out/'bundles'/sid
            export_bundle(state,ctx,registry,target,args.dimension,bundle)
            adapter=DeployedAdapter(bundle)
            prediction=adapter.predict({p:models[p] for p in pair})
            item_loss=core.tv(models[target],prediction)
            if abs(item_loss.mean()-row['validation_tv'])>1e-10:raise ValueError('Validation replay differs')
            losses.append(item_loss)
            catalog.append({'system_id':sid,'target_index':target,'target_id':registry[target]['model_id'],
                'pair':list(pair),'family':row['family'],'actual_bits':row['actual_bits'],'state_digest':digest,
                'validation_tv':float(item_loss.mean()),'bundle':str(bundle),'source_state':str(state.resolve()),
                'eligible_binary_caps':[b for b in PRIMARY if row['actual_bits']<=b],**adapter.accounting})
            systems.append({'system_id':sid,'bundle':str(bundle)})
    loss_matrix=np.column_stack(losses)
    np.savez_compressed(out/'validation_losses.npz',losses=loss_matrix,item_id=validation.item_id.to_numpy(dtype=str),cluster_id=validation.item_cluster.to_numpy(dtype=str),systems=np.asarray([r['system_id'] for r in catalog]))
    lock={'status':'resource_candidate_catalog_frozen_before_test','created_from':'validation-only state replay',
        'task':args.task,'dimension':args.dimension,'seed':args.seed,'targets':args.targets,'model_registry':registry,
        'candidate_pool':'Every complete q2 pair x5 adapter-family frozen validation winner at each primary cap and unbounded; duplicate byte states deduplicated. No test-dependent pruning.',
        'candidate_count':len(catalog),'candidates':catalog,'fixed_pairs':fixed_pairs,'source_freeze_hashes':freeze_hashes,
        'validation_loss_sha256':sha(out/'validation_losses.npz'),'manifest_sha256':sha(args.manifest),
        'primary_resource_configuration':{'batch_size':1,'context_tokens':0,'context_law':'natural untruncated validation prompts'},
        'secondary_resource_configurations':'batch8 and128/512 token controlled profiles; not assumed numerically equivalent to batch1 quality',
        'quality_margins':[0,.0025,.005,.01],'primary_quality_margin':.005,
        'validation_eligibility':'Bonferroni approximate paired bootstrap upper bound vs validation-best unbounded target-absent reference, <=margin; paired clusters',
        'cost_selection':'Lowest development mean complete-predictor end_to_end milliseconds per1000contexts among eligible candidates; identical catalog for joint/fixed, fixed subset pair imposed. Peer GPU elapsed reported separately.',
        'test_used_for_selection':False}
    write(out/'candidate_catalog.json',lock)
    spec={'measurement_role':'development_operating_point_profile','selected_on_validation_only':True,'seed':81724039,'dimension':args.dimension,
        'prompts':str(out/'development_prompts.parquet'),'peer_response_dir':str(args.response_dir.resolve()),
        'systems':systems,'batch_sizes':[1],'context_lengths':[0],'blocks':2,'warmup_requests':5,'timed_requests_per_block':20}
    write(out/'development_profile_spec.json',spec)
    print(json.dumps({'catalog_candidates':len(catalog),'targets':args.targets,'output':str(out)}))

def select(args):
    out=args.output.resolve();catalog=json.loads((out/'candidate_catalog.json').read_text())
    if (out/'operating_points_lock.json').exists():raise ValueError('Operating points already frozen')
    blocks=[json.loads(x) for x in args.profile.read_text().splitlines()]
    profiles={}
    observed={}
    for row in blocks:
        if row.get('failure_status'):continue
        sid=row['system_id'];block=row['timing_block']
        if block in observed.setdefault(sid,set()):raise ValueError('Duplicate development profile block')
        if row['batch_size']!=1 or row['context_tokens']!=0 or len(row['traces'])<20:raise ValueError('Development profile settings differ')
        observed[sid].add(block)
        times=[r['end_to_end_ms']/r['context_count']*1000 for r in row['traces']]
        if not np.isfinite(times).all() or min(times)<=0:raise ValueError('Invalid measured complete-predictor cost')
        profiles.setdefault(sid,[]).extend(times)
    expected={r['system_id'] for r in catalog['candidates']}
    if set(observed)!=expected or any(observed[s]!={0,1} for s in expected):raise ValueError('Every catalog state requires both balanced development timing blocks')
    with np.load(out/'validation_losses.npz') as z:
        losses=z['losses'];clusters=z['cluster_id'];ids=list(z['systems'])
    unique,ix=np.unique(clusters,return_inverse=True)
    if len(unique)!=len(clusters):raise ValueError('This cohort should have one canonical representative per cluster')
    rng=np.random.default_rng(81724040);bootstrap=5000
    weights=np.zeros((bootstrap,len(clusters)),dtype=float)
    for i in range(bootstrap):weights[i]=np.bincount(rng.integers(len(clusters),size=len(clusters)),minlength=len(clusters))/len(clusters)
    decisions=[];eligibility=[];refs={};cost_fixed={};all_candidates=catalog['candidates']
    for target in catalog['targets']:
        candidates=[r for r in all_candidates if r['target_index']==target]
        ref=min(candidates,key=lambda r:(r['validation_tv'],r['actual_bits'],r['system_id']));refs[str(target)]=ref
        differences=losses[:,[ids.index(r['system_id']) for r in candidates]]-losses[:,ids.index(ref['system_id'])][:,None]
        upper=np.quantile(weights@differences,1-.05/(2*len(candidates)),axis=0)
        for i,row in enumerate(candidates):
            sid=row['system_id'];row['validation_quality_gap_upper']=float(upper[i]);row['development_end_to_end_ms_per1000']=float(np.mean(profiles[sid])) if sid in profiles else None
            eligibility.append(row)
        # Additional strong resource-aware diagnostic: choose ONE pair using
        # all four primary state caps and development cost, then retune its
        # state for every fidelity margin. This reveals a cheap static pair
        # that could erase apparent savings over risk-selected fixed peers.
        admissible=[r for r in candidates if r['validation_quality_gap_upper']<=.005+1e-14 and r['development_end_to_end_ms_per1000'] is not None]
        best_by_cap={b:min((r['development_end_to_end_ms_per1000'] for r in admissible if r['actual_bits']<=b),default=None) for b in PRIMARY}
        diagnostics=[]
        for pair in sorted({tuple(r['pair']) for r in candidates}):
            costs={b:min((r['development_end_to_end_ms_per1000'] for r in admissible if tuple(r['pair'])==pair and r['actual_bits']<=b),default=None) for b in PRIMARY}
            relevant=[b for b in PRIMARY if best_by_cap[b] is not None]
            infeasible=sum(costs[b] is None for b in relevant)
            regret=[costs[b]/best_by_cap[b]-1 for b in relevant if costs[b] is not None]
            diagnostics.append({'pair':list(pair),'infeasible_caps':infeasible,'mean_cost_regret':float(np.mean(regret)) if regret else None,'costs_by_cap':costs})
        chosen=min(diagnostics,key=lambda r:(r['infeasible_caps'],r['mean_cost_regret'] if r['mean_cost_regret'] is not None else float('inf'),r['pair']))
        cost_fixed[str(target)]={'selection_rule':'One pair minimizing infeasible-cap count then mean relative cost regret across fourprimarycaps atvalidationNI margin.005; no test data. Additional diagnostic, not replacing registered four fixed comparators.','chosen_pair':chosen['pair'],'pair_diagnostics':diagnostics,'joint_min_cost_by_cap':best_by_cap}
        for margin in catalog['quality_margins']:
            for policy in ('budget_aware',*FIXED,'fixed_cost_aware_mean_cap'):
                fixed=chosen['pair'] if policy=='fixed_cost_aware_mean_cap' else catalog['fixed_pairs'][str(target)].get(policy)
                possible=[r for r in candidates if (fixed is None or r['pair']==fixed) and r['validation_quality_gap_upper']<=margin+1e-14 and r['development_end_to_end_ms_per1000'] is not None]
                selected=min(possible,key=lambda r:(r['development_end_to_end_ms_per1000'],r['validation_tv'],r['actual_bits'],r['system_id'])) if possible else None
                decisions.append({'target_index':target,'policy':policy,'quality_margin':margin,'reference_system':ref['system_id'],'reference_validation_tv':ref['validation_tv'],'selected_system':selected['system_id'] if selected else None,'status':'eligible' if selected else 'infeasible','selected':selected})
    lock={'status':'frozen_before_final_test','operating_points_frozen_before_test':True,'candidate_catalog_sha256':sha(out/'candidate_catalog.json'),'validation_loss_sha256':sha(out/'validation_losses.npz'),'development_measurement_sha256':sha(args.profile),'primary_margin':.005,'resource_configuration':catalog['primary_resource_configuration'],'decisions':decisions,'references':refs,'cost_aware_fixed_diagnostic':cost_fixed,'eligibility':eligibility,'bootstrap_draws':bootstrap,'validation_ci_rule':'approximate Bonferroni marginal pairedcluster bootstrap, family size number of catalog candidates per target; finite-draw extreme tails are development eligibility only, not confirmatory evidence','test_quality_used':False,'code_sha256':sha(__file__),'created_unix':__import__('time').time()}
    lock['analysis_code_sha256']=sha(Path(__file__).with_name('mcq_resource_analyze.py'))
    lock['benchmark_code_sha256']=sha(Path(__file__).with_name('mcq_resource_benchmark.py'))
    write(out/'operating_points_lock.json',lock)
    ids_needed={d['selected_system'] for d in decisions if d['selected_system'] is not None}|{r['system_id'] for r in refs.values()}
    initial=json.loads((out/'development_profile_spec.json').read_text())
    initial.update({'measurement_role':'final_live_resource_measurement','systems':[s for s in initial['systems'] if s['system_id'] in ids_needed],'blocks':10,'warmup_requests':20,'timed_requests_per_block':20,'batch_sizes':[1,8],'context_lengths':[0,128,512]})
    write(out/'final_resource_spec.json',initial)
    direct=dict(initial)
    direct['systems']=[{'system_id':f'direct_model_{r["index"]}','direct_model':r,'role':'direct_model_contextual_reference'} for r in catalog['model_registry']]
    direct['measurement_role']='contextual_direct_model_resource_measurement'
    write(out/'direct_model_resource_spec.json',direct)
    print(json.dumps({'operating_points':len(decisions),'infeasible':sum(d['status']=='infeasible' for d in decisions),'distinct_final_systems':len(ids_needed)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','select'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--registry',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--response-dir',type=Path)
    p.add_argument('--cells',type=Path,nargs='+');p.add_argument('--targets',type=int,nargs='+');p.add_argument('--task');p.add_argument('--dimension',type=int);p.add_argument('--seed',type=int);p.add_argument('--profile',type=Path)
    a=p.parse_args();prepare(a) if a.stage=='prepare' else select(a)

if __name__=='__main__':main()
