#!/usr/bin/env python3
"""Analyze frozen MCQ operating points, fresh fidelity and paired live costs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

REGISTERED_FIXED=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def intervals(d,comparisons,seed=81724042,draws=10000):
    d=np.asarray(d,dtype=float)
    if d.ndim==1:d=d[:,None]
    n,k=d.shape
    if n<2 or not np.isfinite(d).all():raise ValueError('Need finite paired observations')
    rng=np.random.default_rng(seed);boot=np.empty((draws,k))
    for start in range(0,draws,128):
        batch=min(128,draws-start)
        index=rng.integers(n,size=(batch,n))
        weights=np.bincount((index+np.arange(batch)[:,None]*n).reshape(-1),minlength=batch*n).reshape(batch,n)/n
        boot[start:start+batch]=weights@d
    return {'mean':d.mean(0),'low':np.quantile(boot,.025,axis=0),'high':np.quantile(boot,.975,axis=0),
        'simultaneous_low':np.quantile(boot,.05/(2*comparisons),axis=0),'simultaneous_high':np.quantile(boot,1-.05/(2*comparisons),axis=0)}

def quality(catalog_dir,cells,out):
    lock=json.loads((catalog_dir/'operating_points_lock.json').read_text())
    catalog=json.loads((catalog_dir/'candidate_catalog.json').read_text())
    if sha(catalog_dir/'candidate_catalog.json')!=lock['candidate_catalog_sha256']:raise ValueError('Resource candidate catalog hash drift')
    if sha(catalog_dir/'validation_losses.npz')!=lock['validation_loss_sha256']:raise ValueError('Validation losses hash drift')
    if not lock.get('operating_points_frozen_before_test') or lock.get('test_quality_used'):raise ValueError('Resource operating points were not frozen before final test')
    if len(cells)!=len(catalog['targets']):raise ValueError('Need exactly one primary fresh cell per registered target')
    lookup={r['system_id']:r for r in lock['eligibility']}
    by_target={};inputs={}
    for cell in cells:
        freeze=json.loads((cell/'selection_freeze.json').read_text());target=freeze['target']
        complete=json.loads((cell/'evaluation_complete.json').read_text())
        freeze_path=cell/'selection_freeze.json'
        if complete['freeze_sha256']!=sha(freeze_path):raise ValueError('Fresh source freeze hash drift')
        expected_freeze=catalog['source_freeze_hashes'].get(str(freeze_path.resolve()))
        if expected_freeze!=sha(freeze_path):raise ValueError('Fresh cell differs from the resource catalog source freeze')
        if freeze['seed']!=catalog['seed'] or freeze['task']!=catalog['task'] or freeze['q']!=2:raise ValueError('Resource quality must use the registered primary seed/task/q')
        if target in by_target or target not in catalog['targets']:raise ValueError('Duplicate or unregistered fresh target cell')
        if complete['created_unix']<lock['created_unix']:raise ValueError('Final test evaluation predates operating-point freeze')
        for rel,h in complete['outputs'].items():
            if sha(cell/rel)!=h:raise ValueError('Fresh test output hash drift')
        by_target[target]=pd.read_parquet(cell/'item_predictions_losses.parquet')
        inputs[str(cell/'evaluation_complete.json')]=sha(cell/'evaluation_complete.json')
    if set(by_target)!=set(catalog['targets']):raise ValueError('Fresh target cells incomplete')
    records=[];diffs=[];argmax_diffs=[];indices=[];cluster_orders=[]
    for decision in lock['decisions']:
        row={k:decision[k] for k in ('target_index','policy','quality_margin','reference_system','selected_system','status')}
        row['primary_registered_endpoint']=decision['quality_margin']==.005 and decision['policy']!='fixed_cost_aware_mean_cap'
        if decision['status']!='eligible':records.append(row);continue
        table=by_target[decision['target_index']]
        ref_digest=lookup[decision['reference_system']]['state_digest'];sel_digest=lookup[decision['selected_system']]['state_digest']
        ref=table[table.state_digest==ref_digest].sort_values('item_id').set_index('item_id')
        sel=table[table.state_digest==sel_digest].sort_values('item_id').set_index('item_id')
        if ref.empty or not ref.index.equals(sel.index) or not np.array_equal(ref.item_cluster,sel.item_cluster):raise ValueError('Fresh quality pairs do not align')
        if sel.item_cluster.nunique()!=len(sel):raise ValueError('Expected canonical cluster representatives')
        row.update({'test_clusters':len(sel),'selected_test_TV':float(sel.tv_loss.mean()),'reference_test_TV':float(ref.tv_loss.mean()),
            'quality_gap_TV':float((sel.tv_loss-ref.tv_loss).mean()),'selected_test_KL':float(sel.kl_loss.mean()),
            'selected_argmax_disagreement':float(sel.argmax_disagreement.mean()),'reference_argmax_disagreement':float(ref.argmax_disagreement.mean()),
            'selected_family':lookup[decision['selected_system']]['family'],'selected_pair':lookup[decision['selected_system']]['pair'],
            'selected_actual_bits':lookup[decision['selected_system']]['actual_bits']})
        if not sel.argmax_disagreement.isin([0,1]).all() or not ref.argmax_disagreement.isin([0,1]).all():raise ValueError('Argmax disagreement must be binary')
        row['argmax_disagreement_gap']=float((sel.argmax_disagreement.to_numpy(dtype=float)-ref.argmax_disagreement.to_numpy(dtype=float)).mean())
        row['argmax_noninferiority_margin']=.01
        argmax_diffs.append((sel.argmax_disagreement.to_numpy(dtype=float)-ref.argmax_disagreement.to_numpy(dtype=float)))
        diffs.append((sel.tv_loss-ref.tv_loss).to_numpy());indices.append(len(records));cluster_orders.append(sel.item_cluster.to_numpy());records.append(row)
    if diffs:
        if any(not np.array_equal(cluster_orders[0],x) for x in cluster_orders):raise ValueError('Target cohorts differ; require stratified analysis')
        ci=intervals(np.column_stack(diffs),comparisons=10)
        arg_ci=intervals(np.column_stack(argmax_diffs),comparisons=10)
        for i,index in enumerate(indices):
            row=records[index]
            row.update({k:float(ci[k][i]) for k in ('low','high','simultaneous_low','simultaneous_high')})
            row.update({'argmax_'+k:float(arg_ci[k][i]) for k in ('low','high','simultaneous_low','simultaneous_high')})
            row['tv_noninferiority_pass_primary_adjustment']=row['simultaneous_high']<=row['quality_margin']
            row['argmax_noninferiority_pass_primary_adjustment']=row['argmax_simultaneous_high']<=row['argmax_noninferiority_margin']
            row['noninferiority_pass_primary_adjustment']=row['tv_noninferiority_pass_primary_adjustment'] and row['argmax_noninferiority_pass_primary_adjustment']
            row['registered_joint_fidelity_claim_pass']=row['primary_registered_endpoint'] and row['noninferiority_pass_primary_adjustment']
            row['inference_note']='10000 paired-cluster bootstrap draws shared across TV/argmax and systems; familywise10 for2targets x5registered systems atprimary.005 TV margin. Joint fidelity requires BOTH TV margin and argmax degradation<=.01 (intersection-union test); no additional endpoint multiplicity factor is needed for this conjunctive claim. Othermargins/cost-awarefixed are descriptive sensitivity.'
    pd.DataFrame(records).to_csv(out/'fresh_operating_point_quality.csv',index=False)
    write(out/'quality_inputs.json',inputs)
    return records

def timing(raw_paths,out):
    rows=[];block_rows=[];fail=[]
    for path in raw_paths:
        for line in path.read_text().splitlines():
            block=json.loads(line)
            if block.get('failure_status'):fail.append({'source':str(path),**block});continue
            meta={k:block[k] for k in ('system_id','timing_block','batch_size','context_tokens','context_scope','execution_order','model_parameter_and_buffer_bytes','nominal_q','actual_call_q','system_role')}
            meta.update(block.get('adapter_accounting',{}));meta['hardware']=block['runtime']['GPU'];meta['GPU_uuid']=block['runtime'].get('GPU_uuid','')
            meta['dtype']=block['runtime']['dtype'];meta['resource_scope']='complete_predictor; bothconditional andstandalone accounting available';meta['real_target_count']=1
            traces=block['traces']
            if len(traces)!=block['timed_requests']:raise ValueError('Truncated timing block')
            for trace in traces:rows.append(meta|trace)
            block_rows.append(meta|{'mean_end_to_end_ms_per1000':float(np.mean([r['end_to_end_ms']/r['context_count']*1000 for r in traces])),
                'mean_peer_GPU_ms_per1000':float(np.mean([r['peer_GPU_elapsed_ms']/r['context_count']*1000 for r in traces])),
                'timed_requests':len(traces),'warmup_requests':block['warmup_requests'],'activation_seconds':block['activation_seconds']})
    trace=pd.DataFrame(rows);blocks=pd.DataFrame(block_rows)
    if blocks.duplicated(['system_id','timing_block','batch_size','context_tokens']).any():raise ValueError('Duplicate final timing blocks')
    trace.to_parquet(out/'timing_traces.parquet',index=False);blocks.to_csv(out/'timing_blocks.csv',index=False);write(out/'timing_failures.json',fail)
    summaries=[]
    for key,g in trace.groupby(['system_id','batch_size','context_tokens']):
        b=blocks[(blocks.system_id==key[0])&(blocks.batch_size==key[1])&(blocks.context_tokens==key[2])]
        if len(b)<10 or len(g)<200 or b.warmup_requests.min()<20:raise ValueError(f'Insufficient final timing replication {key}')
        first=g.iloc[0]
        def accounting_value(name):
            value=first.get(name,0)
            if pd.isna(value):
                if first.system_role!='direct_model_contextual_reference':raise ValueError(f'Missing replacement accounting field {name}')
                return 0.
            return float(value)
        summaries.append({'system_id':key[0],'batch_size':key[1],'context_tokens':key[2],'timed_requests':len(g),'timing_blocks':len(b),
            'warm_p50_latency_ms':float(g.end_to_end_ms.median()),'warm_p95_latency_ms':float(g.end_to_end_ms.quantile(.95)),
            'mean_end_to_end_ms_per1000':float(b.mean_end_to_end_ms_per1000.mean()),'mean_peer_GPU_ms_per1000':float(b.mean_peer_GPU_ms_per1000.mean()),
            'throughput_contexts_per_second':float(g.context_count.sum()/(g.end_to_end_ms.sum()/1000)),
            'peak_allocated_VRAM_bytes':int(g.peak_allocated_bytes.max()),'peak_reserved_VRAM_bytes':int(g.peak_reserved_bytes.max()),'peak_observed_host_RSS':int(g.host_RSS.max()),
            'model_parameter_and_buffer_bytes':int(first.model_parameter_and_buffer_bytes),'nominal_q':int(first.nominal_q),'actual_call_q':int(first.actual_call_q),
            'state_binary_bytes':accounting_value('state_binary_bytes'),'bundle_metadata_bytes':accounting_value('target_routing_manifest_bytes'),
            'resident_target_coefficients_bytes':accounting_value('resident_target_coefficients_bytes'),
            'resident_target_encoded_bytes':accounting_value('resident_target_encoded_bytes'),
            'resident_target_state_total_bytes':accounting_value('resident_target_coefficients_bytes')+accounting_value('resident_target_encoded_bytes'),
            'standalone_accounted_weights_context_state_bytes':int(first.model_parameter_and_buffer_bytes)+accounting_value('shared_deployment_resident_bytes')+accounting_value('resident_target_coefficients_bytes')+accounting_value('resident_target_encoded_bytes'),
            'shared_deployment_disk_bytes':accounting_value('shared_deployment_disk_bytes'),
            'shared_deployment_resident_bytes':accounting_value('shared_deployment_resident_bytes'),'GPU_uuid':first.GPU_uuid,
            'fidelity_link':'Freshbatch1naturalquality only' if key[1]==1 and key[2]==0 else 'Resourceprofile only; no transferofbatch1fidelity assumed'})
    pd.DataFrame(summaries).to_csv(out/'resource_summary.csv',index=False)
    return blocks,summaries,fail

def align_paired_blocks(joint,fixed):
    if joint.index.has_duplicates or fixed.index.has_duplicates:raise ValueError('Duplicate paired timing block')
    if set(joint.index)!=set(fixed.index) or len(joint)<10:raise ValueError('Need complete identical paired natural-batch1 block IDs')
    common=sorted(joint.index);a=joint.loc[common];b=fixed.loc[common]
    if 'GPU_uuid' not in a or 'GPU_uuid' not in b:raise ValueError('Paired GPU UUID evidence missing')
    au=a.GPU_uuid.fillna('').astype(str);bu=b.GPU_uuid.fillna('').astype(str)
    if au.str.strip().eq('').any() or bu.str.strip().eq('').any() or not np.array_equal(au,bu):raise ValueError('Joint/fixed timing blocks must run on the same nonempty GPU UUID within each block')
    ja=a.mean_end_to_end_ms_per1000.to_numpy(float);fb=b.mean_end_to_end_ms_per1000.to_numpy(float)
    if not np.isfinite(ja).all() or not np.isfinite(fb).all() or (ja<=0).any() or (fb<=0).any():raise ValueError('Invalid positive paired complete-predictor timing')
    return ja,fb,common,sorted(set(au))

def paired_resource_statistics(joint,fixed,comparisons=8,draws=10000):
    ja=np.asarray(joint,dtype=float);fb=np.asarray(fixed,dtype=float)
    if ja.shape!=fb.shape or ja.ndim!=1 or len(ja)<2 or not np.isfinite(ja).all() or not np.isfinite(fb).all() or (ja<=0).any() or (fb<=0).any():raise ValueError('Invalid paired cost vectors')
    ci=intervals(fb-ja,comparisons=comparisons,draws=draws)
    rng=np.random.default_rng(81724044);ix=rng.integers(len(ja),size=(draws,len(ja)))
    reduction=1-ja[ix].mean(1)/fb[ix].mean(1)
    return {'joint_ms_per1000':float(ja.mean()),'fixed_ms_per1000':float(fb.mean()),
        'resource_reduction':float(1-ja.mean()/fb.mean()),'resource_reduction_low':float(np.quantile(reduction,.05/(2*comparisons))),
        'resource_reduction_high':float(np.quantile(reduction,1-.05/(2*comparisons))),
        'cost_difference_ms_per1000':float(ci['mean'][0]),'cost_difference_simultaneous_low':float(ci['simultaneous_low'][0]),
        'cost_difference_simultaneous_high':float(ci['simultaneous_high'][0])}

def contrasts(catalog_dir,blocks,out,quality_records=None):
    lock=json.loads((catalog_dir/'operating_points_lock.json').read_text());records=[]
    quality_lookup={(r['target_index'],r['policy'],r['quality_margin']):r for r in (quality_records or [])}
    for target in map(int,lock['references']):
      for margin in (0,.0025,.005,.01):
        decisions={r['policy']:r for r in lock['decisions'] if r['target_index']==target and r['quality_margin']==margin}
        joint=decisions['budget_aware']
        for policy in (*REGISTERED_FIXED,'fixed_cost_aware_mean_cap'):
            fixed=decisions[policy];row={'target_index':target,'quality_margin':margin,'comparator':policy,'joint_system':joint['selected_system'],'fixed_system':fixed['selected_system']}
            if joint['status']!='eligible' or fixed['status']!='eligible':records.append(row|{'status':'infeasible_operating_point'});continue
            a=blocks[(blocks.system_id==joint['selected_system'])&(blocks.batch_size==1)&(blocks.context_tokens==0)].set_index('timing_block')
            b=blocks[(blocks.system_id==fixed['selected_system'])&(blocks.batch_size==1)&(blocks.context_tokens==0)].set_index('timing_block')
            ja,fb,common,gpu_ids=align_paired_blocks(a,b)
            stats=paired_resource_statistics(ja,fb,comparisons=8)
            jq=quality_lookup.get((target,'budget_aware',margin),{});fq=quality_lookup.get((target,policy,margin),{})
            joint_pass=bool(jq.get('noninferiority_pass_primary_adjustment',False));fixed_pass=bool(fq.get('noninferiority_pass_primary_adjustment',False))
            primary=margin==.005 and policy in REGISTERED_FIXED
            evidence=primary and joint_pass and fixed_pass and stats['resource_reduction']>=.2 and stats['resource_reduction_low']>0
            records.append(row|stats|{'status':'measured','paired_blocks':len(common),'paired_GPU_uuids':gpu_ids,
                'primary_registered_endpoint':primary,'joint_fresh_TV_and_argmax_NI':joint_pass,'fixed_fresh_TV_and_argmax_NI':fixed_pass,
                'registered_operating_point_evidence_pass':bool(evidence),
                'inference_note':'10000 paired timing-block draws; primaryfamily8 across2targets x4registeredfixedpolicies atmargin.005. Otherrowsdescriptive; sameGPU within each pairedblock. Evidenceguard additionally requires fresh TV and argmax NI for BOTH systems, mean resource reduction>=.20 and adjusted lowerbound>0. This is a per-comparison result within the frozen catalog, not a global algorithm/deployment claim.'})
    pd.DataFrame(records).to_csv(out/'paired_resource_contrasts.csv',index=False)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--cells',type=Path,nargs='+',required=True)
    p.add_argument('--raw',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    q=quality(a.catalog,a.cells,a.output);blocks,summary,fail=timing(a.raw,a.output);c=contrasts(a.catalog,blocks,a.output,q)
    outputs={str(f.resolve()):sha(f) for f in a.output.iterdir() if f.is_file() and f.name!='resource_complete.json'}
    write(a.output/'resource_complete.json',{'deployment_measured':True,'quality_evaluated_on_fresh_clusters':True,'outputs':outputs,'timing_settings':len(summary),'timing_blocks':len(blocks),'failed_blocks':len(fail),'primary_fidelity_link':'batch1canonicaluntruncated only','scientific_outcome':'See frozenoperating-point qualityandpairedcostcontrasts; scriptcompletion isnotoperationalbenefit','input_locks':{str(a.catalog/'operating_points_lock.json'):sha(a.catalog/'operating_points_lock.json')},'raw_measurements':{str(f):sha(f) for f in a.raw}})
    print(json.dumps({'quality_rows':len(q),'timing_settings':len(summary),'cost_contrasts':len(c),'failed_blocks':len(fail)}))

if __name__=='__main__':main()
