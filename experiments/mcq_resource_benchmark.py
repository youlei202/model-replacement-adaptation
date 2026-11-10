#!/usr/bin/env python3
"""Live, synchronized MCQ replacement benchmark with randomized paired blocks.

Each isolated worker loads only its callable models and minimal adapter bundle.
Workers offload between randomized blocks, so just one system occupies the GPU.
Cold/activation time is recorded separately from warm complete-predictor timing.
Controlled 128/512-token runs are resource profiles, not quality evaluations of
the original untruncated query law. Natural-context replay is reported separately.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))

def execution_records(system,adapter=None):
    if not system.get('bundle'):return [system['direct_model']]
    if adapter is None:
        from mcq_deployment import DeployedAdapter
        adapter=DeployedAdapter(system['bundle'])
    records=adapter.manifest['callable_peers']
    if adapter.encoded.family in ('convex','nnls'):
        nonzero={p for p,w in zip(adapter.pair,adapter.matrix.reshape(-1)) if w!=0}
        records=[r for r in records if r['index'] in nonzero]
    return records

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n');tmp.replace(path)

def worker(spec_path,system_id):
    import gc
    import inspect
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    import mcq_deployment as deploy
    sys.path.insert(0,str(ROOT/'reference_implementation'))
    from kernel_isqed.llm import SemanticOptionScorer
    from kernel_isqed.response import scores_to_coordinates
    spec=json.loads(Path(spec_path).read_text())
    system=next(x for x in spec['systems'] if x['system_id']==system_id)
    adapter=deploy.DeployedAdapter(system['bundle']) if system.get('bundle') else None
    records=execution_records(system,adapter)
    if adapter and any(r['model_id']==adapter.manifest['target_model_id'] for r in records):raise RuntimeError('Target loaded in replacement')
    start=time.monotonic();scorers={}
    for r in records:
        scorer=SemanticOptionScorer(r['model_id'],r['revision'],'cuda:0')
        scorer.model.to('cpu');torch.cuda.empty_cache();scorers[r['index']]=scorer
    load_seconds=time.monotonic()-start
    frame=pd.read_parquet(spec['prompts'])
    if 'split' in frame:frame=frame[frame.split.isin(['fit','validation'])]
    if len(frame)<20:raise ValueError('Need at least20 development prompts')
    frame=frame.sort_values('item_id').reset_index(drop=True)
    dims=adapter.manifest['dimension'] if adapter else int(spec['dimension'])
    codes=[str(i) for i in range(dims+1)]
    weight_bytes={str(k):sum(p.numel()*p.element_size() for p in s.model.parameters())+sum(b.numel()*b.element_size() for b in s.model.buffers()) for k,s in scorers.items()}
    def announce(obj):
        print('BENCH_JSON '+json.dumps(obj,allow_nan=False),flush=True)
    announce({'ready':True,'system_id':system_id,'cold_model_load_seconds':load_seconds,'weight_bytes':weight_bytes,'models':[r['model_id'] for r in records]})
    process=psutil.Process()

    def request(rows,cap):
        coords={};token_counts=[];truncated=0;gpu_events=[]
        for idx,s in scorers.items():
            prompts=[];candidate_ids=[]
            for row in rows:
                ids=s.render(row.system_prompt,row.user_prompt)
                candidates=s._candidate_ids(row.system_prompt,row.user_prompt,codes,ids)
                if any(len(c)!=1 for c in candidates):raise ValueError('Non-single-token option')
                ids=ids[0]
                if cap and len(ids)>cap:ids=ids[-cap:];truncated+=1
                prompts.append(ids);candidate_ids.append([c[0] for c in candidates]);token_counts.append(len(ids))
            length=cap or max(len(p) for p in prompts)
            pad=s.tokenizer.pad_token_id
            if pad is None:pad=s.tokenizer.eos_token_id
            if pad is None:pad=0
            ids=torch.full((len(rows),length),int(pad),dtype=torch.long,device='cuda:0')
            mask=torch.zeros_like(ids)
            for i,prompt in enumerate(prompts):ids[i,-len(prompt):]=prompt;mask[i,-len(prompt):]=1
            kwargs={'input_ids':ids,'attention_mask':mask,'use_cache':False,'return_dict':True}
            position_ids=(mask.cumsum(-1)-1).clamp_min(0)
            model_start=torch.cuda.Event(enable_timing=True);model_end=torch.cuda.Event(enable_timing=True)
            model_start.record()
            with torch.inference_mode():
                if s.is_gemma:
                    kwargs['logits_to_keep']=1
                    if 'position_ids' in inspect.signature(s.model.forward).parameters:kwargs['position_ids']=position_ids
                    logits=s.model(**kwargs).logits[:,-1,:].float()
                    selected=torch.stack([logits[i,c] for i,c in enumerate(candidate_ids)])
                else:
                    if 'position_ids' in inspect.signature(s.backbone.forward).parameters:kwargs['position_ids']=position_ids
                    hidden=s.backbone(**kwargs).last_hidden_state[:,-1,:]
                    selected=torch.stack([s._selected_logits(hidden[i:i+1],c)[0] for i,c in enumerate(candidate_ids)])
            model_end.record();gpu_events.append((model_start,model_end))
            displayed=selected.detach().cpu().double().numpy()
            semantic=np.empty_like(displayed)
            for i,row in enumerate(rows):semantic[i,np.asarray(json.loads(row.permutation),int)]=displayed[i]
            coords[idx]=scores_to_coordinates(semantic)
        if adapter:
            for p in adapter.pair:
                if p not in coords:coords[p]=np.zeros((len(rows),dims))
        prediction=adapter.predict(coords) if adapter else next(iter(coords.values()))
        gpu_ms=sum(a.elapsed_time(b) for a,b in gpu_events)
        return prediction,token_counts,truncated,coords,gpu_ms

    for line in sys.stdin:
        command=json.loads(line)
        if command['command']=='quit':break
        try:
            requested_id=command.get('system_id',system_id)
            if requested_id!=system_id:
                system=next(x for x in spec['systems'] if x['system_id']==requested_id)
                candidate=deploy.DeployedAdapter(system['bundle']) if system.get('bundle') else None
                new_records=execution_records(system,candidate)
                if [(r['index'],r['model_id'],r['revision']) for r in new_records]!=[(r['index'],r['model_id'],r['revision']) for r in records]:raise ValueError('Cannot switch callable models in a worker')
                adapter=candidate;system_id=requested_id;gc.collect()
            t0=time.monotonic()
            for s in scorers.values():s.model.to('cuda:0')
            torch.cuda.synchronize();activation=time.monotonic()-t0
            if command['command']=='replay':
                selected_rows=list(frame.head(8).itertuples(index=False))
                # Only peer banks are opened. The retired target bank is absent
                # from this verification and from the deployed worker inputs.
                expected={}
                for r in records:
                    cache=pd.read_parquet(Path(spec['peer_response_dir'])/(r['model_id'].replace('/','--')+'.parquet'))
                    ordered=pd.DataFrame({'task_id':[x.task_id for x in selected_rows],'item_id':[x.item_id for x in selected_rows],'prompt_hash':[x.prompt_hash for x in selected_rows]}).merge(cache,on=['task_id','item_id','prompt_hash'],validate='one_to_one')
                    if len(ordered)!=len(selected_rows):raise ValueError('Replay keys incomplete')
                    expected[r['index']]=ordered[[f'coord_{j}' for j in range(dims)]].to_numpy()
                if adapter:
                    for p in adapter.pair:
                        if p not in expected:expected[p]=np.zeros((len(selected_rows),dims))
                expected_prediction=adapter.predict(expected) if adapter else next(iter(expected.values()))
                checks=[]
                for batch in (1,8):
                    predictions=[];peer_errors={str(k):0. for k in scorers}
                    for offset in range(0,len(selected_rows),batch):
                        pred,_,_,coord,_=request(selected_rows[offset:offset+batch],0);predictions.append(pred)
                        for k,v in coord.items():peer_errors[str(k)]=max(peer_errors.get(str(k),0.),float(np.max(np.abs(v-expected[k][offset:offset+batch]))))
                    actual=np.concatenate(predictions)
                    checks.append({'batch_size':batch,'max_prediction_coordinate_difference':float(np.max(abs(actual-expected_prediction))),'mean_prediction_TV_difference':float(deploy.core.tv(actual,expected_prediction).mean()),'peer_coordinate_max_differences':peer_errors,'query_count':len(selected_rows)})
                for s in scorers.values():s.model.to('cpu')
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                announce({'system_id':system_id,'replay_checks':checks,'target_loaded':adapter is None,'failure_status':None})
                continue
            block=int(command['block']);batch=int(command['batch_size']);cap=int(command['context_tokens'])
            count=int(command.get('timed_requests',20));warmup=int(command.get('warmup_requests',20))
            rng=np.random.default_rng(int(spec['seed'])+block*1000+batch*100+cap)
            order=rng.integers(len(frame),size=(count+warmup,batch))
            traces=[]
            for i,index in enumerate(order):
                rows=list(frame.iloc[index].itertuples(index=False))
                torch.cuda.synchronize()
                if i==warmup:torch.cuda.reset_peak_memory_stats()
                ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();ev0.record()
                prediction,tokens,truncated,_,gpu_ms=request(rows,cap)
                ev1.record();torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                if i>=warmup:
                    traces.append({'request_index':i-warmup,'end_to_end_ms':wall,'GPU_stream_elapsed_ms':ev0.elapsed_time(ev1),
                        'peer_GPU_elapsed_ms':gpu_ms,
                        'context_count':batch,'minimum_active_tokens':min(tokens,default=0),'maximum_active_tokens':max(tokens,default=0),'truncated_peer_prompts':truncated,
                        'prediction_sha256':hashlib.sha256(np.asarray(prediction,dtype='<f8').tobytes()).hexdigest(),
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_RSS':process.memory_info().rss})
            result={'system_id':system_id,'timing_block':block,'batch_size':batch,'context_tokens':cap,
                'context_scope':'natural_untruncated' if cap==0 else 'controlled_truncated_or_left_padded_resource_profile',
                'warmup_requests':warmup,'timed_requests':count,'execution_order':command['execution_order'],
                'activation_seconds':activation,'cold_model_load_seconds':load_seconds,
                'model_parameter_and_buffer_bytes':sum(weight_bytes.values()),'weight_bytes':weight_bytes,
                'nominal_q':len(adapter.pair) if adapter else 1,'actual_call_q':len(records),'exact_zero_weight_pruning':True,
                'adapter_accounting':adapter.accounting if adapter else {},'models':[r['model_id'] for r in records],
                'target_called':False if adapter else None,'system_role':'target_free_replacement' if adapter else 'direct_model_contextual_reference','failure_status':None,'traces':traces,
                'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'GPU_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','')),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES',''),'host':__import__('socket').gethostname(),'dtype':'bfloat16'},
                'GPU_metric_definition':'CUDA-event elapsed on synchronized default stream; includes stream idle intervals during CPU preprocessing. Not hardware utilization or FLOPs.'}
            if not command.get('keep_active',False):
                for s in scorers.values():s.model.to('cpu')
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            announce(result)
        except Exception as e:
            for s in scorers.values():s.model.to('cpu')
            gc.collect();torch.cuda.empty_cache()
            announce({'system_id':system_id,'failure_status':repr(e),'command':command})

def run(spec_path,output,only_block=None):
    spec=json.loads(Path(spec_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if spec.get('selected_on_validation_only') is not True:raise ValueError('Freeze validation-selected systems before profiling')
    if len({s['system_id'] for s in spec['systems']})!=len(spec['systems']):raise ValueError('Duplicate systems')
    profiling=spec.get('measurement_role')=='development_operating_point_profile'
    if not profiling and (spec.get('blocks',10)<10 or spec.get('timed_requests_per_block',20)*spec.get('blocks',10)<200):raise ValueError('Insufficient measurement replication')
    if (output/'raw_blocks.jsonl').exists():raise ValueError('Existing measurements preserved; choose a new output namespace')
    if only_block is not None and not profiling:raise ValueError('Block sharding is restricted to balanced development profiles')
    dump(output/'measurement_lock.json',{'spec_sha256':hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),'benchmark_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'spec':spec,'only_block':only_block,'test_quality_used_to_select':False})
    workers={};startup=[];worker_for={};group_for={}
    for system in spec['systems']:
        records=execution_records(system)
        key=tuple((r['index'],r['model_id'],r['revision']) for r in records)
        group_for.setdefault(key,system['system_id']);worker_for[system['system_id']]=group_for[key]
    def response(proc):
        for line in proc.stdout:
            if line.startswith('BENCH_JSON '):return json.loads(line[len('BENCH_JSON '):])
        raise RuntimeError('Benchmark worker exited unexpectedly')
    usage=[]
    def close_worker(sid):
        proc,err=workers.pop(sid)
        if proc.poll() is None:
            try:proc.stdin.write('{"command":"quit"}\n');proc.stdin.flush();proc.wait(timeout=30)
            except Exception:proc.terminate();proc.wait(timeout=30)
        err.close()
    def get_worker(sid):
        representative=worker_for[sid]
        if representative in usage:usage.remove(representative)
        if representative not in workers:
            while len(workers)>=int(spec.get('maximum_worker_processes',4)):
                close_worker(usage.pop(0))
            err=(output/f'{representative}.stderr.log').open('a')
            sid=representative
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker','--spec',str(Path(spec_path).resolve()),'--system-id',sid],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
            workers[sid]=(proc,err);ready=response(proc);startup.append(ready)
            if not ready.get('ready'):raise RuntimeError(ready)
            dump(output/'startup.json',startup);print(json.dumps({'started':sid}),flush=True)
            if spec.get('peer_response_dir'):
                proc.stdin.write('{"command":"replay"}\n');proc.stdin.flush();replay=response(proc)
                dump(output/f'{sid}.replay.json',replay)
                if replay.get('failure_status'):raise RuntimeError(replay)
        usage.append(representative)
        return workers[representative][0]
    try:
        schedule=[];rng=random.Random(spec['seed'])
        for batch in spec.get('batch_sizes',[1,8]):
            for cap in spec.get('context_lengths',[128,512,0]):
                for block in range(spec.get('blocks',10)):
                    order=[s['system_id'] for s in spec['systems']];rng.shuffle(order)
                    if profiling:
                        # Development profiles use randomized pair groups to
                        # amortize checkpoint loading; final timings interleave
                        # all systems independently in each paired block.
                        groups=list(group_for.values());rng.shuffle(groups)
                        order=[sid for group in groups for sid in order if worker_for[sid]==group]
                    for ix,sid in enumerate(order):
                        keep=ix+1<len(order) and worker_for[sid]==worker_for[order[ix+1]]
                        schedule.append({'system_id':sid,'command':'run','block':block,'batch_size':batch,'context_tokens':cap,'execution_order':ix,'warmup_requests':spec.get('warmup_requests',20),'timed_requests':spec.get('timed_requests_per_block',20),'keep_active':keep})
        if only_block is not None:schedule=[s for s in schedule if s['block']==only_block]
        dump(output/'schedule.json',schedule)
        with (output/'raw_blocks.jsonl').open('a') as f:
            for k,cmd in enumerate(schedule):
                proc=get_worker(cmd['system_id']);proc.stdin.write(json.dumps(cmd)+'\n');proc.stdin.flush()
                result=response(proc);f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
                dump(output/'progress.json',{'completed_blocks':k+1,'planned_blocks':len(schedule),'last':{x:result.get(x) for x in ('system_id','timing_block','batch_size','context_tokens','failure_status')},'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                print(json.dumps({'block_completed':k+1,'total':len(schedule),'system':cmd['system_id'],'failed':result.get('failure_status')}),flush=True)
    finally:
        for proc,err in workers.values():
            if proc.poll() is None:
                try:proc.stdin.write('{"command":"quit"}\n');proc.stdin.flush();proc.wait(timeout=30)
                except Exception:proc.terminate()
            err.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--system-id');p.add_argument('--only-block',type=int);a=p.parse_args()
    if a.worker:worker(a.spec,a.system_id)
    else:run(a.spec,a.output,a.only_block)

if __name__=='__main__':main()
