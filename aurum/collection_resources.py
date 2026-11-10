"""Summarize measured canonical response collection; do not call it serving."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .sprint import sha,dump


def summarize(cohort,output):
    cohort=Path(cohort);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];timings={};sources={}
    for p in sorted((cohort/'logits').glob('model_*.json')):
        m=json.loads(p.read_text())
        if not m.get('complete'):raise ValueError('Incomplete response bank')
        observations=m['batch_timings'];seconds=np.array([x['seconds'] for x in observations])
        if any(x['batch']!=1 for x in observations):raise ValueError('Expected canonical microbatch1')
        timings[m['model_index']]=seconds;sources[str(p)]=sha(p)
        rows.append({'model_index':m['model_index'],'model_id':m['model_id'],'revision':m['model_revision'],
                     'role':m['role'],'documents':m['expected'],'device':m['runtime']['device'],
                     'inference_seconds':float(seconds.sum()),'seconds_including_load_and_io':m['seconds_including_load'],
                     'per_document_median_ms':float(np.median(seconds)*1000),'per_document_p95_ms':float(np.quantile(seconds,.95)*1000),
                     'per_document_mean_ms':float(seconds.mean()*1000),'GPU_peak_allocated_bytes':m['memory_peak_allocated'],
                     'GPU_peak_reserved_bytes':m['memory_peak_reserved'],
                     'cache_disk_bytes':p.with_suffix('.npy').stat().st_size,'cohort_sha256':m['cohort_sha256'],
                     'scope':'Measured collection forward timings on fixed512token prefixes, BF16 microbatch1; includes initial cold observation; excludes co-resident pair serving'})
    pd.DataFrame(rows).to_csv(out/'MEASURED_COLLECTION_RESOURCES.csv',index=False)
    pairs=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        if a in timings and b in timings:
            if len(timings[a])!=len(timings[b]):raise ValueError('Unaligned collection cohorts')
            summed=timings[a]+timings[b]
            pairs.append({'pair':f'{a},{b}','sum_of_separate_jobs_mean_ms_per_document':float(summed.mean()*1000),
                          'sum_of_separate_jobs_median_ms_per_document':float(np.median(summed)*1000),
                          'scope':'Conditional accounting sum of independently measured model forwards; NOT measured end-to-end pair serving latency or total resident VRAM'})
    pd.DataFrame(pairs).to_csv(out/'CONDITIONAL_PEER_FORWARD_SUMS.csv',index=False)
    dump(out/'collection_resources_provenance.json',{'sources':sources,'timing_interpretation':'Historical recorded wall-clock forward intervals, one per document; no new inference in this summarizer',
         'total_worker_seconds_including_load_and_io':sum(r['seconds_including_load_and_io'] for r in rows),
         'total_GPU_forward_seconds':sum(r['inference_seconds'] for r in rows),
         'parallel_wall_time_is_not_sum_of_worker_seconds':True,'scope':'collection, not deployment service benchmark'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args();summarize(a.cohort,a.output)
