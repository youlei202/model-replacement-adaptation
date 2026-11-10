"""Measured adapter replay costs, with peer-model collection kept separate."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from .deployment import DeployedPredictor
from .data import RichDevelopmentCache
from .sprint import ROOT, sha, dump


def measure(catalog, output, device='cuda:0', repeats=50, batches=(1,16,64)):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve prior resource run; use new directory')
    out.mkdir(parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device(device);cache=RichDevelopmentCache();records=[]
    decisions=pd.DataFrame(json.loads(Path(catalog).read_text()))
    unique=decisions.drop_duplicates(['target_index','pair','candidate_id','state_format'])
    public_payloads={}
    for row in unique.to_dict('records'):
        pair=tuple(map(int,row['pair'].split(',')))
        gc.collect()
        if dev.type=='cuda':torch.cuda.empty_cache();torch.cuda.synchronize(dev)
        before=torch.cuda.memory_allocated(dev) if dev.type=='cuda' else None
        model=DeployedPredictor(row,dev).eval()
        public_arrays={name:buf.detach().cpu().numpy() for name,buf in
                       [('basis',model.basis),('mean',model.mean),('scale',model.scale)] if buf is not None}
        digest=row.get('public_context_digest')
        numeric_public=sum(x.nbytes for x in public_arrays.values())
        serialized_public=0
        if public_arrays:
            key=str(digest)+'_rank'+str(model.state['basis_rank'])
            if key not in public_payloads:
                path=out/'public_context_payloads'/f'{key}.npz';path.parent.mkdir(exist_ok=True)
                np.savez(path,**public_arrays)
                public_payloads[key]={'path':str(path),'sha256':sha(path),'actual_bytes':path.stat().st_size,
                                      'numeric_bytes':numeric_public,'scope':'only deployed public basis/normalizer; excludes cached training features'}
            serialized_public=public_payloads[key]['actual_bytes']
        residency=torch.cuda.memory_allocated(dev)-before if dev.type=='cuda' else None
        for batch in batches:
            peer=torch.stack([torch.tensor(np.array(cache.logits('validation',p)[:batch],copy=True),device=dev) for p in pair],1)
            with torch.inference_mode():
                for _ in range(5):model(peer)
                if dev.type=='cuda':torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev)
                base_alloc=torch.cuda.memory_allocated(dev) if dev.type=='cuda' else None
                durations=[]
                for _ in range(repeats):
                    if dev.type=='cuda':torch.cuda.synchronize(dev)
                    started=time.perf_counter();prediction=model(peer)
                    if dev.type=='cuda':torch.cuda.synchronize(dev)
                    durations.append(time.perf_counter()-started)
                    del prediction
            record={'target_index':int(row['target_index']),'pair':row['pair'],'candidate_id':row['candidate_id'],
                    'family':row['family'],'state_format':row['state_format'],'codec':row['codec'],
                    'target_serialized_bytes':int(row['actual_bytes']),
                    'target_decoded_numeric_bytes':sum(x.nbytes for x in model.state['arrays']),
                    'public_numeric_bytes':numeric_public,'public_deployment_npz_bytes':serialized_public,
                    'device':str(dev),'device_name':torch.cuda.get_device_name(dev) if dev.type=='cuda' else platform.processor(),
                    'batch':batch,'repeats':repeats,'median_ms':float(np.median(durations)*1000),
                    'p95_ms':float(np.quantile(durations,.95)*1000),'mean_ms':float(np.mean(durations)*1000),
                    'model_load_GPU_resident_increment_bytes':residency,
                    'inference_GPU_peak_increment_bytes':torch.cuda.max_memory_allocated(dev)-base_alloc if dev.type=='cuda' else None,
                    'peer_response_input_bytes':peer.numel()*peer.element_size(),
                    'scope':'adapter only: selected-peer responses already resident; includes public feature projection, normalization, correction, full log_softmax, and current decoder coefficient placement',
                    'peer_model_forward_included':False,'training_response_cache_deployed':False}
            records.append(record);pd.DataFrame(records).to_csv(out/'ADAPTER_RESOURCE_MEASUREMENTS.csv',index=False)
            del peer
        del model
    dump(out/'PUBLIC_DEPLOYMENT_PAYLOADS.json',public_payloads)
    dump(out/'complete.json',{'complete':True,'states':len(unique),'measurements':len(records),
         'source_sha256':sha(Path(__file__)),'policy_decisions_sha256':sha(catalog),
         'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),'device':str(dev),
         'limitations':'Adapter-only wall latency with warmups. No peer-model end-to-end serving, queueing, throughput, exact energy, or total service VRAM claim. Decoded NumPy coefficients are placed by the current predictor during forward; resident GPU context measured separately.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--catalog',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--repeats',type=int,default=50);parser.add_argument('--batches',type=int,nargs='+',default=[1,16,64]);args=parser.parse_args()
    measure(args.catalog,args.output,args.device,args.repeats,tuple(args.batches))
