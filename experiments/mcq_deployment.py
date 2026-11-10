"""Minimal inference-only MCQ adapter; target outputs are never an input.

The public KRR support and used eigenvector columns are deployed and charged.
Validation projections/eigenvalues/unused basis columns are not deployed.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import adapters_v2 as adapters
core=adapters.core

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def export_bundle(state_path,context_path,registry,target_index,dimension,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    state_path=Path(state_path)
    encoded,matrix=core.decode(state_path.read_bytes())
    if target_index in encoded.pair:raise ValueError('Target is a callable peer')
    if any(registry[p]['model_id']==registry[target_index]['model_id'] and registry[p]['revision']==registry[target_index]['revision'] for p in encoded.pair):
        raise ValueError('Target checkpoint leaked under an alias')
    shutil.copyfile(state_path,output/'state.bin')
    public={}
    if encoded.family in ('krr','linear_ridge','nnls_plus_low_rank_residual'):
        with np.load(context_path,allow_pickle=False) as data:
            for name in ('mean','scale'):public[name]=data[name]
            if encoded.family=='krr':
                for name in ('sigma','features'):public[name]=data[name]
                public['eigenvectors']=data['eigenvectors'][:,:encoded.rank]
        np.savez(output/'public_context.npz',**public)
    manifest={'format':'replacement_mcq_deploy_v2','dimension':int(dimension),'target_index':int(target_index),
        'target_model_id':registry[target_index]['model_id'],'target_revision':registry[target_index]['revision'],
        'callable_peers':[{**registry[p],'index':p} for p in encoded.pair],
        'registry_size':len(registry),'adapter_family':encoded.family,'state_sha256':sha(output/'state.bin'),
        'state_binary_bytes':len(encoded.blob),'meaningful_bits':encoded.meaningful_bits,
        'decoded_coefficient_bytes':matrix.nbytes,'public_arrays_resident_bytes':sum(v.nbytes for v in public.values()),
        'public_context_sha256':sha(output/'public_context.npz') if public else None,
        'source_context_sha256':sha(context_path) if public else None,
        'source_state_sha256':sha(state_path),
        'code_sha256':{Path(__file__).name:sha(__file__),Path(adapters.__file__).name:sha(adapters.__file__),Path(core.__file__).name:sha(core.__file__)},
        'execution':'Only callable peer models and decoded state. No target loading, target calls, fit/validation responses, or experimental projections.',
        'accounting':'Binary state and manifest routing metadata counted separately. Shared context charged in full; model weights charged separately in standalone accounting.'}
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest

class DeployedAdapter:
    def __init__(self,bundle):
        self.bundle=Path(bundle)
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['format']!='replacement_mcq_deploy_v2':raise ValueError('Unknown format')
        for source in (Path(__file__),Path(adapters.__file__),Path(core.__file__)):
            if sha(source)!=self.manifest['code_sha256'][source.name]:raise ValueError('Deployment code changed since export')
        if sha(self.bundle/'state.bin')!=self.manifest['state_sha256']:raise ValueError('State changed')
        self.encoded,self.matrix=core.decode((self.bundle/'state.bin').read_bytes())
        self.pair=self.encoded.pair
        if self.manifest['target_index'] in self.pair:raise ValueError('Target is a peer')
        if self.pair!=tuple(x['index'] for x in self.manifest['callable_peers']):raise ValueError('Peer order mismatch')
        self.context=None
        context_path=self.bundle/'public_context.npz'
        if self.manifest['public_context_sha256']:
            if sha(context_path)!=self.manifest['public_context_sha256']:raise ValueError('Public context changed')
            with np.load(context_path,allow_pickle=False) as z:
                self.context=SimpleNamespace(**{k:z[k] for k in z.files})
        self.accounting={'state_binary_bytes':len(self.encoded.blob),
            'target_routing_manifest_bytes':(self.bundle/'manifest.json').stat().st_size,
            'resident_target_coefficients_bytes':self.matrix.nbytes,
            'resident_target_encoded_bytes':len(self.encoded.blob),
            'shared_deployment_disk_bytes':context_path.stat().st_size if context_path.exists() else 0,
            'shared_deployment_resident_bytes':sum(v.nbytes for v in vars(self.context).values()) if self.context else 0}

    def predict(self,peer_coordinates):
        if set(peer_coordinates)!=set(self.pair):raise ValueError('Provide exactly the callable peer coordinates')
        arrays=[np.asarray(peer_coordinates[p],dtype=np.float64) for p in self.pair]
        n,d=arrays[0].shape
        if d!=self.manifest['dimension'] or any(a.shape!=(n,d) for a in arrays):raise ValueError('Coordinate shape mismatch')
        if not all(np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite coordinates')
        # Unused slots carry no model outputs and are never accessed by a family.
        peers=np.zeros((n,self.manifest['registry_size'],d),dtype=float)
        for p,a in zip(self.pair,arrays):peers[:,p,:]=a
        projection=None
        if self.encoded.family=='krr':
            x=peers[:,self.pair,:].reshape(n,-1)
            x=(x-self.context.mean)/self.context.scale
            projection=core.rbf(x,self.context.features,float(self.context.sigma))@self.context.eigenvectors
        return adapters.predict(self.encoded,self.matrix,peers,self.context,projection)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,required=True);p.add_argument('--context',type=Path)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--target-index',type=int,required=True)
    p.add_argument('--dimension',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();obj=json.loads(a.registry.read_text());registry=obj['model_registry'] if isinstance(obj,dict) else obj
    export_bundle(a.state,a.context,registry,a.target_index,a.dimension,a.output)
    print(json.dumps(DeployedAdapter(a.output).accounting,indent=2))

if __name__=='__main__':main()
