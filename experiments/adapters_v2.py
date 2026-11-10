"""Literal codecs and symmetric adapter menu for decisive replacement campaign.

Source RBTL families remain byte-compatible. Families7/8 add linear ridge and
NNLS+factorized reduced-rank residual; all target-dependent bias/factors/weights
are in the state. Input mean/scale are peer-only public deployment context.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import selection_core as core
core.FAMILIES.update({'linear_ridge':7,'nnls_plus_low_rank_residual':8})
FAMILIES=tuple(core.FAMILIES)

def pack_fast(values,width):
    vals=np.asarray(values).reshape(-1)
    if width>64: return original_pack(vals,width)
    vals=vals.astype(np.uint64)
    # Chunking bounds transient bit matrix memory even for unbounded full KRR.
    chunks=[]
    block=max(8,65536//width//8*8)
    for start in range(0,len(vals),block):
        bits=((vals[start:start+block,None] >> np.arange(width-1,-1,-1,dtype=np.uint64)) & 1).astype(np.uint8)
        chunks.append(np.packbits(bits.reshape(-1),bitorder='big').tobytes())
    return b''.join(chunks)

def unpack_fast(payload,count,width):
    if width>64:return original_unpack(payload,count,width)
    bits=np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder='big')[:count*width]
    if len(bits)!=count*width:raise ValueError('Truncated payload')
    return bits.reshape(count,width).astype(np.uint64) @ (np.uint64(1)<<np.arange(width-1,-1,-1,dtype=np.uint64))
original_pack=core._pack;original_unpack=core._unpack
core._pack=pack_fast;core._unpack=unpack_fast

def decode_values(matrix,family,codec):
    """Exact transmitted numerical values without materializing a candidate blob."""
    matrix=np.asarray(matrix,dtype=np.float64).reshape(len(matrix),-1)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite state')
    if family=='convex' and codec.startswith('q'):
        levels=(1<<int(codec[1:]))-1
        w=np.maximum(matrix.reshape(-1),0);w/=w.sum()
        raw=w*levels; counts=np.floor(raw).astype(np.int64)
        counts[np.argsort(-(raw-counts),kind='stable')[:levels-int(counts.sum())]]+=1
        result=(counts/levels).reshape(matrix.shape)
        return result/result.sum()
    if codec.startswith('native_fp'):
        dtype=np.dtype('<f'+str(int(codec.removeprefix('native_fp'))//8))
        if np.max(np.abs(matrix),initial=0)>np.finfo(dtype).max:raise ValueError('Native precision overflow')
        result=matrix.astype(dtype).astype(float)
        if family=='convex': result=np.maximum(result,0);result/=result.sum()
        return result
    width=int(codec[1:]);maximum=(1<<(width-1))-1 if width>1 else 1
    scale=np.max(np.abs(matrix),axis=0)/maximum
    scale=np.where(scale==0,1,scale)
    if np.max(scale)>np.finfo(np.float32).max:raise ValueError('Scale overflow')
    integers=np.where(matrix>=0,1,-1) if width==1 else np.clip(np.rint(matrix/scale),-maximum,maximum)
    return integers*scale.astype('<f4').astype(float)

def standardized(peers,pair,context):
    return (peers[:,pair,:].reshape(len(peers),-1)-context.mean)/context.scale

def residual_shape(matrix,q,d):
    r=(matrix.size-q-d)//(q*d+d)
    if q+d+r*(q*d+d)!=matrix.size or r<1:raise ValueError('Invalid residual state dimensions')
    return r

def predict(encoded,matrix,peers,context=None,projection=None):
    family=encoded.family;pair=encoded.pair
    if family=='linear_ridge':
        x=standardized(peers,pair,context)
        return x@matrix[:-1]+matrix[-1]
    if family=='nnls_plus_low_rank_residual':
        q=len(pair);d=peers.shape[-1];r=residual_shape(matrix,q,d)
        a=matrix.reshape(-1);u=a[q:q+q*d*r].reshape(q*d,r)
        v=a[q+q*d*r:-d].reshape(r,d);bias=a[-d:]
        return np.einsum('npd,p->nd',peers[:,pair,:],a[:q])+standardized(peers,pair,context)@u@v+bias
    return core.predict(encoded,matrix,peers,projection)

def linear_candidates(peers,target,pair,context,lambdas=core.LAMBDAS):
    x=standardized(peers,pair,context);n=len(x);bias=target.mean(0)
    yc=target-bias;gram=x.T@x;rhs=x.T@yc
    w=core.fit_weights(peers,target,pair,'nnls')
    residual=target-np.einsum('npd,p->nd',peers[:,pair,:],w)
    rbias=residual.mean(0);rrhs=x.T@(residual-rbias)
    for li,lam in enumerate(lambdas):
        reg=gram+n*lam*np.eye(gram.shape[0])
        matrix=np.linalg.solve(reg,rhs)
        yield 'linear_ridge',np.vstack((matrix,bias)),li,min(matrix.shape)
        rmatrix=np.linalg.solve(reg,rrhs)
        # Ridge reduced-rank prediction: truncate in fit-output metric, not just
        # coefficient Euclidean norm. V from fitted residual responses is the
        # least-squares reduced-rank output projection at this ridge solution.
        _,_,vt=np.linalg.svd(x@rmatrix,full_matrices=False)
        for rank in sorted({min(r,min(rmatrix.shape)) for r in (1,2,4,8)}):
            v=vt[:rank];u=rmatrix@v.T
            state=np.r_[w,u.reshape(-1),v.reshape(-1),rbias].reshape(-1,1)
            yield 'nnls_plus_low_rank_residual',state,li,rank
