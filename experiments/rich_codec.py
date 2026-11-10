"""Version2 explicit state codec: global peer mask, uint32 shapes, literal byte accounting."""
from __future__ import annotations
import struct
import numpy as np
MAGIC=b'RBV2';VERSION=2
FAMILIES={'logit_mean':0,'probability_mean':1,'nnls':2,'temperature':3,'vocabulary_bias':4,'ridge_residual':5,'factored_residual':6,'convex':7}
HEADER=struct.Struct('<4sBBBBIIII') #24 bytes
ARRAY=struct.Struct('<IIIIf') #20 bytes shape, count and scale
CODECS={'fp32':0,'fp16':1,'int8':2,'int4':3}

def encode(family,peers,basis_rank,arrays,codec='fp32',vocab=151936):
 if family not in FAMILIES or codec not in CODECS:raise ValueError('Unknown family/codec')
 if not peers or sorted(set(peers))!=list(peers) or max(peers)>7:raise ValueError('Ascending global peer IDs required')
 payload=bytearray()
 for a in arrays:
  a=np.asarray(a,dtype=np.float32)
  if a.ndim not in (1,2) or not np.isfinite(a).all():raise ValueError('Finite arrays1/2D required')
  shape=(len(a),0) if a.ndim==1 else a.shape
  if max(shape)>2**32-1:raise ValueError('uint32 dimensions exceeded')
  f=a.ravel();scale=1.
  if codec=='fp32':b=f.astype('<f4').tobytes()
  elif codec=='fp16':
   if np.max(np.abs(f),initial=0)>65504:raise ValueError('FP16 overflow')
   b=f.astype('<f2').tobytes()
  else:
   bits=8 if codec=='int8' else 4;limit=(1<<(bits-1))-1;scale=float(np.max(np.abs(f),initial=0))/limit or 1.;v=np.rint(f/scale).clip(-limit,limit).astype(np.int8)
   if bits==8:b=v.tobytes()
   else:
    nib=(v.astype(np.int16)&15).astype(np.uint8)
    if len(nib)%2:nib=np.r_[nib,np.uint8(0)]
    b=(nib[::2]|(nib[1::2]<<4)).tobytes()
  payload.extend(ARRAY.pack(shape[0],shape[1],len(f),len(b),scale));payload.extend(b)
 return HEADER.pack(MAGIC,VERSION,FAMILIES[family],sum(1<<p for p in peers),CODECS[codec],int(basis_rank),int(vocab),len(arrays),len(payload))+payload

def decode(blob):
 if len(blob)<HEADER.size:raise ValueError('Truncated header')
 magic,version,fid,mask,cid,rank,vocab,narr,size=HEADER.unpack_from(blob)
 if magic!=MAGIC or version!=VERSION or size!=len(blob)-HEADER.size:raise ValueError('Invalid state header')
 family=next((k for k,v in FAMILIES.items() if v==fid),None);codec=next((k for k,v in CODECS.items() if v==cid),None)
 if family is None or codec is None:raise ValueError('Unknown encoded IDs')
 off=HEADER.size;arrays=[]
 for _ in range(narr):
  rows,cols,n,nb,scale=ARRAY.unpack_from(blob,off);off+=ARRAY.size
  if n!=rows*(cols or 1) or off+nb>len(blob) or scale<=0:raise ValueError('Invalid payload descriptor')
  payload=blob[off:off+nb];off+=nb
  if codec in ('fp32','fp16'):v=np.frombuffer(payload,dtype='<f4' if codec=='fp32' else '<f2').astype(np.float32)
  elif codec=='int8':v=np.frombuffer(payload,dtype=np.int8).astype(np.float32)*scale
  else:
   b=np.frombuffer(payload,dtype=np.uint8);v=np.empty(len(b)*2,dtype=np.int8);v[::2]=b&15;v[1::2]=b>>4;v=np.where(v>=8,v-16,v).astype(np.float32)[:n]*scale
  if len(v)!=n:raise ValueError('Decoded size mismatch')
  arrays.append(v.reshape((rows,cols)) if cols else v)
 if off!=len(blob):raise ValueError('Trailing data')
 if family=='convex':
  if not arrays or np.any(arrays[0]<0) or arrays[0].sum()<=0:raise ValueError('Invalidconvexweights')
  arrays[0]=arrays[0]/arrays[0].sum()
 return {'family':family,'peers':[p for p in range(8) if mask&(1<<p)],'basis_rank':rank,'vocab':vocab,'codec':codec,'arrays':arrays,'serialized_bytes':len(blob),'resident_array_bytes':sum(a.nbytes for a in arrays)}

def self_test():
 for codec in CODECS:
  for shape in [(3,),(2,300),(70000,1)]:
   a=np.linspace(-2,3,np.prod(shape),dtype=np.float32).reshape(shape);b=encode('ridge_residual',[0,2],16,[a],codec);d=decode(b)
   assert d['arrays'][0].shape==shape and d['peers']==[0,2] and d['serialized_bytes']==len(b)
   tol={'fp32':0,'fp16':.002,'int8':.013,'int4':.22}[codec]
   assert np.max(np.abs(a-d['arrays'][0]))<=tol
 print('rich codec tests passed')
if __name__=='__main__':self_test()
