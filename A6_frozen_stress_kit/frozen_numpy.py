"""Restricted tensor decoding and independent NumPy inference, without PyTorch."""
import collections,io,pickle,zipfile,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import full_history as f
import stgnn_a6 as b
from threadpoolctl import threadpool_limits
threadpool_limits(1)

def checkpoint(path):
 z=zipfile.ZipFile(path);prefix=z.namelist()[0].split('/')[0]
 def tensor(storage,offset,size,stride,*args):
  # The file is inspected to contain float tensors only. Bounds checked explicitly.
  hi=offset+sum((n-1)*s for n,s in zip(size,stride))+1
  if offset<0 or any(s<0 for s in stride) or hi>len(storage):raise ValueError('Unsafe tensor bounds')
  return np.lib.stride_tricks.as_strided(storage[offset:],shape=size,strides=tuple(s*4 for s in stride)).copy()
 class Reader(pickle.Unpickler):
  def find_class(self,module,name):
   if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
   if (module,name)==('torch._utils','_rebuild_tensor_v2'):return tensor
   if (module,name)==('torch','FloatStorage'):return 'float32'
   raise ValueError((module,name))
  def persistent_load(self,identity):
   kind,dtype,key,device,length=identity
   if kind!='storage' or dtype!='float32':raise ValueError('Unexpected storage')
   a=np.frombuffer(z.read(prefix+'/data/'+key),dtype='<f4')
   if len(a)!=length:raise ValueError('Storage length mismatch')
   return a
 return Reader(io.BytesIO(z.read(prefix+'/data.pkl'))).load()

def sigmoid(x):return np.exp(-np.logaddexp(0,-x))
def infer(g,c):
 s=c['state'];x=g['_normalized_x'];e=(g['edge']/np.array([100,100,np.pi,1])).astype('f4');adj=g['adj'];n=len(x)
 h=np.zeros((n,64),dtype='f4')
 for t in range(x.shape[1]):
  ir,iz,inn=np.split(x[:,t]@s['gru.weight_ih_l0'].T+s['gru.bias_ih_l0'],3,axis=-1)
  hr,hz,hn=np.split(h@s['gru.weight_hh_l0'].T+s['gru.bias_hh_l0'],3,axis=-1)
  r=sigmoid(ir+hr);zz=sigmoid(iz+hz);nn=np.tanh(inn+r*hn);h=(1-zz)*nn+zz*h
 for layer in ['gat1','gat2']:
  own=(h@s[layer+'.self_map.weight'].T).reshape(n,4,32)
  neigh=np.broadcast_to(h[None],(n,n,h.shape[-1]));msg=(np.concatenate([neigh,e],-1)@s[layer+'.msg.weight'].T).reshape(n,n,4,32)
  v=np.concatenate([np.broadcast_to(own[:,None],(n,n,4,32)),msg],-1)
  logits=(v*s[layer+'.att']).sum(-1);logits=np.where(logits>=0,logits,.2*logits);logits=np.where(adj[:,:,None],logits,-np.inf)
  w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
  h=np.maximum(0,own+(w[:,:,:,None]*msg).sum(1)).reshape(n,128)
 for layer in ['0','3']:
  h=np.maximum(0,h@s['mlp.'+layer+'.weight'].T+s['mlp.'+layer+'.bias'])
 return (h@s['mlp.5.weight'].T+s['mlp.5.bias']).ravel()

