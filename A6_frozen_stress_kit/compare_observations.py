"""Matched development comparison. Test seeds deliberately excluded."""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
import observation_adapter as adapter
import full_history as history
import stgnn_a6 as base
TRAIN=set(range(2041,2047));VAL={2047,2048,2049}
def metrics(y,s):
 y=np.asarray(y);s=np.asarray(s);p=s>=.5
 tp=int(sum(p&(y==1)));fp=int(sum(p&(y==0)));tn=int(sum(~p&(y==0)));fn=int(sum(~p&(y==1)))
 return dict(n=len(y),tp=tp,fp=fp,tn=tn,fn=fn,f1=2*tp/max(1,2*tp+fp+fn),recall=tp/max(1,tp+fn),false_positive_rate=fp/max(1,fp+tn))
def load(root):
 out={};sources={}
 for p,rows in history.discover(root):
  seed=rows[0]['seed']
  if seed not in TRAIN|VAL:raise ValueError('Only development seeds 2041–2049 allowed')
  if seed in out or any(r['reporter_is_byzantine'] or r['reporter_attack']!='none' for r in rows):raise ValueError('Need one control per seed')
  obs=adapter.load_observations(p,rows);gs=adapter.examples(rows,20,obs)
  if not gs:raise ValueError('No complete windows')
  out[seed]=gs;sources[seed]={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in p.parent.glob(p.name+'*') if q.is_file()}
  print(f'Built seed {seed}: {len(gs)} graphs',flush=True)
 if set(out)!=TRAIN|VAL:raise ValueError('Need all nine development controls')
 return out,sources
def selected(g):return [i for i,r in g['selected']]
def rule(groups):
 benign=[]
 for seed in TRAIN:
  for g in groups[seed]:
   for i in selected(g):
    if g['y'][i]==0 and g['obs_x'][i,-1,5]==0:benign.append(g['obs_x'][i,-1,2])
 if not benign:raise ValueError('No training benign residuals')
 threshold=float(np.quantile(benign,.95,method='higher'))
 result={}
 for seed,gs in groups.items():
  y=[];s=[];missing=0
  for g in gs:
   for i in selected(g):
    if g['obs_x'][i,-1,5]:missing+=1;continue
    y.append(g['y'][i]);s.append(float(g['obs_x'][i,-1,2]>threshold))
  result[seed]=dict(**metrics(y,s),abstained=missing)
 return dict(threshold_m=threshold,selection='95th percentile of benign training residuals; positive iff greater',per_seed=result)
def raw(g,variant):return g['x'] if variant=='baseline' else np.concatenate([g['x'],g['obs_x']],axis=-1)
def stats(groups,variant):
 x=np.concatenate([raw(g,variant).reshape(-1,7 if variant=='baseline' else 13) for seed in TRAIN for g in groups[seed]])
 mean=np.zeros(x.shape[1]);std=np.ones(x.shape[1])
 for cols,mask in [(range(6),6)]+([(range(7,12),12)] if variant=='assisted' else []):
  present=x[:,mask]==0
  for j in cols:
   mean[j]=x[present,j].mean();std[j]=max(float(x[present,j].std()),1e-3)
 return dict(mean=mean.tolist(),std=std.tolist())
def inputs(g,variant,norm):
 x=(raw(g,variant)-norm['mean'])/norm['std'];x=x.astype('float32')
 x[:,:,:6][g['x'][:,:,6]!=0]=0
 if variant=='assisted':x[:,:,7:12][g['obs_x'][:,:,5]!=0]=0
 return x,(g['edge']/np.array([100,100,np.pi,1])).astype('float32'),g['adj']
def train(groups,sources,args):
 import torch
 torch.set_num_threads(2)
 results={}
 for variant in ['baseline','assisted']:
  torch.manual_seed(42);random.seed(42);np.random.seed(42)
  net=base.make_model()
  if variant=='assisted':net.gru=torch.nn.GRU(13,64,batch_first=True)
  norm=stats(groups,variant)
  labels=np.concatenate([g['y'][selected(g)] for seed in TRAIN for g in groups[seed]])
  lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((labels==0).sum()/(labels==1).sum())))
  opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-3)
  training=[g for seed in sorted(TRAIN) for g in groups[seed]]
  def tensors(g):return [torch.from_numpy(v) for v in inputs(g,variant,norm)]
  def evaluate():
   net.eval();per={};total=0;count=0
   with torch.no_grad():
    for seed in sorted(VAL):
     y=[];s=[]
     for g in groups[seed]:
      idx=selected(g);logit=net(*tensors(g))[idx];target=torch.tensor(g['y'][idx]);total+=float(lossfn(logit,target))*len(idx);count+=len(idx)
      y.extend(target.tolist());s.extend(torch.sigmoid(logit).tolist())
     per[seed]=metrics(y,s)
   return dict(loss=total/count,per_seed=per)
  best=float('inf');bad=0;records=[];best_epoch=0
  def save(epoch):
   torch.save(dict(state=net.state_dict(),stats=norm,variant=variant,window=20,best_epoch=epoch,train_seeds=sorted(TRAIN),val_seeds=sorted(VAL),sources=sources,protocol='selected-report supervision; simulated observations; not exact TRE'),args.output/(variant+'_model.pt'))
  initial=evaluate();best=initial['loss'];save(0)
  for epoch in range(1,args.epochs+1):
   net.train();random.shuffle(training);total=0
   for start in range(0,len(training),64):
    batch=training[start:start+64];opt.zero_grad()
    for g in batch:
     idx=selected(g);loss=lossfn(net(*tensors(g))[idx],torch.tensor(g['y'][idx]));(loss/len(batch)).backward();total+=float(loss.detach())
    torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
   val=evaluate();row=dict(epoch=epoch,train_loss=total/len(training),validation=val);records.append(row)
   if val['loss']<best-1e-4:best=val['loss'];bad=0;best_epoch=epoch;save(epoch)
   else:bad+=1
   print(variant,json.dumps(row),flush=True)
   results[variant]=dict(initial_validation=initial,epochs=records,best_epoch=best_epoch)
   (args.output/'training_history.json').write_text(json.dumps(results,indent=2))
   if bad>=args.patience:break
  print(f'{variant}: saved epoch {best_epoch}',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--train',action='store_true');ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--patience',type=int,default=3)
 a=ap.parse_args()
 if a.output.exists():raise ValueError('Use a fresh output folder')
 groups,sources=load(a.input);result=rule(groups);a.output.mkdir(parents=True)
 (a.output/'residual_baseline.json').write_text(json.dumps(result,indent=2));(a.output/'sources.json').write_text(json.dumps(sources,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 if a.train:train(groups,sources,a)
