"""Development-only ST-GNN/A6 offline integration. See README before use."""
import argparse, copy, csv, hashlib, json, math, random
from pathlib import Path
import numpy as np

FEATURES=['claimed_target_x_m','claimed_target_y_m','claimed_target_speed_mps','claimed_target_heading_deg']
def seed_list(s): return {int(v) for v in s.split(',')}
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def discover(root):
    found=[];seen=set()
    for p in sorted(Path(root).rglob('*.csv')):
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not set(FEATURES+['report_sequence','report_created_time_s']).issubset(reader.fieldnames or []): continue
            rows=list(reader)
        if not rows: continue
        evp=Path(str(p)+'.evaluation.json');opp=Path(str(p)+'.operational.json')
        if not evp.exists() or not opp.exists(): continue
        ev=json.loads(evp.read_text());op=json.loads(opp.read_text())
        if ev.get('interrupted'): continue
        if op.get('schema')!='a6-independent-rsu-v3': continue
        key=(int(rows[0]['seed']),rows[0]['map'],rows[0]['collusion_fraction'],tuple(sorted({r['reporter_attack'] for r in rows})))
        if key in seen: raise ValueError(f'Duplicate scenario/attempt: {p}; keep one verified attempt per condition')
        seen.add(key)
        for r in rows:
            for k in FEATURES+['beacon_rx_time_s','report_created_time_s']:
                r[k]=float(r[k])
                if not math.isfinite(r[k]): raise ValueError(f'Nonfinite {k} in {p}')
            for k in ['seed','reporter_id','target_id','beacon_sequence','report_sequence','target_is_malicious','reporter_is_byzantine']:r[k]=int(r[k])
        ids=[r['report_sequence'] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate report IDs: {p}')
        found.append((p,rows))
    if not found: raise ValueError('No complete v3 CSV + operational/evaluation sidecar sets found')
    return found

def graph(history,current,window):
    """Observable claimed kinematics only; labels are returned separately."""
    end=current[0]['beacon_sequence'];cut=min(r['report_created_time_s'] for r in current)
    avail=[r for r in history if end-window<r['beacon_sequence']<=end and r['beacon_rx_time_s']<=cut]
    nodes=sorted({r['target_id'] for r in avail});index={n:i for i,n in enumerate(nodes)}
    x=np.zeros((len(nodes),window,7),np.float32);x[:,:,6]=1
    for r in sorted(avail,key=lambda r:r['beacon_rx_time_s']):
        i=index[r['target_id']];t=r['beacon_sequence']-(end-window+1)
        x[i,t,:4]=[r[k] for k in FEATURES];x[i,t,6]=0
    present=x[:,:,6]==0;raw=x[:,:,:2].copy()
    # Common coordinate origin preserves between-vehicle offsets.
    if present.any(): x[:,:,:2][present]-=raw[present].mean(axis=0)
    for t in range(1,window):
        ok=present[:,t]&present[:,t-1];x[ok,t,4:6]=raw[ok,t]-raw[ok,t-1]
    n=len(nodes);edge=np.zeros((n,n,4),np.float32);adj=np.eye(n,dtype=bool)
    for i in range(n):
        for j in range(n):
            both=present[i]&present[j];count=int(both.sum())
            if count:
                dxy=raw[j,both]-raw[i,both];d=np.linalg.norm(dxy,axis=1);angle=np.arctan2(dxy[:,1],dxy[:,0])
                edge[i,j]=[d.mean(),d.std(),np.arctan2(np.sin(angle).mean(),np.cos(angle).mean()),count/window]
                adj[i,j]|=bool(d.min()<=100 or count>=math.ceil(window/2))
    labels={r['target_id']:r['target_is_malicious'] for r in avail}
    y=np.array([labels[n] for n in nodes],np.float32)
    selected=[(index[r['target_id']],r) for r in current if r['target_id'] in index]
    return dict(x=x,edge=edge,adj=adj,y=y,selected=selected)

def examples(rows,window):
    from collections import defaultdict
    histories=defaultdict(list);groups=defaultdict(list)
    for r in rows:
        histories[r['reporter_id']].append(r)
        if r['beacon_sequence']>=window:
            groups[(r['reporter_id'],r['beacon_sequence'],r['report_created_time_s'])].append(r)
    return [graph(histories[k[0]],v,window) for k,v in sorted(groups.items())]

def make_model():
    import torch
    from torch import nn
    class EdgeGAT(nn.Module):
        def __init__(self,dim):
            super().__init__();self.self_map=nn.Linear(dim,128,bias=False)
            self.msg=nn.Linear(dim+4,128,bias=False);self.att=nn.Parameter(torch.empty(4,64));nn.init.xavier_uniform_(self.att)
        def forward(self,h,e,adj):
            n=len(h);self_h=self.self_map(h).view(n,4,32)
            neigh=h[None,:,:].expand(n,n,-1)
            msg=self.msg(torch.cat([neigh,e],-1)).view(n,n,4,32)
            own=self_h[:,None].expand(n,n,4,32)
            logits=torch.nn.functional.leaky_relu((torch.cat([own,msg],-1)*self.att).sum(-1),.2)
            weights=logits.masked_fill(~adj[:,:,None],float('-inf')).softmax(dim=1)
            return torch.relu(self_h+(weights[:,:,:,None]*msg).sum(1)).reshape(n,128)
    class Network(nn.Module):
        def __init__(self):
            super().__init__();self.gru=nn.GRU(7,64,batch_first=True);self.gat1=EdgeGAT(64);self.gat2=EdgeGAT(128)
            self.drop=nn.Dropout(.3);self.mlp=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,x,e,adj):
            _,h=self.gru(x);h=self.drop(self.gat1(h[0],e,adj));h=self.drop(self.gat2(h,e,adj));return self.mlp(h).squeeze(-1)
    return Network()

def tensors(g,stats):
    import torch
    x=(g['x']-np.array(stats['mean']))/np.array(stats['std'])
    # Fixed unit conversion, not statistics fitted on test data.
    e=g['edge']/np.array([100,100,np.pi,1],np.float32)
    return torch.tensor(x,dtype=torch.float32),torch.tensor(e,dtype=torch.float32),torch.tensor(g['adj'])

def train(a):
    import torch
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    tr,va=seed_list(a.train_seeds),seed_list(a.val_seeds)
    if tr&va:raise ValueError('Training and validation seeds overlap')
    sets={'train':[],'val':[]};sources=[]
    for p,rows in discover(a.input):
        seed=rows[0]['seed'];split='train' if seed in tr else 'val' if seed in va else None
        if split is None:continue
        # Train only non-Byzantine-report conditions to avoid repeated attack variants.
        if any(r['reporter_is_byzantine'] for r in rows):continue
        gs=examples(rows,a.window);sets[split].extend(gs);sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sidecar_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(p.parent.glob(p.name+'.*')) if q.is_file()}))
    if not all(sets.values()):raise ValueError('Need control runs for both seed sets and enough rounds for window')
    flat=np.concatenate([g['x'].reshape(-1,7) for g in sets['train']]);stats=dict(mean=flat.mean(0).tolist(),std=np.maximum(flat.std(0),1e-6).tolist())
    labels=np.concatenate([g['y'] for g in sets['train']]);pos=labels.sum();neg=len(labels)-pos
    if not pos or not neg:raise ValueError('Training requires both classes')
    net=make_model();optim=torch.optim.Adam(net.parameters(),lr=3e-4);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos))
    best=float('inf');history=[];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        net.train();random.shuffle(sets['train']);total=0
        for start in range(0,len(sets['train']),64):
            batch=sets['train'][start:start+64];optim.zero_grad()
            for g in batch:
                loss=lossfn(net(*tensors(g,stats)),torch.tensor(g['y']));(loss/len(batch)).backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
        net.eval();vl=0
        with torch.no_grad():
            for g in sets['val']:vl+=float(lossfn(net(*tensors(g,stats)),torch.tensor(g['y'])))
        vl/=len(sets['val']);history.append(dict(epoch=epoch+1,train_loss=total/len(sets['train']),val_loss=vl))
        print(history[-1],flush=True)
        if vl<best:
            best=vl;torch.save(dict(state=net.state_dict(),stats=stats,window=a.window,train_seeds=sorted(tr),val_seeds=sorted(va),sources=sources,kind='CARLA_FULL_HISTORY_DEVELOPMENT_NOT_EXACT_TRE'),out/'model.pt')
    write_json(out/'training_history.json',history)

def submitted(row,neural):
    """Replay an existing attack intervention. Truth never enters neural inputs."""
    if not row['reporter_is_byzantine'] or row['reporter_attack']=='none':return neural
    attack=row['reporter_attack'];old=float(row['submitted_risk_score']);local=float(row['local_risk_score'])
    if attack in ('coordinated','random_score'):return old
    if attack=='on_off':
        # Recorded active outputs are exactly .01/.99. Ambiguous cases are refused.
        if min(abs(old-.01),abs(old-.99))<1e-5:
            if abs(old-local)<1e-5:raise ValueError('Ambiguous on/off intervention; need explicit attack-active logging')
            return old
        if abs(old-local)>1e-4:raise ValueError('Unrecognized on/off score intervention')
        return neural
    raise ValueError(f'Unsupported attack replay: {attack}')

def predict(a):
    import torch
    from evaluate_rules import operational_decisions,dump_csv
    from v3_reference import metric
    torch.set_num_threads(a.threads)
    ck=torch.load(a.model,map_location='cpu',weights_only=True);seeds=seed_list(a.test_seeds)
    if ck.get('kind')!='CARLA_FULL_HISTORY_DEVELOPMENT_NOT_EXACT_TRE':raise ValueError('Use a checkpoint trained with full_history.py, not the old CSV-only model')
    if seeds & set(ck['train_seeds']+ck['val_seeds']):raise ValueError('Test seeds overlap model development seeds')
    net=make_model();net.load_state_dict(ck['state']);net.eval();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    metrics=[];predictions=[];audits=[]
    for p,rows in discover(a.input):
        if rows[0]['seed'] not in seeds:continue
        scores={}
        with torch.no_grad():
            for g in examples(rows,ck['window']):
                prob=net(*tensors(g,ck['stats'])).sigmoid().tolist()
                for i,r in g['selected']:scores[r['report_sequence']]=submitted(r,prob[i])
        op=json.loads(Path(str(p)+'.operational.json').read_text());ev=json.loads(Path(str(p)+'.evaluation.json').read_text())
        before=len(op['reports']);op['reports']=[dict(r,submitted_score=scores[r['report_id']]) for r in op['reports'] if r['report_id'] in scores]
        schedule=[{k:s[k] for k in ('beacon_sequence','target_id','beacon_tx_time_s')} for s in ev['transmitted_beacons'] if s['beacon_sequence']>=ck['window']]
        decisions,_=operational_decisions(op,schedule);malicious=set(ev['malicious_target_ids']);run=p.relative_to(a.input).as_posix()
        # Report only the explicitly implemented baselines, never label mean as TRE fuzzy.
        keep={'Mean','Median','RSU_only','Static_capped','Temporal_only','Temporal_capped'}
        for d in decisions:d.update(run=run,seed=rows[0]['seed'],label=int(d['target_id'] in malicious))
        for method in sorted(keep):
            rr=[d for d in decisions if d['method']==method];metrics.append(dict(run=run,seed=rows[0]['seed'],method=method,**metric(rr)));predictions.extend(rr)
        shared={ (d['beacon_sequence'],d['target_id']) for d in decisions if d['method']=='Temporal_capped' and d['prediction'] is not None }
        shared &= {(d['beacon_sequence'],d['target_id']) for d in decisions if d['method']=='RSU_only' and d['prediction'] is not None}
        for method in ('Temporal_capped','RSU_only'):
            rr=[d for d in decisions if d['method']==method and (d['beacon_sequence'],d['target_id']) in shared]
            metrics.append(dict(run=run,seed=rows[0]['seed'],method=method+'_common',**metric(rr)))
        audits.append(dict(run=run,original_reports=before,neural_reports=len(op['reports']),scheduled_cases=len(schedule),csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if not audits:raise ValueError('No matching test runs')
    dump_csv(out/'metrics.csv',metrics);dump_csv(out/'decisions.csv',predictions)
    write_json(out/'audit.json',dict(mode='offline_full_history_counterfactual',model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),runs=audits))
    print(f'Saved {len(audits)} run comparisons to {out}')

def audit(a):
    report=[]
    for p,rows in discover(a.input):
        seq=sorted({r['beacon_sequence'] for r in rows});times={r['beacon_sequence']:float(r['beacon_tx_time_s']) for r in rows}
        dt=np.median([(times[j]-times[i])/(j-i) for i,j in zip(seq,seq[1:])]) if len(seq)>1 else None
        report.append(dict(file=str(p),seed=rows[0]['seed'],rows=len(rows),rounds=len(seq),slot_seconds=dt,window_slots=a.window,window_seconds=None if dt is None else dt*a.window,missing_observer_self_node=True,rsu_survivor_bias=True))
    write_json(a.output,report);print(f'Found {len(report)} runs. Audit saved to {a.output}')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('audit');q.add_argument('--input',type=Path,required=True);q.add_argument('--output',default='data_audit.json');q.add_argument('--window',type=int,default=20)
    q=sub.add_parser('train');q.add_argument('--input',type=Path,required=True);q.add_argument('--train-seeds',required=True);q.add_argument('--val-seeds',required=True);q.add_argument('--output',default='trained');q.add_argument('--window',type=int,default=20);q.add_argument('--epochs',type=int,default=30);q.add_argument('--seed',type=int,default=42);q.add_argument('--threads',type=int,default=2)
    q=sub.add_parser('predict');q.add_argument('--input',type=Path,required=True);q.add_argument('--model',required=True);q.add_argument('--test-seeds',required=True);q.add_argument('--output',default='comparison');q.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if hasattr(a,'window') and a.window<2:p.error('window must be at least 2')
    if hasattr(a,'epochs') and a.epochs<1:p.error('epochs must be positive')
    {'audit':audit,'train':train,'predict':predict}[a.command](a)
if __name__=='__main__':main()
