"""Frozen, unused-seed CARLA local-detector test. No training or threshold tuning.
Use --phase plan first. All results, including failures, remain recorded.
"""
import argparse,csv,hashlib,json,random,statistics,subprocess,sys,time
from pathlib import Path

CONDITIONS=['a1_constant_position','a2_constant_offset','a3_random_offset','a4_eventual_stop','a5_collusive_offset','none']
NAMES=dict(zip(CONDITIONS,['A1','A2','A3','A4','A5','Benign']))
VARIANTS=['full','no_gru','no_gat','no_edges','no_observations']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(p.read_text())
def csvout(p,rows):
 if rows:
  with p.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(rows):
 c=dict(tp=0,fp=0,tn=0,fn=0)
 for r in rows:
  y=int(r['y_true']);pos=float(r['score'])>=.5;c[('tp' if y else 'fp') if pos else ('fn' if y else 'tn')]+=1
 div=lambda x,y:x/y if y else None
 return dict(n=len(rows),**c,precision=div(c['tp'],c['tp']+c['fp']),recall=div(c['tp'],c['tp']+c['fn']),
   f1=div(2*c['tp'],2*c['tp']+c['fp']+c['fn']) if c['tp']+c['fn'] else None,fpr=div(c['fp'],c['fp']+c['tn']))
def execute(cmd,log,timeout):
 with log.open('w',encoding='utf-8') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  start=time.monotonic()
  try:
   while p.poll() is None:
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:print('  running',int(time.monotonic()-start),'s',flush=True)
    if time.monotonic()-start>timeout and p.poll() is None:
     p.kill();p.wait();return dict(return_code=p.returncode,timed_out=True)
  except KeyboardInterrupt:
   p.terminate();p.wait();raise
 return dict(return_code=p.returncode,timed_out=False)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--phase',choices=['plan','collect','evaluate','all'],default='plan')
 ap.add_argument('--kit',type=Path,default=Path(__file__).parent)
 ap.add_argument('--collector-dir',type=Path,default=Path('../tre_target_attack_kit'))
 ap.add_argument('--checkpoints',type=Path,default=Path('ablation_development_v1'))
 ap.add_argument('--output',type=Path,default=Path('final_test_2050_2054'))
 ap.add_argument('--seeds',default='2050,2051,2052,2053,2054')
 ap.add_argument('--tm-start',type=int,default=11000)
 ap.add_argument('--duration',type=float,default=60)
 ap.add_argument('--timeout',type=int,default=600)
 a=ap.parse_args()
 seeds=[int(v) for v in a.seeds.split(',')]
 if not seeds or len(seeds)!=len(set(seeds)):ap.error('Use distinct simulation seeds.')
 if any(s in range(2030,2050) for s in seeds):ap.error('Seeds 2030-2049 were previously used. Select unused seeds.')
 if a.duration<30:ap.error('Use duration >=30 s for 20-slot eligibility.')
 a.kit=a.kit.resolve();a.collector_dir=a.collector_dir.resolve();a.checkpoints=a.checkpoints.resolve();a.output=a.output.resolve()
 required=['compare_observations.py','observation_adapter.py','full_history.py','stgnn_a6.py','check_observations.py','run_ablations.py']
 for name in required:
  if not (a.kit/name).exists():raise FileNotFoundError(a.kit/name)
 for name in ['carla_a6_v6_target_attacks.py','carla_a6_v6_moving_a4.py','check_target_attacks.py']:
  if not (a.collector_dir/name).exists():raise FileNotFoundError(a.collector_dir/name)
 tp=load(a.checkpoints/'protocol.json')
 if tp.get('smoke_only'):raise ValueError('Do not use smoke checkpoints.')
 if set(tp['variants'])!=set(VARIANTS):raise ValueError('Expected five variants.')
 if set(seeds)&set(tp['train_simulation_seeds']+tp['validation_simulation_seeds']):raise ValueError('Test seed overlaps development.')
 for name,h in tp['adapter_sha256'].items():
  if digest(a.kit/name)!=h:raise ValueError('Training adapter changed: '+name)
 checkpoints={f'{v}_init{i}':digest(a.checkpoints/f'{v}_init{i}'/'model.pt') for i in tp['training_initializations'] for v in VARIANTS}
 protocol=dict(rpc_timeout_s=180,seeds=seeds,conditions=CONDITIONS,duration=a.duration,map='Town10HD_Opt',vehicles=8,
  window=20,threshold=.5,tm_start=a.tm_start,training_protocol_sha256=digest(a.checkpoints/'protocol.json'),
  checkpoint_sha256=checkpoints,code_sha256={str(p):digest(p) for p in list(a.collector_dir.glob('*.py'))+[a.kit/n for n in required]},
  scope='Unused-seed local-detector evaluation; A4 post-deceleration; synthetic observations; no RSU replay claim')
 print(f'{len(seeds)*6} collection runs; frozen checkpoints; no retraining.',flush=True)
 print('Only call these untouched tests if these seeds/episodes have NEVER informed your decisions.',flush=True)
 if a.phase=='plan':
  for seed in seeds:
   for condition in CONDITIONS:print(seed,NAMES[condition],condition)
  print('Plan only: no simulator calls or output files.');return
 a.output.mkdir(parents=True,exist_ok=True)
 pp=a.output/'frozen_protocol.json'
 if pp.exists():
  saved_protocol = load(pp)

  if saved_protocol != protocol:
      for key in sorted(set(saved_protocol) | set(protocol)):
          old = saved_protocol.get(key)
          new = protocol.get(key)

          if old != new:
              print(f"\nPROTOCOL DIFFERENCE: {key}", flush=True)

              if isinstance(old, dict) and isinstance(new, dict):
                  for subkey in sorted(set(old) | set(new)):
                      if old.get(subkey) != new.get(subkey):
                          print(f"  {subkey}", flush=True)
                          print(f"    saved:   {old.get(subkey)}", flush=True)
                          print(f"    current: {new.get(subkey)}", flush=True)
              else:
                  print(f"  saved:   {old}", flush=True)
                  print(f"  current: {new}", flush=True)

      raise ValueError("Frozen protocol differs; inspect differences above.")
 else:dump(pp,protocol)
 if a.phase in ['collect','all']:
  for si,seed in enumerate(seeds):
   for ci,condition in enumerate(CONDITIONS):
    run=a.output/'collections'/f'seed{seed}_{NAMES[condition]}'
    run.mkdir(parents=True,exist_ok=True);success=run/'accepted.json'
    if success.exists():print('SKIP verified collection',run.name);continue
    attempt=run/f'attempt_{len(list(run.glob("attempt_*")))+1:03d}';attempt.mkdir()
    collector=a.collector_dir/('carla_a6_v6_moving_a4.py' if NAMES[condition]=='A4' else 'carla_a6_v6_target_attacks.py')
    event=attempt/'events.csv'
    cmd=[sys.executable,'-u',str(collector),
     '--rpc-timeout','180',
     '--map','Town10HD_Opt','--tm-port',str(a.tm_start+si*6+ci),
     '--seed',str(seed),'--num-vehicles','8','--duration',str(a.duration),'--warmup','5','--spawn-radius','150',
     '--filter-distance','1000','--beacon-rate','1','--target-malicious-fraction','0' if condition=='none' else '.25',
     '--target-attack',condition,'--collusion-fraction','0','--reporter-attack','none','--output',str(event)]
    dump(attempt/'command.json',cmd);print('COLLECT',run.name,flush=True)
    status=execute(cmd,attempt/'collector.log',a.timeout);dump(attempt/'status.json',status)
    if status['return_code']!=0 or status['timed_out']:raise RuntimeError(f'Collection failed. Inspect {attempt}/collector.log. Re-running resumes; failed attempts remain.')
    hm=load(Path(str(event)+'.history_manifest.json'));om=load(Path(str(event)+'.observations.manifest.json'))
    if hm['interrupted'] or not hm.get('fixed_fleet_valid',False) or hm.get('actor_losses') or hm['beacon_queue_remaining'] or om['interrupted']:
     raise ValueError('Run integrity failed; do not evaluate: '+str(attempt))
    if digest(Path(str(event)+'.observations.jsonl'))!=om['sha256']:raise ValueError('Observation hash mismatch')
    if condition!='none':
     st=execute([sys.executable,str(a.collector_dir/'check_target_attacks.py'),'--input',str(event)],attempt/'attack_audit.log',a.timeout)
     if st['return_code']!=0:raise ValueError('Attack-equation audit failed: '+str(attempt))
    dump(success,dict(attempt=attempt.name,files={f.name:digest(f) for f in attempt.glob('events.csv*') if f.is_file()}))
 if a.phase=='collect':print('Collection complete. Run --phase evaluate next.');return
 result=a.output/'evaluation'
 if (result/'COMPLETE.json').exists():print('Evaluation already complete:',result);return
 result.mkdir(exist_ok=True)
 sys.path.insert(0,str(a.kit))
 import torch
 import run_ablations as train
 import compare_observations as original
 import full_history as history
 import observation_adapter as adapter
 torch.set_num_threads(2);models={}
 for init in tp['training_initializations']:
  for v in VARIANTS:
   ck=torch.load(a.checkpoints/f'{v}_init{init}'/'model.pt',map_location='cpu',weights_only=True)
   if json.loads(json.dumps(ck['protocol']))!=tp:raise ValueError('Checkpoint protocol mismatch')
   if ck['variant']!=v or ck['initialization']!=init:raise ValueError('Checkpoint identity mismatch')
   m=train.make_model(torch,v);m.load_state_dict(ck['state'],strict=True);m.eval();models[(init,v)]=(m,ck)
 metrics_rows=[];audit=[];all_predictions=[]
 for seed in seeds:
  for condition in CONDITIONS:
   run=a.output/'collections'/f'seed{seed}_{NAMES[condition]}'
   accepted=load(run/'accepted.json');attempt=run/accepted['attempt']
   for name,h in accepted['files'].items():
    if digest(attempt/name)!=h:raise ValueError('Collected source changed: '+str(attempt/name))
   found=history.discover(attempt)
   if len(found)!=1:raise ValueError('Expected one accepted run')
   path,rows=found[0]
   if rows[0]['seed']!=seed or any(r['reporter_is_byzantine'] for r in rows):raise ValueError('Wrong seed/reporter condition')
   obs=adapter.load_observations(path,rows);graphs=adapter.examples(rows,20,obs)
   expected={str(r['report_sequence']):int(r['target_is_malicious']) for r in rows if r['beacon_sequence']>=20}
   ids=[str(r['report_sequence']) for g in graphs for _,r in g['selected']]
   if not expected or len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Incomplete eligible graph coverage')
   if condition=='none' and any(expected.values()):raise ValueError('Benign run contains positive labels')
   audit.append(dict(seed=seed,attack=NAMES[condition],generated=len(rows),eligible=len(expected),graphs=len(graphs)))
   for (init,v),(model,ck) in models.items():
    preds=[]
    with torch.inference_mode():
     for g in graphs:
      tensors=[torch.from_numpy(t) for t in original.inputs(g,'baseline' if v=='no_observations' else 'assisted',ck['stats'])]
      score=torch.sigmoid(model(*tensors)[original.selected(g)])
      if not torch.isfinite(score).all():raise ValueError('Nonfinite prediction')
      for (_,r),value in zip(g['selected'],score.tolist()):
       preds.append(dict(seed=seed,attack=NAMES[condition],initialization=init,variant=v,report_id=str(r['report_sequence']),
        y_true=int(r['target_is_malicious']),score=value,trained_checkpoint_selected=ck['best_epoch']>0))
    assert {r['report_id']:r['y_true'] for r in preds}==expected
    all_predictions+=preds
    metrics_rows.append(dict(seed=seed,attack=NAMES[condition],initialization=init,variant=v,best_epoch=ck['best_epoch'],
      trained_checkpoint_selected=ck['best_epoch']>0,parameters=sum(p.numel() for p in model.parameters()),**metrics(preds)))
   print('EVALUATED',seed,NAMES[condition],len(expected),'matched reports',flush=True)
 for seed in seeds:
  for (init,v),(_,ck) in models.items():
   rows=[r for r in all_predictions if r['seed']==seed and r['initialization']==init and r['variant']==v and r['attack']!='Benign']
   metrics_rows.append(dict(seed=seed,attack='Pooled_A1_A5',initialization=init,variant=v,best_epoch=ck['best_epoch'],
    trained_checkpoint_selected=ck['best_epoch']>0,parameters=sum(p.numel() for p in models[(init,v)][0].parameters()),**metrics(rows)))
 csvout(result/'predictions.csv',all_predictions);csvout(result/'metrics_by_seed_and_initialization.csv',metrics_rows)
 summary=[]
 for condition in ['A1','A2','A3','A4','A5','Benign','Pooled_A1_A5']:
  for v in VARIANTS:
   rr=[r for r in metrics_rows if r['variant']==v and r['attack']==condition]
   row=dict(attack=condition,variant=v,simulation_seeds=len(seeds),training_initializations=len(tp['training_initializations']),
    trained_initializations=sum(models[(i,v)][1]['best_epoch']>0 for i in tp['training_initializations']),parameters=rr[0]['parameters'])
   for key in ['precision','recall','f1','fpr']:
    # First average initializations per episode; SD is then across simulation seeds.
    vals=[]
    for seed in seeds:
     vv=[r[key] for r in rr if r['seed']==seed and r[key] is not None]
     if vv:vals.append(statistics.mean(vv))
    row[key+'_mean']=statistics.mean(vals) if vals else None
    row[key+'_sd_across_simulation_seeds']=statistics.stdev(vals) if len(vals)>1 else None
   summary.append(row)
 csvout(result/'summary.csv',summary);dump(result/'data_audit.json',audit)
 def fmt(row,key,scale=1):
  v=row[key+'_mean'];sd=row[key+'_sd_across_simulation_seeds'];return '--' if v is None else f'{v*scale:.4f}'+(f' $\\pm$ {sd*scale:.4f}' if sd is not None else '')
 lines=[]
 for v in VARIANTS:
  rows={r['attack']:r for r in summary if r['variant']==v};r=rows['Pooled_A1_A5'];mark=r' $^{\dagger}$' if r['trained_initializations']<len(tp['training_initializations']) else ''
  lines.append(train.LABELS[v]+mark+' & '+' & '.join(fmt(r,k) for k in ['precision','recall','f1'])+' & '+fmt(rows['Benign'],'fpr',100)+r' \\')
 (result/'main_table_rows.tex').write_text('\n'.join(lines)+'\n% FPR is on separate all-benign episodes. Dagger: initialization retained.\n')
 lines=[]
 for v in VARIANTS:
  rr={r['attack']:r for r in summary if r['variant']==v};mark=r' $^{\dagger}$' if rr['A1']['trained_initializations']<len(tp['training_initializations']) else ''
  lines.append(train.LABELS[v]+mark+' & '+' & '.join(fmt(rr[k],'f1') for k in ['A1','A2','A3','A4','A5'])+r' \\')
 (result/'per_attack_rows.tex').write_text('\n'.join(lines)+'\n')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
  for i,v in enumerate(VARIANTS):
   rr=[next(r for r in summary if r['variant']==v and r['attack']==k) for k in ['A1','A2','A3','A4','A5']]
   label=train.LABELS[v]+(' [initialization retained]' if rr[0]['trained_initializations']<len(tp['training_initializations']) else '')
   ax.plot(range(1,6),[r['f1_mean'] for r in rr],marker='o',label=label)
  ax.set_xticks(range(1,6),['A1','A2','A3','A4','A5']);ax.set_ylim(0,1.03);ax.set_ylabel('Mean F1');ax.grid(alpha=.2)
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8)
  for ext in ['pdf','png']:fig.savefig(result/f'per_attack.{ext}',dpi=300)
  plt.close(fig)
 except ImportError:print('Plots unavailable; install matplotlib. All numeric results are saved.')
 dump(result/'COMPLETE.json',dict(completed=True,scope=protocol['scope'],threshold=.5,
  note='Mean first over training initializations within each episode, then over simulation seeds. SD across seeds. No retuning performed.'))
 print('COMPLETE:',result,flush=True)

if __name__=='__main__':main()
