"""Frozen local predictions -> measured offline RSU reporter-attack replay.
Not an exact TRE A6 reproduction. No model training or CARLA calls.
"""
import argparse,csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path
from evaluate_rules import operational_decisions,CONFIG
from v3_reference import metric

KEEP=['Mean','RSU_only','Static_capped','Temporal_capped']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def measure(rows):
 m=metric(rows)
 if not any(r['label'] for r in rows):m['f1_on_decided']=None
 return m

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data',type=Path,required=True)
 ap.add_argument('--predictions',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--window',type=int,default=20)
 ap.add_argument('--attack-seed',type=int,default=7301)
 ap.add_argument('--on-period',type=float,default=10)
 a=ap.parse_args()
 if a.output.exists():raise ValueError('Use a fresh output folder.')
 if a.window<1 or a.on_period<=0:raise ValueError('Invalid window/period')
 with a.predictions.open(newline='',encoding='utf-8-sig') as f:pred=list(csv.DictReader(f))
 lookup={}
 for r in pred:
  k=(r['run_id'],str(r['item_id']))
  if k in lookup:raise ValueError('Duplicate prediction')
  if r['stage']!='local' or r['method']!='Proposed':raise ValueError('Expected one local Proposed model')
  if int(r['seed']) not in [2047,2048,2049]:raise ValueError('Development seeds only')
  if not 0<=float(r['score'])<=1:raise ValueError('Invalid neural score')
  lookup[k]=r
 summaries=[];decisions=[];traces=[];sources=[];used=set()
 for manifest in sorted(a.data.rglob('*.csv.history_manifest.json')):
  p=Path(str(manifest).removesuffix('.history_manifest.json'))
  run=p.parent.relative_to(a.data).as_posix()
  if not any(k[0]==run for k in lookup):continue
  h=read(manifest);ev=read(str(p)+'.evaluation.json');op=read(str(p)+'.operational.json')
  if h['interrupted'] or ev['interrupted'] or not h.get('fixed_fleet_valid') or h['beacon_queue_remaining']:
   raise ValueError('Incomplete run')
  if h['seed'] not in [2047,2048,2049]:raise ValueError('Development only')
  attacks=[json.loads(x) for x in Path(str(p)+'.attacks.evaluation.jsonl').read_text().splitlines() if x.strip()]
  if any(r['reporter_is_byzantine'] or r['reporter_attack']!='none' for r in attacks):raise ValueError('Use honest-reporter source runs')
  eligible={str(r['report_id']):r for r in attacks if r['beacon_sequence']>=a.window}
  provided={k[1] for k in lookup if k[0]==run}
  if set(eligible)!=provided:raise ValueError('Prediction eligibility mismatch')
  malicious=set(ev['malicious_target_ids'])
  for rid,r in eligible.items():
   n=lookup[(run,rid)]
   if (int(n['reporter_id']),int(n['target_id']))!=(r['reporter_id'],r['target_id']) or int(n['y_true'])!=int(r['target_id'] in malicious):raise ValueError('Prediction identity/label mismatch')
   used.add((run,rid))
  reports=[]
  for r in op['reports']:
   if r['beacon_sequence']<a.window:continue
   n=lookup[(run,str(r['report_id']))]
   reports.append(dict(r,submitted_score=float(n['score'])))
  reporter_ids=sorted({r['reporter_id'] for r in attacks})
  random.Random(a.attack_seed+int(h['seed'])).shuffle(reporter_ids)
  nreporters=len(reporter_ids)
  if nreporters!=8:raise ValueError('This sweep expects the observed eight-reporter fleet')
  schedule=[{k:r[k] for k in ['beacon_sequence','target_id','beacon_tx_time_s']} for r in ev['transmitted_beacons'] if r['beacon_sequence']>=a.window]
  if len({(s['beacon_sequence'],s['target_id']) for s in schedule})!=len(schedule):raise ValueError('Duplicate schedule')
  expected={(seq,t) for seq in range(a.window,ev['scheduled_rounds']+1) for t in ev['target_ids']}
  actual={(s['beacon_sequence'],s['target_id']) for s in schedule}
  if not actual<=expected:raise ValueError('Unexpected schedule')
  sources.append(dict(run=run,operational_sha256=hashlib.sha256(Path(str(p)+'.operational.json').read_bytes()).hexdigest(),reporter_order=reporter_ids))
  for count in range(6):
   coalition=set(reporter_ids[:count])
   for mode in ['coordinated','on_off']:
    manipulated=[]
    for r in reports:
     t=float(r['received_time_s'])-float(h['start_sim_time_s'])
     active=mode=='coordinated' or int(max(0,t)//a.on_period)%2==0
     score=r['submitted_score']
     # Labels are used ONLY by the adversary to construct a target-aware attack.
     # The aggregator receives no labels or coalition identities.
     if r['reporter_id'] in coalition and active:
      score=0.01 if r['target_id'] in malicious else 0.99
     manipulated.append(dict(r,submitted_score=score))
    operational={**op,'reports':manipulated}
    ds,trace=operational_decisions(operational,schedule)
    ds=[d for d in ds if d['method'] in KEEP]
    for seq,target in sorted(expected-actual):
     for method in KEEP:ds.append(dict(beacon_sequence=seq,target_id=target,method=method,score=None,prediction=None))
    context=dict(run=run,seed=h['seed'],reporter_attack=mode,malicious_reporters=count,rho=count/nreporters)
    # Join truth only after operational decisions/trust updates have completed.
    for d in ds:d.update(context,label=int(d['target_id'] in malicious))
    common=None
    for method in KEEP:
     keys={(d['beacon_sequence'],d['target_id']) for d in ds if d['method']==method and d['prediction'] is not None}
     common=keys if common is None else common & keys
    for method in KEEP:
     rr=[d for d in ds if d['method']==method]
     for scope,items in [('all_scheduled',rr),('common_decided',[d for d in rr if (d['beacon_sequence'],d['target_id']) in common])]:
      summaries.append(dict(**context,method=method,comparison_scope=scope,**measure(items)))
    decisions.extend(ds)
    traces.extend([{**context,**r} for r in trace])
  print('Completed',run,flush=True)
 if used!=set(lookup):raise ValueError('Some predictions were not matched')
 a.output.mkdir(parents=True)
 dump(a.output/'metrics.csv',summaries);dump(a.output/'decisions.csv',decisions);dump(a.output/'trust_trace.csv',traces)
 (a.output/'protocol.json').write_text(json.dumps(dict(parameters=CONFIG,window=a.window,attack_seed=a.attack_seed,on_period=a.on_period,
  scope='Offline target-aware reporter-score intervention on fixed received messages; no spatial coalition restriction; not exact TRE A6; no A1-A5 claim.',
  trust_ablation='Static_capped versus Temporal_capped: same RSU anchor and 20% reporter influence; temporal weights only differ.',
  limitation='Original transmission timing/delivery frozen. On-off phase uses received time. Zero-rho rows repeat across attack modes; do not treat them as independent replicates.',
  sources=sources,predictions_sha256=hashlib.sha256(a.predictions.read_bytes()).hexdigest()),indent=2))
 print('Saved measured replay metrics, decisions and trust traces:',a.output)
if __name__=='__main__':main()
