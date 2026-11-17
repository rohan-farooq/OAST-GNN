"""Frozen local-detector stress evaluation; synthetic observation dropout only.
Physical offset/noise changes require fresh collector runs, never label rewriting.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import full_history as history
import observation_adapter as adapter
from compare_observations import inputs,selected,metrics as _metrics
from frozen_numpy import checkpoint,infer,sigmoid
RULE_THRESHOLD=10.48297593814976

def metrics(y,s):
 result=_metrics(y,s)
 if not len(y):
  for key in ('f1','recall','false_positive_rate'):result[key]=None
 return result

def evaluate(args):
 if hashlib.sha256(Path(args.model).read_bytes()).hexdigest()!='c879cccae42c18e1cf763174dd62bd8684787b73ac384fcba5fea9a40d0fbc74':raise ValueError('Checkpoint differs from the uploaded frozen model')
 c=checkpoint(args.model)
 if c.get('variant')!='assisted' or c.get('best_epoch')!=2 or c.get('window')!=20:
  raise ValueError('Expected frozen assisted epoch-2, 20-slot checkpoint')
 if c['state']['gru.weight_ih_l0'].shape!=(192,13):raise ValueError('Expected 13 inputs')
 allowed={2047,2048,2049}
 if set(c['train_seeds'])!=set(range(2041,2047)) or set(c['val_seeds'])!=allowed:raise ValueError('Unexpected checkpoint split')
 if args.output.exists():raise ValueError('Use a fresh output filename')
 rates=[float(v) for v in args.drop_rates.split(',')]
 if any(v<0 or v>1 for v in rates):raise ValueError('Drop probabilities must be in [0,1]')
 results=[];processed=0
 for p,rows in history.discover(args.input):
  seed=rows[0]['seed']
  if seed not in allowed:continue
  if any(r['reporter_is_byzantine'] or r['reporter_attack']!='none' for r in rows):raise ValueError('Only honest-reporter controls supported')
  obs=adapter.load_observations(p,rows);meta=json.loads(Path(str(p)+'.metadata.json').read_text())
  for rate in rates:
   altered=copy.deepcopy(obs);drops=0
   for o in altered:
    # Stable per-event draw, reused in every overlapping graph; nested masks.
    token=f"{args.mask_seed}:{seed}:{o['reporter_id']}:{o['target_id']}:{o['beacon_sequence']}".encode()
    u=int.from_bytes(hashlib.sha256(token).digest()[:8],'big')/2**64
    if u<rate:o['available']=False;drops+=1
   gs=adapter.examples(rows,20,altered)
   if not gs:raise ValueError('No complete 20-slot graphs')
   y=[];scores=[];ry=[];rp=[];paired=[]
   for g in gs:
    g['_normalized_x']=inputs(g,'assisted',c['stats'])[0]
    pred=sigmoid(infer(g,c))
    for i,r in g['selected']:
     label=g['y'][i];y.append(label);scores.append(pred[i])
     if g['obs_x'][i,-1,5]==0:
      ry.append(label);rp.append(float(g['obs_x'][i,-1,2]>RULE_THRESHOLD));paired.append(pred[i])
   row=dict(run=str(p),seed=seed,offset_m=meta['attack_model']['target_position_offset_m'],
            source_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in p.parent.glob(p.name+'.*') if q.is_file()},
            base_noise_m=meta['measurement_model']['base_measurement_noise_m'],drop_rate=rate,
            dropped_records=drops,total_records=len(obs),graphs=len(gs),
            neural_all_reports=metrics(y,scores),neural_rule_covered_reports=metrics(ry,paired),
            residual_rule_covered_reports=metrics(ry,rp),rule_abstentions=len(y)-len(ry),
            rule_coverage=len(ry)/len(y))
   results.append(row);print(json.dumps(row),flush=True)
  processed+=1
  if args.max_runs and processed>=args.max_runs:break
 if not results:raise ValueError('No validation-seed runs found')
 payload=dict(protocol='Frozen development stress; local generated reports; not held-out or RSU evaluation',
              model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),model_epoch=2,
              rule_threshold_m=RULE_THRESHOLD,neural_threshold=.5,mask_seed=args.mask_seed,
              note='Dropout masks observations only; does not model packet loss. AUC/CI not inferred from repeated windows.',results=results)
 args.output.write_text(json.dumps(payload,indent=2));print('Saved',args.output)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--drop-rates',default='0,0.25,0.5');p.add_argument('--mask-seed',type=int,default=7301);p.add_argument('--max-runs',type=int,default=0)
 evaluate(p.parse_args())
