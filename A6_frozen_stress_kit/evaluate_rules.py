"""Offline, chronological development evaluation. Python standard library only."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from v3_reference import metric, predict as original_predict

CONFIG = dict(deadline_s=0.75, threshold=0.5, reporter_influence=0.20,
              prior_mass=4.0, half_life_s=30.0, learning_confidence=0.60,
              ambiguity_margin=0.15)
METHODS = ['Mean','Median','RSU_only','A6_v3','Static_capped',
           'Temporal_only','Temporal_capped','Temporal_selective']


def unit(value):
    v=float(value)
    if not math.isfinite(v) or not 0<=v<=1: raise ValueError('Invalid score')
    return v


class Memory:
    """Decayed agreement with an imperfect RSU reference, not true honesty."""
    def __init__(self): self.state={}
    def evidence(self, reporter, now):
        good,total,last=self.state.get(reporter,(0.0,0.0,now))
        if now<last: raise ValueError('Nonchronological update')
        decay=2**(-(now-last)/CONFIG['half_life_s'])
        return good*decay,total*decay
    def reliability(self, reporter, now):
        good,total=self.evidence(reporter,now)
        prior=CONFIG['prior_mass']
        return (prior*0.5+good)/(prior+total)
    def update(self, reporter, score, anchor, now):
        confidence=abs(2*anchor-1)
        if confidence<CONFIG['learning_confidence']: return
        good,total=self.evidence(reporter,now)
        agreement=1-abs(score-anchor)
        self.state[reporter]=(good+confidence*agreement,total+confidence,now)


def score_case(reports, evidence, tx, memory):
    """Restricted operational inputs only. Does not update memory."""
    now=tx+CONFIG['deadline_s']
    reports=[r for r in reports if tx<=r['received_time_s']<=now]
    # Reject repeated votes rather than letting one reporter multiply its weight.
    ids=[r['reporter_id'] for r in reports]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate reporter-target-round vote')
    base=original_predict(reports,evidence,tx,CONFIG['deadline_s'])
    anchor=base['RSU_only']
    result=dict(Mean=base['Mean'],Median=base['Median'],RSU_only=anchor,A6_v3=base['A6_RSU_trust'])
    result.update(dict.fromkeys(METHODS[4:]))
    updates=[]
    used=[]
    if anchor is not None:
        # No reports: use RSU alone for capped methods. No evidence: abstain.
        result['Static_capped']=anchor
        result['Temporal_capped']=anchor
        result['Temporal_selective']=anchor
        if reports:
            values=[unit(r['submitted_score']) for r in reports]
            weights=[memory.reliability(r['reporter_id'],now)**2 for r in reports]
            reporter_score=sum(v*w for v,w in zip(values,weights))/sum(weights)
            mix=CONFIG['reporter_influence']
            capped=(1-mix)*anchor+mix*reporter_score
            result['Static_capped']=(1-mix)*anchor+mix*statistics.mean(values)
            result['Temporal_only']=reporter_score
            result['Temporal_capped']=capped
            conflict=(anchor>=0.5)!=(reporter_score>=0.5)
            result['Temporal_selective']=None if conflict and abs(anchor-0.5)<CONFIG['ambiguity_margin'] else capped
            for r,v,w in zip(reports,values,weights):
                updates.append((r['reporter_id'],v,anchor,now))
                used.append(dict(reporter_id=r['reporter_id'],time_s=now,
                    reliability_before_decision=math.sqrt(w),submitted_score=v,rsu_score=anchor))
    return result,updates,used


def operational_decisions(operational, schedule):
    """No evaluation labels accepted; memory is reset for every run."""
    if operational['schema']!='a6-independent-rsu-v3': raise ValueError('Expected v3 schema')
    by_key=defaultdict(list)
    seen=set()
    for r in operational['reports']:
        if r['report_id'] in seen: raise ValueError('Duplicate report ID')
        seen.add(r['report_id']);unit(r['submitted_score'])
        if not math.isfinite(r['received_time_s']): raise ValueError('Invalid timestamp')
        by_key[(r['beacon_sequence'],r['target_id'])].append(r)
    anchors={}
    for e in operational['evidence']:
        key=(e['beacon_sequence'],e['target_id'])
        if key in anchors: raise ValueError('Duplicate evidence key')
        unit(e['rsu_score'])
        if not all(math.isfinite(e[k]) for k in ('available_time_s','sample_time_s')): raise ValueError('Invalid evidence timestamp')
        anchors[key]=e
    memory=Memory();decisions=[];history=[]
    scheduled=sorted(schedule,key=lambda s:(s['beacon_tx_time_s'],s['beacon_sequence'],s['target_id']))
    # Same-time cases cannot teach one another; all updates follow the batch.
    for _,batch in itertools.groupby(scheduled,key=lambda s:s['beacon_tx_time_s']):
        pending=[]
        for s in batch:
            key=(s['beacon_sequence'],s['target_id'])
            result,updates,used=score_case(by_key[key],anchors.get(key),s['beacon_tx_time_s'],memory)
            pending.extend(updates);history.extend(used)
            for method,score in result.items():
                decisions.append(dict(beacon_sequence=key[0],target_id=key[1],method=method,
                    score=score,prediction=None if score is None else int(score>=CONFIG['threshold'])))
        for update in pending: memory.update(*update)
    return decisions,history


def dump_csv(path,rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate_file(file):
    op=json.loads(file.read_text(encoding='utf-8'))
    ep=Path(str(file).replace('.operational.json','.evaluation.json'))
    ev=json.loads(ep.read_text(encoding='utf-8'))
    if ev['interrupted']: raise ValueError('Interrupted run requires separate review')
    # Strip truth before prediction. Schedule/time is evaluation instrumentation,
    # not an input feature used to infer labels.
    schedule=[{k:s[k] for k in ('beacon_sequence','target_id','beacon_tx_time_s')}
              for s in ev['transmitted_beacons']]
    keys={(s['beacon_sequence'],s['target_id']) for s in schedule}
    if len(keys)!=len(schedule): raise ValueError('Duplicate scheduled beacon')
    decisions,history=operational_decisions(op,schedule)
    expected=set(itertools.product(range(1,ev['scheduled_rounds']+1),ev['target_ids']))
    if not keys<=expected: raise ValueError('Unexpected beacon')
    for seq,target in sorted(expected-keys):
        for method in METHODS:
            decisions.append(dict(beacon_sequence=seq,target_id=target,method=method,score=None,prediction=None))
    # Labels enter only here, after ALL predictions and reliability updates.
    malicious=set(ev['malicious_target_ids'])
    for d in decisions: d.update(run=file.name,label=int(d['target_id'] in malicious))
    grouped={m:[d for d in decisions if d['method']==m] for m in METHODS}
    metrics=[dict(run=file.name,method=m,**metric(rows)) for m,rows in grouped.items()]
    reference={(r['beacon_sequence'],r['target_id']):r for r in grouped['RSU_only']}
    pairs=[]
    for m in METHODS:
        rows=grouped[m]
        common=[(r,reference[(r['beacon_sequence'],r['target_id'])]) for r in rows
                if r['prediction'] is not None and reference[(r['beacon_sequence'],r['target_id'])]['prediction'] is not None]
        fixed=sum(r['prediction']==r['label'] and b['prediction']!=b['label'] for r,b in common)
        harmed=sum(r['prediction']!=r['label'] and b['prediction']==b['label'] for r,b in common)
        correct=sum(r['prediction']==r['label'] for r,b in common)
        refcorrect=sum(b['prediction']==b['label'] for r,b in common)
        pairs.append(dict(run=file.name,method=m,common_cases=len(common),
            accuracy_common=correct/len(common) if common else None,
            rsu_accuracy_common=refcorrect/len(common) if common else None,
            rsu_errors_corrected=fixed,rsu_correct_cases_damaged=harmed,
            net_errors_removed=fixed-harmed))
    for r in history: r['run']=file.name
    return metrics,decisions,pairs,history


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): p.error('Use a new empty output directory to preserve earlier results')
    args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.input.glob('*.csv.operational.json'))
    if not paths: p.error('No v3 operational JSON files found')
    tables=[[],[],[],[]]
    for file in paths:
        for target,rows in zip(tables,evaluate_file(file)): target.extend(rows)
    for name,rows in zip(['metrics','decisions','paired_comparisons','reliability_trace'],tables):
        dump_csv(args.output/(name+'.csv'),rows)
    manifest=dict(purpose='DEVELOPMENT ONLY; no performance tuning inside script',parameters=CONFIG,
        methods=METHODS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        evaluation_inputs={str(f).replace('.operational.json','.evaluation.json'):hashlib.sha256(Path(str(f).replace('.operational.json','.evaluation.json')).read_bytes()).hexdigest() for f in paths})
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for r in tables[0]:
        print(r['run'],r['method'],'decided',r['decided_cases'],'FP',r['fp'],'FN',r['fn'])


if __name__=='__main__': main()
