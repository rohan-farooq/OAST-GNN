"""Standard-library evaluator; deliberately refuses the old privileged CSVs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METHODS = ['Mean', 'Median', 'A6_RSU_trust', 'RSU_only']


def predict(reports, evidence, tx_time, deadline=0.75):
    """No labels, hidden reporter scores or simulator truth enter this function.

    Decisions are made once at tx_time + deadline. Late information is excluded.
    The RSU anchor is shared across reporters; it is not independent corroboration
    per reporter. Without a valid anchor A6 explicitly abstains.
    """
    import statistics
    values = [float(r['submitted_score']) for r in reports
              if tx_time <= float(r['received_time_s']) <= tx_time + deadline]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid wire score')
    anchor = None
    if evidence is not None:
        sampled = float(evidence['sample_time_s'])
        available = float(evidence['available_time_s'])
        if sampled <= available <= tx_time + deadline and 0 <= tx_time+deadline-sampled <= deadline+1e-6:
            anchor = float(evidence['rsu_score'])
            if not math.isfinite(anchor) or not 0 <= anchor <= 1:
                raise ValueError('Invalid RSU score')
    a6 = None
    if anchor is not None and values:
        weights = [max(0.0, 1-abs(v-anchor)) for v in values]
        if sum(weights) > 1e-12:
            a6 = sum(v*w for v,w in zip(values,weights))/sum(weights)
    return {'Mean': statistics.mean(values) if values else None,
            'Median': statistics.median(values) if values else None,
            'A6_RSU_trust': a6, 'RSU_only': anchor}


def metric(rows):
    tp=tn=fp=fn=missing=missing_positive=0
    for row in rows:
        y,p=row['label'],row['prediction']
        if p is None:
            missing+=1; missing_positive+=y
        elif y==1 and p==1: tp+=1
        elif y==0 and p==0: tn+=1
        elif y==0: fp+=1
        else: fn+=1
    decided=tp+tn+fp+fn
    div=lambda n,d: n/d if d else None
    return dict(scheduled_cases=len(rows), decided_cases=decided, no_decision=missing,
        malicious_no_decision=missing_positive, tp=tp,tn=tn,fp=fp,fn=fn,
        decision_coverage=div(decided,len(rows)), accuracy_on_decided=div(tp+tn,decided),
        correct_decisions_per_scheduled_case=div(tp+tn,len(rows)),
        f1_on_decided=div(2*tp,2*tp+fp+fn), recall_on_decided=div(tp,tp+fn),
        scheduled_malicious_detection=div(tp,tp+fn+missing_positive),
        fpr_on_decided=div(fp,fp+tn))


def write_csv(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,default=0.75)
    a=p.parse_args()
    if a.deadline<=0: p.error('deadline must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    files=sorted(a.input.glob('*.csv.operational.json'))
    if not files: p.error('No v3 operational files found; old CSVs cannot be converted to independent RSU evidence')
    summaries=[]; all_predictions=[]
    for file in files:
        o=json.loads(file.read_text(encoding='utf-8'))
        e=json.loads(Path(str(file).replace('.operational.json','.evaluation.json')).read_text(encoding='utf-8'))
        if o['schema']!='a6-independent-rsu-v3' or e['interrupted']:
            raise ValueError(f'Wrong schema or interrupted run: {file}')
        reports=defaultdict(list)
        report_ids=set()
        for r in o['reports']:
            if r['report_id'] in report_ids: raise ValueError('Duplicate report ID')
            report_ids.add(r['report_id'])
            reports[(r['beacon_sequence'],r['target_id'])].append(r)
        evidence={(v['beacon_sequence'],v['target_id']):v for v in o['evidence']}
        sent={(v['beacon_sequence'],v['target_id']):v for v in e['transmitted_beacons']}
        malicious=set(e['malicious_target_ids'])
        by_method=defaultdict(list)
        for round_id in range(1,e['scheduled_rounds']+1):
            for target in e['target_ids']:
                key=(round_id,target); beacon=sent.get(key)
                scores=predict(reports[key],evidence.get(key),beacon['beacon_tx_time_s'],a.deadline) if beacon else dict.fromkeys(METHODS)
                for method,score in scores.items():
                    row=dict(run=file.name,method=method,beacon_sequence=round_id,target_id=target,
                        label=int(target in malicious),score=score,
                        prediction=int(score>=0.5) if score is not None else None)
                    by_method[method].append(row); all_predictions.append(row)
        for method,rows in by_method.items():
            summaries.append(dict(run=file.name,seed=o['seed'],method=method,**metric(rows)))
        # Fair A6 versus RSU comparison on exactly the same decided cases.
        common=[(x,y) for x,y in zip(by_method['A6_RSU_trust'],by_method['RSU_only'])
                if x['prediction'] is not None and y['prediction'] is not None]
        for method,index in [('A6_common',0),('RSU_common',1)]:
            summaries.append(dict(run=file.name,seed=o['seed'],method=method,**metric([v[index] for v in common])))
    write_csv(a.output/'metrics_v3.csv',summaries)
    write_csv(a.output/'decisions_v3.csv',all_predictions)
    print(f'Analyzed {len(files)} v3 runs. Saved metrics_v3.csv and decisions_v3.csv.')


if __name__=='__main__': main()
