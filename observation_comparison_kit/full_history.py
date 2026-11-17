"""Full observer-history adapter for the GRU/GAT development model."""
import bisect,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import stgnn_a6 as base
CACHE={}

def loadlines(p):
    with p.open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]

def discover(root):
    CACHE.clear();found=[];seen=set()
    for manifest in sorted(Path(root).rglob('*.csv.history_manifest.json')):
        p=Path(str(manifest).removesuffix('.history_manifest.json'))
        m=json.loads(manifest.read_text());ev=json.loads(Path(str(p)+'.evaluation.json').read_text());meta=json.loads(Path(str(p)+'.metadata.json').read_text())
        if m['interrupted'] or ev['interrupted']:raise ValueError(f'Interrupted run: {p}')
        if m['beacon_queue_remaining']:raise ValueError(f'Undrained beacon queue: {p}')
        parts={k:loadlines(Path(str(p)+suffix)) for k,suffix in {'beacons':'.beacons.jsonl','self':'.self.jsonl','attacks':'.attacks.evaluation.jsonl'}.items()}
        if any(len(parts[k])!=m['counts'][k] for k in parts):raise ValueError('Manifest count mismatch')
        key=(m['seed'],meta['map'],meta['collusion_fraction_requested'],meta['reporter_attack'],meta['target_attack'])
        if key in seen:raise ValueError(f'Duplicate condition: {p}')
        seen.add(key);labels=set(ev['malicious_target_ids']);byobserver=defaultdict(list);lookup={}
        for b in parts['beacons']:
            v=b['payload']
            if v.get('k')!='b':continue
            r=dict(zip(base.FEATURES,[float(v[k]) for k in ('x','y','v','h')]))
            r.update(target_id=int(v['i']),reporter_id=int(b['observer_id']),beacon_sequence=int(v['n']),
                beacon_rx_time_s=max(b['received_time_s'],b['processed_time_s']),target_is_malicious=int(v['i'] in labels))
            if not all(math.isfinite(r[k]) for k in base.FEATURES+['beacon_rx_time_s']):raise ValueError('Nonfinite observation')
            if r['target_id']==r['reporter_id']:continue
            byobserver[r['reporter_id']].append(r)
            k=(r['reporter_id'],r['target_id'],r['beacon_sequence'])
            if k in lookup:raise ValueError('Repeated observer/target/beacon; explicit duplicate policy required')
            lookup[k]=r
        selfs=defaultdict(list)
        for s in parts['self']:selfs[s['observer_id']].append(s)
        for ss in selfs.values():ss.sort(key=lambda s:s['time_s'])
        rows=[]
        for a in parts['attacks']:
            k=(a['reporter_id'],a['target_id'],a['beacon_sequence'])
            if k not in lookup:raise ValueError('Attack/report missing beacon input')
            r=dict(lookup[k]);r.update(report_sequence=a['report_id'],report_created_time_s=a['created_time_s'],
                seed=m['seed'],reporter_is_byzantine=int(a['reporter_is_byzantine']),reporter_attack=a['reporter_attack'],
                attack_active=a['attack_active'],submitted_risk_score=a['submitted_score'],local_risk_score=a['local_score'],
                _source=str(p),beacon_tx_time_s=m['start_sim_time_s']+(a['beacon_sequence']-1)/m['beacon_rate_requested_hz'])
            if r['beacon_rx_time_s']>r['report_created_time_s']+1e-8:raise ValueError('Report precedes available beacon')
            rows.append(r)
        if not rows:raise ValueError('No generated reports')
        if len({r['report_sequence'] for r in rows})!=len(rows):raise ValueError('Duplicate report IDs')
        op=json.loads(Path(str(p)+'.operational.json').read_text())
        if not {r['report_id'] for r in op['reports']} <= {r['report_sequence'] for r in rows}:raise ValueError('RSU report has no local report record')
        period=1/m['beacon_rate_requested_hz']
        # Verify requested slots are supported by measured transmissions; no fabricated 10 Hz.
        tx=defaultdict(list)
        for b in ev['transmitted_beacons']:tx[b['target_id']].append((b['beacon_sequence'],b['beacon_tx_time_s']))
        gaps=[(b[1]-a[1])/(b[0]-a[0]) for events in tx.values() for a,b in zip(sorted(events),sorted(events)[1:]) if b[0]>a[0]]
        if gaps and abs(float(base.np.median(gaps))-period)>max(.01,period*.1):raise ValueError('Observed cadence differs from requested cadence')
        CACHE[str(p)]=dict(byobserver=byobserver,selfs=selfs,period=period,start=m['start_sim_time_s'],parts=parts)
        found.append((p,rows))
    if not found:raise ValueError('No full-history manifests found')
    return found

def examples(rows,window):
    ctx=CACHE[rows[0]['_source']];groups=defaultdict(list);out=[]
    for r in rows:
        if r['beacon_sequence']>=window:groups[(r['reporter_id'],r['beacon_sequence'],r['report_created_time_s'])].append(r)
    for (observer,end,cut),current in sorted(groups.items()):
        history=list(ctx['byobserver'][observer]);ss=ctx['selfs'][observer];times=[s['time_s'] for s in ss]
        for seq in range(end-window+1,end+1):
            when=min(cut,ctx['start']+seq*ctx['period'])
            pos=bisect.bisect_right(times,when)-1
            if pos<0:raise ValueError('Missing causal ego state')
            s=ss[pos]
            if when-s['time_s']>ctx['period']:raise ValueError('Stale ego state')
            history.append(dict(target_id=observer,reporter_id=observer,beacon_sequence=seq,
                beacon_rx_time_s=s['time_s'],target_is_malicious=0,
                claimed_target_x_m=s['x_m'],claimed_target_y_m=s['y_m'],
                claimed_target_speed_mps=math.sqrt(s['vx_mps']**2+s['vy_mps']**2+s['vz_mps']**2),
                claimed_target_heading_deg=s['heading_deg']))
        g=base.graph(history,current,window);out.append(g)
    return out

def submitted(row,neural):
    # Attack instrumentation is used ONLY by this counterfactual replay boundary.
    if not row['attack_active']:return neural
    if row['reporter_attack'] in ('coordinated','on_off','random_score','false_accusation','false_reassurance'):
        return float(row['submitted_risk_score'])
    raise ValueError('Replay attack requires chronological neural-score history; not implemented')

def audit(a):
    result=[]
    for p,rows in discover(a.input):
        ctx=CACHE[str(p)];rounds=len({r['beacon_sequence'] for r in rows});gs=examples(rows,a.window)
        result.append(dict(run=str(p),seed=rows[0]['seed'],beacon_records=len(ctx['parts']['beacons']),
            self_records=len(ctx['parts']['self']),generated_reports=len(rows),rounds=rounds,
            window_slots=a.window,window_seconds=a.window*ctx['period'],graphs=len(gs),
            ready_for_window=bool(gs),observer_self_nodes=True,explicit_attack_flags=True))
    base.write_json(a.output,result);print(json.dumps(result,indent=2))
    print('AUDIT COMPLETE. Zero graphs means this run is shorter than the requested window; no padding fabricated.')

if __name__=='__main__':
    base.discover=discover;base.examples=examples;base.submitted=submitted;base.audit=audit
    base.main()
