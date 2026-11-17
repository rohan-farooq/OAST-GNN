"""Independent checks of recorded attack equations and actual motion. No CARLA needed."""
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path

def audit(path):
    path=Path(path)
    m=json.loads(Path(str(path)+'.target_attack_manifest.json').read_text())
    q=path.parent/m['trace_file']
    if hashlib.sha256(q.read_bytes()).hexdigest()!=m['trace_sha256']:raise ValueError('Trace hash mismatch')
    rows=[json.loads(x) for x in q.read_text().splitlines() if x.strip()]
    if m['interrupted'] or not rows or len(rows)!=m['successful_transmissions']:raise ValueError('Incomplete attack trace')
    h=json.loads(Path(str(path)+'.history_manifest.json').read_text())
    if h['interrupted'] or not h['fixed_fleet_valid'] or h['actor_losses']:raise ValueError('Invalid fleet')
    members=set(m['malicious_target_ids']);groups=defaultdict(list)
    seen=set();attack=m['attack'];offsets=[]
    def close(a,b):
        if not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7):raise ValueError(f'Equation mismatch: {a} != {b}')
    for r in rows:
        key=(r['target_id'],r['beacon_sequence'])
        if key in seen:raise ValueError('Duplicate target beacon')
        seen.add(key);t=r['true_state'];c=r['claimed_state'];w=r['wire_state']
        if r['malicious']!=(r['target_id'] in members):raise ValueError('Incorrect attack label')
        if not all(math.isfinite(float(v)) for s in [t,c,w] for v in s.values()):raise ValueError('Nonfinite state')
        for k in ['x','y','speed','heading']:close(w[k],round(c[k],1 if k=='heading' else 2))
        if not r['malicious']:
            for k in t:close(c[k],t[k])
            continue
        groups[r['target_id']].append(r)
        initial=m['initial_states'][str(r['target_id'])];s=initial['state']
        if attack=='a1_constant_position':
            close(c['x'],s['x']);close(c['y'],s['y']);close(c['speed'],t['speed']);close(c['heading'],t['heading'])
        elif attack=='a4_eventual_stop':
            duration=m['stop_seconds'];dt=min(max(0,r['time_s']-initial['time_s']),duration)
            close(c['speed'],s['speed']*(1-dt/duration));close(c['heading'],s['heading'])
            d=s['speed']*(dt-dt*dt/(2*duration));angle=math.radians(s['heading'])
            close(c['x'],s['x']+d*math.cos(angle));close(c['y'],s['y']+d*math.sin(angle))
        else:
            dx,dy=c['x']-t['x'],c['y']-t['y']
            if max(abs(dx),abs(dy))>m['offset_bound_m']+1e-7:raise ValueError('Offset outside bounds')
            close(c['speed'],t['speed']);close(c['heading'],t['heading'])
            offsets.append((r['target_id'],dx,dy))
    if set(groups)!=members:raise ValueError('Missing malicious-target transmissions')
    if attack=='a2_constant_offset':
        for actor in members:
            v=[(dx,dy) for i,dx,dy in offsets if i==actor]
            for dx,dy in v:close(dx,v[0][0]);close(dy,v[0][1])
    if attack=='a3_random_offset':
        for actor in members:
            if len({(round(dx,7),round(dy,7)) for i,dx,dy in offsets if i==actor})<2:raise ValueError('Random offset did not vary')
    if attack=='a5_collusive_offset':
        for _,dx,dy in offsets:close(dx,m['shared_offset_xy_m'][0]);close(dy,m['shared_offset_xy_m'][1])
        co=m['coalition'];pos=co['reference_positions'];anchor=pos[str(co['anchor_id'])]
        for actor in members:
            xy=pos[str(actor)]
            if math.dist(xy,anchor)>co['radius_m']+1e-7:raise ValueError('Coalition outside radius')
        if len(members)<2:raise ValueError('A5 needs a coalition')
    diagnostics=[]
    for actor,rr in groups.items():
        rr.sort(key=lambda r:r['time_s']);first=rr[0]['true_state']
        movement=max(math.hypot(r['true_state']['x']-first['x'],r['true_state']['y']-first['y']) for r in rr)
        error=max(math.hypot(r['claimed_state']['x']-r['true_state']['x'],r['claimed_state']['y']-r['true_state']['y']) for r in rr)
        diagnostics.append(dict(target_id=actor,true_displacement_m=movement,max_claim_error_m=error))
    print(json.dumps(dict(attack=attack,equations_pass=True,transmissions=len(rows),
        realised_fraction=m['actual_target_fraction'],motion_diagnostics=diagnostics,
        note='Low motion can make A1/A4 uninformative; an equation pass is not a detection result.'),indent=2))
    return diagnostics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();audit(a.input)
