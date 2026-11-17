"""Check a completed observation run. Does not train or expose labels as features."""
import argparse,json,math,hashlib
from pathlib import Path

def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def check(output):
    p=Path(output)
    m=json.loads(Path(str(p)+'.observations.manifest.json').read_text())
    h=json.loads(Path(str(p)+'.history_manifest.json').read_text())
    if m['interrupted'] or h['interrupted'] or not h.get('fixed_fleet_valid',False):
        raise ValueError('Interrupted or invalid fleet: do not train')
    q=Path(str(p)+'.observations.jsonl');rows=lines(q)
    if not rows or len(rows)!=m['count'] or hashlib.sha256(q.read_bytes()).hexdigest()!=m['sha256']:
        raise ValueError('Observation count/hash mismatch')
    attacks=lines(Path(str(p)+'.attacks.evaluation.jsonl'))
    a={r['report_id']:r for r in attacks}
    if len(a)!=len(attacks) or len({r['report_id'] for r in rows})!=len(rows) or set(a)!={r['report_id'] for r in rows}:
        raise ValueError('Generated report/observation IDs differ')
    for r in rows:
        if any(k in r for k in ('target_is_malicious','reporter_is_byzantine','true_x','true_y','local_score','submitted_score')):
            raise ValueError('Forbidden evaluation field')
        t=a[r['report_id']]
        if any(r[k]!=t[k] for k in ('reporter_id','target_id','beacon_sequence')):raise ValueError('Join mismatch')
        if not r['available'] or r['reference_time_s']>r['available_time_s']+1e-8 or r['available_time_s']>t['created_time_s']+1e-8:
            raise ValueError('Unavailable or future observation')
        nums=[v for v in r.values() if isinstance(v,(float,int))]
        if not all(math.isfinite(v) for v in nums) or r['noise_std_m']<0:raise ValueError('Invalid numeric value')
        for actual,expected in [(r['residual_x_m'],r['claimed_x_m']-r['observed_x_m']),
                                (r['residual_y_m'],r['claimed_y_m']-r['observed_y_m']),
                                (r['residual_norm_m'],math.hypot(r['residual_x_m'],r['residual_y_m'])),
                                (r['age_s'],r['available_time_s']-r['reference_time_s'])]:
            if not math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Residual/age mismatch')
    print(f'PASS: {len(rows)} observations; complete generated-report joins; causal times; finite residuals; valid fleet.')
    print('Synthetic observation logging verified; detector improvement not evaluated.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);args=ap.parse_args();check(args.input)
