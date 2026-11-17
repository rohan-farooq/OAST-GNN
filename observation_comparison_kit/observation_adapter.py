"""Causal observation channels for the existing full-history graph adapter.
Returns original x plus obs_x; does not alter the seven-feature model or train it.
"""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import full_history as history
import stgnn_a6 as base
from check_observations import check

CHANNELS=['residual_x_m','residual_y_m','residual_norm_m','noise_std_m','age_at_slot_s','missing']

def load_observations(path,rows):
    check(path)
    observations=[json.loads(s) for s in Path(str(path)+'.observations.jsonl').read_text().splitlines() if s.strip()]
    reports={r['report_sequence']:r for r in rows}
    if set(reports)!={r['report_id'] for r in observations}:raise ValueError('Report IDs do not match graph records')
    for o in observations:
        r=reports[o['report_id']]
        if (o['reporter_id'],o['target_id'],o['beacon_sequence'])!=(r['reporter_id'],r['target_id'],r['beacon_sequence']):
            raise ValueError('Observation identity mismatch')
        if not math.isclose(o['claimed_x_m'],r['claimed_target_x_m'],abs_tol=1e-8) or not math.isclose(o['claimed_y_m'],r['claimed_target_y_m'],abs_tol=1e-8):
            raise ValueError('Observation and beacon claims differ')
    return observations

def examples(rows,window,observations):
    if window<1:raise ValueError('Window must be positive')
    ctx=history.CACHE[rows[0]['_source']]
    lookup=defaultdict(list)
    for o in observations:lookup[(o['reporter_id'],o['target_id'],o['beacon_sequence'])].append(o)
    original=base.graph
    def augment(past,current,width):
        g=original(past,current,width)
        end=current[0]['beacon_sequence'];cut=min(r['report_created_time_s'] for r in current)
        avail=[r for r in past if end-width<r['beacon_sequence']<=end and r['beacon_rx_time_s']<=cut]
        nodes=sorted({r['target_id'] for r in avail});observer=current[0]['reporter_id']
        ox=np.zeros((len(nodes),width,len(CHANNELS)),dtype=np.float32);ox[:,:,5]=1
        for i,target in enumerate(nodes):
            # No independent target measurement is invented for the ego node.
            if target==observer:continue
            for t,seq in enumerate(range(end-width+1,end+1)):
                if g['x'][i,t,6]!=0:continue
                slot_cut=min(cut,ctx['start']+seq*ctx['period'])
                candidates=[o for o in lookup[(observer,target,seq)] if o['available'] and o['available_time_s']<=slot_cut and o['reference_time_s']<=slot_cut]
                if not candidates:continue
                o=max(candidates,key=lambda v:v['available_time_s'])
                ox[i,t]=[o['residual_x_m'],o['residual_y_m'],o['residual_norm_m'],o['noise_std_m'],slot_cut-o['reference_time_s'],0]
        g['obs_x']=ox
        return g
    # Existing adapter calls base.graph; restore it even if graph building fails.
    base.graph=augment
    try:return history.examples(rows,window)
    finally:base.graph=original

def audit(root,window):
    results=[]
    for p,rows in history.discover(root):
        obs=load_observations(p,rows);gs=examples(rows,window,obs)
        results.append(dict(run=str(p),seed=rows[0]['seed'],observations=len(obs),window_slots=window,
                            window_seconds=window*history.CACHE[str(p)]['period'],graphs=len(gs),
                            present_observation_slots=sum(int((g['obs_x'][:,:,5]==0).sum()) for g in gs),
                            original_node_channels=7,additional_channels=CHANNELS,
                            ready_for_window=bool(gs)))
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--window',type=int,default=20);ap.add_argument('--output',default='observation_audit.json')
    a=ap.parse_args();result=audit(a.input,a.window)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    print('Audit only: no training, no padded history, no claim of improved detection.')
