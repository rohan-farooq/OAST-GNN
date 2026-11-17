"""Frozen A1-A5 development comparison. Place in observation_comparison_kit.
Uses existing adapters; never trains, tunes thresholds or merges attempts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ATTACKS = ['a1_constant_position', 'a2_constant_offset', 'a3_random_offset',
           'a4_eventual_stop', 'a5_collusive_offset']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp, fp, tn, fn = [int(x.sum()) for x in
        [(y == 1)&(p == 1), (y == 0)&(p == 1),
         (y == 0)&(p == 0), (y == 1)&(p == 0)]]
    def div(a, b): return a / b if b else None
    return dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn,
        precision=div(tp,tp+fp), recall=div(tp,tp+fn),
        f1=div(2*tp,2*tp+fp+fn) if tp+fn else None, fpr=div(fp,fp+tn))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attack-root', type=Path, required=True)
    ap.add_argument('--a4-root', type=Path, required=True)
    ap.add_argument('--attempt', default='attempt_002')
    ap.add_argument('--seed', type=int, default=2047)
    ap.add_argument('--assisted-checkpoint', type=Path,
                    default=Path('observation_comparison_smoke_v1/assisted_model.pt'))
    ap.add_argument('--mlp-checkpoint', type=Path,
                    default=Path('current_mlp_60epoch_v1/current_mlp.pt'))
    ap.add_argument('--residual-rule', type=Path,
                    default=Path('observation_comparison_smoke_v1/residual_baseline.json'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audit-only', action='store_true')
    a = ap.parse_args()
    if a.seed not in {2047,2048,2049}: raise ValueError('Development seeds only.')
    if a.output.exists(): raise ValueError('Use a fresh output directory.')
    import full_history as history
    import observation_adapter as adapter
    import compare_observations as training
    import stgnn_a6 as base
    rule = json.loads(a.residual_rule.read_text())
    threshold = float(rule['threshold_m'])
    if not np.isfinite(threshold) or threshold < 0: raise ValueError('Invalid rule.')
    models, provenance = {}, {}
    if not a.audit_only:
        import torch
        torch.set_num_threads(2)
        for method, path, variant in [('Assisted_GRU_GAT', a.assisted_checkpoint, 'assisted'),
                                     ('Current_MLP', a.mlp_checkpoint, 'current_feature_mlp')]:
            cp = torch.load(path, map_location='cpu', weights_only=True)
            if cp.get('variant') != variant or cp.get('best_epoch',0) <= 0 or cp.get('window') != 20:
                raise ValueError(f'Incompatible or untrained checkpoint: {path}')
            if a.seed in cp.get('train_seeds',[]): raise ValueError('Evaluation seed in training.')
            norm = {k: np.asarray(cp['stats'][k],dtype=float) for k in ['mean','std']}
            if any(v.shape != (13,) or not np.isfinite(v).all() for v in norm.values()) or (norm['std'] <= 0).any():
                raise ValueError('Invalid normalization.')
            if method == 'Assisted_GRU_GAT':
                net = base.make_model(); net.gru = torch.nn.GRU(13,64,batch_first=True)
            else:
                net = torch.nn.Sequential(torch.nn.Linear(13,64),torch.nn.ReLU(),
                    torch.nn.Dropout(.3),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
            net.load_state_dict(cp['state'],strict=True); net.eval()
            models[method] = (net,norm)
            provenance[method] = dict(sha256=sha(path),best_epoch=cp['best_epoch'],
                train_seeds=cp.get('train_seeds'),val_seeds=cp.get('val_seeds'),threshold=.5)
    audits, predictions, table = [], [], []
    for attack in ATTACKS:
        root = a.a4_root if attack.startswith('a4') else a.attack_root/f'seed{a.seed}_{attack}'/a.attempt
        runs = history.discover(root)
        if len(runs) != 1: raise ValueError(f'Expected exactly one run in {root}; found {len(runs)}')
        path, rows = runs[0]
        manifest = json.loads(Path(str(path)+'.target_attack_manifest.json').read_text())
        if manifest['attack'] != attack or manifest['seed'] != a.seed or manifest['interrupted']:
            raise ValueError('Attack manifest mismatch.')
        trace = path.parent/manifest['trace_file']
        if sha(trace) != manifest['trace_sha256']: raise ValueError('Trace hash mismatch.')
        if attack.startswith('a4'):
            gate = manifest['moving_start']['min_speed_mps']
            if any(s['state']['speed'] < gate for s in manifest['initial_states'].values()):
                raise ValueError('A4 did not start moving.')
        if any(r['reporter_is_byzantine'] or r['reporter_attack'] != 'none' for r in rows):
            raise ValueError('This stage requires honest reporters.')
        obs = adapter.load_observations(path, rows)
        graphs = adapter.examples(rows,20,obs)
        if not graphs: raise ValueError('No eligible graphs.')
        seen, labels = set(), []
        decisions = {m: [] for m in [*models,'Fixed_residual']}
        for g in graphs:
            scores = {}
            for method,(net,norm) in models.items():
                arrays = training.inputs(g,'assisted',norm)
                if not all(np.isfinite(x).all() for x in arrays): raise ValueError('Nonfinite input.')
                with torch.inference_mode():
                    logits = net(*[torch.from_numpy(x) for x in arrays]) if method == 'Assisted_GRU_GAT' else net(torch.from_numpy(arrays[0][:,-1,:])).squeeze(-1)
                    scores[method] = torch.sigmoid(logits).reshape(-1).tolist()
            for i,row in g['selected']:
                item = str(row['report_sequence'])
                if item in seen: raise ValueError('Duplicate report.')
                seen.add(item)
                if g['obs_x'][i,-1,5] != 0:
                    raise ValueError('Missing current observation: cannot use full-coverage matched protocol.')
                y = int(row['target_is_malicious']); labels.append(y)
                residual = float(g['obs_x'][i,-1,2])
                if not np.isfinite(residual): raise ValueError('Nonfinite residual.')
                for method in decisions:
                    value = residual if method == 'Fixed_residual' else float(scores[method][i])
                    if not np.isfinite(value): raise ValueError('Nonfinite score.')
                    cut = threshold if method == 'Fixed_residual' else .5
                    pred = int(value > cut) if method == 'Fixed_residual' else int(value >= cut)
                    decisions[method].append(pred)
                    predictions.append(dict(attack=attack,seed=a.seed,method=method,item_id=item,
                        reporter_id=row['reporter_id'],target_id=row['target_id'],y_true=y,
                        raw_score=value,score_units='metres' if method == 'Fixed_residual' else 'probability',
                        threshold=cut,prediction=pred))
        expected = {str(r['report_sequence']) for r in rows if r['beacon_sequence'] >= 20}
        if seen != expected: raise ValueError('Incomplete eligible-report coverage.')
        audit = dict(attack=attack,path=str(path.resolve()),generated_reports=len(rows),
            eligible_reports=len(seen),excluded_before_window=len(rows)-len(seen),
            graphs=len(graphs),current_observation_coverage=1.0,window_slots=20,
            window_seconds=20*history.CACHE[str(path)]['period'],
            sources={q.name:sha(q) for q in path.parent.glob(path.name+'*') if q.is_file()})
        audits.append(audit)
        for method,p in decisions.items(): table.append(dict(attack=attack,method=method,**metrics(labels,p)))
        print(f'PASS {attack}: {len(seen)} identical eligible reports',flush=True)
    a.output.mkdir(parents=True)
    for name, records in [('predictions.csv',predictions),('per_attack_metrics.csv',table)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report = dict(scope='Development-seed attack transfer; local reports; not TRE reproduction or A6/RSU evaluation. A4 onset excluded by history window.',
        audit_only=a.audit_only,models=provenance,residual_rule_sha256=sha(a.residual_rule),
        residual_threshold_m=threshold,runs=audits)
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',a.output.resolve())

if __name__ == '__main__': main()
