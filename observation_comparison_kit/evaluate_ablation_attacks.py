"""Frozen A1-A5 development-transfer evaluation of run_ablations.py checkpoints.
No training, normalization fitting, threshold selection or CARLA calls.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ATTACKS = {
    'a1_constant_position':'A1', 'a2_constant_offset':'A2',
    'a3_random_offset':'A3', 'a4_eventual_stop':'A4',
    'a5_collusive_offset':'A5',
}
VARIANTS = ['full','no_gru','no_gat','no_edges','no_observations']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def csvwrite(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def locate(root, attempt, moving=False):
    candidates=[]
    for p in root.rglob('*.target_attack_manifest.json'):
        if not moving and attempt not in p.parts:
            continue
        d=json.loads(p.read_text());attack=d.get('attack')
        if attack not in ATTACKS or (moving and attack!='a4_eventual_stop'):
            continue
        if not moving and attack=='a4_eventual_stop':
            continue  # A4 must come from the explicitly selected moving-start root.
        candidates.append((attack,p,d))
    return candidates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit',type=Path,default=Path(__file__).parent)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--attack-root',type=Path,required=True)
    ap.add_argument('--attempt',default='attempt_002')
    ap.add_argument('--a4-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args()
    if a.output.exists():ap.error('Use a fresh output folder.')
    for p in [a.kit,a.checkpoints,a.attack_root,a.a4_root]:
        if not p.is_dir():ap.error(f'Directory not found: {p}')
    if a.threads<1:ap.error('--threads must be positive')
    sys.path.insert(0,str(a.kit.resolve()))
    import numpy as np
    import torch
    import run_ablations as trainer
    import compare_observations as original
    import observation_adapter as adapter
    import full_history as history
    torch.set_num_threads(a.threads)
    training_protocol=json.loads((a.checkpoints/'protocol.json').read_text())
    if training_protocol.get('smoke_only'):
        raise ValueError('Refusing smoke-only checkpoints for this measured transfer evaluation.')
    for name,expected in training_protocol['adapter_sha256'].items():
        if sha(a.kit/name)!=expected:
            raise ValueError(f'Adapter differs from training: {name}. Use the original training adapter.')
    variants=training_protocol['variants'];initializations=training_protocol['training_initializations']
    if set(variants)!=set(VARIANTS):raise ValueError('Expected all five ablation variants.')
    selected=locate(a.attack_root,a.attempt)+locate(a.a4_root,a.attempt,True)
    counts={key:sum(attack==key for attack,_,_ in selected) for key in ATTACKS}
    if any(v!=1 for v in counts.values()):
        raise ValueError(f'Need exactly one run per attack; selected counts: {counts}. Narrow --a4-root if necessary.')
    selected.sort(key=lambda t:ATTACKS[t[0]])
    # Preflight all checkpoints before beginning evaluation. Only trusted local artifacts.
    models={};checkpoint_info=[]
    for init in initializations:
        for variant in variants:
            cp=a.checkpoints/f'{variant}_init{init}'/'model.pt'
            ck=torch.load(cp,map_location='cpu',weights_only=True)
            if ck['variant']!=variant or ck['initialization']!=init:raise ValueError('Checkpoint identity mismatch')
            checkpoint_protocol = json.loads(json.dumps(ck["protocol"]))

            if checkpoint_protocol != training_protocol:
                differing_keys = [
                    key
                    for key in set(checkpoint_protocol) | set(training_protocol)
                    if checkpoint_protocol.get(key) != training_protocol.get(key)
                ]
                raise ValueError(
                    f"Checkpoint protocol mismatch in: {sorted(differing_keys)}"
                )
            model=trainer.make_model(torch,variant);model.load_state_dict(ck['state'],strict=True);model.eval()
            models[(init,variant)]=(model,ck)
            checkpoint_info.append(dict(variant=variant,initialization=init,best_epoch=ck['best_epoch'],
                                        trained_checkpoint_selected=ck['best_epoch']>0,sha256=sha(cp)))
    a.output.mkdir(parents=True)
    dump(a.output/'protocol.json',dict(scope='Frozen A1-A5 development transfer, not untouched testing',
        completed=False,threshold=.5,window=20,checkpoint_info=checkpoint_info,
        training_protocol=training_protocol,attempt=a.attempt,a4_root=str(a.a4_root.resolve())))
    metrics=[];predictions=[];audit=[];paired=[]
    for attack,manifest,attack_meta in selected:
        name=ATTACKS[attack]
        if attack_meta.get('interrupted'):raise ValueError(f'Interrupted attack run: {manifest}')
        base=Path(str(manifest).removesuffix('.target_attack_manifest.json'))
        hm=json.loads(Path(str(base)+'.history_manifest.json').read_text())
        if not hm.get('fixed_fleet_valid',False) or hm.get('actor_losses'):
            raise ValueError(f'Invalid or unverified fixed fleet: {base}')
        found=history.discover(base.parent)
        if len(found)!=1 or found[0][0].resolve()!=base.resolve():
            raise ValueError(f'Expected one source run in {base.parent}')
        path,rows=found[0]
        if any(r['reporter_is_byzantine'] or r['reporter_attack']!='none' for r in rows):
            raise ValueError('Local attack transfer requires honest reporters.')
        obs=adapter.load_observations(path,rows)
        graphs=adapter.examples(rows,20,obs)
        expected={str(r['report_sequence']):r for r in rows if r['beacon_sequence']>=20}
        graph_ids=[str(r['report_sequence']) for g in graphs for _,r in g['selected']]
        if not expected or len(graph_ids)!=len(set(graph_ids)) or set(graph_ids)!=set(expected):
            raise ValueError('Graphs do not cover the identical complete-window report population.')
        seed=int(rows[0]['seed'])
        run_info=dict(attack=name,attack_name=attack,simulation_seed=seed,source=str(base.resolve()),
                      generated_reports=len(rows),eligible_reports=len(expected),graphs=len(graphs),
                      a4_post_deceleration_only=name=='A4',source_sha256={q.name:sha(q) for q in base.parent.glob(base.name+'*') if q.is_file()})
        audit.append(run_info)
        variant_predictions={}
        for init in initializations:
            for variant in variants:
                model,ck=models[(init,variant)];local=[]
                mode='baseline' if variant=='no_observations' else 'assisted'
                with torch.inference_mode():
                    for g in graphs:
                        x,e,adj=original.inputs(g,mode,ck['stats'])
                        logits=model(torch.from_numpy(x),torch.from_numpy(e),torch.from_numpy(adj))
                        scores=torch.sigmoid(logits[original.selected(g)]).tolist()
                        for (_,r),score in zip(g['selected'],scores):
                            if not np.isfinite(score):raise ValueError('Nonfinite prediction')
                            local.append(dict(attack=name,variant=variant,initialization=init,simulation_seed=seed,
                                report_id=str(r['report_sequence']),reporter_id=r['reporter_id'],target_id=r['target_id'],
                                beacon_sequence=r['beacon_sequence'],y_true=int(r['target_is_malicious']),score=float(score),
                                threshold=.5,best_epoch=ck['best_epoch'],trained_checkpoint_selected=ck['best_epoch']>0))
                ids={r['report_id']:r for r in local}
                assert len(local)==len(expected) and set(ids)==set(expected)
                assert all(ids[k]['y_true']==int(expected[k]['target_is_malicious']) for k in ids)
                variant_predictions[(init,variant)]=ids
                predictions.extend(local)
                record=dict(attack=name,variant=variant,initialization=init,best_epoch=ck['best_epoch'],
                    trained_checkpoint_selected=ck['best_epoch']>0,parameters=sum(p.numel() for p in model.parameters()),**trainer.counts(local))
                metrics.append(record)
                print(name,variant,'init',init,'F1',record['f1'],'epoch',ck['best_epoch'],flush=True)
        for init in initializations:
            full=variant_predictions[(init,'full')]
            for variant in variants:
                if variant=='full':continue
                other=variant_predictions[(init,variant)];corrected=introduced=0
                for k,r in full.items():
                    fc=(r['score']>=.5)==r['y_true'];oc=(other[k]['score']>=.5)==r['y_true']
                    corrected+=int(fc and not oc);introduced+=int(oc and not fc)
                paired.append(dict(attack=name,variant=variant,initialization=init,n=len(full),
                    comparator_trained=models[(init,variant)][1]['best_epoch']>0,
                    errors_corrected_by_full=corrected,errors_introduced_by_full=introduced))
        print('MATCHED',name,len(expected),'reports across all models.',flush=True)
    for init in initializations:
        for variant in variants:
            rows=[r for r in predictions if r['initialization']==init and r['variant']==variant]
            ck=models[(init,variant)][1]
            metrics.append(dict(attack='Pooled',variant=variant,initialization=init,best_epoch=ck['best_epoch'],
                trained_checkpoint_selected=ck['best_epoch']>0,parameters=sum(p.numel() for p in models[(init,variant)][0].parameters()),**trainer.counts(rows)))
    summary=[]
    for attack in ['A1','A2','A3','A4','A5','Pooled']:
        for variant in variants:
            rr=[r for r in metrics if r['attack']==attack and r['variant']==variant]
            entry=dict(attack=attack,variant=variant,initializations=len(rr),
                trained_initializations=sum(r['trained_checkpoint_selected'] for r in rr),n_per_initialization=rr[0]['n'],parameters=rr[0]['parameters'])
            for key in ['precision','recall','f1','fpr']:
                vals=[r[key] for r in rr if r[key] is not None]
                entry[key+'_mean']=statistics.mean(vals) if vals else None
                entry[key+'_sd']=statistics.stdev(vals) if len(vals)>1 else None
            summary.append(entry)
    csvwrite(a.output/'predictions.csv',predictions)
    csvwrite(a.output/'per_attack_metrics.csv',metrics)
    csvwrite(a.output/'ablation_transfer_summary.csv',summary)
    csvwrite(a.output/'paired_errors.csv',paired)
    dump(a.output/'data_audit.json',audit)
    # Table marks initialization-only results rather than silently treating them as trained.
    table=[]
    for variant in variants:
        rr={r['attack']:r for r in summary if r['variant']==variant}
        status='' if rr['Pooled']['trained_initializations']==len(initializations) else r' $^{\dagger}$'
        vals=[]
        for attack in ['A1','A2','A3','A4','A5']:
            r=rr[attack];v=r['f1_mean'];sd=r['f1_sd']
            vals.append('--' if v is None else f'{v:.4f}'+(f' $\\pm$ {sd:.4f}' if sd is not None else ''))
        table.append(trainer.LABELS[variant]+status+' & '+' & '.join(vals)+r' \\')
    (a.output/'per_attack_rows.tex').write_text('\n'.join(table)+'\n% dagger: one or more retained checkpoints are initialization-only.\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
        colors=['#557C70','#967A9B','#B78C5B','#63758E','#9B9B9B']
        x=np.arange(5);width=.16
        for i,variant in enumerate(variants):
            rr=[next(r for r in summary if r['variant']==variant and r['attack']==attack) for attack in ['A1','A2','A3','A4','A5']]
            status=' [init retained]' if any(r['trained_initializations']<len(initializations) for r in rr) else ''
            ax.bar(x+(i-2)*width,[r['f1_mean'] or 0 for r in rr],width,
                yerr=[r['f1_sd'] or 0 for r in rr],capsize=2,label=trainer.LABELS[variant]+status,
                color=colors[i],edgecolor='black',linewidth=.4,hatch=['','//','xx','..','\\\\'][i])
        ax.set_xticks(x,['A1','A2','A3','A4','A5']);ax.set_ylim(0,1.08);ax.set_ylabel('F1')
        ax.set_title('Frozen development transfer; SD across training initializations')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for ext in ['png','pdf']:fig.savefig(a.output/f'ablation_transfer.{ext}',dpi=300)
        plt.close(fig)
    except ImportError:
        print('Install matplotlib for plots; tables/predictions were saved.')
    protocol=json.loads((a.output/'protocol.json').read_text());protocol.update(completed=True,runs=audit)
    dump(a.output/'protocol.json',protocol)
    print('COMPLETE',a.output.resolve())
    print('No fitting performed. This is development transfer, not untouched testing.')


if __name__=='__main__':
    main()
