"""Controlled OAST-GNN development ablations; no CARLA server required.
Run beside the supplied adapters, or pass --kit PATH.
First run: python run_ablations.py --self-test
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

VARIANTS = ('full', 'no_gru', 'no_gat', 'no_edges', 'no_observations')
LABELS = dict(full='Full OAST-GNN', no_gru='Without GRU', no_gat='Without GAT',
              no_edges='Without edge attributes', no_observations='Without observations')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def counts(rows):
    tp = fp = tn = fn = 0
    for r in rows:
        y, p = int(r['y_true']), float(r['score']) >= 0.5
        tp += int(y == 1 and p)
        fp += int(y == 0 and p)
        tn += int(y == 0 and not p)
        fn += int(y == 1 and not p)
    div = lambda a, b: a / b if b else None
    return dict(n=len(rows), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=div(tp, tp+fp), recall=div(tp, tp+fn),
                f1=div(2*tp, 2*tp+fp+fn), fpr=div(fp, fp+tn))


def make_model(torch, variant):
    nn = torch.nn

    class GAT(nn.Module):
        def __init__(self, dim, edges):
            super().__init__()
            self.edges = edges
            self.self_map = nn.Linear(dim, 128, bias=False)
            self.msg = nn.Linear(dim + (4 if edges else 0), 128, bias=False)
            self.att = nn.Parameter(torch.empty(4, 64))
            nn.init.xavier_uniform_(self.att)

        def forward(self, h, e, adjacency):
            n = h.shape[0]
            own = self.self_map(h).view(n, 4, 32)
            neighbor = h[None, :, :].expand(n, n, -1)
            msg = self.msg(torch.cat([neighbor, e], -1) if self.edges else neighbor)
            msg = msg.view(n, n, 4, 32)
            pair = torch.cat([own[:, None].expand(n, n, 4, 32), msg], -1)
            logits = torch.nn.functional.leaky_relu((pair * self.att).sum(-1), .2)
            weights = logits.masked_fill(~adjacency[:, :, None], float('-inf')).softmax(1)
            return torch.relu(own + (weights[..., None] * msg).sum(1)).reshape(n, 128)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 7 if variant == 'no_observations' else 13
            if variant == 'no_gru':
                self.project = nn.Linear(channels, 64)
            else:
                self.gru = nn.GRU(channels, 64, batch_first=True)
            if variant == 'no_gat':
                # Preserve the 128-wide classifier input without mixing nodes.
                self.node_project = nn.Linear(64, 128)
            else:
                self.gat1 = GAT(64, variant != 'no_edges')
                self.gat2 = GAT(128, variant != 'no_edges')
            self.drop = nn.Dropout(.3)
            self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, e, adjacency):
            # Slice explicitly so an observation perturbation cannot enter this variant.
            if variant == 'no_observations':
                x = x[..., :7]
            h = torch.relu(self.project(x[:, -1])) if variant == 'no_gru' else self.gru(x)[1][0]
            if variant == 'no_gat':
                h = self.drop(torch.relu(self.node_project(h)))
            else:
                h = self.drop(self.gat1(h, e, adjacency))
                h = self.drop(self.gat2(h, e, adjacency))
            return self.mlp(h).squeeze(-1)
    return Model()


def self_test(torch):
    torch.manual_seed(42)
    x = torch.randn(4, 20, 13)
    e = torch.randn(4, 4, 4)
    adj = torch.ones(4, 4, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    for variant in VARIANTS:
        model = make_model(torch, variant).eval()
        out = model(x, e, adj)
        assert out.shape == (4,) and torch.isfinite(out).all()
        shuffled = model(x[permutation], e[permutation][:, permutation], adj[permutation][:, permutation])
        torch.testing.assert_close(shuffled, out[permutation], atol=1e-5, rtol=1e-5)
        altered = x.clone()
        if variant == 'no_gru':
            altered[:, :-1] += 10
            torch.testing.assert_close(out, model(altered, e, adj))
        if variant == 'no_observations':
            altered[..., 7:] += 10
            torch.testing.assert_close(out, model(altered, e, adj))
        if variant == 'no_edges':
            torch.testing.assert_close(out, model(x, e+10, adj))
        if variant == 'no_gat':
            torch.testing.assert_close(out, model(x, e+10, torch.eye(4, dtype=torch.bool)))
        out.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print('PASS', variant, flush=True)
    print('PASS: shapes, finite gradients, graph permutation and ablation isolation.')


def plot(output, metrics, histories):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('Plots skipped: install matplotlib; all CSV results remain available.')
        return
    order = list(dict.fromkeys(r['variant'] for r in metrics))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, scale, title in [(axes[0], 'f1', 1, 'Validation F1'),
                                   (axes[1], 'fpr', 100, 'Validation FPR (%)')]:
        for i, v in enumerate(order):
            values = [r[key]*scale for r in metrics if r['variant']==v and r[key] is not None]
            mean = float(np.mean(values))
            ax.bar(i, mean, color='#73938A', edgecolor='#303030', hatch=['','//','xx','..','\\\\'][i%5])
            ax.scatter([i]*len(values), values, c='#5A365B', s=24, zorder=3)
            if len(values)>1:
                ax.errorbar(i, mean, yerr=np.std(values, ddof=1), color='black', capsize=4)
        ax.set_xticks(range(len(order)), [LABELS[v] for v in order], rotation=25, ha='right')
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Development controls — not A1–A5 transfer or untouched test')
    for ext in ['png','pdf']:
        fig.savefig(output/f'ablation_validation.{ext}', dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, default=Path(__file__).parent)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--training-seeds', default='42,43,44')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--graph-batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Two epochs, initialization 42; execution check only')
    a = p.parse_args()
    import numpy as np
    import torch
    if a.self_test:
        self_test(torch)
        return
    if a.input is None or a.output is None:
        p.error('--input and --output are required for training')
    variants = a.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in VARIANTS for v in variants):
        p.error('Use unique variants from '+','.join(VARIANTS))
    seeds = [int(s) for s in a.training_seeds.split(',')]
    if len(set(seeds))!=len(seeds):
        p.error('Training seeds must be unique')
    if min(a.epochs,a.patience,a.graph_batch_size,a.threads)<1:
        p.error('Epochs, patience, batch size and threads must be positive')
    if a.smoke:
        seeds=[42];a.epochs=2
    if a.output.exists():
        p.error('Use a fresh output folder; existing results are never overwritten')
    kit=a.kit.resolve()
    required=['compare_observations.py','observation_adapter.py','full_history.py','stgnn_a6.py','check_observations.py']
    for name in required:
        if not (kit/name).is_file():
            p.error(f'Missing {kit/name}; point --kit to observation_comparison_kit')
    sys.path.insert(0,str(kit))
    import compare_observations as source
    torch.set_num_threads(a.threads)
    groups, sources = source.load(a.input.resolve())
    # Loader enforces precisely one honest-reporter control per development seed.
    a.output.mkdir(parents=True)
    normalizations={v:source.stats(groups,'baseline' if v=='no_observations' else 'assisted') for v in variants}
    labels=[int(g['y'][i]) for seed in sorted(source.TRAIN) for g in groups[seed] for i in source.selected(g)]
    positives=sum(labels);negatives=len(labels)-positives
    if not positives or not negatives:
        raise ValueError('Both training classes are required')
    protocol=dict(scope='development-control ablation; not untouched test or per-attack transfer',
                  smoke_only=a.smoke,variants=variants,training_initializations=seeds,
                  train_simulation_seeds=sorted(source.TRAIN),validation_simulation_seeds=sorted(source.VAL),
                  epochs=a.epochs,patience=a.patience,graph_batch_size=a.graph_batch_size,
                  optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-3,gradient_clip=5,
                  positive_weight=negatives/positives,training_reports=len(labels),
                  threshold=.5,min_loss_improvement=1e-4,
                  batch_loss='mean of graph means; selected-report BCE within graph',
                  selection='report-weighted validation BCE; initialization eligible',
                  sources=sources,adapter_sha256={n:hashlib.sha256((kit/n).read_bytes()).hexdigest() for n in required})
    write_json(a.output/'protocol.json',protocol)
    all_metrics=[];all_history=[];paired=[];reference={}
    for initialization in seeds:
        for variant in variants:
            random.seed(initialization);np.random.seed(initialization);torch.manual_seed(initialization)
            rng=random.Random(initialization)
            model=make_model(torch,variant)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
            lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives))
            norm=normalizations[variant]
            mode='baseline' if variant=='no_observations' else 'assisted'
            def tensors(g):
                return [torch.from_numpy(t) for t in source.inputs(g,mode,norm)]
            def evaluate():
                model.eval();rows=[];loss_sum=0.;count=0;identities=set()
                with torch.no_grad():
                    for seed in sorted(source.VAL):
                        for g in groups[seed]:
                            idx=source.selected(g)
                            logits=model(*tensors(g))[idx]
                            y=torch.tensor(g['y'][idx])
                            loss_sum+=float(lossfn(logits,y))*len(idx);count+=len(idx)
                            for (_,r),score in zip(g['selected'],torch.sigmoid(logits).tolist()):
                                identity=(seed,str(r['report_sequence']))
                                if identity in identities:raise ValueError('Duplicate validation report')
                                identities.add(identity)
                                rows.append(dict(simulation_seed=seed,report_id=identity[1],y_true=int(r['target_is_malicious']),score=score,threshold=.5))
                return loss_sum/count,rows
            initial_loss,_=evaluate();best_loss=initial_loss;best_epoch=0;bad=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            directory=a.output/f'{variant}_init{initialization}';directory.mkdir()
            history=[]
            training=[g for seed in sorted(source.TRAIN) for g in groups[seed]]
            for epoch in range(1,a.epochs+1):
                model.train();rng.shuffle(training);total=0.
                for start in range(0,len(training),a.graph_batch_size):
                    batch=training[start:start+a.graph_batch_size];opt.zero_grad()
                    for g in batch:
                        idx=source.selected(g)
                        loss=lossfn(model(*tensors(g))[idx],torch.tensor(g['y'][idx]))
                        (loss/len(batch)).backward();total+=float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                val_loss,rows=evaluate()
                record=dict(variant=variant,initialization=initialization,epoch=epoch,train_loss=total/len(training),validation_loss=val_loss,**counts(rows))
                history.append(record);print(json.dumps(record),flush=True)
                if val_loss<best_loss-1e-4:
                    best_loss=val_loss;best_epoch=epoch;bad=0
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:bad+=1
                write_json(directory/'training_history.json',dict(initial_validation_loss=initial_loss,best_epoch=best_epoch,history=history))
                if bad>=a.patience:break
            model.load_state_dict(best_state);val_loss,rows=evaluate()
            ids={(r['simulation_seed'],r['report_id']):r for r in rows}
            if initialization in reference:
                old=reference[initialization]
                assert old.keys()==ids.keys() and all(old[k]['y_true']==ids[k]['y_true'] for k in ids),'Unmatched validation cases'
            else:reference[initialization]=ids
            summary=dict(variant=variant,initialization=initialization,best_epoch=best_epoch,
                         trained_checkpoint_selected=best_epoch>0,parameters=sum(t.numel() for t in model.parameters()),validation_loss=val_loss,**counts(rows))
            write_csv(directory/'validation_predictions.csv',rows)
            per_seed=[dict(simulation_seed=seed,**counts([r for r in rows if r['simulation_seed']==seed])) for seed in sorted(source.VAL)]
            write_json(directory/'metrics.json',dict(pooled=summary,per_seed=per_seed))
            torch.save(dict(state=best_state,variant=variant,stats=norm,best_epoch=best_epoch,protocol=protocol,initialization=initialization),directory/'model.pt')
            all_metrics.append(summary);all_history+=history
            write_csv(a.output/'validation_metrics.csv',all_metrics)
            print('SAVED',directory,'best epoch',best_epoch,flush=True)
    for initialization in seeds:
        full=a.output/f'full_init{initialization}/validation_predictions.csv'
        if not full.exists():continue
        ref={(r['simulation_seed'],r['report_id']):r for r in csv.DictReader(full.open())}
        for variant in variants:
            if variant=='full':continue
            rows=list(csv.DictReader((a.output/f'{variant}_init{initialization}/validation_predictions.csv').open()))
            corrected=introduced=0
            for r in rows:
                f=ref[(r['simulation_seed'],r['report_id'])];y=int(r['y_true'])
                fc=(float(f['score'])>=.5)==y;vc=(float(r['score'])>=.5)==y
                corrected+=int(not vc and fc);introduced+=int(vc and not fc)
            paired.append(dict(initialization=initialization,comparison='full versus '+variant,n=len(rows),errors_corrected_by_full=corrected,errors_introduced_by_full=introduced))
    write_csv(a.output/'paired_errors.csv',paired)
    summary=[]
    for variant in variants:
        r=[x for x in all_metrics if x['variant']==variant]
        item=dict(variant=variant,initializations=len(r),parameters=r[0]['parameters'])
        for key in ['precision','recall','f1','fpr']:
            values=[x[key] for x in r if x[key] is not None]
            item[key+'_mean']=float(np.mean(values)) if values else None
            item[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(item)
    write_csv(a.output/'ablation_summary.csv',summary)
    lines=[]
    for r in summary:
        values=[]
        for k in ['precision','recall','f1','fpr']:
            factor=100 if k=='fpr' else 1
            v=r[k+'_mean'];sd=r[k+'_sd']
            values.append('--' if v is None else f'{v*factor:.4f}'+(f' $\\pm$ {sd*factor:.4f}' if sd is not None else ''))
        lines.append(LABELS[r['variant']]+' & '+' & '.join(values)+f" & {r['parameters']} "+r'\\')
    (a.output/'ablation_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(a.output,all_metrics,all_history)
    print('COMPLETE:',a.output.resolve())
    print('SMOKE CHECK ONLY.' if a.smoke else 'Development-control results only. Do not label these A1–A5 transfer or final-test results.')


if __name__=='__main__':
    main()
