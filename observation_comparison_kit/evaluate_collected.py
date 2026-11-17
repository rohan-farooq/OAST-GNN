"""Frozen local assisted-model evaluation using the original training adapters.
Place beside compare_observations.py. No training and no RSU aggregation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    def ratio(a, b):
        return a / b if b else None
    return dict(n=len(labels), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn) if tp+fn else None,
                fpr=ratio(fp, fp+tn))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import torch
    import full_history as history
    import observation_adapter as adapter
    import compare_observations as training
    import stgnn_a6 as base

    if args.output.exists():
        raise ValueError('Use a fresh output folder; previous results are preserved.')
    torch.set_num_threads(2)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if cp.get('variant') != 'assisted':
        raise ValueError('Expected the assisted checkpoint.')
    if cp.get('best_epoch', 0) <= 0:
        raise ValueError('Checkpoint retained initialization; refusing trained-model label.')
    window = int(cp['window'])
    norm = {k: np.asarray(cp['stats'][k], dtype=float) for k in ['mean', 'std']}
    if any(v.shape != (13,) or not np.isfinite(v).all() for v in norm.values()) or (norm['std'] <= 0).any():
        raise ValueError('Invalid saved normalization statistics.')
    net = base.make_model()
    net.gru = torch.nn.GRU(13, 64, batch_first=True)
    net.load_state_dict(cp['state'], strict=True)
    net.eval()

    predictions, summaries = [], []
    for path, rows in history.discover(args.input):
        seed = int(rows[0]['seed'])
        if seed not in {2047, 2048, 2049}:
            raise ValueError('This compatibility check is restricted to development seeds 2047-2049.')
        if any(r['reporter_is_byzantine'] or r['reporter_attack'] != 'none' for r in rows):
            raise ValueError('This evaluator checks honest-reporter controls only.')
        meta = json.loads(Path(str(path)+'.metadata.json').read_text())
        observations = adapter.load_observations(path, rows)
        graphs = adapter.examples(rows, window, observations)
        if not graphs:
            raise ValueError(f'No complete windows: {path}')
        labels_present = {int(r['target_is_malicious']) for r in rows}
        if meta['target_attack'] == 'none' and labels_present != {0}:
            raise ValueError('Benign run has malicious target labels.')
        condition = 'all_benign' if labels_present == {0} else 'target_offset_honest_reporters'
        run_id = path.parent.relative_to(args.input).as_posix()
        seen, labels, decisions = set(), [], []
        current_available = 0
        with torch.inference_mode():
            for g in graphs:
                arrays = training.inputs(g, 'assisted', norm)
                if not all(np.isfinite(a).all() for a in arrays):
                    raise ValueError('Nonfinite model input.')
                scores = torch.sigmoid(net(*[torch.from_numpy(a) for a in arrays])).tolist()
                for i, row in g['selected']:
                    item = str(row['report_sequence'])
                    if item in seen:
                        raise ValueError('Duplicate selected report.')
                    seen.add(item)
                    y = int(row['target_is_malicious'])
                    score = float(scores[i])
                    if not np.isfinite(score):
                        raise ValueError('Nonfinite prediction.')
                    labels.append(y); decisions.append(int(score >= 0.5))
                    available = int(g['obs_x'][i, -1, 5] == 0)
                    current_available += available
                    predictions.append(dict(run_id=run_id, seed=seed, condition=condition,
                        attack='benign' if labels_present == {0} else meta['target_attack'],
                        rho=0.0, stage='local', method='Proposed', item_id=item,
                        y_true=y, score=score, threshold=0.5,
                        reporter_id=row['reporter_id'], target_id=row['target_id'],
                        current_observation_available=available))
        expected = {str(r['report_sequence']) for r in rows if r['beacon_sequence'] >= window}
        if seen != expected:
            raise ValueError('Selected-report coverage differs from complete-window eligibility.')
        summaries.append(dict(run_id=run_id, seed=seed, condition=condition,
            generated_reports=len(rows), eligible_reports=len(expected),
            excluded_before_window=len(rows)-len(expected), observations=len(observations),
            graphs=len(graphs), window_slots=window,
            window_seconds=window*history.CACHE[str(path)]['period'],
            current_observation_coverage=current_available/len(seen),
            metrics=metrics(labels, decisions),
            observation_sha256=hashlib.sha256(Path(str(path)+'.observations.jsonl').read_bytes()).hexdigest()))
        print(json.dumps(summaries[-1], indent=2), flush=True)

    args.output.mkdir(parents=True)
    with (args.output/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_epoch=cp['best_epoch'], threshold=0.5,
        scope='Development only; local generated reports after history warm-up; no RSU voting or attack replay; not a TRE reproduction.',
        checkpoint_train_seeds=cp.get('train_seeds'), checkpoint_val_seeds=cp.get('val_seeds'),
        runs=summaries)
    (args.output/'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('Saved predictions.csv and evaluation.json to', args.output.resolve())


if __name__ == '__main__':
    main()
