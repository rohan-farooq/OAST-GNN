"""Current-feature MLP baseline using the existing observation adapters."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import compare_observations as original


TRAIN = list(range(2041, 2047))
VAL = [2047, 2048, 2049]


def calculate_metrics(y, scores):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(scores) >= 0.5

    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())

    def divide(a, b):
        return a / b if b else None

    return {
        "n": len(y),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn)
              if tp + fn else None,
        "fpr": divide(fp, fp + tn),
    }


def make_model():
    return nn.Sequential(
        nn.Linear(13, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError("Use a new output folder.")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive.")

    torch.set_num_threads(2)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    random.seed(args.training_seed)

    # Original loader verifies observations and builds 20-slot histories.
    # This also preserves the original selected-report eligibility.
    groups, sources = original.load(args.input)

    # Same normalization procedure as the assisted model.
    # original.stats fits statistics using TRAIN seeds only.
    norm = original.stats(groups, "assisted")

    datasets = {}
    for seed in TRAIN + VAL:
        features, labels, report_ids = [], [], []

        for graph in groups[seed]:
            normalized, _, _ = original.inputs(
                graph, "assisted", norm
            )

            for node_index, report in graph["selected"]:
                # Only the current slot: no GRU, GAT or history input.
                features.append(normalized[node_index, -1].copy())
                labels.append(int(report["target_is_malicious"]))
                report_ids.append(str(report["report_sequence"]))

        if len(report_ids) != len(set(report_ids)):
            raise ValueError(f"Duplicate selected reports in seed {seed}")

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != 13:
            raise ValueError("Expected 13 current-time features.")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features.")

        datasets[seed] = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            report_ids,
        )
        print(
            f"Seed {seed}: {len(y)} selected reports; "
            f"{int(y.sum())} malicious",
            flush=True,
        )

    train_x = torch.cat([datasets[s][0] for s in TRAIN])
    train_y = torch.cat([datasets[s][1] for s in TRAIN])

    positives = int((train_y == 1).sum())
    negatives = int((train_y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training requires both target classes.")

    model = make_model()
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-3
    )

    generator = torch.Generator().manual_seed(args.training_seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_count = 0
        per_seed = {}
        prediction_rows = []

        with torch.inference_mode():
            for seed in VAL:
                x, y, report_ids = datasets[seed]
                score_parts = []

                for start in range(0, len(y), args.batch_size):
                    xb = x[start:start + args.batch_size]
                    yb = y[start:start + args.batch_size]
                    logits = model(xb).squeeze(-1)

                    total_loss += float(
                        loss_function(logits, yb)
                    ) * len(yb)
                    total_count += len(yb)
                    score_parts.extend(torch.sigmoid(logits).tolist())

                per_seed[str(seed)] = calculate_metrics(
                    y.numpy(), score_parts
                )

                for report_id, label, score in zip(
                    report_ids, y.tolist(), score_parts
                ):
                    prediction_rows.append({
                        "seed": seed,
                        "item_id": report_id,
                        "method": "Current-feature MLP",
                        "y_true": int(label),
                        "score": float(score),
                        "threshold": 0.5,
                    })

        return {
            "loss": total_loss / total_count,
            "per_seed": per_seed,
        }, prediction_rows

    # Include initialization so training must genuinely improve
    # validation loss before being labelled a trained comparator.
    initial, _ = evaluate()
    best_loss = initial["loss"]
    best_epoch = 0
    best_state = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }
    bad_epochs = 0
    history = []

    args.output.mkdir(parents=True)
    print("Initial validation:", json.dumps(initial), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = loss_function(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(y)
            total_count += len(y)

        validation, _ = evaluate()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if validation["loss"] < best_loss - 1e-4:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping.", flush=True)
            break

    model.load_state_dict(best_state)
    final_validation, predictions = evaluate()

    torch.save({
        "state": best_state,
        "stats": norm,
        "architecture": "13-64-32-1 ReLU MLP; dropout 0.3",
        "variant": "current_feature_mlp",
        "window": 20,
        "input_slot": "last",
        "best_epoch": best_epoch,
        "train_seeds": TRAIN,
        "val_seeds": VAL,
        "training_seed": args.training_seed,
        "threshold": 0.5,
        "sources": sources,
    }, args.output / "current_mlp.pt")

    summary = {
        "initial_validation": initial,
        "history": history,
        "best_epoch": best_epoch,
        "trained_checkpoint_selected": best_epoch > 0,
        "final_validation": final_validation,
        "scope": (
            "Development comparison on original controls. "
            "Current selected-target features only; "
            "no temporal or neighbour aggregation; not a TRE reproduction."
        ),
    }

    (args.output / "training_history.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with (args.output / "validation_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved checkpoint from epoch {best_epoch}.")
    if best_epoch == 0:
        print(
            "Training did not beat initialization on validation loss. "
            "Do not report this as a trained baseline."
        )
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()