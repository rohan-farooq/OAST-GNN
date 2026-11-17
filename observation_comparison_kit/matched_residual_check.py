"""Compare saved neural predictions with the frozen residual-rule baseline."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kit = Path(args.kit).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output)

    sys.path.insert(0, str(kit))
    import observation_adapter as adapter
    import full_history as history
    from evaluate_collected import metrics

    with open(args.predictions, newline="", encoding="utf-8-sig") as f:
        predictions = list(csv.DictReader(f))

    lookup = {
        (r["run_id"], r["item_id"]): r
        for r in predictions
    }
    if len(lookup) != len(predictions):
        raise ValueError("Duplicate prediction IDs.")

    threshold_file = (
        kit / "observation_comparison_smoke_v1" / "residual_baseline.json"
    )
    threshold = float(
        json.loads(threshold_file.read_text())["threshold_m"]
    )

    results = []
    used = set()

    for path, rows in history.discover(data):
        run_id = path.parent.relative_to(data).as_posix()
        observations = adapter.load_observations(path, rows)
        graphs = adapter.examples(rows, 20, observations)

        labels, neural_decisions, rule_decisions = [], [], []

        for graph in graphs:
            for node_index, row in graph["selected"]:
                key = (run_id, str(row["report_sequence"]))

                if key in used:
                    raise ValueError(f"Repeated report: {key}")
                if key not in lookup:
                    raise ValueError(f"Missing neural prediction: {key}")

                prediction = lookup[key]
                used.add(key)

                label = int(row["target_is_malicious"])
                if int(prediction["y_true"]) != label:
                    raise ValueError("Target-label mismatch.")

                if (
                    int(prediction["reporter_id"]) != row["reporter_id"]
                    or int(prediction["target_id"]) != row["target_id"]
                ):
                    raise ValueError("Reporter/target identity mismatch.")

                # Preserve the original adapter's float32 feature values.
                current = graph["obs_x"][node_index, -1]

                if current[5] != 0:
                    raise ValueError(
                        "Missing current observation. "
                        "An explicit abstention policy is needed."
                    )

                residual = float(current[2])

                labels.append(label)
                neural_decisions.append(
                    int(
                        float(prediction["score"])
                        >= float(prediction["threshold"])
                    )
                )
                rule_decisions.append(int(residual > threshold))

        if not labels:
            raise ValueError(f"No eligible reports: {run_id}")

        result = {
            "run_id": run_id,
            "matched_reports": len(labels),
            "neural": metrics(labels, neural_decisions),
            "residual_rule": metrics(labels, rule_decisions),
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if used != set(lookup):
        raise ValueError("Some prediction rows were not matched.")

    report = {
        "threshold_m": threshold,
        "rule": "Current observation residual strictly greater than threshold",
        "threshold_source": str(threshold_file),
        "predictions_sha256": hashlib.sha256(
            Path(args.predictions).read_bytes()
        ).hexdigest(),
        "scope": (
            "Matched local development evaluation; "
            "no retraining or threshold tuning; not a TRE comparison."
        ),
        "runs": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()