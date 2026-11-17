import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector", required=True)
    parser.add_argument("--output", default="outputs/tre_comparison")
    parser.add_argument("--seeds", default="2047,2048,2049")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--tm-start", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    collector = Path(args.collector).resolve()
    if not collector.is_file():
        raise FileNotFoundError(collector)

    output = Path(args.output).resolve()
    seeds = [int(x) for x in args.seeds.split(",")]

    # Target attack is held fixed in the reporter-corruption experiment.
    # These are position-offset experiments, NOT all TRE attack classes.
    conditions = [
        ("target_offset_honest_reporters", 0.25, "position_offset", 0.0, "none"),
        ("all_benign", 0.0, "none", 0.0, "none"),
    ]

    for fraction in [0.125, 0.25, 0.375, 0.5, 0.625]:
        for reporter_attack in ["coordinated", "on_off"]:
            conditions.append((
                f"offset_{reporter_attack}_{fraction}",
                0.25,
                "position_offset",
                fraction,
                reporter_attack,
            ))

    # Benign targets with malicious reporters making false accusations.
    for reporter_attack in ["coordinated", "on_off"]:
        conditions.append((
            f"benign_targets_{reporter_attack}",
            0.0,
            "none",
            0.625,
            reporter_attack,
        ))

    jobs = [
        (seed, condition)
        for seed in seeds
        for condition in conditions
    ]
    if args.max_runs is not None:
        jobs = jobs[:args.max_runs]

    print(f"Selected {len(jobs)} runs")

    for index, (seed, condition) in enumerate(jobs):
        name, target_fraction, target_attack, rho, reporter_attack = condition
        run_id = f"seed{seed}_{name}"
        run_dir = output / run_id

        command_base = [
            sys.executable, "-u", str(collector),
            "--map", "Town10HD_Opt",
            "--tm-port", str(args.tm_start + index),
            "--num-vehicles", "8",
            "--duration", str(args.duration),
            "--warmup", "5",
            "--seed", str(seed),
            "--spawn-radius", "150",
            "--filter-distance", "1000",
            "--beacon-rate", "1",
            "--target-malicious-fraction", str(target_fraction),
            "--target-attack", target_attack,
            "--target-position-offset", "20",
            "--collusion-fraction", str(rho),
            "--reporter-attack", reporter_attack,
        ]

        print(f"[{index + 1}/{len(jobs)}] {run_id}")

        if args.dry_run:
            print(subprocess.list2cmdline(
                command_base + ["--output", str(run_dir / "events.csv")]
            ))
            continue

        # Preserve every attempt instead of overwriting previous results.
        attempt_number = 1
        while (run_dir / f"attempt_{attempt_number:03d}").exists():
            attempt_number += 1

        attempt_dir = run_dir / f"attempt_{attempt_number:03d}"
        attempt_dir.mkdir(parents=True)

        command = command_base + [
            "--output", str(attempt_dir / "events.csv")
        ]

        config = {
            "run_id": run_id,
            "seed": seed,
            "condition": name,
            "requested_collusion": rho,
            "num_vehicles": 8,
            "target_fraction": target_fraction,
            "target_attack": target_attack,
            "reporter_attack": reporter_attack,
            "command": command,
        }

        (attempt_dir / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        started = time.monotonic()
        timed_out = False

        with (attempt_dir / "collector.log").open(
            "w", encoding="utf-8"
        ) as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                return_code = process.wait()

        status = {
            "return_code": return_code,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            # Exit zero is necessary, but does not validate data completeness.
            "process_completed_cleanly": return_code == 0 and not timed_out,
            "data_validation": "pending",
        }
        (attempt_dir / "process_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )

        print(status)

        # Do not keep launching experiments into an unhealthy simulator.
        if not status["process_completed_cleanly"]:
            print(f"Stopped. Inspect: {attempt_dir / 'collector.log'}")
            break


if __name__ == "__main__":
    main()