# CARLA target attack verification kit

Extract all files into one new folder. Keep your existing v5 collector unchanged. This contains a derived v6 collector, the unchanged uploaded observation_logger.py, a simulator-independent attack engine, an auditor and a short-run driver. No model training occurs.

Run `python test_target_attacks.py` without CARLA. Then start CARLA and run `python run_target_checks.py --dry-run` followed by `python run_target_checks.py --max-runs 1`. Review the first run before running all five. Use a fresh --output root when starting the complete five-run series, or note that attempts are preserved. Default seed 2047 is development only; do not use final test seeds here.

## Implemented interventions

- a1_constant_position: freeze XY at the target's first attempted post-warmup beacon; retain its true speed and heading fields. This is a position-only intervention.
- a2_constant_offset: draw each malicious target's DX and DY independently from U[-50,50] once per episode.
- a3_random_offset: draw DX and DY independently from U[-50,50] for each target beacon.
- a4_eventual_stop: freeze initial heading, reduce initial reported speed linearly over --stop-seconds (default 5), and analytically integrate reported XY. Claimed position remains fixed after stopping. The real vehicle continues under Traffic Manager. A stationary initial target can make this test uninformative.
- a5_collusive_offset: at the post-warmup snapshot, select a random anchor and a random subset within --coalition-radius (default 500 m); all share one Cartesian offset drawn from U[-50,50]^2. Fail rather than silently reduce a requested coalition. With eight vehicles and fraction .25, two are selected. Coalition fraction controls target corruption; --collusion-fraction separately controls reporter corruption, set to zero for these checks.

Legacy none/position_offset/random_offset modes remain available. --target-position-offset affects the legacy modes; new A2/A3/A5 use --tre-offset-bound (default 50). Actor-ID-independent random keys use stable sorted fleet indices. Requested and actual fractions and cohort snapshot are logged. A1/A4 onset is per-target first attempted beacon, not one global physical event.

These definitions are based on the supplied TRE manuscript's A1-A5 concepts; this is a CARLA implementation, not exact VeReMi/protocol reproduction. In particular, the A4 deceleration duration/profile is explicitly chosen here. These 30-second runs verify attacks, not publishable detector performance.

## Audit and information boundaries

New evaluation-only files: events.csv.target_attacks.evaluation.jsonl and events.csv.target_attack_manifest.json. They contain simulator truth, initial attack states, full-precision claims and actual rounded wire claims. Never use these files as operational model features. The original beacon payload structure and observation logger are preserved; every observer receives the same sender claim rather than independently sampled attacks.

The auditor checks manifest integrity, fleet survival, packet rounding, honest claims, attack equations, coalition geometry and displacement. Inspect true_displacement_m and max_claim_error_m: near-zero movement can make A1/A4 ineffective even if equations pass. A successful audit does not assert successful callback delivery or detection. Labels retain assigned malicious identities throughout the episode, including initial frames with zero displacement; this policy is explicitly logged.

Do not run the earlier evaluate_collected.py unchanged for paper attack-class tables: its condition naming assumes the earlier position-offset checks. Update the evaluation mapping and define label/aggregation policies before full benchmarking. Your frozen model was trained on position offsets; performance on other attack types is initially an unseen-attack transfer test, not equivalent to TRE's trained multiclass protocol.

## Validation performed here

Eight attack-dynamics unit tests pass; source syntax checks pass. The new collector was derived from your uploaded v5 source. CARLA integration has not run in this workspace; verify the first local run before proceeding. Existing sensor logging and cleanup logic are retained. The attack trace is new and records only successful send calls; failed/interrupted collections must not be used for model results.
