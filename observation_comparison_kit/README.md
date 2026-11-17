# Matched observation comparison

Extract into C:\CARLA_0.9.16\PythonAPI\examples\observation_comparison_kit.
Keep the nine new control bundles in ..\outputs\a6_observations_60s. Use carla_env.

Audit and residual-only evaluation (no PyTorch needed):

    python -u compare_observations.py --input ..\outputs\a6_observations_60s --output residual_check_v1

Two-epoch neural smoke comparison (requires the existing PyTorch installation):

    python -u compare_observations.py --input ..\outputs\a6_observations_60s --train --epochs 2 --output observation_comparison_smoke_v1

This executes both variants sequentially and saves baseline_model.pt, assisted_model.pt, training_history.json, residual_baseline.json and sources.json. Upload training_history.json and both model checkpoints after completion. Choose a fresh output directory each time. CARLA is not needed for training. Test seeds 2050–2052 are deliberately excluded. The audit accepts exactly nine development control runs and rejects repeated seeds.

## Methods

1. Baseline: original seven node channels and existing GRU/GAT architecture.
2. Assisted: same architecture, with GRU input expanded from 7 to 13 channels using obs_x. Extra channels are signed position residual x/y, residual magnitude, noise scale, age at slot cutoff and missing mask.
3. Rule: compare the current available residual magnitude to a threshold selected only from benign training reports (95th percentile, higher empirical quantile; alarm strictly greater). This design targets approximately 5% training false positives; it does not guarantee the same rate elsewhere. Missing current observations cause an explicit abstention.

Train seeds: 2041–2046. Validation seeds: 2047–2049. Window: 20 samples at 1 Hz (20 seconds). Both neural models supervise only the selected report target, not every graph node; this deliberately corrects alignment between the training/evaluation unit and operational reports. Therefore these training losses are not directly comparable to the earlier all-node runs. This remains a development baseline, not an exact TRE reproduction.

Both variants use AdamW, learning rate 1e-4, weight decay 1e-3, accumulation of 64 graphs, gradient clipping 5, seed 42, dropout inherited from the original architecture and no rotation augmentation. Train-only normalization ignores absent observations, keeps missing masks binary, and zeros missing normalized physical channels. No ground-truth position, label, attack flag, report ID or precomputed risk score enters neural tensors. Edges remain the original claimed-geometry edges.

Checkpoint selection uses validation weighted BCE; best_epoch=0 explicitly means initialization was retained. Three non-improving epochs stop training by default. F1/recall/false alarms use threshold 0.5, not a threshold optimized on validation. Validation thresholds and model settings must be frozen before any held-out evaluation. All report samples share vehicles and overlapping windows; they are not independent experimental replicates.

The residual baseline abstains when the current observation is missing, while the neural model may still predict from history. Check abstention counts before calling the populations identical. No A6/RSU aggregation is evaluated here; this is local detection on generated reports before RSU delivery.

## Limits

Observations are synthetic noisy positions with ideal target association, based on simulator beacon-state coordinates and distance-dependent noise. They are not camera/radar perception. Improved performance can reflect the additional information alone. The residual-only baseline is essential: if it already performs well, neural benefit is not established. Do not compare these numbers directly to TRE's published experiments.

## Verification

The bundled adapter was executed on all nine new control runs and the residual rule evaluated. Python compilation and finite/masked 7/13-channel normalization checks passed. PyTorch is unavailable in the analysis environment, so the new neural training loop has NOT been executed here. Run the two-epoch smoke command before longer training. It does not load the old seven-feature checkpoints into the 13-feature model.
