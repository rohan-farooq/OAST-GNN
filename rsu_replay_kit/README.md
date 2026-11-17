# Measured RSU replay

Run replay_neural_rsu.py with --data pointing to one collection root, --predictions to its evaluate_collected.py CSV, and --output to a new directory. Standard-library only. Plot afterwards with plot_replay.py --input OUTPUT (requires matplotlib).

This package retains the earlier evaluate_rules.py and v3_reference.py implementation. Trust estimates decayed agreement with imperfect RSU evidence, not proven honesty. The prior mass is 4, half-life 30 seconds, squared reliability weights, confidence gate 0.60, and capped reporter contribution 20%. Decisions precede trust updates; same-time decisions share prior trust. No labels enter the operational predictor.

The added replay selects a nested uniformly shuffled coalition of 0 through 5 of eight observed reporters, and changes received scores to .99 for benign targets and .01 for malicious targets. On-off replay alternates ten-second active/inactive intervals using receive time relative to recorded collection start. Labels enter attack construction as explicitly assumed attacker knowledge and evaluation after prediction. Delivery and local neural outputs are frozen. This is not an exact TRE A6 attack: spatial coalition/victim selection is absent. It does not generate A1-A5 target attacks.

Methods: Mean; RSU_only; Static_capped (RSU plus unweighted reports); Temporal_capped (same RSU mixing plus temporal reporter weights). Static_capped versus Temporal_capped is the trust ablation. Always report coverage; all_scheduled metrics describe each method's decided subset, common_decided metrics use the intersection. All-benign F1 is undefined. Zero-rho results repeat across modes and are not independent repetitions.

IMPORTANT: With capped reporter influence 0.20, strong RSU anchors can dominate decisions. A tie between static and temporal methods is a valid outcome, not a reason to tune until trust wins. These runs use development seeds only. No claim of GRU/GAT superiority or full-system improvement over TRE is supported by this package alone.

Local validation completed on the supplied seed2047 controls using saved assisted-model predictions. Neural inference was not rerun. Results can differ under other source runs. Keep your original files unchanged.
