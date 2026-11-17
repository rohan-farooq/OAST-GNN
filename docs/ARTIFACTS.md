# Required experiment artifacts

These artifacts were not present in the source archive. Supply the actual files or versioned download links with SHA-256 checksums. Never regenerate a historical protocol from memory.

- Development controls: seeds 2041–2049, event CSVs and beacon/self/attack/observation sidecars and manifests.
- Ablation checkpoint root: original `protocol.json`, five variants x initializations 42/43/44, each `model.pt`, training histories and validation metrics.
- Final test: original `frozen_protocol.json`, accepted episode manifests, unavailable-episode record, failed A4 attempt logs and statuses, final evaluation CSV/JSON files and completion marker. Preserve coverage and counting units.
- Earlier assisted and current-feature MLP checkpoints plus configurations if reporting the earlier matched-baseline figures.
- Stress and RSU replay inputs, configurations, predictions and metrics if those experiments appear in the paper.
- Figure source scripts and their exact source tables. Distinguish development figures from final held-out results.
- A small complete example episode with all sidecars for a quick end-to-end check.

Keep large data and weights outside ordinary Git history. Publish a versioned archive and provide its URL, checksum, size, scope and expected extraction path. Selected small metrics/protocol files can be copied into `results/` and `configs/` after reviewing private paths. Preserve unmodified originals in the research archive; document any public redaction separately.
