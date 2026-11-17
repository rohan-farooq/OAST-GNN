# OAST-GNN

Research code for observation-assisted spatio-temporal graph misbehavior detection in CARLA 0.9.16. Local inference combines temporal encoding, graph attention and an MLP prediction head. The controlled variants remove the GRU, GAT, edge attributes or observation channels.

## Release status

This is a source-code release candidate. Trained checkpoints, frozen experiment protocols, raw episodes and measured result files are **not included**. Do not interpret successful source tests as reproduced detection results. The repository does not claim direct superiority over published methods evaluated under different datasets or protocols.

Observation inputs are synthetic simulator-derived measurements with ideal association; they are not camera/radar measurements. The attack implementation is inspired by published definitions, not an exact VeReMi protocol replication. Historical folder names are retained to preserve paths.

## Contents

| Folder | Purpose |
|---|---|
| `observation_comparison_kit` | Local detectors, MLP baseline, training, ablations and frozen evaluation |
| `tre_target_attack_kit` | A1–A5 definitions, collectors, moving-start A4 and equation tests |
| `A6_frozen_stress_kit` | Earlier frozen-checkpoint observation sensitivity experiments |
| `rsu_replay_kit` | Separate reporter-score replay and aggregation experiments |

Root scripts collect observation controls and comparison episodes. Read `docs/SOURCE_REVIEW.md` and `docs/ARTIFACTS.md` before reproducing results. Existing subfolder READMEs describe earlier stages and may name older output folders; they are retained as historical instructions.

## Environment

The supplied `environment.yml` is the author's Windows Conda export, not a portable lockfile. Package availability and a fresh installation have not been validated for this release. CARLA's server application and maps are installed separately.

From an Anaconda Prompt:

```powershell
conda env create -f environment.yml
conda activate carla_env
```

If the existing environment is already installed, activate it instead. Do not replace a working experimental environment merely to publish the repository. Direct source dependencies include CARLA, NumPy, PyTorch, Matplotlib and threadpoolctl. Record hardware, accelerator and driver details alongside released results.

## Source checks (no CARLA server)

From the repository root:

```powershell
python tre_target_attack_kit/test_target_attacks.py
Push-Location observation_comparison_kit
python run_ablations.py --self-test
Pop-Location
```

## Collect development controls

Start the separately installed CARLA server on port 2000. From the repository root:

```powershell
python collect_observation_controls.py --dry-run
python collect_observation_controls.py
```

The default output is `outputs/a6_observations_60s`. Inspect manifests and audits before training. These are development controls, not final held-out evaluation.

## Controlled ablation training

```powershell
Push-Location observation_comparison_kit
python -u run_ablations.py --input ../outputs/a6_observations_60s --output ablation_development_v1
Pop-Location
```

The source fixes training simulation seeds 2041–2046 and validation simulation seeds 2047–2049. Default model initializations are 42, 43 and 44. Defaults are 60 maximum epochs, patience 8, graph batch size 64, AdamW learning rate 1e-4 and weight decay 1e-3. Selection uses validation loss and can retain initialization; inspect checkpoint metadata rather than assuming every selected checkpoint trained successfully. Use a fresh output directory.

## Final local-detector evaluation

After obtaining the full ablation checkpoint tree and `protocol.json`:

```powershell
Push-Location observation_comparison_kit
python final_test_resume.py --phase plan --checkpoints ablation_development_v1
# Start CARLA before collection. Use a fresh output path for a new reproduction.
python -u final_test_resume.py --phase all --checkpoints ablation_development_v1 --output final_test_2050_2054_reproduction
Pop-Location
```

Even `--phase plan` requires checkpoints and their protocol. The schedule is five simulation seeds (2050–2054), A1–A5 and benign conditions, with a fixed threshold of 0.5. These seeds are held-out only relative to the original development procedure; rerunning them does not create new independent test data.

`final_test_resume.py` supports evidence-based recording of an unavailable A4 episode after two moving-start failures. It does not replace the seed or treat the episode as successful. The author's earlier run reported this failure for seed 2053; the logs and unavailable-episode manifest must accompany the final results before that coverage is independently verified here.

**Do not move an existing frozen run and expect it to resume unchanged.** Frozen protocols contain absolute paths and code hashes. Reproduce into a fresh directory; preserve historical artifacts and protocols verbatim. Do not disable mismatch checks or rewrite evidence to force a pass.

## Other experiment stages

`evaluate_target_attacks.py` evaluates the earlier assisted checkpoint and current-feature MLP on matched attack reports. It is distinct from the five-variant final evaluation. `A6_frozen_stress_kit/stress_evaluate.py` specifically expects the earlier assisted epoch-2, 20-slot checkpoint. Final ablation weights are not interchangeable with that checkpoint. RSU replay metrics are a separate evaluation layer and must not be presented as local-detector metrics.

## Results and artifact access

See `docs/ARTIFACTS.md` for the release manifest. Download links and measured result tables are pending. No missing experimental numbers have been filled in with illustrative values.

## Attribution and license

See `THIRD_PARTY_NOTICES.md`. A repository-wide license and author/citation metadata have not yet been supplied. Retain all existing notices and resolve distribution permissions before a public release. This package does not assign an open-source license on the authors' behalf.
