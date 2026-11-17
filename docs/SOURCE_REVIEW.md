# Source review

## Changes

All 45 supplied files are retained byte-for-byte. Added `evaluate_rules.py` and `v3_reference.py` to both observation_comparison_kit and A6_frozen_stress_kit, copying the supplied rsu_replay_kit versions. These satisfy imports inside the legacy stgnn_a6 prediction path; that path was not executed. Added root documentation, ignore rules and Git attributes preserving Python byte hashes.

## Checks

- Parsed all Python files successfully.
- Ran all eight target-attack unit tests successfully.
- Static import scan found the two legacy local helper omissions described above.
- No CARLA collection, neural training, checkpoint inference or end-to-end reproduction was run in this environment. PyTorch was unavailable.
- The author's environment export is retained without substitutions; installation/package availability is unverified.

## Reproduction constraints

final_test.py and final_test_resume.py both remain present. The latter contains explicit unavailable-A4 handling. Determine the authoritative historical invocation from released command logs and protocols, not filenames alone.

Frozen protocol code_sha256 keys include absolute filesystem paths. Relocating the project can cause protocol mismatch despite unchanged source bytes. Existing checkpoints also hash adapter bytes. The Git attributes prevent newline normalization of Python source, but do not solve absolute-path relocation. Do not change frozen checks to bypass mismatches.

The original ZIP contains no checkpoints, datasets, final metrics or frozen protocols. Full research reproducibility is pending their release. Licensing, authorship and artifact access details are also pending. This is a source release candidate, not verification of the paper's empirical claims.
