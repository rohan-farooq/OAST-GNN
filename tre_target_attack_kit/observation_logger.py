"""Causal synthetic observation records; no evaluation labels or true coordinates."""
import json, math, hashlib
from pathlib import Path

class ObservationLogger:
    def __init__(self, output, seed):
        self.path = Path(str(output) + '.observations.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental overwrite of an earlier observation history.
        with self.path.open('x', encoding='utf-8'): pass
        self.seed = seed
        self.count = 0

    def record(self, report_id, reporter_id, target_id, sequence, available, reference,
               claimed_x, claimed_y, observed_x, observed_y, sigma):
        values = [available, reference, claimed_x, claimed_y, observed_x, observed_y, sigma]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Nonfinite observation')
        if reference > available + 1e-8 or sigma < 0:
            raise ValueError('Invalid observation time or noise scale')
        dx, dy = claimed_x-observed_x, claimed_y-observed_y
        row = dict(schema='a6-local-observation-v1', seed=self.seed,
                   report_id=report_id, reporter_id=reporter_id, target_id=target_id,
                   beacon_sequence=sequence, available_time_s=available,
                   reference_time_s=reference, age_s=max(0.0,available-reference),
                   available=True, claimed_x_m=claimed_x, claimed_y_m=claimed_y,
                   observed_x_m=observed_x, observed_y_m=observed_y,
                   residual_x_m=dx, residual_y_m=dy, residual_norm_m=math.hypot(dx,dy),
                   noise_std_m=sigma,
                   source='synthetic noisy position at beacon state; sampled at report processing')
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        self.count += 1

    def finish(self, output, interrupted):
        manifest = dict(schema='a6-local-observation-manifest-v1',
                        file=self.path.name, count=self.count, interrupted=bool(interrupted),
                        sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                        note='One sampled observation per generated report, before RSU delivery. '
                             'Ideal association; synthetic observation is not a camera/radar measurement. '
                             'Noise scale uses simulator-derived distance in the existing observation model.')
        Path(str(output)+'.observations.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
