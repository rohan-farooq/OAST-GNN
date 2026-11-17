"""Fresh CARLA runs for physical offset/noise changes; no observation rewrites."""
import argparse,json,subprocess,sys
from pathlib import Path
from check_observations import check
CASES=[('reference',20,.5),('offset2',2,.5),('offset5',5,.5),('offset10',10,.5),('noise2',20,2),('noise4',20,4)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--collector',default='../carla_a6_v5_observations.py');p.add_argument('--output',default='../outputs/a6_frozen_stress');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-runs',type=int,default=0);p.add_argument('--tm-base',type=int,default=10020);a=p.parse_args()
 collector=Path(a.collector).resolve();root=Path(a.output).resolve()
 if not collector.is_file():raise FileNotFoundError(collector)
 jobs=[(seed,name,offset,noise) for name,offset,noise in CASES for seed in (2047,2048,2049)]
 if a.max_runs:jobs=jobs[:a.max_runs]
 print(f'{len(jobs)} selected development jobs. No test seeds. No fitting.')
 for index,(seed,name,offset,noise) in enumerate(jobs):
  folder=root/f'{name}_seed{seed}';out=folder/'events.csv';status=folder/'completed.json'
  config=dict(seed=seed,offset=offset,base_noise=noise)
  if status.exists():
   if json.loads(status.read_text())!=config:raise ValueError('Configuration mismatch')
   check(out);print('SKIP',folder.name,flush=True);continue
  if folder.exists():raise ValueError(f'Incomplete folder {folder}; preserve it elsewhere before retrying')
  cmd=[sys.executable,'-u',str(collector),'--map','Town10HD_Opt','--tm-port',str(a.tm_base+index),
       '--num-vehicles','8','--duration','60','--warmup','5','--seed',str(seed),
       '--spawn-radius','150','--filter-distance','1000','--beacon-rate','1',
       '--target-malicious-fraction','0.25','--collusion-fraction','0','--reporter-attack','none',
       '--target-attack','position_offset','--target-position-offset',str(offset),
       '--base-measurement-noise',str(noise),'--output',str(out)]
  print(subprocess.list2cmdline(cmd),flush=True)
  if a.dry_run:continue
  folder.mkdir(parents=True)
  with (folder/'collector.log').open('w') as log:
   try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
   except subprocess.TimeoutExpired:raise RuntimeError(f'Timeout: inspect {folder}/collector.log and CARLA')
  if r.returncode:raise RuntimeError(f'Exit {r.returncode}; inspect {folder}/collector.log')
  check(out);status.write_text(json.dumps(config,indent=2));print('PASS',folder.name,flush=True)
 print('Dry run complete.' if a.dry_run else 'Collection complete; run stress_evaluate.py on the output root.')
if __name__=='__main__':main()
