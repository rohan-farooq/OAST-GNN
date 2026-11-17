"""Collect development controls only. Preserves existing complete runs."""
import argparse,json,subprocess,sys,zipfile
from pathlib import Path

def validate(p,seed):
    from check_observations import check
    check(p)
    m=json.loads(Path(str(p)+'.metadata.json').read_text())
    if m['seed']!=seed or m['duration_seconds']!=60 or m['reporter_attack']!='none' or m['target_attack']!='position_offset':
        raise ValueError('Run configuration mismatch')
    h=json.loads(Path(str(p)+'.history_manifest.json').read_text())
    if h['beacon_queue_remaining']:raise ValueError('Beacon queue not drained')
    # Ensure files needed by the graph adapter exist before declaring success.
    for suffix in ('.beacons.jsonl','.self.jsonl','.attacks.evaluation.jsonl','.evaluation.json','.operational.json'):
        if not Path(str(p)+suffix).is_file():raise ValueError('Missing '+suffix)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--collector',default='carla_a6_v5_observations.py')
    ap.add_argument('--output',default='outputs/a6_observations_60s')
    ap.add_argument('--tm-base',type=int,default=9820)
    ap.add_argument('--timeout',type=float,default=600)
    a=ap.parse_args();root=Path(a.output).resolve();collector=Path(a.collector).resolve()
    if not collector.is_file():raise FileNotFoundError(collector)
    completed=[]
    for seed in range(2041,2050):
        p=root/f'seed{seed}_control.csv'
        if p.exists():
            validate(p,seed);print(f'SKIP seed {seed}: valid existing files',flush=True);completed.append(p);continue
        if list(root.glob(p.name+'.*')):raise ValueError(f'Incomplete existing files for {p.name}; preserve them elsewhere before retrying')
        cmd=[sys.executable,'-u',str(collector),'--map','Town10HD_Opt','--tm-port',str(a.tm_base+seed-2041),
             '--num-vehicles','8','--duration','60','--warmup','5','--seed',str(seed),
             '--spawn-radius','150','--filter-distance','1000','--beacon-rate','1',
             '--target-malicious-fraction','0.25','--target-attack','position_offset','--target-position-offset','20',
             '--collusion-fraction','0','--reporter-attack','none','--output',str(p)]
        print(subprocess.list2cmdline(cmd),flush=True)
        if a.dry_run:continue
        root.mkdir(parents=True,exist_ok=True);log=root/f'seed{seed}_collector.log'
        print(f'RUN {seed}; log: {log}',flush=True)
        with log.open('w',encoding='utf-8') as handle:
            try:r=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,timeout=a.timeout)
            except subprocess.TimeoutExpired:
                print('STOP: collector timed out. Inspect the log and CARLA before retrying.',flush=True);return 1
        if r.returncode!=0:
            print(f'STOP: seed {seed} exit code {r.returncode}; see {log}',flush=True);return 1
        validate(p,seed);completed.append(p);print(f'PASS seed {seed}',flush=True)
    if a.dry_run:print('Dry run finished; no simulator calls or files written.');return 0
    archive=root/'observation_controls_2041_2049.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in completed:
            z.write(p,p.name)
            for q in sorted(root.glob(p.name+'.*')):
                if q.is_file():z.write(q,q.name)
    print(f'COMPLETE: {len(completed)} development controls. Upload {archive}',flush=True)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print(f'STOP: {exc}',file=sys.stderr);raise SystemExit(1)
