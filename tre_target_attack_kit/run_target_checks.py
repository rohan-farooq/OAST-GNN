"""Five short attack-verification runs. Stops on process/audit failure."""
import argparse,json,subprocess,sys
from pathlib import Path
from target_attacks import ATTACKS

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/target_attack_checks'))
    p.add_argument('--seed',type=int,default=2047);p.add_argument('--duration',type=float,default=30)
    p.add_argument('--tm-start',type=int,default=10400);p.add_argument('--max-runs',type=int,default=5)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.seed not in [2047,2048,2049]:raise ValueError('Development checks only')
    if a.duration<=0 or not 1<=a.max_runs<=5:raise ValueError('Invalid duration/run count')
    here=Path(__file__).resolve().parent
    for i,attack in enumerate(ATTACKS[:a.max_runs]):
        folder=a.output/f'seed{a.seed}_{attack}'
        j=1
        while (folder/f'attempt_{j:03d}').exists():j+=1
        attempt=folder/f'attempt_{j:03d}';out=attempt/'events.csv'
        cmd=[sys.executable,'-u',str(here/'carla_a6_v6_target_attacks.py'),
             '--map','Town10HD_Opt','--tm-port',str(a.tm_start+i),'--seed',str(a.seed),
             '--num-vehicles','8','--duration',str(a.duration),'--warmup','5',
             '--spawn-radius','150','--filter-distance','1000','--beacon-rate','1',
             '--target-malicious-fraction','0.25','--target-attack',attack,
             '--collusion-fraction','0','--reporter-attack','none','--output',str(out)]
        print(subprocess.list2cmdline(cmd),flush=True)
        if a.dry_run:continue
        attempt.mkdir(parents=True)
        (attempt/'command.json').write_text(json.dumps(cmd,indent=2))
        with (attempt/'collector.log').open('w') as log:
            try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            except subprocess.TimeoutExpired:
                print('TIMEOUT. Inspect',attempt/'collector.log');return 1
        if r.returncode:
            print('Collector failed:',r.returncode,attempt/'collector.log');return 1
        with (attempt/'attack_audit.log').open('w') as log:
            audit=subprocess.run([sys.executable,str(here/'check_target_attacks.py'),'--input',str(out)],stdout=log,stderr=subprocess.STDOUT)
        print((attempt/'attack_audit.log').read_text(),flush=True)
        if audit.returncode:return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
