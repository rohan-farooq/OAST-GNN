import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('csv');a=p.parse_args();base=Path(a.csv)
m=json.loads(Path(str(base)+'.history_manifest.json').read_text())
assert not m['interrupted'],'Interrupted run'
for kind,name in m['files'].items():
 count=0
 with (base.parent/name).open() as f:
  for line in f:json.loads(line);count+=1
 assert count==m['counts'][kind],(kind,count,m['counts'][kind])
 assert count>0,f'Empty {kind} history'
 print(f'{kind}: {count} records; valid JSON')
assert m['beacon_queue_remaining']==0,'Beacon callback queue not fully drained'
print('PASS: history files parsed, counts match, non-interrupted, no queued beacon callbacks at save')
print('This check does not certify wireless delivery completeness or detection performance.')
