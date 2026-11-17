"""Plot measured per-run replay metrics, with decision coverage visible."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
rows=list(csv.DictReader((a.input/'metrics.csv').open()))
groups=defaultdict(list)
for r in rows:
 if r['comparison_scope']=='all_scheduled':groups[(r['run'],r['reporter_attack'])].append(r)
plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42})
for index,((run,attack),rs) in enumerate(groups.items()):
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for method in ['Mean','RSU_only','Static_capped','Temporal_capped']:
  mm=sorted([r for r in rs if r['method']==method],key=lambda r:float(r['rho']))
  for ax,key,label in zip(axes,['f1_on_decided','fpr_on_decided','decision_coverage'],['F1 on decided cases','FPR on decided cases','Decision coverage']):
   ax.plot([100*float(r['rho']) for r in mm],[float(r[key]) if r[key] else float('nan') for r in mm],marker='o',label=method)
   ax.set(xlabel='Malicious reporters (%)',ylabel=label,ylim=(-.02,1.02));ax.grid(alpha=.3)
 axes[0].legend(fontsize=7)
 fig.suptitle(f'{run} | {attack}\nMeasured offline replay; not a TRE reproduction',fontsize=9)
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(a.input/f'comparison_{index:02d}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
print('Figures saved beside metrics.csv. Empty F1 curves in all-benign runs are intentional.')
