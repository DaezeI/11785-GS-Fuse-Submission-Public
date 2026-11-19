"""Regenerate the aggregate paired gain figure from the saved metrics."""
import json
from pathlib import Path

out = Path(__file__).resolve().parent
d = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.3,3.6),layout='constrained')
reds=['#127a74','#ac4437','#127a74','#2e485e']
ys=[3,2,1,0]; vals=[row['improvement_bp'] for row in d['comparisons']]+[d['means']['improvement_bp']]
ax.barh(ys,vals,color=reds,height=.52,zorder=3)
for y,x in zip(ys,vals):
    offset = .014 if x>=0 else -.012
    ax.text(x+offset,y,f'{x:+.3f}',ha='left' if x>=0 else 'right',va='center',fontsize=10)
ax.axvline(0,color='#344154',lw=1.3,zorder=2)
ax.set_yticks(ys,['Seed 0','Seed 1','Seed 2','Mean'])
ax.set_xlim(-.115,.64)
ax.set_xlabel('L35 MAE minus L140 MAE (bp); positive favors L140')
ax.set_title('Longer price history: small, uneven validation gain',loc='left',fontweight='bold',pad=12)
ax.grid(axis='x',color='#e4e9ed',zorder=0)
ax.spines[['right','top','left']].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.savefig(out/'context-gain.png',dpi=165,facecolor='white')
