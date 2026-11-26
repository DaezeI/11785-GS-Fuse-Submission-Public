"""Plot the shared scorer's seed-average forecasts and paired validation gains."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARM = 'Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add'
scores = json.loads((HERE / 'paired-metrics.json').read_text())['results']['ALL']
methods = [('Persistence', 'Zero return'), ('none+CNN~none~none', 'Frozen price CNN'),
           (ARM, 'CNN + matched 8B text'), (ARM + '[text=zero]', 'CNN + zero text features')]
fig, ax = plt.subplots(figsize=(7.1, 4.0))
for i, (label, display) in enumerate(methods):
    metrics = scores['forecasters'][label]['metrics_seed_average_forecast']
    bars = ax.bar(np.arange(2) + (i - 1.5) * .2,
                  [metrics[str(h)]['mae_bp'] for h in (20, 30)], .2, label=display)
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
ax.set_xticks([0, 1], ['20-minute return', '30-minute return'])
ax.set_ylabel('Weighted MAE (basis points; lower is better)')
ax.set_title('FED-PXTX-ALL-REL5-v2: exploratory validation')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
ax.set_ylim(0, max(p.get_height() for p in ax.patches) * 1.27)
fig.tight_layout()
fig.savefig(HERE / 'validation-mae.png', dpi=180)
plt.close(fig)

display = {'Persistence': 'Zero return', 'none+CNN~none~none': 'Frozen price CNN',
           ARM + '[text=zero]': 'Zero text features'}
for seed in (11, 23, 37):
    display[ARM + f'[text=shuffled-{seed}]'] = f'Past-text reassignment {seed}'
pairs = [p for p in scores['comparisons'] if p['label'] == ARM and p['other'] in display]
fig, ax = plt.subplots(figsize=(8.2, 4.5))
for i, h in enumerate((20, 30)):
    for y, pair in enumerate(pairs):
        value = pair['gain_bp_20_30'][i]
        ci = pair['paired_gain_ci95_bp_by_block_20_30']['10'][i]
        ax.plot(ci, [y + (i - .5) * .2] * 2, color=f'C{i}', linewidth=2)
        ax.plot(value, y + (i - .5) * .2, 'o', color=f'C{i}',
                label=f'{h} minutes' if y == 0 else None)
ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
ax.set_yticks(range(len(pairs)), [f"{display[p['other']]} (n={p['rows']})" for p in pairs])
ax.invert_yaxis()
ax.set_xlabel('MAE gain from matched text (bp); positive favors matched text')
ax.set_title('Paired gains: 95% intervals using 10-date blocks')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='best')
fig.tight_layout()
fig.savefig(HERE / 'paired-gains.png', dpi=180)
plt.close(fig)
