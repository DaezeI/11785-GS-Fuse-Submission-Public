#!/usr/bin/env python3
"""Export saved volatility subgroup results and optionally recompute block sensitivity.

Run from repository root. NPZ inputs are private and remain outside the repository.
"""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gsfuse-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/price-volatility-v1'
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_text_predictions import validate, pre_event_volatility, block_bootstrap_improvement


def export_saved():
    source = json.loads((OUT / 'saved_aggregate.json').read_text())
    a, b = source['zeroed_article_tokens'], source['fusion_bypass_ts_inst']
    assert a['volatility_train_cutoffs'] == b['volatility_train_cutoffs']
    rows = []
    for h in ('35', '70', '140'):
        for group in ('low', 'medium', 'high'):
            z, f = [x['groups'][h]['volatility:' + group] for x in (a, b)]
            assert z['events'] == f['events'] and z['distinct_dates'] == f['distinct_dates']
            assert z['mse']['real_text'] == f['mse']['real_text'] and z['comparisons']['shuffled_text'] == f['comparisons']['shuffled_text']
            for name, item, control in [('shuffled', z, 'shuffled_text'), ('zeroed tokens', z, 'zero_text'), ('fusion bypass', f, 'zero_text')]:
                c = item['comparisons'][control]
                rows.append(dict(horizon=int(h), volatility=group, events=z['events'], dates=z['distinct_dates'], control=name, real_mse=z['mse']['real_text'], control_mse=item['mse'][control], gain_pct=c['real_text_mse_gain_pct'], ci_low_pct=c['ci95_pct'][0], ci_high_pct=c['ci95_pct'][1]))
    with (OUT / 'saved_subgroups.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8)); fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.32, wspace=.35)
    colors = ['#226c8a', '#d47d30', '#6e598f']
    for ax, control in zip(axes, ('shuffled', 'zeroed tokens', 'fusion bypass')):
        for i, (group, color) in enumerate(zip(('low','medium','high'),colors)):
            subset = [r for r in rows if r['control'] == control and r['volatility'] == group]
            x = np.arange(3) + (i-1)*0.20
            y = np.array([r['gain_pct'] for r in subset]); lo = np.array([r['ci_low_pct'] for r in subset]); hi = np.array([r['ci_high_pct'] for r in subset])
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=3,color=color,label=group.title())
        ax.axhline(0,color='#777777',lw=.8); ax.set_xticks(range(3),['35','70','140']); ax.set_title('vs ' + control); ax.set_xlabel('Forecast horizon'); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Matched-text MSE gain vs control (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Pre-event volatility', ncol=3, loc='lower center', bbox_to_anchor=(.5,.14), frameon=False, fontsize=9, title_fontsize=9)
    axes[0].set_ylim(-.16,.20); axes[1].set_ylim(-3.15,.35); axes[2].set_ylim(-44,5)
    for ax in axes: ax.text(.98,.03,'Independent y scale',transform=ax.transAxes,ha='right',fontsize=7,color='#555555')
    fig.suptitle('Matched text vs inference controls by prior volatility',fontsize=13)
    fig.text(.5,.035,'Points: saved estimates; bars: 95% date-block bootstrap intervals (20 dates, 1,000 replicates).\nPositive means lower matched-text MSE; the three panels use different controls.',ha='center',fontsize=9)
    for ext in ('png','pdf'): fig.savefig(OUT / ('volatility-comparisons.'+ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return source


def sensitivity(paths, source):
    rows = []
    hashes = {'zeroed_article_tokens': 'fcc5f06018ef63c5137be2b74d7b3777c8a7ce3bc0f92775f056c1be5921a84f',
              'fusion_bypass_ts_inst': '000274d307ee4e6f952102ffc651c0945356a6413aea8e924752d9b3fa03945c'}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Prediction checksum mismatch: {name}')
        with np.load(path, allow_pickle=False) as data:
            meta = validate(data)
            assert meta['checkpoint_id'] == '5358bbf48b4061ab325eb2a13390d7acdcb75ad7601f6969d3bb73152a2a2a52'
            cuts = np.quantile(data['train_volatility'],[1/3,2/3])
            np.testing.assert_allclose(cuts,source[name]['volatility_train_cutoffs'],rtol=0,atol=1e-12)
            labels = np.searchsorted(cuts,pre_event_volatility(data['raw_seqs']),side='right')
            for h in (35,70,140):
                # Float64 is deliberate; saved source used input dtype (float32).
                truth = data['targets'][:,:h].astype(np.float64)
                real = np.mean((data['real_text'][:,:h].astype(np.float64)-truth)**2,axis=(1,2))
                zero = np.mean((data['zero_text'][:,:h].astype(np.float64)-truth)**2,axis=(1,2))
                shuffle = np.mean((data['shuffled_text'][:,:,:h].astype(np.float64)-truth[None])**2,axis=(0,2,3))
                for i, group in enumerate(('low','medium','high')):
                    mask=labels==i; old=source[name]['groups'][str(h)]['volatility:'+group]
                    assert int(mask.sum())==old['events'] and len(np.unique(data['event_dates'][mask]))==old['distinct_dates']
                    for control, loss in [('zeroed tokens' if name.startswith('zeroed') else 'fusion bypass',zero),('shuffled',shuffle)]:
                        if control=='shuffled' and not name.startswith('zeroed'): continue
                        oldkey='shuffled_text' if control=='shuffled' else 'zero_text'
                        np.testing.assert_allclose([real[mask].mean(),loss[mask].mean()],[old['mse']['real_text'],old['mse'][oldkey]],rtol=2e-6,atol=1e-10)
                        gain=100*(1-real[mask].mean()/loss[mask].mean())
                        for block in (5,10,20,40):
                            ci=block_bootstrap_improvement(real,loss,data['event_dates'],mask,reps=1000,block=block,seed=23)['ci95_pct']
                            rows.append(dict(horizon=h,volatility=group,control=control,block_dates=block,gain_pct=gain,ci_low_pct=ci[0],ci_high_pct=ci[1]))
    with (OUT/'block_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--zero-npz',type=Path); p.add_argument('--bypass-npz',type=Path); args=p.parse_args()
    if bool(args.zero_npz)!=bool(args.bypass_npz): p.error('supply both NPZ files')
    source=export_saved()
    if args.zero_npz: sensitivity({'zeroed_article_tokens':args.zero_npz,'fusion_bypass_ts_inst':args.bypass_npz},source)

if __name__=='__main__': main()
