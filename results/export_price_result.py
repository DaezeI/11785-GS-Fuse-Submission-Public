"""Rebuild small paper assets from the preserved aggregate metrics; no GPU needed."""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gsfuse-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run = Path(__file__).resolve().parent / 'screened-price-v1'
    source = json.loads((run / 'source_metrics.json').read_text())
    rows = []
    table = [r'\begin{tabular}{rrrrr}', r'\toprule',
             r'Horizon & Real & Shuffled & Zero article & Fusion bypass \\', r'\midrule']
    for horizon, modes in source['horizons'].items():
        zero = modes['zeroed_article_tokens']
        bypass = modes['fusion_bypass_ts_inst']
        values = [zero['mse'][k] for k in ('real_text', 'shuffled_text', 'zero_text')]
        values.append(bypass['mse']['zero_text'])
        table.append(horizon + ' & ' + ' & '.join(f'{v:.8f}' for v in values) + r' \\')
        for control, item, key in [('shuffled_text', zero, 'shuffled_text'),
                                   ('zeroed_article_tokens', zero, 'zero_text'),
                                   ('fusion_bypass_ts_inst', bypass, 'zero_text')]:
            comparison = item['comparisons'][key]
            low, high = comparison['ci95_pct']
            rows.append(dict(horizon=int(horizon), control=control,
                             events=item['events'], distinct_dates=item['distinct_dates'],
                             real_mse=item['mse']['real_text'], control_mse=item['mse'][key],
                             real_mse_gain_pct=comparison['real_text_mse_gain_pct'],
                             ci95_low_pct=low, ci95_high_pct=high,
                             real_endpoint_close_dhr=item['endpoint_close_dhr']['real_text'],
                             always_up_dhr=item['endpoint_close_dhr']['always_up']))
    with (run / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    table += [r'\bottomrule', r'\end{tabular}']
    (run / 'metrics.tex').write_text('\n'.join(table) + '\n')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained')
    titles = [('shuffled_text', 'vs shuffled articles'),
              ('zeroed_article_tokens', 'vs zero article tokens'),
              ('fusion_bypass_ts_inst', 'vs fusion bypass')]
    for axis, (control, title) in zip(axes, titles):
        values = [row for row in rows if row['control'] == control]
        for row in values:
            y = row['real_mse_gain_pct']
            axis.errorbar(row['horizon'], y,
                          yerr=[[y-row['ci95_low_pct']], [row['ci95_high_pct']-y]],
                          fmt='o', color='#165b86', capsize=4, linewidth=1.5)
        axis.axhline(0, color='#555555', linewidth=1, linestyle='--')
        axis.set(title=title, xlabel='Forecast prefix (bars)', xticks=[35, 70, 140])
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=0.15)
    axes[0].set_ylabel('Real-text MSE improvement (%)\nPositive favors correctly matched text')
    fig.suptitle('Screened C1 price model: paired text controls', fontsize=12)
    fig.supxlabel('95% date-block bootstrap intervals; one checkpoint, seed 0; 2,236 events', fontsize=8)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(run / f'text-control-gains.{suffix}', dpi=180)
    plt.close(fig)
    svg = run / 'text-control-gains.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    checksums = [hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name
                 for p in sorted(run.iterdir()) if p.is_file() and p.name != 'SHA256SUMS']
    (run / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    print('Generated metrics.csv, metrics.tex and PDF/SVG/PNG figure from source_metrics.json')


if __name__ == '__main__':
    main()
