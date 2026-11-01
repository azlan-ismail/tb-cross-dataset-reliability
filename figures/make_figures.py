"""Reproduce publication figures from the included, unrounded saved estimates.

Usage: python make_figures.py
No training, resampling, or confidence-interval calculations are performed.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

BASE = Path(__file__).resolve().parent
OUT = BASE / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 11.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 11,
    'legend.fontsize': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'path', 'axes.linewidth': 0.7,
    'axes.edgecolor': '#777777', 'text.color': '#222222',
    'axes.labelcolor': '#222222', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})

COHORT = {'montgomery': ('#0072B2', 'o', 'Montgomery'), 'tbx11k': ('#D55E00', 's', 'TBX11K')}
ENCODER = {'rad': ('#009E73', 'o', 'RAD-DINO'), 'dino': ('#CC79A7', 'D', 'DINOv2')}
SUBGROUP = {'health': ('#009E73', 'o', 'Healthy (n = 800)'), 'sick': ('#CC79A7', 'D', 'Other illness (n = 794)')}
MODELS = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
MODEL_LABELS = ['Linear', 'Neural: seed 42', 'Neural: seed 43', 'Neural: seed 44']
audit = []

def read(number):
    with (BASE / 'data' / f'figure{number}_data.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ['estimate', 'ci_low', 'ci_high']:
            row[key] = float(row[key]) if row[key] else None
    return rows

def select(rows, **filters):
    found = [row for row in rows if all(row[key] == value for key, value in filters.items())]
    assert len(found) == 1, filters
    return found[0]

def panel(ax, title, labels, limits, ticks):
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.set_ylim(len(labels)-0.45, -0.55)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', length=3, pad=5)
    ax.grid(axis='x', color='#e1e4e8', linewidth=0.65, zorder=0)
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i-0.45, i+0.45, color='#f5f6f7', zorder=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=34)

def point(ax, row, y, color, marker, scale=1):
    value = row['estimate'] * scale
    if row['ci_low'] is not None:
        lo, hi = row['ci_low'] * scale, row['ci_high'] * scale
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt=marker,
                    color=color, markersize=6.4, markeredgewidth=0.8,
                    elinewidth=1.5, capsize=3, capthick=1.1, zorder=3)
    else:
        ax.plot(value, y, marker=marker, color=color, markersize=6.4, linestyle='none', zorder=3)
    audit.append({'figure': row['figure'], 'panel': row['panel'], 'cohort': row['cohort'],
                  'model': row['model'], 'group': row['group'], 'source_value': row['estimate'],
                  'display_value': value, 'display_scale': scale, 'ci_rendered': row['ci_low'] is not None})

def handles(styles):
    return [Line2D([0], [0], color=color, marker=marker, markersize=6.5, linewidth=1.4, label=label)
            for color, marker, label in styles.values()]

def legend(ax, styles, ncol=2):
    ax.legend(handles=handles(styles), loc='lower left', bbox_to_anchor=(-0.02, 1.005),
              frameon=False, ncol=ncol, columnspacing=1.2, handletextpad=0.5,
              handlelength=1.2, borderaxespad=0)

def export(fig, stem):
    fig.canvas.draw()
    # Check every visible text object fits the PDF canvas before exporting.
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    bad = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            bad.append(item.get_text())
    assert not bad, f'Text outside figure: {bad}'
    for extension in ['pdf', 'svg', 'png']:
        kwargs = {'dpi': 600} if extension == 'png' else {}
        if extension == 'pdf':
            kwargs['metadata'] = {'Title': stem.replace('_', ' '), 'Creator': 'Matplotlib; saved PulmoVision evaluation outputs'}
        fig.savefig(OUT / f'{stem}.{extension}', **kwargs)
    plt.close(fig)

def figure2():
    rows = read(2)
    fig = plt.figure(figsize=(7.6, 6.4))
    axa = fig.add_axes([0.195, 0.59, 0.335, 0.29])
    axb = fig.add_axes([0.645, 0.59, 0.325, 0.29])
    axc = fig.add_axes([0.195, 0.12, 0.775, 0.275])
    panel(axa, 'A  External discrimination', MODEL_LABELS, (0.85, 1.0), [0.85, 0.90, 0.95, 1.0])
    panel(axb, 'B  Transferred sensitivity', ['']*4, (60, 100), [60, 70, 80, 90, 100])
    panel(axc, 'C  TBX11K negative-case composition', MODEL_LABELS, (0, 100), [0, 20, 40, 60, 80, 100])
    axa.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axa.set_xlabel('AUROC')
    axb.set_xlabel('Sensitivity (%)')
    axc.set_xlabel('Specificity (%)')
    for cohort, offset in [('montgomery', -0.17), ('tbx11k', 0.17)]:
        color, marker, _ = COHORT[cohort]
        for i, model in enumerate(MODELS):
            point(axa, select(rows, panel='A', cohort=cohort, model=model), i+offset, color, marker)
            point(axb, select(rows, panel='B', cohort=cohort, model=model), i+offset, color, marker, 100)
    axb.axvline(90, color='#555555', linewidth=1.2, linestyle=(0, (3, 3)), zorder=2.6)
    legend(axa, COHORT)
    axb.text(0.0, 1.035, 'Dashed line: source target (90%)', transform=axb.transAxes, fontsize=9.5)
    for group, offset in [('health', -0.17), ('sick', 0.17)]:
        color, marker, _ = SUBGROUP[group]
        for i, model in enumerate(MODELS):
            point(axc, select(rows, panel='C', group=group, model=model), i+offset, color, marker, 100)
    legend(axc, SUBGROUP)
    fig.text(0.195, 0.016, 'A/B: bootstrap 95% intervals   |   C: Wilson 95% intervals', fontsize=9.5, color='#555555')
    export(fig, 'figure2_external_performance')

def figure3():
    rows = read(3)
    fig = plt.figure(figsize=(7.6, 6.6))
    axa = fig.add_axes([0.225, 0.585, 0.325, 0.285])
    axb = fig.add_axes([0.66, 0.585, 0.31, 0.285])
    axc = fig.add_axes([0.225, 0.12, 0.745, 0.245])
    groups = [('montgomery','224'), ('montgomery','518'), ('tbx11k','224'), ('tbx11k','518')]
    labels = ['Montgomery\n224 pixels', 'Montgomery\n518 pixels', 'TBX11K\n224 pixels', 'TBX11K\n518 pixels']
    panel(axa, 'A  Matched encoders', labels, (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    panel(axb, 'B  Paired AUROC contrast', ['']*4, (-0.025, 0.4), [0, 0.1, 0.2, 0.3, 0.4])
    axa.set_xlabel('AUROC')
    axb.set_xlabel('RAD-DINO minus DINOv2', fontsize=10.3)
    axb.axvline(0, color='#555555', linewidth=1.2, linestyle=(0, (3, 3)), zorder=2.6)
    for i, (cohort, resolution) in enumerate(groups):
        for encoder, offset in [('rad', -0.16), ('dino', 0.16)]:
            color, marker, _ = ENCODER[encoder]
            point(axa, select(rows, panel='A', cohort=cohort, model=f'{encoder}_{resolution}'), i+offset, color, marker)
        point(axb, select(rows, panel='B', cohort=cohort, group=resolution), i, '#374151', 's')
    legend(axa, ENCODER)
    axb.text(0.0, 1.035, 'Positive values favour RAD-DINO', transform=axb.transAxes, fontsize=9.5)
    conditions = ['hf256_crop224', 'hf224_crop224', 'tv256_crop224']
    panel(axc, 'C  DINOv2 preprocessing diagnostic', ['HF 256 → 224', 'HF 224 → 224', 'TV 256 → 224'],
          (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axc.set_xlabel('AUROC')
    for cohort, offset in [('montgomery', -0.17), ('tbx11k', 0.17)]:
        color, marker, _ = COHORT[cohort]
        for i, condition in enumerate(conditions):
            row = select(rows, panel='C', cohort=cohort, group=condition)
            point(axc, row, i+offset, color, marker)
            axc.annotate(f"{row['estimate']:.3f}", (row['estimate'], i+offset), xytext=(7, 0),
                         textcoords='offset points', ha='left', va='center', fontsize=10, color=color)
    legend(axc, COHORT)
    fig.text(0.225, 0.018, 'A/B: bootstrap 95% intervals   |   C: post hoc point estimates; no intervals', fontsize=9.2, color='#555555')
    export(fig, 'figure3_encoder_preprocessing')

figure2()
figure3()
assert len(audit) == 42 and sum(x['ci_rendered'] for x in audit) == 36
(BASE / 'PLOT_CHECK.json').write_text(json.dumps({'all_text_within_canvas': True, 'plotted_points': audit}, indent=2), encoding='utf-8')
print('Exported 2 vector PDFs, 2 vector SVGs and 2 PNGs at 600 dpi; 42 estimates, 36 saved intervals.')
