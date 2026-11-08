"""Figure 2: external performance at the unchanged development thresholds.

Usage: python manuscript/figure2_external_performance.py REPO OUT

Plots the saved estimates in evidence/manuscript/external_performance_estimates.csv:
AUROC and transferred sensitivity with saved percentile-bootstrap intervals
(2,000 resamples for Montgomery, 10,000 for TBX11K) and TBX11K subgroup
specificity with Wilson intervals. No resampling or model fitting is performed.
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 11.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 11,
    'legend.fontsize': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.7,
    'axes.edgecolor': '#777777', 'text.color': '#222222',
    'axes.labelcolor': '#222222', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
COHORT = {'montgomery': ('#0072B2', 'o', 'Montgomery'), 'tbx11k': ('#D55E00', 's', 'TBX11K')}
SUBGROUP = {'health': ('#009E73', 'o', 'Healthy (n = 800)'), 'sick': ('#CC79A7', 'D', 'Other illness (n = 794)')}
MODELS = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
LABELS = ['Linear', 'Neural: seed 42', 'Neural: seed 43', 'Neural: seed 44']

with open(R / 'evidence/manuscript/external_performance_estimates.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    for key in ['estimate', 'ci_low', 'ci_high']:
        row[key] = float(row[key])

def select(**filters):
    found = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    assert len(found) == 1, filters
    return found[0]

def panel(ax, title, labels, limits, ticks):
    ax.set_xlim(*limits); ax.set_xticks(ticks)
    ax.set_ylim(len(labels) - 0.45, -0.55); ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis='y', length=0, pad=8); ax.tick_params(axis='x', length=3, pad=5)
    ax.grid(axis='x', color='#e1e4e8', linewidth=0.65, zorder=0)
    for i in range(0, len(labels), 2):
        ax.axhspan(i - 0.45, i + 0.45, color='#f5f6f7', zorder=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=34)

def point(ax, row, y, color, marker, scale=1):
    v, lo, hi = row['estimate'] * scale, row['ci_low'] * scale, row['ci_high'] * scale
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=marker, color=color, markersize=6.4, markeredgewidth=0.8,
                elinewidth=1.5, capsize=3, capthick=1.1, zorder=3)

def legend(ax, styles):
    handles = [Line2D([0], [0], color=c, marker=m, markersize=6.5, linewidth=1.4, label=l) for c, m, l in styles.values()]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.02, 1.005), frameon=False, ncol=2,
              columnspacing=1.2, handletextpad=0.5, handlelength=1.2, borderaxespad=0)

fig = plt.figure(figsize=(7.6, 6.4))
axa = fig.add_axes([0.195, 0.59, 0.335, 0.29])
axb = fig.add_axes([0.645, 0.59, 0.325, 0.29])
axc = fig.add_axes([0.195, 0.12, 0.775, 0.275])
panel(axa, 'a  External discrimination', LABELS, (0.85, 1.0), [0.85, 0.90, 0.95, 1.0])
panel(axb, 'b  Transferred sensitivity', [''] * 4, (60, 100), [60, 70, 80, 90, 100])
panel(axc, 'c  TBX11K negative-case composition', LABELS, (0, 100), [0, 20, 40, 60, 80, 100])
axa.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axa.set_xlabel('AUROC'); axb.set_xlabel('Sensitivity (%)'); axc.set_xlabel('Specificity (%)')
for cohort, offset in [('montgomery', -0.17), ('tbx11k', 0.17)]:
    color, marker, _ = COHORT[cohort]
    for i, m in enumerate(MODELS):
        point(axa, select(panel='A', cohort=cohort, model=m), i + offset, color, marker)
        point(axb, select(panel='B', cohort=cohort, model=m), i + offset, color, marker, 100)
axb.axvline(90, color='#555555', linewidth=1.2, linestyle=(0, (3, 3)), zorder=2.6)
legend(axa, COHORT)
axb.text(0.0, 1.035, 'Dashed line: source target (90%)', transform=axb.transAxes, fontsize=9.5)
for group, offset in [('health', -0.17), ('sick', 0.17)]:
    color, marker, _ = SUBGROUP[group]
    for i, m in enumerate(MODELS):
        point(axc, select(panel='C', group=group, model=m), i + offset, color, marker, 100)
legend(axc, SUBGROUP)
fig.text(0.195, 0.016, 'a/b: bootstrap 95% intervals   |   c: Wilson 95% intervals', fontsize=9.5, color='#555555')
fig.savefig(OUT / 'Fig2.pdf', metadata={'Title': 'Figure 2 external performance', 'Creator': 'Matplotlib; saved evaluation outputs'})
fig.savefig(OUT / 'Fig2.png', dpi=200)
plt.close(fig)
