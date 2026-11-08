"""Figure and LaTeX table rows for the uncertainty-aware threshold analysis."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
J, R, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
res = json.load(open(J)); M = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
# primary (empirical, development-only) referral rate and specificity from tier-1 analysis
T = json.load(open(f'{R}/evidence/revision/tier1_analysis.json'))['models']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.labelsize': 11, 'pdf.fonttype': 42,
                     'axes.linewidth': 0.7, 'axes.edgecolor': '#777777', 'savefig.facecolor': 'white'})
GREY, BLUE, ORANGE, GREEN, PINK = '#555555', '#0072B2', '#D55E00', '#009E73', '#CC79A7'
loc = res['local']['tbx11k']; ns = [e['n_cal_pos'] for e in loc['linear']['np']]
def series(rule, key, stat='median'):
    a = np.array([[ (e[key] if key == 'attainment' else e[key][stat]) for e in loc[m][rule]] for m in M])
    return a.mean(0), a.min(0), a.max(0)
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.3), gridspec_kw={'wspace': 0.45})
def style(ax, title, ylab, ylim):
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=8); ax.set_ylabel(ylab)
    ax.set_xlabel('Local TB positives (n)'); ax.set_ylim(*ylim); ax.set_xlim(20, 158); ax.set_xticks([29, 50, 75, 100, 150])
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e1e4e8', linewidth=0.6)
for rule, col, mk in [('empirical', GREY, 'o'), ('np', BLUE, 's')]:
    m_, lo, hi = series(rule, 'attainment')
    axs[0].fill_between(ns, 100 * lo, 100 * hi, color=col, alpha=0.15, linewidth=0)
    axs[0].plot(ns, 100 * m_, color=col, marker=mk, markersize=5, linewidth=1.6)
    m_, lo, hi = series(rule, 'referral_rate')
    axs[1].fill_between(ns, 100 * lo, 100 * hi, color=col, alpha=0.15, linewidth=0)
    axs[1].plot(ns, 100 * m_, color=col, marker=mk, markersize=5, linewidth=1.6)
axs[0].axhline(95, color='#999999', linestyle=(0, (3, 3)), linewidth=1)
axs[1].axhline(100 * 200 / 1794, color='#999999', linestyle=(0, (3, 3)), linewidth=1)
axs[1].text(ns[-1], 100 * 200 / 1794 + 2, 'TB prevalence', ha='right', fontsize=8.5, color='#666666')
style(axs[0], 'A  Target met', 'Splits meeting 90% (%)', (0, 100))
style(axs[1], 'B  Referral cost', 'Images flagged (%)', (0, 80))
for key, col, mk, lab in [('spec_healthy', GREEN, 'o', 'Healthy'), ('spec_other_illness', PINK, 'D', 'Other illness')]:
    m_, lo, hi = series('np', key)
    axs[2].fill_between(ns, 100 * lo, 100 * hi, color=col, alpha=0.18, linewidth=0)
    axs[2].plot(ns, 100 * m_, color=col, marker=mk, markersize=5, linewidth=1.6)
style(axs[2], 'C  New-rule specificity', 'Specificity (%)', (0, 100))
fig.legend(handles=[Line2D([0], [0], color=GREY, marker='o', label='Empirical rule (current)'),
                    Line2D([0], [0], color=BLUE, marker='s', label='Uncertainty-aware rule'),
                    Line2D([0], [0], color=GREEN, marker='o', label='Healthy negatives'),
                    Line2D([0], [0], color=PINK, marker='D', label='Other-illness negatives')],
           loc='lower center', ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.0)
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.27)
fig.savefig(OUT / 'Fig4.pdf'); fig.savefig(OUT / 'Fig4.png', dpi=200); plt.close(fig)

def rng_(vals, f=lambda v: f'{100*v:.1f}'):
    lo, hi = min(vals), max(vals)
    return f(lo) if f(lo) == f(hi) else f'{f(lo)}--{f(hi)}'
rows = []
dev = res['development_only']
for c, name in [('montgomery', 'Montgomery'), ('tbx11k', 'TBX11K')]:
    emp = {'sens': [T[m][c]['sens_at_transfer'] for m in M], 'spec': [T[m][c]['spec_at_transfer'] for m in M],
           'ref': [T[m][c]['positive_rate_at_transfer'] for m in M]}
    npd = {'sens': [dev[m][c + '_np']['sensitivity'] for m in M], 'spec': [dev[m][c + '_np']['specificity'] for m in M],
           'ref': [dev[m][c + '_np']['referral_rate'] for m in M]}
    oth_e = rng_([dev[m][c + '_empirical']['spec_other_illness'] for m in M]) if c == 'tbx11k' else '---'
    oth_n = rng_([dev[m][c + '_np']['spec_other_illness'] for m in M]) if c == 'tbx11k' else '---'
    rows.append(f"Development only, {name} & Empirical & --- & {rng_(emp['sens'])} & {rng_(emp['spec'])} & {oth_e} & {rng_(emp['ref'])} \\\\")
    rows.append(f" & Uncertainty-aware & --- & {rng_(npd['sens'])} & {rng_(npd['spec'])} & {oth_n} & {rng_(npd['ref'])} \\\\")
for c, name, n in [('montgomery', 'Montgomery', 29), ('tbx11k', 'TBX11K', 100)]:
    for rule, lab in [('empirical', 'Empirical'), ('np', 'Uncertainty-aware')]:
        E = [next(e for e in res['local'][c][m][rule] if e['n_cal_pos'] == n) for m in M]
        oth = rng_([e['spec_other_illness']['median'] for e in E]) if c == 'tbx11k' else '---'
        first = f"Local, {name} ($n={n}$)" if rule == 'empirical' else ''
        rows.append(f"{first} & {lab} & {rng_([e['attainment'] for e in E], lambda v: f'{100*v:.0f}')} & "
                    f"{rng_([e['sensitivity']['median'] for e in E])} & {rng_([e['specificity']['median'] for e in E])} & {oth} & "
                    f"{rng_([e['referral_rate']['median'] for e in E])} \\\\")
open(OUT / 'remedy_table_rows.tex', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
