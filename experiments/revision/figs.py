import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
R, A, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10.5, 'legend.fontsize': 10, 'pdf.fonttype': 42, 'axes.linewidth': 0.7,
    'axes.edgecolor': '#777777', 'text.color': '#222222', 'axes.labelcolor': '#222222',
    'xtick.color': '#333333', 'ytick.color': '#333333', 'savefig.facecolor': 'white'})
BLUE, ORANGE, GREEN, PINK, GREY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#333333'
MODELS = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
LABELS = ['Linear', 'Neural: seed 42', 'Neural: seed 43', 'Neural: seed 44']
res = json.load(open(A))['models']
def rd(p): return list(csv.DictReader(open(p, newline='')))

def panel(ax, title, labels, limits, ticks, pad=30):
    ax.set_xlim(*limits); ax.set_xticks(ticks)
    ax.set_ylim(len(labels) - 0.45, -0.55); ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis='y', length=0, pad=8); ax.tick_params(axis='x', length=3, pad=4)
    ax.grid(axis='x', color='#e1e4e8', linewidth=0.65, zorder=0)
    for i in range(len(labels)):
        if i % 2 == 0: ax.axhspan(i - 0.45, i + 0.45, color='#f5f6f7', zorder=0)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=pad)

def pt(ax, v, lo, hi, y, color, marker, lw=1.5, ms=6, cap=3, alpha=1, mfc=None):
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=marker, color=color, markersize=ms, elinewidth=lw,
                capsize=cap, capthick=1.0, zorder=3, alpha=alpha, markerfacecolor=mfc or color, markeredgewidth=0.9)

def leg(ax, items, ncol=3, y=1.005):
    hs = [Line2D([0], [0], color=c, marker=m, markersize=6, linewidth=1.3, label=l, markerfacecolor=f or c)
          for c, m, l, f in items]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(-0.02, y), frameon=False, ncol=ncol,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.2, borderaxespad=0)

def save(fig, stem):
    fig.savefig(OUT / f'{stem}.pdf'); fig.savefig(OUT / f'{stem}.png', dpi=200); plt.close(fig)

# ---------- Figure 2: operating-point uncertainty ----------
fig = plt.figure(figsize=(7.6, 7.0))
axa = fig.add_axes([0.20, 0.60, 0.33, 0.28]); axb = fig.add_axes([0.64, 0.60, 0.33, 0.28])
axc = fig.add_axes([0.20, 0.11, 0.77, 0.28])
panel(axa, 'A  Threshold for 90% sensitivity', LABELS, (0, 1), [0, 0.25, 0.5, 0.75, 1.0], pad=44)
panel(axb, 'B  Transferred sensitivity', [''] * 4, (40, 100), [40, 60, 80, 100], pad=44)
panel(axc, 'C  TBX11K specificity at a locally selected threshold', LABELS, (40, 100), [40, 50, 60, 70, 80, 90, 100])
axa.set_xlabel('TB-score threshold'); axb.set_xlabel('Sensitivity (%)'); axc.set_xlabel('Specificity (%)')
for i, m in enumerate(MODELS):
    o = res[m]
    pt(axa, o['validation_threshold'], *o['val']['threshold_ci'], i - 0.24, GREY, 'o')
    pt(axa, o['montgomery']['cohort_t90'], *o['montgomery']['cohort_t90_ci'], i, BLUE, 'o')
    pt(axa, o['tbx11k']['cohort_t90'], *o['tbx11k']['cohort_t90_ci'], i + 0.24, ORANGE, 's')
leg(axa, [(GREY, 'o', 'Shenzhen val.', None), (BLUE, 'o', 'Montgomery', None), (ORANGE, 's', 'TBX11K', None)], ncol=2)
saved = {(r['cohort'], r['model']): r for r in rd(f'{R}/figures/data/figure2_data.csv') if r['panel'] == 'B'}
for i, m in enumerate(MODELS):
    for c, off, col, mk in [('montgomery', -0.17, BLUE, 'o'), ('tbx11k', 0.17, ORANGE, 's')]:
        o = res[m][c]; v = 100 * o['sens_at_transfer']
        lo2, hi2 = [100 * x for x in o['sens_selection_plus_eval_ci']]
        axb.plot([lo2, hi2], [i + off, i + off], color=col, linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=2)
        s = saved[(c, m)]
        pt(axb, v, 100 * float(s['ci_low']), 100 * float(s['ci_high']), i + off, col, mk, lw=2.2, cap=0)
axb.axvline(90, color='#555555', linewidth=1.1, linestyle=(0, (3, 3)), zorder=2.5)
axb.text(0.0, 1.03, 'Solid: conditional on threshold\nDashed: + threshold selection', transform=axb.transAxes, fontsize=9.5)
from math import sqrt
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
tbx = rd(f'{R}/evidence/tbx11k/predictions.csv')
colmap = {'linear': 'linear_score', 'head_seed42': 'head_seed42_score', 'head_seed43': 'head_seed43_score', 'head_seed44': 'head_seed44_score'}
sub = {}
for i, m in enumerate(MODELS):
    t = res[m]['tbx11k']['cohort_t90']
    for g, off, col, mk in [('health', -0.17, GREEN, 'o'), ('sick', 0.17, PINK, 'D')]:
        neg = [float(r[colmap[m]]) for r in tbx if r['label'] == '0' and r['original_group'] == g]
        k = sum(x < t for x in neg); lo, hi = wilson(k, len(neg)); sub[(m, g)] = (k, len(neg), lo, hi)
        pt(axc, 100 * k / len(neg), 100 * lo, 100 * hi, i + off, col, mk)
leg(axc, [(GREEN, 'o', 'Healthy (n = 800)', None), (PINK, 'D', 'Other illness (n = 794)', None)], ncol=2)
fig.text(0.20, 0.015, 'A, B (dashed): 10,000 class-stratified bootstrap resamples   |   C: Wilson 95% intervals',
         fontsize=8.6, color='#555555')
save(fig, 'Fig3')
json.dump({f'{m}|{g}': v for (m, g), v in sub.items()}, open(OUT / 'fig3c_subgroups.json', 'w'), indent=1)

# ---------- Figure 3: score distributions + reliability (seed 42) ----------
m = 'head_seed42'; t = res[m]['validation_threshold']
val = rd(f'{R}/evidence/development/models/{m}_validation_predictions.csv')
al = rd(f'{R}/evidence/comparators/aligned_predictions.csv')
grp = {r['image_id']: r['original_group'] for r in tbx}
groups = [
    ('Shenzhen val.: TB (69)', [float(r['tb_probability']) for r in val if r['label'] == '1'], ORANGE),
    ('Shenzhen val.: normal (64)', [float(r['tb_probability']) for r in val if r['label'] == '0'], GREEN),
    ('Montgomery: TB (58)', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'montgomery' and r['label'] == '1'], ORANGE),
    ('Montgomery: normal (80)', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'montgomery' and r['label'] == '0'], GREEN),
    ('TBX11K: TB (200)', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'tbx11k' and r['label'] == '1'], ORANGE),
    ('TBX11K: healthy (800)', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'tbx11k' and grp[r['image_id']] == 'health'], GREEN),
    ('TBX11K: other illness (794)', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'tbx11k' and grp[r['image_id']] == 'sick'], PINK),
]
assert [len(g[1]) for g in groups] == [69, 64, 58, 80, 200, 800, 794]
fig = plt.figure(figsize=(7.6, 7.4))
axa = fig.add_axes([0.30, 0.50, 0.66, 0.42]); axb = fig.add_axes([0.30, 0.08, 0.40, 0.31])
def logit(p): p = np.clip(np.asarray(p), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
rng = np.random.default_rng(7)
for i, (lab, vals, col) in enumerate(groups):
    x = logit(vals); y = i + rng.uniform(-0.22, 0.22, len(x))
    axa.scatter(x, y, s=7, color=col, alpha=0.45, linewidths=0, zorder=2)
    q = np.percentile(x, [10, 50, 90])
    axa.plot([q[0], q[2]], [i, i], color='#222222', linewidth=1.4, zorder=3)
    axa.plot(q[1], i, marker='|', color='#222222', markersize=12, markeredgewidth=2, zorder=3)
axa.axvline(logit(t), color='#555555', linewidth=1.2, linestyle=(0, (3, 3)), zorder=2.5)
ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
axa.set_xticks(logit(ticks), [str(v) for v in ticks]); axa.set_xlim(logit(0.0005), logit(0.9995))
axa.set_yticks(range(len(groups)), [g[0] for g in groups]); axa.set_ylim(len(groups) - 0.5, -0.6)
axa.tick_params(axis='y', length=0)
for s in ['top', 'right', 'left']: axa.spines[s].set_visible(False)
axa.grid(axis='x', color='#e1e4e8', linewidth=0.65, zorder=0)
axa.set_xlabel('TB score (logit-spaced axis)')
axa.set_title('A  Score distributions, neural classifier (seed 42)', loc='left', fontsize=11.5, fontweight='bold', pad=22)
axa.text(logit(t), -0.75, f' transferred threshold {t:.3f}', fontsize=9, color='#555555', ha='left', va='bottom')
# reliability
def rel(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y); idx = np.minimum((p * bins).astype(int), bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        k = idx == b
        if k.sum() >= 5: xs.append(p[k].mean()); ys.append(y[k].mean()); ns.append(k.sum())
    return xs, ys, ns
sets = [('Shenzhen val.', [float(r['tb_probability']) for r in val], [int(r['label']) for r in val], GREY, 'o'),
        ('Montgomery', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'montgomery'], [int(r['label']) for r in al if r['cohort'] == 'montgomery'], BLUE, 'o'),
        ('TBX11K', [float(r['rad_head_seed42']) for r in al if r['cohort'] == 'tbx11k'], [int(r['label']) for r in al if r['cohort'] == 'tbx11k'], ORANGE, 's')]
axb.plot([0, 1], [0, 1], color='#999999', linewidth=1, linestyle=(0, (3, 3)))
for lab, p, y, col, mk in sets:
    xs, ys, ns = rel(p, y); axb.plot(xs, ys, color=col, marker=mk, markersize=5.5, linewidth=1.6, label=lab)
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.set_aspect('equal')
axb.set_xlabel('Mean predicted TB score'); axb.set_ylabel('Observed TB fraction')
for s in ['top', 'right']: axb.spines[s].set_visible(False)
axb.grid(color='#e1e4e8', linewidth=0.6)
axb.set_title('B  Reliability (10 bins, n ≥ 5)', loc='left', fontsize=11.5, fontweight='bold', pad=8)
axb.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), frameon=False)
save(fig, 'Fig4')

# ---------- Figure 4: matched encoders (panels A/B of the archived figure) ----------
rows = rd(f'{R}/figures/data/figure3_data.csv')
def sel(**f):
    x = [r for r in rows if all(r[k] == v for k, v in f.items())]; assert len(x) == 1, f; return x[0]
fig = plt.figure(figsize=(7.6, 3.7))
axa = fig.add_axes([0.20, 0.17, 0.35, 0.62]); axb = fig.add_axes([0.64, 0.17, 0.33, 0.62])
gs = [('montgomery', '224'), ('montgomery', '518'), ('tbx11k', '224'), ('tbx11k', '518')]
labs = ['Montgomery\n224 pixels', 'Montgomery\n518 pixels', 'TBX11K\n224 pixels', 'TBX11K\n518 pixels']
panel(axa, 'A  Matched encoders', labs, (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
panel(axb, 'B  Paired AUROC contrast', [''] * 4, (-0.025, 0.4), [0, 0.1, 0.2, 0.3, 0.4])
axa.set_xlabel('AUROC'); axb.set_xlabel('RAD-DINO minus DINOv2')
axb.axvline(0, color='#555555', linewidth=1.1, linestyle=(0, (3, 3)))
for i, (c, r_) in enumerate(gs):
    for enc, off, col, mk in [('rad', -0.16, GREEN, 'o'), ('dino', 0.16, PINK, 'D')]:
        r = sel(panel='A', cohort=c, model=f'{enc}_{r_}')
        pt(axa, float(r['estimate']), float(r['ci_low']), float(r['ci_high']), i + off, col, mk)
    r = sel(panel='B', cohort=c, group=r_)
    pt(axb, float(r['estimate']), float(r['ci_low']), float(r['ci_high']), i, '#374151', 's')
leg(axa, [(GREEN, 'o', 'RAD-DINO', None), (PINK, 'D', 'DINOv2', None)], ncol=2)
save(fig, 'Fig5')
print('ok'); print(json.dumps({f'{m}|{g}': [round(v[0]/v[1],4), round(v[2],4), round(v[3],4)] for (m, g), v in sub.items()}))
