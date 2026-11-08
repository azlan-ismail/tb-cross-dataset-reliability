"""LaTeX row blocks for Tables 2-4 of the article and Tables S4-S5 of the supplement.

Usage: python manuscript/tables.py REPO UNCERTAINTY_JSON SUBGROUPS_JSON CALIBRATION_JSON OUT

Each table body (the rows between \\midrule and \\bottomrule) is written to
OUT/tables/<name>.tex. The files in evidence/manuscript/tables/ match the
published manuscript exactly. Uses only saved estimates and standard-library Python.
"""
import csv, json, sys
from pathlib import Path

R = Path(sys.argv[1])
U = json.load(open(sys.argv[2]))['models']        # threshold_uncertainty.json
SUB = json.load(open(sys.argv[3]))                # local_threshold_subgroups.json
CAL = json.load(open(sys.argv[4]))                # threshold_calibration.json
OUT = Path(sys.argv[5]) / 'tables'; OUT.mkdir(parents=True, exist_ok=True)

M = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
LABEL = {'linear': 'Linear', 'head_seed42': 'Neural, 42', 'head_seed43': 'Neural, 43', 'head_seed44': 'Neural, 44'}
COHORTS = [('montgomery', 'Montgomery'), ('tbx11k', 'TBX11K')]

def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
def pct(x): return f'{100 * x:.1f}'
def f3(x): return f'{x:.3f}'
def f2(x): return f'{x:.2f}'
def cell(v, lo, hi, f=pct): return f'\\cell{{{f(v)}\\\\({f(lo)}--{f(hi)})}}'
def span(vals, f=pct):
    lo, hi = f(min(vals)), f(max(vals))
    return lo if lo == hi else f'{lo}--{hi}'
def write(name, rows):
    (OUT / f'{name}.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# Saved external metrics at the transferred thresholds (initial evaluation outputs).
saved = {}
for r in rd(R / 'evidence/development/results/summary.csv'):           # Montgomery
    saved[('montgomery', r['model'])] = {
        k: float(r[k]) for k in ['auroc', 'average_precision', 'sensitivity', 'specificity', 'brier', 'fn', 'fp']} | {
        f'{k}_ci': (float(r[f'{k}_ci_low']), float(r[f'{k}_ci_high'])) for k in ['auroc', 'sensitivity', 'specificity']}
for m in M:                                                             # TBX11K
    j = json.load(open(R / f'evidence/tbx11k/{m}_metrics.json'))
    saved[('tbx11k', m)] = dict(j['metrics']) | {f'{k}_ci': tuple(j['bootstrap_95ci'][k]) for k in ['auroc', 'sensitivity', 'specificity']}
wilson = {(r['model'], r['group']): r for r in rd(R / 'evidence/manuscript/external_performance_estimates.csv') if r['panel'] == 'C'}

def blocks(make):
    rows = []
    for i, (c, name) in enumerate(COHORTS):
        if i: rows.append('\\midrule')
        for j, m in enumerate(M):
            rows.append(f"{name if j == 0 else ''} & {LABEL[m]} & {make(c, m)} \\\\")
    return rows

# Table 2: external performance at the unchanged development thresholds.
def t2(c, m):
    s = saved[(c, m)]
    return ' & '.join([cell(s['auroc'], *s['auroc_ci'], f3), cell(s['sensitivity'], *s['sensitivity_ci']),
                       cell(s['specificity'], *s['specificity_ci']), f"{int(s['fn'])}", f"{int(s['fp'])}"])
write('table2_external_performance', blocks(t2))

# Table 3: conventional versus uncertainty-aware threshold rule.
dev, loc = CAL['development_only'], CAL['local']
rows = []
for c, name in COHORTS:
    rows.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{Development only, transferred to {name}}}}} \\\\')
    for rule, lab in [('empirical', 'Conventional'), ('np', 'Uncertainty-aware')]:
        if rule == 'empirical':   # the primary transferred thresholds
            sens = [U[m][c]['sens_at_transfer'] for m in M]; spec = [U[m][c]['spec_at_transfer'] for m in M]
            ref = [U[m][c]['positive_rate_at_transfer'] for m in M]
        else:
            sens = [dev[m][c + '_np']['sensitivity'] for m in M]; spec = [dev[m][c + '_np']['specificity'] for m in M]
            ref = [dev[m][c + '_np']['referral_rate'] for m in M]
        oth = span([dev[m][f'{c}_{rule}']['spec_other_illness'] for m in M]) if c == 'tbx11k' else '---'
        rows.append(f'\\quad {lab} & --- & {span(sens)} & {span(spec)} & {oth} & {span(ref)} \\\\')
    rows.append('\\addlinespace[3pt]')
for k, (c, name, n) in enumerate([('montgomery', 'Montgomery', 29), ('tbx11k', 'TBX11K', 100)]):
    rows.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{Local calibration, {name} ($n={n}$ positives)}}}} \\\\')
    for rule, lab in [('empirical', 'Conventional'), ('np', 'Uncertainty-aware')]:
        E = [next(e for e in loc[c][m][rule] if e['n_cal_pos'] == n) for m in M]
        oth = span([e['spec_other_illness']['median'] for e in E]) if c == 'tbx11k' else '---'
        rows.append(f"\\quad {lab} & {span([e['attainment'] for e in E], lambda v: f'{100 * v:.0f}')} & "
                    f"{span([e['sensitivity']['median'] for e in E])} & {span([e['specificity']['median'] for e in E])} & "
                    f"{oth} & {span([e['referral_rate']['median'] for e in E])} \\\\")
    if k == 0: rows.append('\\addlinespace[3pt]')
write('table3_threshold_rules', rows)

# Table 4: TBX11K referral rate and specificity by negative subgroup.
rows = []
for m in M:
    o = U[m]['tbx11k']; parts = [cell(o['positive_rate_at_transfer'], *o['positive_rate_ci'])]
    for g in ['health', 'sick']:
        w = wilson[(m, g)]; parts.append(cell(float(w['estimate']), float(w['ci_low']), float(w['ci_high'])))
    for g in ['health', 'sick']:
        k, n, lo, hi = SUB[f'{m}|{g}']; parts.append(cell(k / n, lo, hi))
    rows.append(f"{LABEL[m]} & {' & '.join(parts)} \\\\")
write('table4_negative_subgroups', rows)

# Table S4: threshold-selection uncertainty and transferred sensitivity.
def s4(c, m):
    o, s = U[m][c], saved[(c, m)]
    return ' & '.join([cell(o['cohort_t90'], *o['cohort_t90_ci'], f3), cell(s['sensitivity'], *s['sensitivity_ci']),
                       span(o['sens_selection_plus_eval_ci']), f"{100 * o['share_resamples_sens_ge_90']:.0f}"])
write('tableS4_threshold_uncertainty', blocks(s4))

# Table S5: probability error and calibration.
def s5(c, m):
    o, s = U[m][c], saved[(c, m)]
    return ' & '.join([f3(s['average_precision']), f3(s['brier']), cell(o['ece'], *o['ece_ci'], f3),
                       cell(o['cal_slope'], *o['cal_slope_ci'], f2), f"\\cell{{{f3(o['mean_prob'])}\\\\({f3(o['prevalence'])})}}"])
write('tableS5_calibration', blocks(s5))

print(f'Wrote {len(list(OUT.glob("*.tex")))} table bodies to {OUT}')
