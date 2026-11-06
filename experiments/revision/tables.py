import json, csv, sys
A = json.load(open(sys.argv[1]))['models']; F = json.load(open(sys.argv[2])); R = sys.argv[3]
M = ['linear', 'head_seed42', 'head_seed43', 'head_seed44']
L = {'linear': 'Linear', 'head_seed42': 'Neural, 42', 'head_seed43': 'Neural, 43', 'head_seed44': 'Neural, 44'}
saved = {(r['cohort'], r['model']): r for r in csv.DictReader(open(f'{R}/figures/data/figure2_data.csv')) if r['panel'] == 'B'}
def p(x): return f'{100*x:.1f}'
def c(a, f=p): return f'{f(a[0])}--{f(a[1])}'
t3 = lambda x: f'{x:.3f}'
t2 = lambda x: f'{x:.2f}'
print('% Table 2 rows')
for coh, name in [('montgomery', 'Montgomery'), ('tbx11k', 'TBX11K')]:
    for m in M:
        o = A[m][coh]; s = saved[(coh, m)]
        print(f"{name} & {L[m]} & \\shortstack{{{t3(o['cohort_t90'])}\\\\({c(o['cohort_t90_ci'], t3)})}} & "
              f"\\shortstack{{{p(o['sens_at_transfer'])}\\\\({p(float(s['ci_low']))}--{p(float(s['ci_high']))})}} & "
              f"{c(o['sens_selection_plus_eval_ci'])} & {100*o['share_resamples_sens_ge_90']:.0f} \\\\")
print('% Table 3 rows')
for m in M:
    o = A[m]['tbx11k']; h = F[f'{m}|health']; k = F[f'{m}|sick']
    print(f"{L[m]} & \\shortstack{{{p(o['positive_rate_at_transfer'])}\\\\({c(o['positive_rate_ci'])})}} & "
          f"\\shortstack{{{p(h[0]/h[1])}\\\\({p(h[2])}--{p(h[3])})}} & \\shortstack{{{p(k[0]/k[1])}\\\\({p(k[2])}--{p(k[3])})}} & "
          f"{p(o['spec_at_cohort_t90'])} \\\\")
print('% Table 4 rows (ECE, slope, mean score)')
for coh, name in [('montgomery', 'Montgomery'), ('tbx11k', 'TBX11K')]:
    for m in M:
        o = A[m][coh]
        print(f"{name} & {L[m]} & \\shortstack{{{t3(o['ece'])}\\\\({c(o['ece_ci'], t3)})}} & \\shortstack{{{t2(o['cal_slope'])}\\\\({c(o['cal_slope_ci'], t2)})}} & {t3(o['mean_prob'])} & {t3(o['prevalence'])} \\\\")
print('% val')
for m in M:
    v = A[m]['val']; print(m, 'thr', t3(A[m]['validation_threshold']), c(v['threshold_ci'], t3), 'ECE', t3(v['ece']), c(v['ece_ci'], t3), 'slope', t2(v['cal_slope']), 'mean', t3(v['mean_prob']), 'prev', t3(v['prevalence']))
for coh in ['montgomery', 'tbx11k']:
    for m in M:
        o = A[m][coh]; print(coh, m, 'shift', f"{o['threshold_shift']:+.3f}", c(o['threshold_shift_ci'], lambda x: f'{x:+.3f}'), 'selonly', c(o['sens_selection_only_ci']), 'spec@t90', p(o['spec_at_cohort_t90']), c(o['spec_at_cohort_t90_ci']))
