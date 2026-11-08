"""Uncertainty-aware threshold calibration (Neyman-Pearson order-statistic rule).

Threshold t = k-th smallest score among n calibration positives, with k the largest
rank such that P(Binomial(n, alpha) <= k-1) <= delta. Under exchangeability this
bounds P(miss rate > alpha) by delta (Tong, Feng & Li, Sci Adv 2018).
Compared with the empirical rule (highest threshold with sample sensitivity >= 1-alpha).
Uses only saved per-image scores.

Usage: python analysis/threshold_calibration.py REPO OUT
Writes OUT/threshold_calibration.json (2,000 random draws per n, seed 20261108).
"""
import csv, json, math, sys
import numpy as np
from math import comb
R, OUT = sys.argv[1], sys.argv[2]
ALPHA, DELTA, SPLITS, SEED = 0.10, 0.05, 2000, 20261108
MODELS = {'linear': 'rad_linear', 'head_seed42': 'rad_head_seed42', 'head_seed43': 'rad_head_seed43', 'head_seed44': 'rad_head_seed44'}
VALF = {m: f'{R}/evidence/development/models/{m}_validation_predictions.csv' for m in MODELS}

def binom_cdf(x, n, p):
    return sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in range(0, x + 1))

def np_rank(n, alpha=ALPHA, delta=DELTA):
    """Largest k (1-based) with P(Bin(n, alpha) <= k-1) <= delta; None if none."""
    k_best = None
    for k in range(1, n + 1):
        if binom_cdf(k - 1, n, alpha) <= delta: k_best = k
        else: break
    return k_best

def np_threshold(pos_scores):
    k = np_rank(len(pos_scores))
    if k is None: return None
    return float(np.sort(pos_scores)[k - 1])

def emp_threshold(pos_scores, target=1 - ALPHA):
    s = np.sort(pos_scores)[::-1]; return float(s[math.ceil(target * len(s) - 1e-12) - 1])

def rd(p): return list(csv.DictReader(open(p, newline='')))
al = rd(f'{R}/evidence/comparators/aligned_predictions.csv')
grp = {r['image_id']: r['original_group'] for r in rd(f'{R}/evidence/tbx11k/predictions.csv')}
coh = {}
for c in ['montgomery', 'tbx11k']:
    rr = [r for r in al if r['cohort'] == c]
    coh[c] = {'y': np.array([int(r['label']) for r in rr]),
              'g': np.array([grp.get(r['image_id'], 'normal') if r['label'] == '0' else 'tb' for r in rr]),
              's': {m: np.array([float(r[col]) for r in rr]) for m, col in MODELS.items()}}

res = {'settings': {'alpha': ALPHA, 'delta': DELTA, 'target_sensitivity': 1 - ALPHA, 'confidence': 1 - DELTA,
                    'splits': SPLITS, 'seed': SEED,
                    'design': 'calibration positives sampled without replacement from the external cohort; '
                              'sensitivity on held-out positives; specificity on all negatives (none used for calibration); '
                              'referral rate = prevalence*sens + (1-prevalence)*(1-spec) at the full-cohort prevalence'},
       'min_positives': {}, 'development_only': {}, 'local': {}}
for a in [0.05, 0.10, 0.15]:
    for d in [0.05, 0.10]:
        res['min_positives'][f'sens{1-a:.2f}_conf{1-d:.2f}'] = math.ceil(math.log(d) / math.log(1 - a) - 1e-12)

def evaluate(c, m, t, pos_mask=None):
    y, g, s = coh[c]['y'], coh[c]['g'], coh[c]['s'][m]
    pm = (y == 1) if pos_mask is None else pos_mask
    sens = float(np.mean(s[pm] >= t)); spec = float(np.mean(s[y == 0] < t))
    prev = float(np.mean(y))
    out = {'threshold': t, 'sensitivity': sens, 'specificity': spec,
           'referral_rate': prev * sens + (1 - prev) * (1 - spec)}
    if c == 'tbx11k':
        out['spec_healthy'] = float(np.mean(s[g == 'health'] < t)); out['spec_other_illness'] = float(np.mean(s[g == 'sick'] < t))
    return out

# A. development-only: NP rule on Shenzhen validation positives, transferred unchanged
for m in MODELS:
    v = rd(VALF[m]); vp = np.array([float(r['tb_probability']) for r in v if r['label'] == '1'])
    t_np, t_emp = np_threshold(vp), emp_threshold(vp)
    res['development_only'][m] = {'n_val_pos': int(len(vp)), 'np_rank': np_rank(len(vp)),
                                  'threshold_np': t_np, 'threshold_empirical': t_emp,
                                  **{f'{c}_np': evaluate(c, m, t_np) for c in coh},
                                  **{f'{c}_empirical': evaluate(c, m, t_emp) for c in coh}}

# B. local calibration with n positives from the target cohort
rng = np.random.default_rng(SEED)
NS = {'montgomery': [29, 35], 'tbx11k': [29, 40, 50, 75, 100, 150]}
for c, ns in NS.items():
    y = coh[c]['y']; P = np.flatnonzero(y == 1)
    for n in ns:
        idx = [rng.choice(P, n, replace=False) for _ in range(SPLITS)]   # shared splits across models/rules
        for m in MODELS:
            s = coh[c]['s'][m]; agg = {'np': [], 'empirical': []}
            for cp in idx:
                held = np.zeros(len(y), bool); held[P] = True; held[cp] = False
                for rule, fn in [('np', np_threshold), ('empirical', emp_threshold)]:
                    agg[rule].append(evaluate(c, m, fn(s[cp]), pos_mask=held))
            for rule, L in agg.items():
                d = {k: np.array([e[k] for e in L]) for k in L[0]}
                summ = {'n_cal_pos': n, 'n_test_pos': int(len(P) - n),
                        'attainment': float(np.mean(d['sensitivity'] >= 1 - ALPHA - 1e-12))}
                for k, a in d.items():
                    summ[k] = {'median': float(np.median(a)), 'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90))}
                res['local'].setdefault(c, {}).setdefault(m, {}).setdefault(rule, []).append(summ)
json.dump(res, open(f'{OUT}/threshold_calibration.json', 'w'), indent=1)

print('min positives', res['min_positives'])
for m, o in res['development_only'].items():
    print(f"DEV {m}: k={o['np_rank']}/{o['n_val_pos']} t_np={o['threshold_np']:.3f} (emp {o['threshold_empirical']:.3f})",
          ' | '.join(f"{c}: sens {o[c+'_np']['sensitivity']:.3f} spec {o[c+'_np']['specificity']:.3f} ref {o[c+'_np']['referral_rate']:.3f}" for c in coh))
for c in res['local']:
    for m in MODELS:
        for rule in ['empirical', 'np']:
            print(f"LOC {c:10s} {m:12s} {rule:9s} " + ' | '.join(
                f"n={e['n_cal_pos']}: att {e['attainment']:.2f} sens {e['sensitivity']['median']:.3f} spec {e['specificity']['median']:.3f} ref {e['referral_rate']['median']:.3f}"
                + (f" oth {e['spec_other_illness']['median']:.3f} hl {e['spec_healthy']['median']:.3f}" if c == 'tbx11k' else '')
                for e in res['local'][c][m][rule]))
