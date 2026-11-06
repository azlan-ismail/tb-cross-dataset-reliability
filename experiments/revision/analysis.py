"""Tier-1 descriptive analyses from saved per-image scores (no model fitting)."""
import csv, json, math, sys
import numpy as np
R, OUT = sys.argv[1], sys.argv[2]
B = 10000
SEED = 20261106
MODELS = {'linear': 'rad_linear', 'head_seed42': 'rad_head_seed42',
          'head_seed43': 'rad_head_seed43', 'head_seed44': 'rad_head_seed44'}
SAVED_T = {'linear': 0.5611574877253318, 'head_seed42': 0.7429928779602051,
           'head_seed43': 0.6160100698471069, 'head_seed44': 0.6283349394798279}

def rd(p): return list(csv.DictReader(open(p, newline='')))
aligned = rd(f'{R}/evidence/comparators/aligned_predictions.csv')
tbx = {r['image_id']: r['original_group'] for r in rd(f'{R}/evidence/tbx11k/predictions.csv')}
data = {}
for m, col in MODELS.items():
    v = rd(f'{R}/evidence/development/models/{m}_validation_predictions.csv')
    d = {'val': (np.array([float(r['tb_probability']) for r in v]), np.array([int(r['label']) for r in v]), None)}
    for c in ['montgomery', 'tbx11k']:
        rr = [r for r in aligned if r['cohort'] == c]
        g = np.array([tbx.get(r['image_id'], '') for r in rr])
        d[c] = (np.array([float(r[col]) for r in rr]), np.array([int(r['label']) for r in rr]), g)
    data[m] = d

def select_t(p, y, target=0.90):
    pos = np.sort(p[y == 1])[::-1]
    k = math.ceil(target * len(pos) - 1e-12)
    return pos[k - 1]          # highest threshold with sensitivity >= target (max specificity)

def sens(p, y, t): return float(np.mean(p[y == 1] >= t))
def spec(p, y, t): return float(np.mean(p[y == 0] < t))

def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def cal_slope(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6); x = np.log(pc / (1 - pc))
    X = np.column_stack([np.ones_like(x), x]); w = np.zeros(2)
    for _ in range(50):
        eta = np.clip(X @ w, -30, 30); mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None]); g = X.T @ (y - mu)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w += step
        if np.max(np.abs(step)) < 1e-8: break
    return float(w[1])

def strat_idx(rng, y):
    out = []
    for cls in (0, 1):
        ii = np.flatnonzero(y == cls)
        out.append(rng.choice(ii, size=len(ii), replace=True))
    return np.concatenate(out)

def ci(a): a = np.asarray(a); return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

res = {'settings': {'bootstrap_resamples': B, 'seed': SEED, 'resampling': 'class-stratified image resamples',
                    'ece_bins': 10, 'threshold_rule': 'highest threshold with empirical sensitivity >= 0.90'}, 'models': {}}
rng = np.random.default_rng(SEED)
for m, d in data.items():
    pv, yv, _ = d['val']
    tv = select_t(pv, yv)
    assert abs(tv - SAVED_T[m]) < 1e-9, (m, tv, SAVED_T[m])
    out = {'validation_threshold': float(tv), 'n_val_pos': int(yv.sum())}
    # point estimates
    for c in ['val', 'montgomery', 'tbx11k']:
        p, y, g = d[c]
        o = {'n': int(len(y)), 'prevalence': float(y.mean()), 'mean_prob': float(p.mean()),
             'ece': ece(p, y), 'cal_slope': cal_slope(p, y),
             'sens_at_transfer': sens(p, y, tv), 'spec_at_transfer': spec(p, y, tv),
             'positive_rate_at_transfer': float(np.mean(p >= tv))}
        if c != 'val':
            tc = select_t(p, y)
            o.update({'cohort_t90': float(tc), 'spec_at_cohort_t90': spec(p, y, tc),
                      'sens_at_cohort_t90': sens(p, y, tc)})
            if c == 'tbx11k':
                for grp in ['health', 'sick']:
                    mm = (y == 1) | (g == grp)
                    o[f'spec_{grp}_at_cohort_t90'] = spec(p[mm], y[mm], tc)
        out[c] = o
    # bootstrap
    bs = {k: [] for k in ['val_t', 'val_ece']}
    for c in ['montgomery', 'tbx11k']:
        for k in ['t90', 'dt', 'sens_sel', 'sens_both', 'ece', 'slope', 'posrate', 'spec_t90']:
            bs[f'{c}_{k}'] = []
    for b in range(B):
        iv = strat_idx(rng, yv); tvb = select_t(pv[iv], yv[iv])
        bs['val_t'].append(tvb); bs['val_ece'].append(ece(pv[iv], yv[iv]))
        for c in ['montgomery', 'tbx11k']:
            p, y, _ = d[c]; ie = strat_idx(rng, y); pe, ye = p[ie], y[ie]
            tc = select_t(pe, ye)
            bs[f'{c}_t90'].append(tc); bs[f'{c}_dt'].append(tc - tvb)
            bs[f'{c}_sens_sel'].append(sens(p, y, tvb))      # selection uncertainty only
            bs[f'{c}_sens_both'].append(sens(pe, ye, tvb))   # selection + evaluation sampling
            bs[f'{c}_ece'].append(ece(pe, ye)); bs[f'{c}_slope'].append(cal_slope(pe, ye))
            bs[f'{c}_posrate'].append(float(np.mean(pe >= tv))); bs[f'{c}_spec_t90'].append(spec(pe, ye, tc))
    out['val']['threshold_ci'] = ci(bs['val_t']); out['val']['ece_ci'] = ci(bs['val_ece'])
    for c in ['montgomery', 'tbx11k']:
        o = out[c]
        o['cohort_t90_ci'] = ci(bs[f'{c}_t90']); o['threshold_shift'] = o['cohort_t90'] - tv
        o['threshold_shift_ci'] = ci(bs[f'{c}_dt'])
        o['sens_selection_only_ci'] = ci(bs[f'{c}_sens_sel'])
        o['sens_selection_plus_eval_ci'] = ci(bs[f'{c}_sens_both'])
        o['share_resamples_sens_ge_90'] = float(np.mean(np.array(bs[f'{c}_sens_both']) >= 0.9 - 1e-12))
        o['ece_ci'] = ci(bs[f'{c}_ece']); o['cal_slope_ci'] = ci(bs[f'{c}_slope'])
        o['positive_rate_ci'] = ci(bs[f'{c}_posrate']); o['spec_at_cohort_t90_ci'] = ci(bs[f'{c}_spec_t90'])
    res['models'][m] = out
json.dump(res, open(f'{OUT}/tier1_analysis.json', 'w'), indent=1)
for m, o in res['models'].items():
    print(f"\n== {m}: val t={o['validation_threshold']:.3f} CI {o['val']['threshold_ci'][0]:.3f}-{o['val']['threshold_ci'][1]:.3f}; val ECE {o['val']['ece']:.3f} slope {o['val']['cal_slope']:.2f}")
    for c in ['montgomery', 'tbx11k']:
        x = o[c]
        print(f"  {c}: t90={x['cohort_t90']:.3f} ({x['cohort_t90_ci'][0]:.3f}-{x['cohort_t90_ci'][1]:.3f}) shift={x['threshold_shift']:+.3f} ({x['threshold_shift_ci'][0]:+.3f} to {x['threshold_shift_ci'][1]:+.3f}); "
              f"sens={x['sens_at_transfer']:.3f} selCI {x['sens_selection_only_ci'][0]:.3f}-{x['sens_selection_only_ci'][1]:.3f} bothCI {x['sens_selection_plus_eval_ci'][0]:.3f}-{x['sens_selection_plus_eval_ci'][1]:.3f} P(>=90)={x['share_resamples_sens_ge_90']:.2f}; "
              f"spec@t90={x['spec_at_cohort_t90']:.3f} ({x['spec_at_cohort_t90_ci'][0]:.3f}-{x['spec_at_cohort_t90_ci'][1]:.3f}); posrate={x['positive_rate_at_transfer']:.3f} ({x['positive_rate_ci'][0]:.3f}-{x['positive_rate_ci'][1]:.3f}) prev={x['prevalence']:.3f}; "
              f"ECE={x['ece']:.3f} ({x['ece_ci'][0]:.3f}-{x['ece_ci'][1]:.3f}) slope={x['cal_slope']:.2f} ({x['cal_slope_ci'][0]:.2f}-{x['cal_slope_ci'][1]:.2f}) meanp={x['mean_prob']:.3f}"
              + (f" spec_h@t90={x['spec_health_at_cohort_t90']:.3f} spec_s@t90={x['spec_sick_at_cohort_t90']:.3f}" if c=='tbx11k' else ''))
