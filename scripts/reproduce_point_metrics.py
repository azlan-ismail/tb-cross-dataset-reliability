"""Independently check saved point estimates; standard library only, no model fitting."""
from pathlib import Path
from itertools import groupby
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('linear', 'head_seed42', 'head_seed43', 'head_seed44')


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def metrics(labels, scores, threshold):
    if len(labels) != len(scores) or not labels:
        raise ValueError('Missing or unaligned scores')
    if set(labels) != {0, 1} or not all(math.isfinite(p) and 0 <= p <= 1 for p in scores):
        raise ValueError('Expected binary labels and finite probability scores')
    positives, negatives = sum(labels), len(labels)-sum(labels)
    ordered = sorted(zip(scores, labels))
    previous_negatives = 0
    concordance = 0.0
    groups = []
    for score, entries in groupby(ordered, key=lambda item: item[0]):
        values = [y for _, y in entries]
        p, n = sum(values), len(values)-sum(values)
        concordance += p * (previous_negatives + 0.5*n)
        previous_negatives += n
        groups.append((p, n))
    cumulative_tp = cumulative_n = 0
    ap = 0.0
    for p, n in reversed(groups):
        cumulative_tp += p
        cumulative_n += p+n
        ap += (p/positives) * (cumulative_tp/cumulative_n)
    prediction = [p >= threshold for p in scores]
    tp = sum(y == 1 and z for y, z in zip(labels, prediction))
    fp = sum(y == 0 and z for y, z in zip(labels, prediction))
    tn, fn = negatives-fp, positives-tp
    return dict(auroc=concordance/(positives*negatives), average_precision=ap,
                sensitivity=tp/positives, specificity=tn/negatives,
                brier=math.fsum((p-y)**2 for p, y in zip(scores, labels))/len(labels),
                tp=tp, fp=fp, tn=tn, fn=fn)


def check(name, data, column, threshold, expected):
    ids = [r['image_id'] for r in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate image IDs in {name}')
    calculated = metrics([int(r['label']) for r in data],
                         [float(r[column]) for r in data], threshold)
    errors = {k: abs(v-float(expected[k])) for k, v in calculated.items()}
    # Original neural Brier calculations use float32; this helper uses Python doubles.
    tolerances = {k: (1e-7 if k == 'brier' else 1e-10) for k in errors}
    if any(error > tolerances[k] for k, error in errors.items()):
        raise ValueError(f'{name}: metric discrepancy {errors}')
    return dict(record=name, n=len(data), threshold=threshold, metrics=calculated,
                maximum_absolute_difference=max(errors.values()))


def main():
    checked = []
    initial = ROOT/'evidence/development/results'
    external = ROOT/'evidence/tbx11k'
    tbx = rows(external/'predictions.csv')
    for model in MODELS:
        saved = json.loads((initial/f'{model}_metrics.json').read_text())
        data = rows(initial/f'{model}_predictions.csv')
        if len(data) != 138:
            raise ValueError('Unexpected Montgomery cohort size')
        checked.append(check(f'initial/montgomery/{model}', data, 'tb_probability',
                             saved['threshold'], saved['metrics']))
        saved = json.loads((external/f'{model}_metrics.json').read_text())
        if len(tbx) != 1794:
            raise ValueError('Unexpected TBX11K cohort size')
        checked.append(check(f'initial/tbx11k/{model}', tbx, model+'_score',
                             saved['threshold'], saved['metrics']))
    base = ROOT/'evidence/comparators'
    aligned = rows(base/'aligned_predictions.csv')
    for saved in rows(base/'summary_with_intervals.csv'):
        data = [r for r in aligned if r['cohort'] == saved['cohort']]
        if len(data) != int(saved['n']):
            raise ValueError('Comparator cohort size mismatch')
        checked.append(check('comparison/'+saved['cohort']+'/'+saved['model'], data,
                             saved['model'], float(saved['threshold']), saved))
    if len(checked) != 24:
        raise ValueError('Expected 24 verification records, including repeated initial-model rows')
    report = dict(status='passed', verification_records=len(checked),
                  scope='Saved point estimates only; overlapping records are not independent experiments. No interval or GPU rerun.',
                  tolerances={'brier': 1e-7, 'other_metrics': 1e-10},
                  numeric_note='Brier tolerance accommodates the original float32 neural-score calculation; other metrics use a tighter tolerance.',
                  maximum_absolute_difference=max(r['maximum_absolute_difference'] for r in checked),
                  records=checked)
    out = ROOT/'verification-output'
    out.mkdir(exist_ok=True)
    (out/'point_metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
