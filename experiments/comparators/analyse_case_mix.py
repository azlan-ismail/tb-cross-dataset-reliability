"""Reproduce the 24 September post hoc paired stratified case-mix analysis.

Run from project root. Does not retrain or change predictions.
"""
import csv,json,importlib.util
from pathlib import Path
import numpy as np
root=Path('outputs/pulmovision-tbx11k-results/evidence')
data=list(csv.DictReader((root/'predictions.csv').open()))
y=np.array([int(r['label']) for r in data]);g=np.array([r['original_group'] for r in data])
spec=importlib.util.spec_from_file_location('metrics','outputs/pulmovision-independent-validation/evaluate_tbx11k.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
names=list(m.THRESHOLDS);rng=np.random.default_rng(20260925)
samples={n:{k:[] for k in ['healthy_auc','sick_auc','combined_auc','difference']} for n in names}
ixs={k:np.flatnonzero(g==k) for k in ['tb','health','sick']}
masks={'healthy':g!='sick','sick':g!='health','combined':np.ones(len(y),bool)}
for offset in range(0,5000,100):
 w=np.zeros((100,len(y)))
 for ix in ixs.values():w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),100)
 for name in names:
  score=np.array([float(r[name+'_score']) for r in data]);v={}
  for k,mask in masks.items():
   v[k]=m.weighted_metrics(y[mask],score[mask],m.THRESHOLDS[name],w[:,mask])['auroc'];samples[name][k+'_auc'].extend(v[k])
  samples[name]['difference'].extend(v['healthy']-v['sick'])
summary=[]
for name in names:
 score=np.array([float(r[name+'_score']) for r in data]);r={'model':name}
 for k,mask in masks.items():
  vals=m.weighted_metrics(y[mask],score[mask],m.THRESHOLDS[name],np.ones(sum(mask)))
  r[k+'_auroc']=float(vals['auroc'][0]);r[k+'_95ci']=np.quantile(samples[name][k+'_auc'],[.025,.975]).tolist()
 r['auroc_difference']=r['healthy_auroc']-r['sick_auroc'];r['difference_95ci']=np.quantile(samples[name]['difference'],[.025,.975]).tolist();summary.append(r)
target=Path('outputs/pulmovision-comparators/Case_Mix_Analysis.json')
assert json.loads(target.read_text())==summary,'Reproduction mismatch'
print('All saved case-mix estimates and intervals reproduced exactly.')
