"""Paired uncertainty for four prospectively specified matched-input conditions."""
import csv,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
BASE=Path('/content/drive/MyDrive/PulmoVision_Journal');ROOT=BASE/'Matched_resolution_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 out=ROOT/'analysis';out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():
  for f,h in json.loads((out/'COMPLETE.json').read_text())['files'].items():assert sha(out/f)==h
  print('MATCHED ANALYSIS ALREADY COMPLETE');return
 receipt=json.loads((ROOT/'COMPLETE.json').read_text())
 for f,h in receipt['files'].items():assert sha(ROOT/f)==h
 src=BASE/'TBX11K_evaluation_v1/evaluate_tbx11k.py'
 assert sha(src)=='47faaccc17a645f7f1d3b577324449378ea8db99dc8ca183d5733aaa9555e6b8'
 spec=importlib.util.spec_from_file_location('metrics',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.selfcheck()
 names=['rad_224','dino_224','rad_518','dino_518']
 pairs=[('rad_224','dino_224'),('rad_518','dino_518'),('rad_518','rad_224'),('dino_518','dino_224')]
 allrows={n:read(ROOT/'results'/(n+'_predictions.csv')) for n in names}
 summary=[];comparisons=[];subgroups=[]
 for cohort in ['montgomery','tbx11k']:
  rr={n:[r for r in rs if r['cohort']==cohort] for n,rs in allrows.items()};ref=rr[names[0]];ids=[r['image_id'] for r in ref]
  assert len(ids)==len(set(ids));y=np.array([int(r['label']) for r in ref]);p={};cuts={}
  for n in names:
   assert [(r['image_id'],r['label'],r['original_group']) for r in rr[n]]==[(r['image_id'],r['label'],r['original_group']) for r in ref]
   p[n]=np.array([float(r['score']) for r in rr[n]]);cuts[n]=float(rr[n][0]['threshold']);assert all(float(r['threshold'])==cuts[n] for r in rr[n])
  keys=['auroc','sensitivity','specificity','brier'];point={n:m.weighted_metrics(y,p[n],cuts[n],np.ones(len(y))) for n in names};samples={n:{k:[] for k in keys} for n in names};rng=np.random.default_rng(20260925)
  for offset in range(0,10000,100):
   w=rng.multinomial(len(y),np.full(len(y),1/len(y)),100);w=w[(w[:,y==0].sum(1)>0)&(w[:,y==1].sum(1)>0)]
   for n in names:
    vals=m.weighted_metrics(y,p[n],cuts[n],w)
    for k in keys:samples[n][k].extend(vals[k])
  for n in names:
   r=dict(cohort=cohort,model=n,threshold=cuts[n],n=len(y),**{k:float(v[0]) for k,v in point[n].items()})
   for k in keys:r[k+'_low'],r[k+'_high']=map(float,np.quantile(samples[n][k],[.025,.975]))
   summary.append(r)
  contrasts=[(a+' minus '+b,{a:1,b:-1}) for a,b in pairs]+[('encoder gap at 518 minus encoder gap at 224',{'rad_518':1,'dino_518':-1,'rad_224':-1,'dino_224':1})]
  for label,weights in contrasts:
   for k in keys:
    delta=sum(v*np.array(samples[n][k]) for n,v in weights.items());lo,hi=np.quantile(delta,[.025,.975])
    comparisons.append(dict(cohort=cohort,comparison=label,metric=k,difference=float(sum(v*point[n][k][0] for n,v in weights.items())),ci_low=float(lo),ci_high=float(hi)))
  if cohort=='tbx11k':
   g=np.array([r['original_group'] for r in ref])
   for n in names:
    for group in ['health','sick']:
     mask=(y==1)|(g==group);vals=m.weighted_metrics(y[mask],p[n][mask],cuts[n],np.ones(sum(mask)))
     subgroups.append(dict(model=n,negatives=group,n=int(sum(mask)),**{k:float(v[0]) for k,v in vals.items()}))
  print('Paired intervals complete:',cohort,flush=True)
 write(out/'summary.csv',summary);write(out/'paired_comparisons.csv',comparisons);write(out/'subgroups.csv',subgroups)
 lines=['# Matched-resolution experiment','', 'Exploratory comparison: frozen encoders, identical preprocessing at each resolution, identical train-only scaling and balanced linear classifier. Thresholds selected only on Shenzhen validation. Montgomery n=138; locked TBX11K n=1794.','', '| Cohort | Model | AUROC (95% CI) | Sensitivity | Specificity |','|---|---|---|---|---|']
 for r in summary:lines.append(f"| {r['cohort']} | {r['model']} | {r['auroc']:.3f} ({r['auroc_low']:.3f}–{r['auroc_high']:.3f}) | {r['sensitivity']:.3f} | {r['specificity']:.3f} |")
 lines+=['','## Paired AUROC differences','','| Cohort | Contrast | Difference (95% CI) |','|---|---|---|']
 for r in comparisons:
  if r['metric']=='auroc':lines.append(f"| {r['cohort']} | {r['comparison']} | {r['difference']:.3f} ({r['ci_low']:.3f}–{r['ci_high']:.3f}) |")
 lines+=['','## Negative case mix','','| Model | Negative group | AUROC | Specificity |','|---|---|---|---|']
 for r in subgroups:lines.append(f"| {r['model']} | {r['negatives']} | {r['auroc']:.3f} | {r['specificity']:.3f} |")
 lines+=['','Intervals use 10,000 paired unstratified image bootstrap draws (seed 20260925), conditional on fitted models and thresholds. They do not capture training-split uncertainty. Subgroup values are point estimates. Multiple comparisons are exploratory and unadjusted.','', 'Identical input processing controls the resolution/transform difference in this experiment. It does not isolate a causal effect of medical pretraining: checkpoint training histories differ. Patient/source independence and pretraining exposure are unverified. No expert clinical review was performed. Earlier target outcomes informed the choice of this extension. No target data were used to fit classifiers or choose thresholds.']
 (out/'Matched_Resolution_Results.md').write_text('\n'.join(lines),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,cohort in zip(axs,['montgomery','tbx11k']):
  for encoder,color in [('rad','#146b8c'),('dino','#b95322')]:
   rr=[next(r for r in summary if r['cohort']==cohort and r['model']==encoder+'_'+str(s)) for s in [224,518]];yy=np.array([r['auroc'] for r in rr])
   ax.errorbar([224,518],yy,yerr=[yy-np.array([r['auroc_low'] for r in rr]),np.array([r['auroc_high'] for r in rr])-yy],label='RAD-DINO' if encoder=='rad' else 'DINOv2',color=color,marker='o',capsize=4)
  ax.set_title(cohort.upper());ax.set_xticks([224,518]);ax.set_xlabel('Input width and height (pixels)');ax.set_ylim(.5,1);ax.grid(alpha=.2);ax.legend()
 axs[0].set_ylabel('AUROC with image bootstrap 95% CI');fig.tight_layout();fig.savefig(out/'Matched_Resolution.png',dpi=200);fig.savefig(out/'Matched_Resolution.pdf');plt.close(fig)
 (out/'COMPLETE.json').write_text(json.dumps({'code_sha256':sha(__file__),'inference_receipt_sha256':sha(ROOT/'COMPLETE.json'),'bootstrap_repeats':10000,'seed':20260925,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2))
 print('MATCHED ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
