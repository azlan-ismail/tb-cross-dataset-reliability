"""Run in Colab after comparator inference; no fitting or threshold changes."""
import csv,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
BASE=Path('/content/drive/MyDrive/PulmoVision_Journal');ROOT=BASE/'Comparator_extension_v1';OUT=ROOT/'paired_analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text())
def write(p,rs):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 assert not OUT.exists(),'Preserve existing paired analysis'
 complete=js(ROOT/'COMPLETE.json')
 assert sha(ROOT/'protocol.json')==complete['protocol_sha256']
 for name,h in complete['results'].items():assert sha(ROOT/'results'/name)==h
 for name,h in complete['model_artifacts'].items():assert sha(ROOT/'models'/name)==h
 source=BASE/'TBX11K_evaluation_v1/evaluate_tbx11k.py'
 assert sha(source)=='47faaccc17a645f7f1d3b577324449378ea8db99dc8ca183d5733aaa9555e6b8'
 spec=importlib.util.spec_from_file_location('metrics',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.selfcheck()
 oldnames=['linear','head_seed42','head_seed43','head_seed44'];newnames=['dinov2_linear','densenet121_seed42','densenet121_seed43','densenet121_seed44']
 names=['rad_'+n for n in oldnames]+newnames
 pairs=list(zip(newnames,['rad_'+n for n in oldnames]))
 summary=[];comparisons=[];allpred=[];case=[]
 for cohort in ['montgomery','tbx11k']:
  if cohort=='montgomery':
   originals={n:read(BASE/'PV_classifier_v2_reviewed/study/results'/ (n+'_predictions.csv')) for n in oldnames}
   reference=originals['linear']
  else:
   reference=read(BASE/'TBX11K_evaluation_v1/results/predictions.csv')
  ids=[r['image_id'] for r in reference];y=np.array([int(r['label']) for r in reference]);p={};thresholds={}
  assert len(ids)==len(set(ids))
  for n in oldnames:
   if cohort=='montgomery':
    rr=originals[n];assert [r['image_id'] for r in rr]==ids;p['rad_'+n]=np.array([float(r['tb_probability']) for r in rr]);thresholds['rad_'+n]=float(rr[0]['threshold'])
   else:p['rad_'+n]=np.array([float(r[n+'_score']) for r in reference]);thresholds['rad_'+n]=m.THRESHOLDS[n]
  for n in newnames:
   rr=[r for r in read(ROOT/'results'/(n+'_predictions.csv')) if r['cohort']==cohort];lookup={r['image_id']:r for r in rr};assert len(rr)==len(ids) and set(lookup)==set(ids)
   assert all(int(lookup[i]['label'])==v for i,v in zip(ids,y))
   p[n]=np.array([float(lookup[i]['score']) for i in ids]);thresholds[n]=float(rr[0]['threshold']);assert all(float(r['threshold'])==thresholds[n] for r in rr)
  points={n:{k:float(v[0]) for k,v in m.weighted_metrics(y,p[n],thresholds[n],np.ones(len(y))).items()} for n in names}
  samples={n:{k:[] for k in ['auroc','sensitivity','specificity','brier']} for n in names};rng=np.random.default_rng(20260926)
  for offset in range(0,10000,100):
   w=rng.multinomial(len(y),np.full(len(y),1/len(y)),100);w=w[(w[:,y==1].sum(1)>0)&(w[:,y==0].sum(1)>0)]
   for n in names:
    v=m.weighted_metrics(y,p[n],thresholds[n],w)
    for k in samples[n]:samples[n][k].extend(v[k])
  for n in names:
   r=dict(cohort=cohort,model=n,threshold=thresholds[n],n=len(y),**points[n])
   for k,v in samples[n].items():r[k+'_ci_low'],r[k+'_ci_high']=map(float,np.quantile(v,[.025,.975]))
   summary.append(r)
  for a,b in pairs:
   for metric in ['auroc','sensitivity','specificity','brier']:
    d=np.array(samples[a][metric])-np.array(samples[b][metric]);lo,hi=np.quantile(d,[.025,.975]);comparisons.append(dict(cohort=cohort,comparison=a+' minus '+b,metric=metric,difference=points[a][metric]-points[b][metric],ci_low=float(lo),ci_high=float(hi)))
  for j,i in enumerate(ids):allpred.append(dict(cohort=cohort,image_id=i,label=int(y[j]),**{n:float(p[n][j]) for n in names}))
  if cohort=='tbx11k':
   g=np.array([r['original_group'] for r in reference])
   for n in names:
    for group in ['health','sick']:
     mask=(y==1)|(g==group);v=m.weighted_metrics(y[mask],p[n][mask],thresholds[n],np.ones(sum(mask)))
     case.append(dict(model=n,negatives=group,n=int(sum(mask)),**{k:float(value[0]) for k,value in v.items()}))
  print('Paired analysis completed for',cohort,flush=True)
 OUT.mkdir();write(OUT/'summary_with_intervals.csv',summary);write(OUT/'paired_comparisons.csv',comparisons);write(OUT/'aligned_predictions.csv',allpred);write(OUT/'case_mix_point_metrics.csv',case)
 (OUT/'ANALYSIS_COMPLETE.json').write_text(json.dumps({'bootstrap_repeats':10000,'seed':20260926,'unit':'unstratified unique images; patient identity unverified','scope':'exploratory; no multiple-comparison adjustment; comparisons differ in adaptation/preprocessing','comparisons':pairs,'code_sha256':sha(__file__),'source_completion_sha256':sha(ROOT/'COMPLETE.json'),'files':{f.name:sha(f) for f in OUT.iterdir() if f.is_file()}},indent=2))
 print('PAIRED ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
