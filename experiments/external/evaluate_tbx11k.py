"""External inference with locked models; no fitting or threshold selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import csv,json,hashlib,zipfile,io,shutil,platform,importlib.metadata
import numpy as np

BASE=Path('/content/drive/MyDrive/PulmoVision_Journal')
AUDIT=BASE/'TBX11K_audit_v1'
OUT=BASE/'TBX11K_evaluation_v1'
MODELS=BASE/'PV_classifier_v2_reviewed/study/models'
REV='110cbc18d5133582e320b43d53bf5c44e410c936'
MANIFEST_HASH='629461acc1b839740264e0ba0a17b7e2baf15201b187a4b2da16a6544d3e6d6c'
FIT_HASH='9ec95a6b76262f354e02af724cccc214b8b1669410b8dce2a752e6c3f003dde5'
THRESHOLDS={'linear':.5611574877253318,'head_seed42':.7429928779602051,'head_seed43':.6160100698471069,'head_seed44':.6283349394798279}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text())
def writejson(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False))
def writecsv(p,rows):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def weighted_metrics(y,p,t,w):
 """Tie-aware weighted AUROC/AP and threshold metrics for a matrix of count weights."""
 w=np.atleast_2d(w).astype(float);pos=w*y;neg=w*(1-y)
 P=pos.sum(1);N=neg.sum(1);total=P+N;z=p>=t
 tp=pos[:,z].sum(1);fp=neg[:,z].sum(1);fn=P-tp;tn=N-fp
 order=np.argsort(p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
 cp=np.cumsum(pos[:,order],axis=1)[:,ends];cn=np.cumsum(neg[:,order],axis=1)[:,ends]
 gp=np.diff(np.c_[np.zeros(len(w)),cp],axis=1);gn=np.diff(np.c_[np.zeros(len(w)),cn],axis=1)
 auc=(gp*(cn-.5*gn)).sum(1)/(P*N)
 desc=np.argsort(-p,kind='stable');de=np.r_[np.flatnonzero(np.diff(p[desc])),len(p)-1]
 dp=np.cumsum(pos[:,desc],axis=1)[:,de];dw=np.cumsum(w[:,desc],axis=1)[:,de]
 ap=(np.diff(np.c_[np.zeros(len(w)),dp],axis=1)*np.divide(dp,dw,out=np.zeros_like(dp),where=dw>0)).sum(1)/P
 return {'auroc':auc,'average_precision':ap,'sensitivity':tp/P,'specificity':tn/N,'brier':(w*(p-y)**2).sum(1)/total,'tp':tp,'fp':fp,'tn':tn,'fn':fn}

def selfcheck():
 from sklearn.metrics import roc_auc_score,average_precision_score
 y=np.array([0,1,0,1,0,1]);p=np.array([.1,.4,.4,.8,.8,.9]);w=np.array([[1,1,1,1,1,1],[2,0,1,3,2,1]])
 m=weighted_metrics(y,p,.5,w)
 for i in range(2):
  assert abs(m['auroc'][i]-roc_auc_score(y,p,sample_weight=w[i]))<1e-12
  assert abs(m['average_precision'][i]-average_precision_score(y,p,sample_weight=w[i]))<1e-12
 print('Weighted metrics self-check passed, including tied scores and repeated samples.',flush=True)

def main():
 import torch
 from PIL import Image,ImageOps
 from transformers import AutoImageProcessor,AutoModel
 selfcheck()
 assert torch.cuda.is_available(),'GPU required; choose T4 runtime before running this evaluation.'
 assert sha(AUDIT/'eligible_manifest.csv')==MANIFEST_HASH
 lock=js(AUDIT/'COHORT_LOCK.json');assert lock['manifest_sha256']==MANIFEST_HASH
 assert sha(AUDIT/'priority_pair_review.csv')==lock['review_sha256']
 assert sha(MODELS/'FIT_COMPLETE.json')==FIT_HASH
 fitted=js(MODELS/'FIT_COMPLETE.json')['files']
 for f,h in fitted.items():assert sha(MODELS/f)==h,('Model artifact changed',f)
 rows=readcsv(AUDIT/'eligible_manifest.csv');assert len(rows)==1794 and len({r['group_id'] for r in rows})==1794
 assert sum(int(r['label']) for r in rows)==200
 OUT.mkdir(exist_ok=True)
 if (OUT/'results').exists():raise FileExistsError('Results exist: preserve them; do not rerun evaluation.')
 environment={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0)}
 for name in ['torch','transformers','numpy','scikit-learn','pillow']:environment[name]=importlib.metadata.version(name)
 protocol={'manifest_sha256':MANIFEST_HASH,'fit_complete_sha256':FIT_HASH,'code_sha256':sha(__file__),'backbone':'microsoft/rad-dino','revision':REV,'thresholds':THRESHOLDS,'bootstrap_repeats':10000,'bootstrap_seed':20260924,'environment':environment,'scope':'exploratory dataset-label external benchmark; no tuning; no clinical review','preprocessing':'AutoImageProcessor use_fast=False; EXIF transpose; RGB; no augmentation; float32','uncertainty':'unique retained image/pixel groups, not verified patients; conditional on fitted models and fixed thresholds','limitations':lock['limitations']}
 if (OUT/'protocol.json').exists():assert js(OUT/'protocol.json')==protocol,'Resume environment/code differs'
 else:writejson(OUT/'protocol.json',protocol)
 shutil.copyfile(AUDIT/'eligible_manifest.csv',OUT/'eligible_manifest.csv')
 shutil.copyfile(__file__,OUT/'evaluate_tbx11k.py')
 archive=Path('/content/TBX11K.zip')
 if not archive.exists():
  import gdown
  target=gdown.download(id='1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR',output=str(archive)+'.partial',quiet=False)
  assert target is not None,'Official archive download failed'
  assert sha(target)==lock['archive_sha256'],'Archive differs from audited release'
  Path(target).rename(archive)
 assert sha(archive)==lock['archive_sha256']
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.set_num_threads(2)
 processor=AutoImageProcessor.from_pretrained('microsoft/rad-dino',revision=REV,use_fast=False)
 backbone=AutoModel.from_pretrained('microsoft/rad-dino',revision=REV,use_safetensors=True).eval().cuda()
 for param in backbone.parameters():param.requires_grad_(False)
 processor.save_pretrained(OUT/'processor')
 cache=OUT/'features_partial.npz';x=np.zeros((len(rows),768),dtype=np.float32);completed=0
 if cache.exists():
  with np.load(cache,allow_pickle=False) as c:
   x=c['x'];completed=int(c['completed']);assert str(c['manifest_sha256'])==MANIFEST_HASH
  assert x.shape==(1794,768) and 0<=completed<=len(rows) and np.isfinite(x).all()
 with zipfile.ZipFile(archive) as z,torch.inference_mode():
  for start in range(completed,len(rows),4):
   images=[]
   for r in rows[start:start+4]:
    data=z.read('TBX11K/imgs/'+r['path']);assert hashlib.sha256(data).hexdigest()==r['sha256']
    with Image.open(io.BytesIO(data)) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
   batch=processor(images=images,return_tensors='pt').to('cuda')
   v=backbone(**batch).last_hidden_state[:,0,:].float().cpu().numpy()
   assert v.shape==(len(images),768) and np.isfinite(v).all()
   x[start:start+len(images)]=v;completed=start+len(images)
   if completed%128==0 or completed==len(rows):
    temporary=cache.with_suffix('.partial')
    with temporary.open('wb') as f:np.savez(f,x=x,completed=completed,manifest_sha256=MANIFEST_HASH)
    temporary.replace(cache);print('Encoded and saved',completed,'/',len(rows),flush=True)
 del backbone;torch.cuda.empty_cache()
 probabilities={}
 for name,t in THRESHOLDS.items():
  assert js(MODELS/f'{name}.json')['threshold']==t
  if name=='linear':
   with np.load(MODELS/'linear.npz',allow_pickle=False) as m:
    logits=(((x-m['mean'])/m['scale'])@m['coef'].T+m['intercept']).ravel()
   p=1/(1+np.exp(-np.clip(logits,-700,700)))
  else:
   import torch.nn as nn
   model=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,2))
   model.load_state_dict(torch.load(MODELS/f'{name}.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
   with torch.inference_mode():p=model(torch.from_numpy(x)).softmax(-1)[:,1].numpy()
  assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();probabilities[name]=p
 # No results directory is created until every image/model prediction is available.
 results=OUT/'results';results.mkdir()
 y=np.array([int(r['label']) for r in rows]);groups=np.array([r['original_group'] for r in rows])
 cat=np.array([r['annotation_category_ids'] for r in rows])
 predictions=[{'image_id':r['image_id'],'group_id':r['group_id'],'label':int(r['label']),'original_group':r['original_group'],'annotation_category_ids':r['annotation_category_ids'],**{n+'_score':float(p[i]) for n,p in probabilities.items()},**{n+'_prediction':int(p[i]>=THRESHOLDS[n]) for n,p in probabilities.items()}} for i,r in enumerate(rows)]
 writecsv(results/'predictions.csv',predictions)
 summary=[];sampled={n:default_samples() for n in THRESHOLDS};rng=np.random.default_rng(20260924)
 for offset in range(0,10000,100):
  weights=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=100)
  valid=(weights[:,y==1].sum(1)>0)&(weights[:,y==0].sum(1)>0);weights=weights[valid]
  for name,p in probabilities.items():
   values=weighted_metrics(y,p,THRESHOLDS[name],weights)
   for k in sampled[name]:sampled[name][k].extend(values[k].tolist())
  if (offset+100)%2000==0:print('Paired bootstrap',offset+100,'/ 10000',flush=True)
 for name,p in probabilities.items():
  def point(mask,t):return {k:float(v[0]) for k,v in weighted_metrics(y[mask],p[mask],t,np.ones(sum(mask))).items()}
  m=point(np.ones(len(y),dtype=bool),THRESHOLDS[name]);ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in sampled[name].items()}
  subgroups={}
  for g in ['health','sick']:
   mask=groups==g;fp=int(sum(p[mask]>=THRESHOLDS[name]));subgroups[g]={'n':int(sum(mask)),'false_positives':fp,'specificity':1-fp/sum(mask)}
  for c in ['1','2','1;2']:
   mask=cat==c;tp=int(sum(p[mask]>=THRESHOLDS[name]));subgroups['annotation_'+c]={'n':int(sum(mask)),'true_positives':tp,'sensitivity':tp/sum(mask)}
  record={'model':name,'threshold':THRESHOLDS[name],'metrics':m,'bootstrap_95ci':ci,'valid_bootstrap_samples':len(sampled[name]['auroc']),'subgroups':subgroups,'fixed_0_5_metrics':point(np.ones(len(y),dtype=bool),.5),'category_1_secondary':point(cat!='2',THRESHOLDS[name])}
  writejson(results/f'{name}_metrics.json',record)
  summary.append({'model':name,'threshold':THRESHOLDS[name],**m,'healthy_specificity':float(subgroups['health']['specificity']),'sick_specificity':float(subgroups['sick']['specificity'])})
  print(name,json.dumps(summary[-1]),flush=True)
 paired=[]
 for name in list(THRESHOLDS)[1:]:
  delta=np.array(sampled[name]['auroc'])-np.array(sampled['linear']['auroc'])
  pointdiff=summary[list(THRESHOLDS).index(name)]['auroc']-summary[0]['auroc']
  paired.append({'comparison':name+' minus linear','auroc_difference':pointdiff,'ci_low':float(np.quantile(delta,.025)),'ci_high':float(np.quantile(delta,.975))})
 writecsv(results/'summary.csv',summary);writecsv(results/'paired_auroc_comparisons.csv',paired)
 writejson(results/'EVALUATION_COMPLETE.json',{'protocol_sha256':sha(OUT/'protocol.json'),'manifest_sha256':MANIFEST_HASH,'feature_sha256':sha(cache),'n':len(rows),'models':list(THRESHOLDS),'results_sha256':{p.name:sha(p) for p in results.iterdir() if p.is_file()},'limitations':lock['limitations']})
 print('COMPLETED: saved all four models, paired intervals and subgroup results to',results,flush=True)
def default_samples():return {k:[] for k in ['auroc','average_precision','sensitivity','specificity','brier']}
if __name__=='__main__':main()
