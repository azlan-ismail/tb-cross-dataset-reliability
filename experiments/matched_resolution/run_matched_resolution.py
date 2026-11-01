"""Exploratory comparator extension. Never changes earlier study artifacts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import csv,json,hashlib,zipfile,io,random,importlib.metadata,shutil
from pathlib import Path
import numpy as np
BASE=Path('/content/drive/MyDrive/PulmoVision_Journal')
OUT=BASE/'Matched_resolution_v1'
STUDY=BASE/'PV_classifier_v2_reviewed/study'
AUDIT=BASE/'TBX11K_audit_v1'
DEVHASH='348bf147ce8c8404afc0fcfd8faf439202f913bd6628e404fee91b9cef04160c'
TBXHASH='629461acc1b839740264e0ba0a17b7e2baf15201b187a4b2da16a6544d3e6d6c'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def csvsave(p,rs):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def seed(s):
 import torch
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.set_num_threads(2)
def threshold(y,p):
 from sklearn.metrics import roc_curve
 fpr,tpr,t=roc_curve(y,p,drop_intermediate=False)
 eligible=np.flatnonzero((tpr>=.9)&np.isfinite(t));i=min(eligible,key=lambda i:(fpr[i],-t[i]));return float(t[i])
def metrics(y,p,t):
 from sklearn.metrics import roc_auc_score,average_precision_score
 z=p>=t;tp=int(sum(z&(y==1)));fp=int(sum(z&(y==0)));tn=int(sum(~z&(y==0)));fn=int(sum(~z&(y==1)))
 return dict(auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),brier=float(np.mean((p-y)**2)),tp=tp,fp=fp,tn=tn,fn=fn)
def result(name,rs,p,t):
 folder=OUT/'results';folder.mkdir(exist_ok=True)
 dest=folder/(name+'_predictions.csv')
 if dest.exists():raise FileExistsError(dest)
 csvsave(dest,[dict(image_id=r['image_id'],group_id=r['group_id'],label=r['label'],cohort=r['cohort'],original_group=r.get('original_group',''),score=float(v),threshold=t,prediction=int(v>=t)) for r,v in zip(rs,p)])
 y=np.array([int(r['label']) for r in rs]);cohort=np.array([r['cohort'] for r in rs]);summary={}
 for c in ['montgomery','tbx11k']:
  ix=cohort==c;summary[c]=metrics(y[ix],p[ix],t)
 for g in ['health','sick']:
  ix=np.array([r.get('original_group')==g for r in rs]);summary[g]={'n':int(sum(ix)),'false_positives':int(sum(p[ix]>=t)),'specificity':float(np.mean(p[ix]<t))}
 save(folder/(name+'_metrics.json'),summary);print(name,json.dumps(summary),flush=True)

def main():
 import torch
 from PIL import Image,ImageOps
 from transformers import AutoModel
 from torchvision import transforms as T
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import roc_auc_score
 assert torch.cuda.is_available()
 assert sha(STUDY/'manifest.csv')==DEVHASH and sha(AUDIT/'eligible_manifest.csv')==TBXHASH
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETE.json').exists():
  receipt=js(OUT/'COMPLETE.json')
  for f,h in receipt['files'].items():assert sha(OUT/f)==h
  print('MATCHED INFERENCE ALREADY COMPLETE');return
 revisions={'rad':('microsoft/rad-dino','110cbc18d5133582e320b43d53bf5c44e410c936'),'dino':('facebook/dinov2-base','f9e44c814b77203eaa57a6bdbbd535f21ede1415')}
 protocol={'scope':'Exploratory 2x2 frozen-encoder comparison after earlier external outcomes were seen. No target tuning. Not a causal pretraining study.', 'code_sha256':sha(__file__),'development_manifest':DEVHASH,'tbx_manifest':TBXHASH,'revisions':{k:list(v) for k,v in revisions.items()},'resolutions':[224,518],'preprocessing':'EXIF transpose RGB; PIL bicubic shortest-edge resize to R then center crop R; ToTensor; ImageNet mean .485 .456 .406 and std .229 .224 .225; no augmentation','feature':'768-dimensional CLS; frozen eval float32 eager attention; batch4','fit':'train-only StandardScaler; balanced LogisticRegression C1 max_iter2000 seed42','threshold':'Shenzhen validation sensitivity >= .90, minimum FPR then largest threshold; fixed externally','bootstrap':'10000 paired image draws seed20260925; percentile95%; no multiplicity correction; conditional on fitted models','environment':{k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','numpy','scikit-learn','pillow']},'gpu':torch.cuda.get_device_name(0)}
 proto=OUT/'protocol.json'
 if proto.exists():assert js(proto)==protocol,'Protocol/environment changed'
 else:save(proto,protocol)
 shutil.copyfile(__file__,OUT/'run_matched_resolution.py')
 dev=rows(STUDY/'manifest.csv');tbx=rows(AUDIT/'eligible_manifest.csv')
 assert len(dev)==800 and len(tbx)==1794
 for r in dev:r['cohort']=r['source']
 for r in tbx:r.update(cohort='tbx11k',split='external')
 allrows=dev+tbx;train=np.array([r['split']=='train' for r in allrows]);val=np.array([r['split']=='validation' for r in allrows]);test=~(train|val)
 assert (sum(train),sum(val),sum(test))==(529,133,1932)
 y=np.array([int(r['label']) for r in allrows]);testrows=[r for r,k in zip(allrows,test) if k]
 # Restore only manifest-selected images from the saved archives, checking every image's hash.
 backup=BASE/'validated_source_archives';receipt=js(backup/'archive_receipt.json');archives={}
 for source in ['shenzhen','montgomery']:
  archive=backup/(source+'.zip');assert sha(archive)==receipt[source]['sha256'];archives[source]=zipfile.ZipFile(archive)
 archive=Path('/content/TBX11K.zip')
 expected=js(AUDIT/'COHORT_LOCK.json')['archive_sha256']
 if not archive.exists():
  import gdown
  temp=str(archive)+'.partial';assert gdown.download(id='1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR',output=temp,quiet=True)
  assert sha(temp)==expected;Path(temp).rename(archive)
 assert sha(archive)==expected;archives['tbx11k']=zipfile.ZipFile(archive)
 def read_image(r):
  source=r['cohort'];member='TBX11K/imgs/'+r['path'] if source=='tbx11k' else r['path'].split('/',1)[1]
  data=archives[source].read(member);assert hashlib.sha256(data).hexdigest()==r['sha256'],r['image_id']
  with Image.open(io.BytesIO(data)) as im:return ImageOps.exif_transpose(im).convert('RGB')
 seed(42)
 key=sha(proto);modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
 cropdir=Path('/content/pv_matched_'+key[:16]);cropdir.mkdir(exist_ok=True)
 geometries={r:T.Compose([T.Resize(r,interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.CenterCrop(r)]) for r in [224,518]}
 # Prepare source-verified, shared pixels once. No labels enter preprocessing.
 for i,r in enumerate(allrows):
  targets={size:cropdir/(str(i)+'_'+str(size)+'.png') for size in geometries}
  if all(p.exists() for p in targets.values()):continue
  im=read_image(r)
  for size,p in targets.items():
   tmp=p.with_suffix('.partial');geometries[size](im).save(tmp,format='PNG');tmp.replace(p)
  if (i+1)%128==0:print('Shared images prepared',i+1,'/',len(allrows),flush=True)
 tensor=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 for size in [224,518]:
  for encoder,(repo,rev) in revisions.items():
   name=encoder+'_'+str(size);done=OUT/(name+'_COMPLETE.json')
   if done.exists():
    for f,h in js(done)['files'].items():assert sha(OUT/f)==h
    continue
   cache=OUT/(name+'_features.npz');x=np.zeros((len(allrows),768),np.float32);completed=0
   if cache.exists():
    with np.load(cache) as z:
     assert str(z['protocol'])==key;x=z['x'];completed=int(z['completed'])
    assert x.shape==(2594,768) and 0<=completed<=2594 and np.isfinite(x).all()
   if completed<len(allrows):
    model=AutoModel.from_pretrained(repo,revision=rev,use_safetensors=True,attn_implementation='eager').eval().cuda()
    assert model.config.hidden_size==768 and model.config.patch_size==14
    model.config.save_pretrained(modeldir/(name+'_config'))
    with torch.inference_mode():
     for i in range(completed,len(allrows),4):
      ims=[]
      for j in range(i,min(i+4,len(allrows))):
       with Image.open(cropdir/(str(j)+'_'+str(size)+'.png')) as im:ims.append(tensor(im))
      values=model(pixel_values=torch.stack(ims).cuda()).last_hidden_state[:,0,:].float().cpu().numpy()
      assert np.isfinite(values).all();x[i:i+len(ims)]=values;completed=i+len(ims)
      if completed%128==0 or completed==len(allrows):
       with open(str(cache)+'.partial','wb') as f:np.savez(f,x=x,completed=completed,protocol=key)
       Path(str(cache)+'.partial').replace(cache);print(name,'encoded',completed,'/',len(allrows),flush=True)
    del model;torch.cuda.empty_cache()
   scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42).fit(scaler.transform(x[train]),y[train]);assert max(clf.n_iter_)<2000
   pv=clf.predict_proba(scaler.transform(x[val]))[:,1];cut=threshold(y[val],pv)
   np.savez(modeldir/(name+'.npz'),mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
   save(modeldir/(name+'.json'),{'threshold':cut,'validation_auroc':float(roc_auc_score(y[val],pv))})
   csvsave(modeldir/(name+'_validation.csv'),[dict(image_id=r['image_id'],label=r['label'],score=float(p)) for r,p in zip([r for r,k in zip(allrows,val) if k],pv)])
   scores=clf.predict_proba(scaler.transform(x[test]))[:,1]
   # An interrupted result write is validated before its completion receipt is added.
   pred=OUT/'results'/(name+'_predictions.csv')
   if pred.exists():
    saved=rows(pred);assert len(saved)==len(scores) and all(r['image_id']==s['image_id'] and float(r['score'])==float(p) for r,s,p in zip(saved,testrows,scores))
    assert (OUT/'results'/(name+'_metrics.json')).exists(),'Partial result: inspect before recovery'
   else:result(name,testrows,scores,cut)
   paths=[p for p in (OUT/'results').glob(name+'_*')]+list(modeldir.glob(name+'.*'))+list(modeldir.glob(name+'_validation.csv'))+[cache]
   save(done,{'protocol_sha256':key,'files':{str(p.relative_to(OUT)):sha(p) for p in paths}})
 files=[proto]+list(OUT.glob('*_COMPLETE.json'))+list((OUT/'results').glob('*'))+list(modeldir.rglob('*'))
 save(OUT/'COMPLETE.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in files if p.is_file()}})
 print('MATCHED INFERENCE COMPLETE',flush=True)
if __name__=='__main__':main()
