"""Exploratory comparator extension. Never changes earlier study artifacts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import csv,json,hashlib,zipfile,io,random,importlib.metadata,shutil
from pathlib import Path
import numpy as np
BASE=Path('/content/drive/MyDrive/PulmoVision_Journal')
OUT=BASE/'DINO_preprocessing_audit_v1'
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
 from transformers import AutoModel,AutoImageProcessor
 from torchvision import transforms as T
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 assert torch.cuda.is_available()
 assert sha(STUDY/'manifest.csv')==DEVHASH and sha(AUDIT/'eligible_manifest.csv')==TBXHASH
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETE.json').exists():print('AUDIT ALREADY COMPLETE');return
 repo='facebook/dinov2-base';rev='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
 protocol={'code_sha256':sha(__file__),'revision':rev,'dev':DEVHASH,'tbx':TBXHASH,'conditions':['hf256_crop224','hf224_crop224','tv256_crop224'],'scope':'Post hoc diagnostic replication, not target optimisation. Same eager float32 model batch4, frozen CLS, train-only scaling balanced logistic C1 seed42; thresholds validation-only.', 'environment':{k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','numpy','scikit-learn','pillow']}}
 if (OUT/'protocol.json').exists():assert js(OUT/'protocol.json')==protocol
 else:save(OUT/'protocol.json',protocol)
 shutil.copyfile(__file__,OUT/'audit_dino_preprocessing.py');key=sha(OUT/'protocol.json')
 print('Protocol saved; verifying archives',flush=True)
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
 processor=AutoImageProcessor.from_pretrained(repo,revision=rev,use_fast=False)
 assert processor.size['shortest_edge']==256 and processor.crop_size=={'height':224,'width':224}
 processor.save_pretrained(OUT/'processor')
 transform=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 oldroot=BASE/'Comparator_extension_v1';matched=BASE/'Matched_resolution_v1'
 oldreceipt=js(oldroot/'COMPLETE.json');matchreceipt=js(matched/'dino_224_COMPLETE.json')
 for f,h in oldreceipt['results'].items():assert sha(oldroot/'results'/f)==h
 for f,h in matchreceipt['files'].items():assert sha(matched/f)==h
 oldx=np.load(oldroot/'dinov2_features.npz')['x'];matchx=np.load(matched/'dino_224_features.npz')['x']
 assert oldx.shape==matchx.shape==(2594,768)
 model=AutoModel.from_pretrained(repo,revision=rev,use_safetensors=True,attn_implementation='eager').eval().cuda()
 names=protocol['conditions'];cache=OUT/'features.npz';xs={n:np.zeros((2594,768),np.float32) for n in names};completed=0;pixelmax=0.
 if cache.exists():
  with np.load(cache) as z:
   assert str(z['key'])==key;completed=int(z['completed']);pixelmax=float(z['pixelmax']);xs={n:z[n] for n in names}
 with torch.inference_mode():
  for i in range(completed,len(allrows),4):
   ims=[read_image(r) for r in allrows[i:i+4]]
   inputs={'hf256_crop224':processor(images=ims,return_tensors='pt')['pixel_values'],'hf224_crop224':processor(images=ims,size={'shortest_edge':224},return_tensors='pt')['pixel_values'],'tv256_crop224':torch.stack([transform(im) for im in ims])}
   pixelmax=max(pixelmax,float((inputs['hf256_crop224']-inputs['tv256_crop224']).abs().max()))
   for n in names:xs[n][i:i+len(ims)]=model(pixel_values=inputs[n].cuda()).last_hidden_state[:,0,:].cpu().numpy()
   completed=i+len(ims)
   if completed%128==0 or completed==len(allrows):
    with open(str(cache)+'.partial','wb') as f:np.savez(f,**xs,completed=completed,key=key,pixelmax=pixelmax)
    Path(str(cache)+'.partial').replace(cache);print('Three preprocessing conditions:',completed,'/',len(allrows),flush=True)
 summary=[];replication={};modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
 for n,x in xs.items():
  assert np.isfinite(x).all()
  scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42).fit(scaler.transform(x[train]),y[train]);assert max(clf.n_iter_)<2000
  cut=threshold(y[val],clf.predict_proba(scaler.transform(x[val]))[:,1]);scores=clf.predict_proba(scaler.transform(x[test]))[:,1]
  np.savez(modeldir/(n+'.npz'),mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_,threshold=cut)
  if not (OUT/'results'/(n+'_predictions.csv')).exists():result(n,testrows,scores,cut)
  for cohort in ['montgomery','tbx11k']:
   mask=np.array([r['cohort']==cohort for r in testrows]);summary.append(dict(condition=n,cohort=cohort,threshold=cut,**metrics(y[test][mask],scores[mask],cut)))
  reference=oldx if n=='hf256_crop224' else matchx if n=='hf224_crop224' else xs['hf256_crop224']
  replication[n]={'reference':'original features' if n=='hf256_crop224' else 'matched224 features' if n=='hf224_crop224' else 'HF256 current features','feature_max_abs_difference':float(np.max(np.abs(x-reference))),'feature_mean_abs_difference':float(np.mean(np.abs(x-reference)))}
 csvsave(OUT/'summary.csv',summary);save(OUT/'replication.json',{'pixel_max_abs_hf256_vs_tv256':pixelmax,'features':replication})
 lines=['# DINOv2 preprocessing diagnostic','','Post hoc diagnostic, with source-only fitting and threshold selection. All conditions feed 224x224 tensors. HF denotes the saved checkpoint processor; TV denotes torchvision.','', '| Condition | Cohort | AUROC | Sensitivity | Specificity |','|---|---|---|---|---|']
 for r in summary:lines.append(f"| {r['condition']} | {r['cohort']} | {r['auroc']:.6f} | {r['sensitivity']:.6f} | {r['specificity']:.6f} |")
 lines+=['','## Replication diagnostics','',json.dumps({'pixel_max_abs_hf256_vs_tv256':pixelmax,'features':replication},indent=2),'','Compare HF256 versus HF224 to assess resize/crop geometry under one processor and inference implementation. Compare HF256 versus TV256 to assess implementation differences. Compare regenerated features with prior caches to detect additional numerical/runtime differences. Do not attribute the whole earlier score gap to cropping unless replication agrees. No target-selected replacement model is declared.']
 (OUT/'Preprocessing_Audit.md').write_text('\n'.join(lines))
 save(OUT/'COMPLETE.json',{'files':{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()}})
 print('PREPROCESSING AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
