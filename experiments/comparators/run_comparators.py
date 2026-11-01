"""Exploratory comparator extension. Never changes earlier study artifacts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import csv,json,hashlib,zipfile,io,random,importlib.metadata,shutil
from pathlib import Path
import numpy as np
BASE=Path('/content/drive/MyDrive/PulmoVision_Journal')
OUT=BASE/'Comparator_extension_v1'
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
 from huggingface_hub import HfApi
 from transformers import AutoImageProcessor,AutoModel
 from torchvision.models import densenet121,DenseNet121_Weights
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import roc_auc_score
 assert torch.cuda.is_available(),'Choose T4 GPU'
 assert sha(STUDY/'manifest.csv')==DEVHASH
 assert sha(AUDIT/'eligible_manifest.csv')==TBXHASH
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETE.json').exists():print('Comparator extension already complete; preserved.');return
 env={k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','numpy','scikit-learn','pillow']}
 protofile=OUT/'protocol.json'
 revision=js(protofile)['dinov2_revision'] if protofile.exists() else HfApi().model_info('facebook/dinov2-base').sha
 protocol={'status':'exploratory extension after prior external outcomes seen','code_sha256':sha(__file__),'development_manifest':DEVHASH,'tbx_manifest':TBXHASH,'dinov2_revision':revision,'dinov2':'frozen CLS + train-only StandardScaler + balanced logistic C=1','dense_weights':'IMAGENET1K_V1','dense_training':'full fine-tuning, 20 epochs, seeds42/43/44, AdamW lr1e-4 wd1e-4 batch8 cosine schedule; no augmentation','selection':'earliest highest Shenzhen validation AUROC; sensitivity>=.90 then maximum specificity then highest threshold','preprocessing':'each model official processor/weights transforms, EXIF transpose, RGB; pipelines differ in input resolution','environment':env,'gpu':torch.cuda.get_device_name(0),'no_target_tuning':True}
 if protofile.exists():assert js(protofile)==protocol,'Code/environment changed; preserve run and investigate'
 else:save(protofile,protocol)
 shutil.copyfile(__file__,OUT/'run_comparators.py')
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
 modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
 # DINOv2 uses its own documented preprocessing; not a pretraining-only ablation.
 dinoresult=OUT/'results/dinov2_linear_metrics.json'
 if not dinoresult.exists():
  processor=AutoImageProcessor.from_pretrained('facebook/dinov2-base',revision=revision,use_fast=False)
  processor.save_pretrained(OUT/'dinov2_processor')
  cache=OUT/'dinov2_features.npz';x=np.zeros((len(allrows),768),np.float32);completed=0
  if cache.exists():
   with np.load(cache) as z:x=z['x'];completed=int(z['completed'])
   assert x.shape==(2594,768) and 0<=completed<=2594 and np.isfinite(x).all()
  model=AutoModel.from_pretrained('facebook/dinov2-base',revision=revision,use_safetensors=True).eval().cuda()
  with torch.inference_mode():
   for i in range(completed,len(allrows),8):
    ims=[read_image(r) for r in allrows[i:i+8]];x[i:i+len(ims)]=model(**processor(images=ims,return_tensors='pt').to('cuda')).last_hidden_state[:,0,:].float().cpu().numpy();completed=i+len(ims)
    if completed%128==0 or completed==len(allrows):
     with open(str(cache)+'.partial','wb') as f:np.savez(f,x=x,completed=completed)
     Path(str(cache)+'.partial').replace(cache);print('DINOv2 encoded',completed,'/',len(allrows),flush=True)
  del model;torch.cuda.empty_cache()
  scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42).fit(scaler.transform(x[train]),y[train]);assert max(clf.n_iter_)<2000
  pv=clf.predict_proba(scaler.transform(x[val]))[:,1];t=threshold(y[val],pv)
  np.savez(modeldir/'dinov2_linear.npz',mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
  save(modeldir/'dinov2_linear.json',{'threshold':t,'validation_auroc':float(roc_auc_score(y[val],pv))})
  csvsave(modeldir/'dinov2_validation.csv',[{'image_id':r['image_id'],'label':r['label'],'score':float(v)} for r,v in zip([r for r,k in zip(allrows,val) if k],pv)])
  result('dinov2_linear',testrows,clf.predict_proba(scaler.transform(x[test]))[:,1],t)
  del x
 weights=DenseNet121_Weights.IMAGENET1K_V1;transform=weights.transforms()
 save(OUT/'dense_preprocessing.json',{'weights_url':weights.url,'transform':str(transform)})
 # Cache deterministic 224-pixel tensors once; no target images enter fitting.
 tensorfile=Path('/content/pv_dense_inputs.pt')
 key=sha(protofile)
 if tensorfile.exists():
  cached=torch.load(tensorfile,weights_only=True);assert cached['protocol']==key;inputs=cached['inputs'];del cached
 else:
  inputs=torch.empty((len(allrows),3,224,224),dtype=torch.float32)
  for i,r in enumerate(allrows):
   inputs[i]=transform(read_image(r))
   if (i+1)%256==0:print('DenseNet prepared',i+1,'/',len(allrows),flush=True)
  torch.save({'protocol':key,'inputs':inputs},tensorfile)
 def predict(model,ix):
  model.eval();scores=[]
  with torch.inference_mode():
   for j in range(0,len(ix),16):scores.extend(model(inputs[ix[j:j+16]].cuda()).softmax(-1)[:,1].cpu().numpy())
  return np.array(scores)
 for s in [42,43,44]:
  name='densenet121_seed'+str(s)
  if (OUT/'results'/ (name+'_metrics.json')).exists():continue
  seed(s);model=densenet121(weights=weights);model.classifier=torch.nn.Linear(1024,2);model.cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
  classweights=1/np.bincount(y[train]);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(classweights/classweights.sum(),dtype=torch.float32,device='cuda'))
  generator=torch.Generator().manual_seed(s);trainix=np.flatnonzero(train);valix=np.flatnonzero(val);testix=np.flatnonzero(test)
  loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(trainix)),batch_size=8,shuffle=True,generator=generator)
  logs=[];best=-1.;start=1;last=modeldir/(name+'_resume.pt');bestpath=modeldir/(name+'.pt');bestepoch=None
  if last.exists():
   state=torch.load(last,weights_only=True,map_location='cpu');model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler']);generator.set_state(state['generator']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);logs=state['logs'];best=state['best'];bestepoch=state['bestepoch'];start=state['epoch']+1
  for epoch in range(start,21):
   model.train();loss_sum=0.
   for (ix,) in loader:
    opt.zero_grad(set_to_none=True);loss=lossfn(model(inputs[ix].cuda()),torch.tensor(y[ix.numpy()],device='cuda'));loss.backward();opt.step();loss_sum+=loss.item()*len(ix)
   schedule.step();pv=predict(model,valix);auc=float(roc_auc_score(y[val],pv));logs.append({'epoch':epoch,'train_loss':float(loss_sum/sum(train)),'validation_auroc':auc})
   if auc>best:best=auc;bestepoch=epoch;torch.save(model.state_dict(),bestpath)
   state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'generator':generator.get_state(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'logs':logs,'best':best,'bestepoch':bestepoch,'epoch':epoch}
   torch.save(state,str(last)+'.partial');Path(str(last)+'.partial').replace(last)
   print(name,'epoch',epoch,'validation AUROC',round(auc,4),flush=True)
  model.load_state_dict(torch.load(bestpath,map_location='cuda',weights_only=True));pv=predict(model,valix);t=threshold(y[val],pv)
  save(modeldir/(name+'.json'),{'threshold':t,'selected_epoch':bestepoch,'validation_auroc':best});csvsave(modeldir/(name+'_training.csv'),logs)
  csvsave(modeldir/(name+'_validation.csv'),[{'image_id':allrows[i]['image_id'],'label':int(y[i]),'score':float(v)} for i,v in zip(valix,pv)])
  result(name,testrows,predict(model,testix),t);del model,opt,schedule,state;torch.cuda.empty_cache()
 save(OUT/'COMPLETE.json',{'protocol_sha256':sha(protofile),'results':{p.name:sha(p) for p in (OUT/'results').iterdir() if p.is_file()},'model_artifacts':{p.name:sha(p) for p in modeldir.iterdir() if p.is_file() and 'resume' not in p.name},'note':'Point estimates and predictions complete. Paired intervals against original models are a subsequent analysis, not yet computed here.'})
 print('COMPARATOR INFERENCE COMPLETE',flush=True)
if __name__=='__main__':main()
