"""Audit the official validation subset without fitting or evaluating any model."""
from pathlib import Path
import csv, json, hashlib, zipfile, io
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn

BASE=Path('/content/drive/MyDrive/PulmoVision_Journal')
OUT=BASE/'TBX11K_audit_v1'
ARCHIVE=Path('/content/TBX11K.zip')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def savecsv(name,rows,fields=None):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def signature(data):
    with Image.open(io.BytesIO(data)) as image:
        im=ImageOps.exif_transpose(image).convert('L');arr=np.asarray(im)
        pix=hashlib.sha256(str(im.size).encode()+arr.tobytes()).hexdigest()
        small=np.asarray(im.resize((9,8),Image.Resampling.LANCZOS))
        dh=int.from_bytes(np.packbits((small[:,1:]>small[:,:-1]).ravel()).tobytes(),'big')
        dct=dctn(np.asarray(im.resize((32,32),Image.Resampling.LANCZOS),dtype=float),norm='ortho')[:8,:8]
        ph=int.from_bytes(np.packbits((dct>np.median(dct.ravel()[1:])).ravel()).tobytes(),'big')
        thumb=np.asarray(im.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)
        view=im.resize((256,256),Image.Resampling.LANCZOS).copy()
        return {'sha256':hashlib.sha256(data).hexdigest(),'pixel_sha256':pix,'dhash':f'{dh:016x}','phash':f'{ph:016x}','width':im.width,'height':im.height},thumb,view
def normalise(a):
    a=a.reshape(len(a),-1);a=a-a.mean(1,keepdims=True)
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-10)

OUT.mkdir(exist_ok=True)
if (OUT/'AUDIT_COMPLETE.json').exists():raise FileExistsError('Preserve completed audit')
assert ARCHIVE.stat().st_size==3306757175
protocol=json.loads((BASE/'PV_classifier_v2_reviewed/study/protocol.json').read_text())
manifest_path=BASE/'PV_classifier_v2_reviewed/study/manifest.csv'
assert sha(manifest_path)==protocol['manifest_sha256']
with manifest_path.open() as f:old=list(csv.DictReader(f))
assert len(old)==800
with zipfile.ZipFile(ARCHIVE) as z:
    names=z.namelist();val=z.read('TBX11K/lists/TBX11K_val.txt').decode().splitlines()
    ann=json.loads(z.read('TBX11K/annotations/json/TBX11K_val.json'))
    assert len(val)==len(set(val))==1800
    assert Counter(p.split('/')[0] for p in val)=={'tb':200,'health':800,'sick':800}
    assert not any('extra/' in p for p in val)
    images={r['file_name']:r for r in ann['images']}
    assert len(images)==len(ann['images'])==1800 and set(images)==set(val)
    cats=defaultdict(set)
    for r in ann['annotations']:cats[r['image_id']].add(r['category_id'])
    rows=[];thumbs=[];views=[]
    for i,p in enumerate(val):
        sig,thumb,view=signature(z.read('TBX11K/imgs/'+p));im=images[p]
        assert (sig['width'],sig['height'])==(im['width'],im['height'])
        category_ids=sorted(cats[im['id']]);label=int(p.startswith('tb/'))
        assert (label==1 and category_ids and set(category_ids)<=set([1,2])) or (label==0 and not category_ids)
        row={'image_id':'tbx11k:'+p,'path':p,'original_group':p.split('/')[0],'label':label,'annotation_category_ids':';'.join(map(str,category_ids)),'patient_id':'','patient_identity_status':'not supplied; paper reports one image per person',**sig}
        rows.append(row);thumbs.append(thumb);views.append(view)
        if (i+1)%300==0:print('TBX11K signatures',i+1,flush=True)
savecsv('candidate_inventory.csv',rows)
receipt=json.loads((BASE/'validated_source_archives/archive_receipt.json').read_text())
old_thumbs=[];old_views=[];old_rows=[]
for source in ['montgomery','shenzhen']:
    archive=BASE/'validated_source_archives'/f'{source}.zip'
    assert sha(archive)==receipt[source]['sha256']
    with zipfile.ZipFile(archive) as z:
        for row in [r for r in old if r['source']==source]:
            name='/'.join(Path(row['path']).parts[1:]);data=z.read(name)
            sig,thumb,view=signature(data);assert sig['sha256']==row['sha256']
            old_rows.append({**row,'phash':sig['phash']});old_thumbs.append(thumb);old_views.append(view)
    print('Reference signatures',source,len(old_rows),flush=True)

A=normalise(np.array(thumbs));B=normalise(np.array(old_thumbs))
corr=A@B.T;flipcorr=normalise(np.array(thumbs)[:,:,::-1])@B.T
bestcorr=np.maximum(corr,flipcorr)
candidates=[];nearest=[];exact=[]
for i,r in enumerate(rows):
    j=int(np.argmax(bestcorr[i]));nearest.append({'image_id':r['image_id'],'reference_id':old_rows[j]['image_id'],'correlation':float(bestcorr[i,j]),'horizontal_flip':bool(flipcorr[i,j]>corr[i,j])})
    for j,o in enumerate(old_rows):
        dh=(int(r['dhash'],16)^int(o['dhash'],16)).bit_count();ph=(int(r['phash'],16)^int(o['phash'],16)).bit_count()
        same=r['sha256']==o['sha256'] or r['pixel_sha256']==o['pixel_sha256']
        if same:exact.append({'image_id':r['image_id'],'reference_id':o['image_id']})
        if same or dh<=4 or ph<=6 or bestcorr[i,j]>=.95:
            candidates.append({'image_id':r['image_id'],'reference_id':o['image_id'],'exact':same,'dhash_distance':dh,'phash_distance':ph,'correlation':float(bestcorr[i,j]),'horizontal_flip':bool(flipcorr[i,j]>corr[i,j]),'review_priority':bool(same or bestcorr[i,j]>=.95 or (ph<=6 and bestcorr[i,j]>=.90))})
fields=['image_id','reference_id','exact','dhash_distance','phash_distance','correlation','horizontal_flip','review_priority']
savecsv('cross_source_candidates.csv',candidates,fields)
savecsv('nearest_reference.csv',nearest)
savecsv('exact_overlap.csv',exact,['image_id','reference_id'])
dup=[];seen={}
for r in rows:
    key=r['pixel_sha256']
    if key in seen:dup.append({'image_a':seen[key],'image_b':r['image_id']})
    else:seen[key]=r['image_id']
savecsv('within_cohort_exact_duplicates.csv',dup,['image_a','image_b'])
priority=sorted([r for r in candidates if r['review_priority']],key=lambda r:-r['correlation'])
rowidx={r['image_id']:i for i,r in enumerate(rows)};oldidx={r['image_id']:i for i,r in enumerate(old_rows)}
for start in range(0,len(priority),6):
    sheet=Image.new('RGB',(640,6*290),'white');draw=ImageDraw.Draw(sheet)
    for k,r in enumerate(priority[start:start+6]):
        y=k*290;sheet.paste(views[rowidx[r['image_id']]],(0,y+28));sheet.paste(old_views[oldidx[r['reference_id']]],(320,y+28))
        draw.text((0,y),r['image_id'],fill='black');draw.text((320,y),r['reference_id'],fill='black')
        draw.text((0,y+13),f"corr={r['correlation']:.4f} ph={r['phash_distance']} flip={r['horizontal_flip']}",fill='black')
    sheet.save(OUT/f'priority_review_{start//6+1:03d}.jpg')
summary={'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'manifest_sha256':sha(manifest_path),'cohort':len(rows),'groups':dict(Counter(r['original_group'] for r in rows)),'annotation_category_patterns':dict(Counter(r['annotation_category_ids'] for r in rows)),'reference_images':len(old_rows),'exact_cross_source_matches':len(exact),'within_cohort_exact_pairs':len(dup),'similarity_candidates':len(candidates),'priority_review_pairs':len(priority),'max_reference_correlation':float(bestcorr.max()),'model_inference_performed':False,'decision':'Pending review; not an independence certificate','limitations':['Patient identifiers unavailable in released annotation examples','Similarity screening does not guarantee detection of transformed duplicates','Foundation-model image overlap not audited','No human clinical review; dataset-label benchmark only']}
(OUT/'AUDIT_COMPLETE.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
