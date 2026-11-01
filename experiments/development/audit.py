"""Audit v1 similarity chains; generate image review sheets, never infer identity from dHash."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from prepare import read_csv, write_csv, write_json, digest


def audit_metadata(old, out):
    old, out = Path(old), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(old/'inventory.csv')
    pairs = read_csv(old/'duplicate_candidates.csv')
    manifest = read_csv(old/'manifest.csv')
    summary = {'n_images': len(rows), 'n_candidate_pairs': len(pairs),
        'exact_pairs': sum(p['exact']=='True' for p in pairs),
        'same_patient_pairs': sum(p['same_patient']=='True' for p in pairs),
        'cross_source_pairs': sum(p['cross_source']=='True' for p in pairs),
        'largest_similarity_components': Counter(r['group_id'] for r in rows).most_common(10),
        'excluded_images': sum(r['split']=='excluded_overlap' for r in manifest),
        'source_hashes': {n:digest(old/n) for n in ['inventory.csv','manifest.csv','duplicate_candidates.csv']},
        'finding': 'V1 unioned dHash candidates transitively. Similarity is not evidence of patient identity.',
        'status': 'Metadata audit complete; visual adjudication remains pending.'}
    write_json(out/'metadata_audit.json', summary)
    print(json.dumps(summary, indent=2))
    return rows, pairs


def visual_audit(old, data, out):
    out, data = Path(out), Path(data)
    rows, pairs = audit_metadata(old, out)
    lookup = {r['image_id']:r for r in rows}
    cache = {}
    def get(key):
        if key not in cache:
            r=lookup[key]
            path=data/r['path']
            if digest(path)!=r['sha256']: raise ValueError('Source image changed: '+key)
            with Image.open(path) as im:
                cache[key]=ImageOps.exif_transpose(im).convert('L').resize((256,256))
        return cache[key]
    scored=[]
    for p in pairs:
        a=np.asarray(get(p['image_a']),dtype=float).ravel()
        b=np.asarray(get(p['image_b']),dtype=float).ravel()
        corr=float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else 0.
        scored.append({**p,'normalized_correlation':corr,
                       'priority_review': p['cross_source']=='True' or corr>=.98})
    write_csv(out/'all_scored_candidates.csv',scored)
    priority=[p for p in scored if p['priority_review']]
    review=[]
    sheets=out/'review_sheets'; sheets.mkdir(exist_ok=True)
    for start in range(0,len(priority),4):
        page=Image.new('RGB',(1100,4*330),'white'); draw=ImageDraw.Draw(page)
        for j,p in enumerate(priority[start:start+4]):
            idx=start+j
            y=j*330
            draw.text((10,y+5),f"Pair {idx:04}: dHash={p['dhash_distance']}, corr={p['normalized_correlation']:.4f}; cross-source={p['cross_source']}",fill='black')
            for col,k in enumerate(['image_a','image_b']):
                draw.text((col*550+10,y+25),p[k],fill='black')
                page.paste(get(p[k]).convert('RGB'),(col*550+10,y+50))
            review.append({'pair_id':f'{idx:04}', 'image_a':p['image_a'],'image_b':p['image_b'],
                           'decision':'pending','reviewer':'','reason':'','sheet':f'review_{start//4:03}.jpg'})
        page.save(sheets/f'review_{start//4:03}.jpg',quality=92)
    if not (out/'review_decisions.csv').exists():
        write_csv(out/'review_decisions.csv',review,['pair_id','image_a','image_b','decision','reviewer','reason','sheet'])
    write_json(out/'VISUAL_REVIEW_PENDING.json',{'priority_pairs':len(priority),'all_candidates':len(scored),
        'policy':'All cross-source candidates and within-source correlation>=0.98 prioritized. Scores do not confirm identity. Other candidates retained as unresolved screening evidence.',
        'next':'Review sheets and original-resolution images for ambiguous cases. Decisions: duplicate, distinct, uncertain. No corrected training until review is finalized.'})
    print('Visual review prepared:',len(priority),'priority pairs. Training has NOT started.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--old',required=True); p.add_argument('--out',required=True); p.add_argument('--data')
    a=p.parse_args()
    if a.data: visual_audit(a.old,a.data,a.out)
    else: audit_metadata(a.old,a.out)
