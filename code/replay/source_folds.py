"""Leave each eligible original source drawing out, using published generation logic.

Run after ink_features.py: python source_folds.py OUTPUT_DIRECTORY
The original generators in notebooks/Latest.ipynb at revision 46fcbd90 of
annedranowski/knot-cnn append round(49/N[c]) siblings per source, then make 20
padded, rescaled images per sibling. Source IDs are anonymous per class, since
directory enumeration order does not give an anatomical knot label.
"""
import csv
import sys
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from ink_features import FEATURE_NAMES,dump_csv,run

ORIGINALS={0:1,3:1,4:1,5:2,6:3,7:7,8:21,9:49,10:3}

def source_id(row,counts):
    c=row['expected']
    variants_per_image=round(49/ORIGINALS[c])
    base_count=counts[c]//21
    assert base_count == variants_per_image*ORIGINALS[c]
    i=row['number']
    v=i if i<base_count else (i-base_count)//20
    return (c,v//variants_per_image)


def main(directory):
    with (directory/'global_ink_features.csv').open() as f:
        train=[r for r in csv.DictReader(f) if r['source']=='train']
    for r in train:
        for k in ('expected','number'):r[k]=int(r[k])
        for k in FEATURE_NAMES:r[k]=float(r[k])
    counts=Counter(r['expected'] for r in train)
    ids=[source_id(r,counts) for r in train]
    hashes=defaultdict(set)
    for r,source in zip(train,ids):hashes[r['sha256']].add(source)
    assert max(map(len,hashes.values()))==1
    groups=sorted(set(ids))
    assert len(groups)==88 and len(hashes)==6657
    predictions=[]
    for source in groups:
        c,slot=source
        if ORIGINALS[c]==1:continue
        fitting=[r for r,s in zip(train,ids) if s!=source]
        test=[r for r,s in zip(train,ids) if s==source]
        fold=run(test,fitting,'leave_one_source_out')
        for r in fold:r['source_id']=f'{c}-{slot}'
        predictions+=fold
    dump_csv(directory/'source_fold_predictions.csv',predictions)
    def stat(rows,kind,label):
        return dict(unit=kind,group=label,correct=sum(r['exact'] for r in rows),
                    total=len(rows),mae=np.mean([r['abs_error'] for r in rows]))
    stats=[stat(predictions,'pooled','all')]
    for c in ORIGINALS:
        sub=[r for r in predictions if r['expected']==c]
        if sub:stats.append(stat(sub,'class',c))
    for c,i in groups:
        sub=[r for r in predictions if r['source_id']==f'{c}-{i}']
        if sub:stats.append(stat(sub,'source',f'{c}-{i}'))
    dump_csv(directory/'source_fold_summary.csv',stats)
    print('per class',stats[:7])
    source_rows=[r for r in stats if r['unit']=='source']
    print('Source average',np.mean([r['correct']/r['total'] for r in source_rows]),len(source_rows))
    print('Class average',np.mean([r['correct']/r['total'] for r in stats if r['unit']=='class']))
    print('identical files across source IDs',max(map(len,hashes.values()))-1)

if __name__=='__main__':main(Path(sys.argv[1]))
