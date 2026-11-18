"""Global-ink control for knot-picture crossing-count experiments.

Train a 1-nearest-neighbor classifier on four scalar features of each grayscale image.
Input archive (read-only) and knots.png and its crop manifest as emitted by
table_baseline.py. Requires numpy, sklearn, scipy, Pillow.
Run: python ink_features.py DATASET.tar.gz MANIFEST.csv KNOTS.png OUTPUT_DIRECTORY
"""
import argparse
import csv
import hashlib
import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ['fractional_ink', 'rms_radius', 'eccentricity_squared', 'radial_fourth_moment']


def fingerprint(image):
    image = image.convert('L')
    # Both public collections use dark strokes on white backgrounds.
    a = 255 - np.asarray(image)
    y, x = np.nonzero(a)
    ink = a[y,x].astype(float) / 255.0
    mass = ink.sum()
    if mass == 0:
        raise ValueError('Blank image')
    w, h = image.size
    x = x.astype(float) / w
    y = y.astype(float) / h
    x -= np.dot(x,ink) / mass
    y -= np.dot(y,ink) / mass
    xx = np.dot(x*x,ink) / mass
    yy = np.dot(y*y,ink) / mass
    xy = np.dot(x*y,ink) / mass
    radius2 = x*x + y*y
    trace = xx + yy
    ecc2 = ((xx-yy)**2 + 4*xy**2) / trace**2
    fourth = np.dot(radius2**2,ink) / (mass * trace**2)
    return [mass / (w*h), np.sqrt(trace), ecc2, fourth]


def dump_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_archive_features(dataset):
    rows = []
    with tarfile.open(dataset) as archive:
        for member in archive:
            path = Path(member.name)
            if len(path.parts) != 3 or path.suffix != '.png' or not path.stem.isdigit(): continue
            split, label = path.parts[:2]
            raw = archive.extractfile(member).read()
            with Image.open(io.BytesIO(raw)) as im: feats=fingerprint(im)
            rows.append(dict(path=str(path),source=split,expected=int(label),number=int(path.stem),
                             sha256=hashlib.sha256(raw).hexdigest(),**dict(zip(FEATURE_NAMES,feats))))
    return sorted(rows,key=lambda r:(r['source'],r['expected'],r['number']))


def image_groups(rows):
    counts=Counter(r['expected'] for r in rows)
    groups=[]
    for r in rows:
        c=counts[r['expected']]
        # Legacy notebook, cell 13: retain the pre-resize images 0..c/21-1.
        # Their 20 scaled descendants occupy consecutive blocks in the rest.
        base, rem = divmod(c,21)
        if rem: raise ValueError('Class has unexpected augmentation count')
        i = r['number']
        family = i if i < base else (i-base)//20
        groups.append((r['expected'], family))
    return np.array([f'{c}-{i}' for c,i in groups])


def table_data(manifest, sheet):
    with Image.open(sheet) as source:
        image=source.convert('RGBA')
        white=Image.new('RGBA',image.size,'white')
        white.alpha_composite(image)
        image=white.convert('L')
        with manifest.open() as f:
            rows=[]
            for r in csv.DictReader(f):
                box=[int(r[k]) for k in ('left','top','right','bottom')]
                crop=image.crop(box)
                rows.append(dict(path=r['name'],source='printed_table',expected=int(r['expected']),
                                number=0,sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                                **dict(zip(FEATURE_NAMES,fingerprint(crop)))))
    return rows


def run(rows, training_rows, label):
    train_X=[[r[k] for k in FEATURE_NAMES] for r in training_rows]
    train_y=[r['expected'] for r in training_rows]
    x=[[r[k] for k in FEATURE_NAMES] for r in rows]
    fitted = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(train_X, train_y)
    pred=fitted.predict(x)
    training_hashes={r['sha256'] for r in training_rows}
    return [dict(evaluation=label,path=r['path'],expected=r['expected'],prediction=int(p),
                 exact=int(p==r['expected']),abs_error=abs(int(p)-r['expected']),
                 trained_class=int(r['expected'] in set(train_y)),same_file_as_training=int(r['sha256'] in training_hashes))
            for p,r in zip(pred,rows)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset','manifest','sheet','output'): parser.add_argument(k,type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rows=get_archive_features(args.dataset);table=table_data(args.manifest,args.sheet)
    dump_csv(args.output/'global_ink_features.csv',rows+table)
    tr=[r for r in rows if r['source']=='train']; te=[r for r in rows if r['source']=='test']
    allidx=np.arange(len(tr))
    idx, val=train_test_split(allidx,test_size=.2,random_state=42,stratify=[r['expected'] for r in tr])
    pred=run([tr[i] for i in val],[tr[i] for i in idx],'image_split')
    groups=image_groups(tr)
    splitter = GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    idx,val=next(splitter.split(allidx,groups=groups))
    pred+=run([tr[i] for i in val],[tr[i] for i in idx],'resizing_family_split')
    pred+=run(te,tr,'archive_test')
    pred+=run(table,tr,'table')
    dump_csv(args.output/'global_ink_predictions.csv',pred)
    stats=[]
    for label in ('image_split','resizing_family_split','archive_test','archive_test_trained_classes','table'):
        s=[r for r in pred if r['evaluation']==label or
           (label=='archive_test_trained_classes' and r['evaluation']=='archive_test' and r['trained_class'])]
        stats.append(dict(evaluation=label,correct=sum(r['exact'] for r in s),total=len(s),
                          mean_absolute_error=np.mean([r['abs_error'] for r in s]),
                          same_file_as_training=sum(r['same_file_as_training'] for r in s)))
    dump_csv(args.output/'global_ink_summary.csv',stats)
    print(stats)

if __name__=='__main__':
    main()
