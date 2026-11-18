"""Read the underlying Jkasd base-crop bitmasks with the mixed Euler rule."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, label
from measure_mixed import thin_zhang_suen


def run(corpus, output):
    with (corpus/'observations.csv').open() as stream:
        observations=list(csv.DictReader(stream))
    rows=[]
    for rec in observations:
        name=rec['name']; size=int(rec['side_px']); target=int(rec['expected'])
        with Image.open(corpus/'crops'/f'{name}.png') as original:
            im=original.resize((size,size),Image.Resampling.LANCZOS) if original.width!=size else original
            mask=np.pad(255-np.asarray(im) > int(rec['otsu_ink_cutoff']), 1)
        c=label(mask,structure=np.ones((3,3)))[1]; b=label(~mask)[1]-1
        assert (c,b)==(int(rec['components']),int(rec['holes']))
        skeleton=thin_zhang_suen(mask)
        neighbors=convolve(skeleton,np.ones((3,3),dtype=np.uint8),mode='constant')-skeleton
        e=int(np.sum((skeleton!=0)&(neighbors==1)))
        rows.append(dict(knot=name,width=size,count=target,tips=e,white_bounded=b,ink_components=c,arc_prediction=int(rec['prediction']),face_prediction=b-1,euler_prediction=e+b-c))
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for size in [32,48,64,96,128,160,230]:
        subset=[row for row in rows if row['width']==size]
        for col in ['arc_prediction','face_prediction','euler_prediction']:
            errors=[r[col]-r['count'] for r in subset];print(size,col,errors.count(0),len(errors),sum(map(abs,errors)))


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=root.parent/'corpus')
    parser.add_argument('--output',type=Path,default=root/'jkasd-euler.csv')
    args=parser.parse_args();run(args.corpus,args.output)
