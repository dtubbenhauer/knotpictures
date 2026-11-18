"""Apply the Euler endpoint readout to the exact masks of the Rideout poster census."""
import csv
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import fitz
import numpy as np
from scipy.ndimage import convolve, label
from measure_mixed import thin_zhang_suen


def run(poster_dir, output, sizes):
    sys.path.insert(0,str(poster_dir))
    from poster_census import standalone, NS
    with (poster_dir/'results/poster_counts.csv').open() as stream:
        originals={(r['knot'],int(r['width'])):r for r in csv.DictReader(stream)}
    fitz.TOOLS.set_aa_level(8)
    rows=[]
    for link in ET.parse(poster_dir/'knottable_v1.svg').getroot().findall(NS+'a'):
        name=Path(link.get('href')).stem
        for size in sizes:
            ref=originals[name,size];cutoff=int(ref['otsu_min_foreground'])
            with fitz.open(stream=standalone(link,size),filetype='svg') as document:
                pix=document[0].get_pixmap(colorspace=fitz.csGRAY,alpha=False)
                mask=255-np.frombuffer(pix.samples,dtype=np.uint8).reshape(size,size)>=cutoff
            # Pad to represent the unique exterior component of the plane.
            mask=np.pad(mask,1,constant_values=False)
            # Limit thinning to the nonempty drawing rectangle plus white margin.
            yy,xx=np.where(mask);ink=mask[yy.min()-1:yy.max()+2,xx.min()-1:xx.max()+2]
            n_comp=label(ink,structure=np.ones((3,3)))[1];assert n_comp==int(ref['components_otsu'])
            bounded=label(~ink)[1]-1
            sm=thin_zhang_suen(ink);nb=convolve(sm,np.ones((3,3),dtype=np.uint8),mode='constant')-sm
            tips=np.count_nonzero((sm!=0)&(nb==1));pred=tips+bounded-n_comp
            rows.append(dict(knot=name,width=size,count=int(ref['label']),tips=tips,white_bounded=bounded,ink_components=n_comp,arc_prediction=int(ref['prediction_otsu']),face_prediction=bounded-1,euler_prediction=pred))
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for size in sizes:
        selected=[r for r in rows if r['width']==size]
        for col in ['arc_prediction','face_prediction','euler_prediction']:
            errors=[r[col]-r['count'] for r in selected];print(size,col,errors.count(0),'/',len(selected),'AE sum',sum(map(abs,errors)))


if __name__=='__main__':
    import argparse
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poster-dir',type=Path,default=root.parent/'legacy')
    parser.add_argument('--output',type=Path,default=root/'poster-euler.csv')
    parser.add_argument('--sizes',type=int,nargs='+',default=[64,128,256,512,1024,2048])
    args=parser.parse_args();run(args.poster_dir,args.output,args.sizes)
