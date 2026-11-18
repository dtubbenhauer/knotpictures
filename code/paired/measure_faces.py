"""Retrospective Euler-shadow diagnostic on the fixed 160 paired test geometries."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def measure(root):
    with (root/'manifest.csv').open() as f:
        source=list(csv.DictReader(f))
    rows=[]
    for row in source:
        if row['split']!='test':
            continue
        raster=np.asarray(Image.open(root/row['image']))
        # The background is white, the diagram is strictly inside the frame, so
        # exactly one 4-connected white component touches the frame (exterior).
        labels,n=label(raster>int(row['otsu']))
        border=np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]
        assert len(np.unique(border))==1 and border[0]!=0
        bounded=n-1
        rows.append(dict(image=row['image'],group=row['group'],k=row['k'],curl=row['curl'],style=row['style'],count=row['count'],white_bounded=bounded,face_estimate=bounded-1))
    with (root/'faces.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    for style in ['clean','thin','break']:
        subset=[r for r in rows if r['style']==style];errs=[r['face_estimate']-int(r['count']) for r in subset]
        pred={(r['group'],r['k'],int(r['curl'])):r['face_estimate'] for r in subset}
        diff=[pred[g,k,1]-v for (g,k,c),v in pred.items() if c==0]
        print(style,len(subset),errs.count(0),sum(map(abs,errs)),diff.count(1),'of',len(diff),'pairs')


def illustrate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    colors=['#C1DDFF','#F7C4B4','#CCBDDE','#C5DCB2','#FFD58C','#E9BFD9','#A8DEDA']
    im=np.asarray(Image.open(root/'images/thin/g038_k05_c1.png'))
    with (root/'manifest.csv').open() as f:
        row=next(r for r in csv.DictReader(f) if r['image']=='images/thin/g038_k05_c1.png')
    fg=im<=int(row['otsu']); labels,count=label(~fg); assert count==8
    bounded=[idx for idx in range(1,count+1) if idx!=labels[0,0]]
    rgb=np.ones((*fg.shape,3));rgb[fg]=to_rgb('#242A34')
    for idx,col in zip(bounded,colors): rgb[labels==idx]=to_rgb(col)
    yy,xx=np.where(fg); bounds=(xx.min()-5,xx.max()+5,yy.max()+5,yy.min()-5)
    fig,axes=plt.subplots(1,3,figsize=(6.5,1.70)); titles=['Fine-line drawing','One ink component','Seven bounded white regions']
    for ax,title,data in zip(axes,titles,[im,~fg,rgb]):
        ax.imshow(data,interpolation='nearest',cmap='gray');ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.axis('off');ax.set_title(title,fontsize=9,pad=1)
    fig.subplots_adjust(left=.01,right=.99,top=.89,bottom=.015,wspace=.04)
    fig.savefig(root/'examples/shadow-faces.pdf',dpi=220);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    root=parser.parse_args().root;measure(root);illustrate(root)
