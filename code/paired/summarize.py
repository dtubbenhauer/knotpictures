"""Summarize the held-out layouts and draw the four-panel comparison used in the paper."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image
from scipy.ndimage import label

from generate_pairs import otsu_count


def comparison(root):
    group,k=38,5
    panels=[('Original',0,'clean'),('With one curl',1,'clean'),('With one ink break',1,'break'),('Fine, low-resolution',1,'thin')]
    colors=['#111827','#C34431','#258088','#7B55A5','#B67209','#376AA6','#007047','#B64E7A']
    table=np.vstack(([1,1,1],[to_rgb(c) for c in colors]))
    figure,axes=plt.subplots(2,len(panels),figsize=(7.0,3.20))
    all_images=[Image.open(root/f'images/{style}/g{group:03d}_k{k:02d}_c{curl}.png') for _,curl,style in panels]
    # All images depict the same underlying uncurled/curl pair. Use one window.
    yy,xx=np.where(np.min(np.stack([np.asarray(i) for i in all_images]),axis=0)<225)
    bounds=(xx.min()-7,xx.max()+7,yy.max()+7,yy.min()-7)
    for i,((title,curl,style),im) in enumerate(zip(panels,all_images)):
        x=np.asarray(im); thresh,count=otsu_count(im); labels,_=label(x<=thresh,structure=np.ones((3,3)))
        axes[0,i].imshow(x,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axes[1,i].imshow(table[labels],interpolation='nearest')
        axes[0,i].set_title(title,fontsize=9,pad=1)
        axes[1,i].set_xlabel(f'{count} component'+('s' if count!=1 else ''),fontsize=9,labelpad=1)
        for ax in axes[:,i]:
            ax.set_xticks([]);ax.set_yticks([]);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.spines[:].set_visible(False)
    figure.subplots_adjust(left=.008,right=.992,top=.92,bottom=.067,wspace=.045,hspace=.025)
    figures=root/'examples';figures.mkdir(exist_ok=True)
    figure.savefig(figures/'paired-comparison.pdf',dpi=240);plt.close(figure)


def summarize(root):
    with (root/'manifest.csv').open() as f:
        all_rows=list(csv.DictReader(f))
    summaries=[]
    for split in ['train','val','test']:
        for style in ['clean','thin','break']:
            rows=[r for r in all_rows if r['split']==split and r['style']==style]
            if not rows: continue
            errors=[int(r['components'])-int(r['count']) for r in rows]
            pred={(r['group'],r['k'],int(r['curl'])):int(r['components']) for r in rows}
            slopes=[pred[g,k,1]-p for (g,k,c),p in pred.items() if c==0]
            summaries.append(dict(split=split,style=style,images=len(rows),correct=errors.count(0),abs_error_sum=sum(abs(x) for x in errors),pairs=len(slopes),increments_equal_one=slopes.count(1)))
    with (root/'summary.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(summaries[0]));out.writeheader();out.writerows(summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    root=parser.parse_args().root;summarize(root);comparison(root)
