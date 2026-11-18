"""Make the print figure from the two supplied manuscript PNGs."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import ndimage as ndi
from legacy_audit import FG8, otsu_cutoff, read_ink

def figure(paper, out):
    with ZipFile(paper) as z:
        orig = read_ink(z.read('arXiv/figs/knot_orig.png'))
        final = read_ink(z.read('arXiv/figs/knot_final.png'))
    palette = np.array([[21,74,124],[151,62,91],[1,111,103],[167,92,16],
                        [94,65,139],[63,103,55],[135,72,33],[29,102,150],[180,53,58]])
    labels,n = ndi.label(orig >= otsu_cutoff(orig), FG8)
    binary = final >= otsu_cutoff(final)
    n2 = ndi.label(binary, FG8)[1]
    fig = plt.figure(figsize=(7.15, 3.18), layout='constrained', facecolor='white')
    gs = fig.add_gridspec(2,3,width_ratios=(1,0.93,1.13),wspace=0.06,hspace=.1)
    axs = [fig.add_subplot(gs[:,0]), fig.add_subplot(gs[:,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]
    rgb = np.full((*orig.shape,3),255,np.uint8)
    rgb[labels>0]=palette[(labels[labels>0]-1)%len(palette)]
    axs[0].imshow(rgb, interpolation='nearest')
    axs[0].set(xlim=(55,435),ylim=(402,80))
    axs[0].set_title(f'Original\n{n} ink components',fontsize=10)
    axs[1].imshow(255-final, cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    axs[1].set(xlim=(157,395),ylim=(407,110))
    axs[1].set_title(f'After augmentation\n{n2} ink components',fontsize=10)
    x,y,dx,dy=299,305,50,30
    axs[1].add_patch(Rectangle((x-.5,y-.5),dx,dy,fill=False,edgecolor='#b04528',linewidth=1.1))
    for ax,ar,title in [(axs[2],255-final,'Selected detail'),(axs[3],(~binary)*255,'After thresholding')]:
        ax.imshow(ar[y:y+dy,x:x+dx],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(title,fontsize=9,pad=2)
        ax.add_patch(Rectangle((-.5,-.5),dx,dy,fill=False,edgecolor='#b04528',linewidth=1.1))
    for ax in axs:
        ax.axis('off')
    fig.savefig(out/'legacy_components.pdf',bbox_inches='tight',pad_inches=.02,dpi=300,
                metadata={'Title':'Components in the original and augmented raster'})
    fig.savefig(out/'legacy_components.png',bbox_inches='tight',pad_inches=.02,dpi=200)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True);figure(a.paper,a.out)
