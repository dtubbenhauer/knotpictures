"""Plot the 88-diagram source census and the thresholded rendering of knot 10_3."""
from argparse import ArgumentParser
import csv
from pathlib import Path
from xml.etree import ElementTree as ET
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from hashlib import sha256
from legacy_audit import FG8, otsu_cutoff
from poster_census import DIGEST, NS, SIZES, standalone

PALETTE = np.array([[27,76,110],[161,66,72],[11,108,100],[140,88,21],
                    [100,74,130],[55,102,52],[132,71,39],[28,112,148],
                    [142,67,106],[83,85,91]])

def run(poster, results, out):
    if sha256(poster.read_bytes()).hexdigest()!=DIGEST:
        raise ValueError('Unrecognized poster checksum')
    r=ET.parse(poster).getroot()
    link=next(a for a in r.findall(NS+'a') if a.get('href')=='knots/10_3.svg')
    with (results/'poster_summary.csv').open() as f:
        summary={int(row['size']):int(row['correct']) for row in csv.DictReader(f) if row['threshold']=='otsu'}
    with (results/'poster_counts.csv').open() as f:
        specimen={int(row['width']):int(row['components_otsu']) for row in csv.DictReader(f) if row['knot']=='10_3'}
    fitz.TOOLS.set_aa_level(8)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.7,
                         'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42}):
        fig=plt.figure(figsize=(6.7,4.45),facecolor='white')
        gs=fig.add_gridspec(2,3,height_ratios=[1.06,.84],left=.095,right=.97,
                            top=.915,bottom=.13,wspace=.09,hspace=.31)
        for i,size in enumerate((128,512,2048)):
            ax=fig.add_subplot(gs[0,i])
            with fitz.open(stream=standalone(link,size), filetype='svg') as doc:
                pix=doc[0].get_pixmap(colorspace=fitz.csGRAY,alpha=False)
                ink=255-np.frombuffer(pix.samples,dtype=np.uint8).reshape(size,size)
            label,n=ndi.label(ink>=otsu_cutoff(ink), FG8)
            assert n==specimen[size]
            rgb=np.full((size,size,3),255,np.uint8)
            rgb[label>0]=PALETTE[(label[label>0]-1)%len(PALETTE)]
            # Show the same framing in all three panels, with no rescaling in the measurement.
            ax.imshow(rgb,interpolation='nearest' if size==128 else 'antialiased')
            ax.set(xlim=(.01*size,.99*size),ylim=(.99*size,.01*size))
            ax.set_axis_off()
            ax.set_title(f'{size:,} px\n{n} components',fontsize=10.5, pad=0)
        ax=fig.add_subplot(gs[1,:])
        xs=np.arange(len(SIZES)); y=[summary[w] for w in SIZES]
        ax.plot(xs,y,'o-',lw=1.8,color='#204e6e',ms=4.8,zorder=3)
        ax.set_xticks(xs,[f'{w:,}' for w in SIZES]);ax.set_yticks([0,22,44,66,88])
        ax.set(xlim=(-.15,5.14),ylim=(-6,99))
        ax.set_ylabel('Correct out of 88',labelpad=6)
        ax.set_xlabel('Raster width (pixels)',labelpad=4)
        for x,v in zip(xs,y):
            ax.annotate(str(v),(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        ax.grid(axis='y',color='#d8dadd',linewidth=.55)
        for spine in ('top','right'):ax.spines[spine].set_visible(False)
        for s in ('left','bottom'):ax.spines[s].set_color('#65686b')
        ax.tick_params(direction='out',length=3)
        fig.savefig(out/'poster_resolution.pdf',bbox_inches='tight',pad_inches=.04,dpi=320,
                    metadata={'Title':'Counting crossings at six raster resolutions'})
        fig.savefig(out/'poster_resolution.png',bbox_inches='tight',pad_inches=.04,dpi=190)
        plt.close(fig)

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--poster',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);run(a.poster,a.results,a.out)
