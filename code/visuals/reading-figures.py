"""Draw the labeled planar-diagram example and reproduce the archival band comparison."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
INK = '#252a2e'
TEAL = '#006875'
BROWN = '#8d481a'
TAU = 2*np.pi
VISITS = np.pi/6 + np.arange(6)*np.pi/3
ROWS = [[1,5,2,4],[3,1,4,6],[5,3,6,2]]
plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'stix'})


def curve(t):
    # Reflecting x fixes the handedness required by the printed counterclockwise PD.
    r = 2 + .86*np.cos(3*t)
    u = 2*t + np.pi/2
    return np.stack((-r*np.cos(u), r*np.sin(u)),axis=-1)


def tangent(t):
    return (curve(t+1e-5)-curve(t-1e-5)) / 2e-5


def read_vertex(j,under):
    arms=[]
    for k in (j,(j+3)%6):
        v=tangent(VISITS[k]); incoming=k or 6; outgoing=k+1
        arms += [(np.arctan2(-v[1],-v[0]),incoming),(np.arctan2(v[1],v[0]),outgoing)]
    row=[number for _,number in sorted(arms)]
    start=row.index(under)
    return row[start:]+row[:start]


def head(ax,a,b,color=INK,scale=9.5):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,
                                 shrinkA=0,shrinkB=0,lw=1.15,color=color,zorder=5))


def full_drawing(ax):
    cuts = VISITS[[1,3,5]]
    for a,b in zip(cuts,np.r_[cuts[1:],cuts[0]+TAU]):
        p=curve(np.linspace(a+.064,b-.064,1400))
        ax.plot(*p.T,color=INK,lw=2.25,solid_capstyle='round')
    for i,t in enumerate(VISITS + np.pi/6):
        p=curve(t); factor=1.12 if i in (1,3,5) else .67
        pos=p*factor
        ax.text(*pos,str(i+1),ha='center',va='center',fontsize=10.1,color=INK,weight='bold')
    for i in (0,2,5):
        t=VISITS[i]+.47
        head(ax,curve(t-.038),curve(t+.038))
    for j,name,dxy in [(1,'A',(0.,-.43)),(3,'B',(.48,.04)),(5,'C',(-.48,.04))]:
        p=curve(VISITS[j]);q=p+np.array(dxy)
        ax.text(*q,name,ha='center',va='center',fontsize=9,color=TEAL,weight='bold',
                bbox=dict(boxstyle='circle,pad=.16',facecolor='white',edgecolor=TEAL,linewidth=.6))
    ax.set(xlim=(-3.35,3.35),ylim=(-2.87,3.37),aspect='equal');ax.axis('off')


def local(ax,switched):
    # The coordinate axes point right and up. The under strand originally runs NE to SW.
    ne=np.array([.95,.68]);se=np.array([.95,-.68]);sw=-ne;nw=-se
    endpoints=[(ne,sw),(se,nw)]; under=1 if switched else 0
    for branch,(a,b) in enumerate(endpoints):
        pieces=[(a,a*.18),(b*.18,b)] if branch==under else [(a,b)]
        for p,q in pieces:
            ax.plot(*np.vstack((p,q)).T,color=INK,lw=2.1,solid_capstyle='round')
        head(ax,.70*a,.49*a,scale=9.2)
    for p,n in [(ne,1),(se,4),(sw,2),(nw,5)]:
        ax.text(*(p*1.20),str(n),ha='center',va='center',fontsize=10.8,weight='bold')
    label='After the switch' if switched else 'At crossing A'
    word=r'$X[4,1,5,2]$' if switched else r'$X[1,5,2,4]$'
    ax.text(.5,1.065,label,ha='center',va='bottom',transform=ax.transAxes,fontsize=9.0,color=TEAL)
    ax.text(.5,-.10,word,ha='center',va='top',transform=ax.transAxes,fontsize=12.6)
    ax.set(xlim=(-1.36,1.36),ylim=(-1.05,1.05),aspect='equal');ax.axis('off')


def pd_figure():
    assert [read_vertex(j,j) for j in (1,3,5)]==ROWS
    assert read_vertex(1,4)==[4,1,5,2]
    fig=plt.figure(figsize=(6.05,2.12))
    full_drawing(fig.add_axes((.005,.005,.42,.985)))
    local(fig.add_axes((.46,.23,.245,.56)),False)
    local(fig.add_axes((.75,.23,.245,.56)),True)
    fig.savefig(HERE/'reading-trefoil.pdf',bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def band(archive):
    with ZipFile(archive) as z:
        img=Image.open(BytesIO(z.read('arXiv/figs/knotpicture.png'))).convert('RGB')
        edges=Image.open(BytesIO(z.read('arXiv/figs/knotpicture2.png'))).convert('L')
    # Highlight on the original; the third panel is its actual unmodified crop.
    box=(255,24,455,224)
    fig,axes=plt.subplots(1,3,figsize=(5.98,1.86),gridspec_kw={'wspace':.08})
    axes[0].imshow(img);axes[0].add_patch(patches.Rectangle((box[0],box[1]),200,200,
                     fill=False,lw=.85,edgecolor='white',linestyle=(0,(3,2))))
    edge_array=np.asarray(edges,dtype=float)
    # Normalize black-on-white for print; the source edge map is extremely dark.
    axes[1].imshow(255-np.clip(edge_array*3.3,0,255),cmap='gray',vmin=0,vmax=255)
    axes[2].imshow(img.crop(box))
    for ax,name in zip(axes,['Photograph','Edges of the band','One crossing, enlarged']):
        ax.axis('off');ax.text(.5,-.045,name,ha='center',va='top',transform=ax.transAxes,fontsize=8.8)
    fig.subplots_adjust(left=.002,right=.998,top=.99,bottom=.16)
    fig.savefig(HERE/'reading-band.pdf',bbox_inches='tight',pad_inches=.025,dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--archive',type=Path,default=HERE.parent/'legacy/paper_illustrations.zip')
    args=p.parse_args();pd_figure();band(args.archive)
