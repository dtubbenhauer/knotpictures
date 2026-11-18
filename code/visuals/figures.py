"""Create three print figures. Run from this folder after installing matplotlib, scipy, numpy and Pillow."""
from itertools import product
from pathlib import Path
import argparse
import csv
import hashlib
import io
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from scipy.ndimage import label, maximum_filter

HERE = Path(__file__).resolve().parent
TAU = 2*np.pi
COLORS = ('#166384', '#AE491E', '#32745D')
INK = '#252A2D'
ACCENT = '#707070'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','font.size':9,
                     'pdf.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':False})


def save(fig, path, name):
    fig.savefig(path / (name + '.pdf'), bbox_inches='tight',pad_inches=.03)
    fig.savefig(path / (name + '.png'), bbox_inches='tight',pad_inches=.03,dpi=180)
    plt.close(fig)


def xy(t):
    radius = 2 + .86*np.cos(3*t)
    angle = 2*t + np.pi/2
    return np.stack((radius*np.cos(angle), radius*np.sin(angle)),axis=-1)


def crossing_data(switch=False):
    t = np.pi/6 + np.arange(3)*np.pi/3
    partners = (t + np.pi) % TAU
    assert np.allclose(xy(t),xy(partners))
    first_under = np.sin(3*t) < 0
    if switch:
        first_under[0] = ~first_under[0]
    under = np.where(first_under,t,partners)
    over = np.where(first_under,partners,t)
    bounds = np.sort(under)
    eqs=[]
    for t_under,t_over in zip(under,over):
        i = int(np.argmin(abs(bounds - t_under)))
        j = (np.searchsorted(bounds,t_over)-1)%3
        eqs.append(((i-1)%3,i,j))
    valid = [color for color in product(range(3),repeat=3)
             if all((color[a] + color[b] - 2*color[o])%3==0 for a,b,o in eqs)]
    assert len(valid) == (3 if switch else 9)
    return bounds,xy(t[0]),len(valid)


def knot(axis, switched=False, colored=False, mark=False):
    bounds, center, _ = crossing_data(switched)
    edges = np.r_[bounds,bounds[0]+TAU]
    for i,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
        # The ink is removed on the lower branch. The upper branch is continuous.
        t = np.linspace(start+.062, stop-.062, 950)
        points = xy(t)
        color = COLORS[i] if colored else INK
        axis.plot(points[:,0],points[:,1],color=color,linewidth=2.15,solid_capstyle='round')
        if colored:
            j = int(np.argmax(np.linalg.norm(points,axis=1)))
            pos = points[j]*1.10
            axis.text(*pos,str(i+1),ha='center',va='center',fontsize=9,color=color,fontweight='bold')
    if mark:
        axis.add_patch(patches.Circle(center,.49,fc='none',edgecolor=ACCENT,linewidth=.8,linestyle=(0,(2.4,2))))
    axis.set(xlim=(-3.1,3.1),ylim=(-2.82,3.22),aspect='equal')
    axis.axis('off')


def geometry(path):
    fig, axes = plt.subplots(1,3,figsize=(6.1,2.05),gridspec_kw={'wspace':.09})
    titles = ['Trefoil', 'Three ink arcs', 'One crossing switched']
    for i,axis in enumerate(axes):
        knot(axis, switched=(i==2), colored=(i==1),mark=(i!=0))
        axis.text(.5,-.012,titles[i],transform=axis.transAxes,ha='center',va='top',fontsize=9.6)
    save(fig,path,'switch-arcs')


def archive_views(path,archive):
    files = ['train/6/0.png','train/6/1.png','train/6/59.png','train/6/16.png']
    matched = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            key = member.name.removeprefix('./')
            if key in files:
                matched[key] = tar.extractfile(member).read()
    assert len(matched)==4
    fig, axes = plt.subplots(1,4,figsize=(6.05,1.72),gridspec_kw={'wspace':.06})
    manifest=[]
    for i,(axis,key) in enumerate(zip(axes,files)):
        data = matched[key]
        with Image.open(io.BytesIO(data)) as im:
            gray = np.asarray(im.convert('L'))
        # A one-pixel skeleton is very faint at print scale. Fixed 4x ink gain
        # and one-pixel expansion aid reproduction; these display operations
        # are applied identically and never enter the measured predictions.
        ink = 255 - gray
        shown = maximum_filter(np.minimum(4 * ink.astype(np.uint16),255), size=3)
        axis.imshow(shown, cmap='gray_r',vmin=0,vmax=255,interpolation='hanning')
        axis.set_xlim(59.5,451.5)
        axis.set_ylim(435.5,43.5)
        axis.axis('off')
        manifest.append((key,0 if i<3 else 1,hashlib.sha256(data).hexdigest()))
    fig.subplots_adjust(bottom=.13,top=.99,left=.012,right=.988)
    left=axes[0].get_position().x0+.012;right=axes[2].get_position().x1-.012
    y=.108
    fig.add_artist(Line2D((left,right),(y,y),transform=fig.transFigure,color='#777777',lw=.6))
    fig.text((left+right)/2,y-.045,'Three views of one drawing',ha='center',va='top',fontsize=9.1)
    b=axes[3].get_position();fig.add_artist(Line2D((b.x0+.012,b.x1-.012),(y,y),transform=fig.transFigure,color='#777777',lw=.6))
    fig.text((b.x0+b.x1)/2,y-.045,'Another drawing',ha='center',va='top',fontsize=9.1)
    with (path/'archive-views.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('archive_member','within_six_crossing_source_slot','sha256'));w.writerows(manifest)
    save(fig,path,'archive-views')


def get_components(image):
    ink=255-np.asarray(image)
    hist=np.bincount(ink.ravel(),minlength=256)
    counts=np.cumsum(hist);weights=np.cumsum(np.arange(256)*hist)
    den=counts*(counts[-1]-counts)
    score=np.divide((weights[-1]/counts[-1]*counts-weights)**2,den,out=np.zeros(256),where=den>0)
    return label(ink>np.argmax(score),structure=np.ones((3,3)))


def resolution(path, native):
    sizes=(230,64,32)
    fig, axes=plt.subplots(1,3,figsize=(5.85,2.07),gridspec_kw={'wspace':.12})
    for axis,size,expected in zip(axes,sizes,(7,6,1)):
        img=native if size==230 else native.resize((size,size),Image.Resampling.LANCZOS)
        components,count=get_components(img)
        assert count==expected
        axis.imshow(components>0,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        axis.axis('off')
        axis.text(.5,1.015,f'{size} × {size}',ha='center',va='bottom',transform=axis.transAxes,fontsize=9)
        unit='component' if count==1 else 'components'
        axis.text(.5,-.018,f'{count} ink {unit}',ha='center',va='top',transform=axis.transAxes,fontsize=9.5)
    save(fig,path,'resolution-7-4')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=HERE.parent/'legacy/knot_crossings_dataset.tar.gz')
    parser.add_argument('--crop',type=Path,default=HERE.parent/'corpus/crops/7_4.png')
    parser.add_argument('--out',type=Path,default=HERE)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    geometry(args.out)
    archive_views(args.out,args.archive)
    with Image.open(args.crop) as im:
        resolution(args.out,im.convert('L'))


if __name__=='__main__':
    main()
