"""Plot Rideout's 10_3 beside the per-diagram results for both crossing counters."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / 'legacy'))
from legacy_audit import FG8, otsu_cutoff
from poster_census import DIGEST, NS, SIZES, standalone
from poster_figure import PALETTE


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def numbers(component_path, euler_dir):
    components = load_csv(component_path)
    euler = load_csv(euler_dir / 'poster-euler.csv')
    baseline, corrected = [], []
    for width in SIZES:
        a = {r['knot']: r for r in components if int(r['width']) == width}
        b = {r['knot']: r for r in euler if int(r['width']) == width}
        assert len(a) == len(b) == 88 and a.keys() == b.keys()
        assert all(a[k]['label'] == b[k]['count'] for k in a)
        baseline.append(sum(int(r['prediction_otsu']) == int(r['label']) for r in a.values()))
        corrected.append(sum(int(r['euler_prediction']) == int(r['count']) for r in b.values()))
    assert baseline == [1, 5, 35, 62, 82, 88]
    assert corrected == [23, 75, 77, 81, 73, 55]
    assert len(components) == len(euler) == len(SIZES) * 88
    return baseline, corrected


def picture(node, width):
    with fitz.open(stream=standalone(node, width), filetype='svg') as doc:
        pix = doc[0].get_pixmap(colorspace=fitz.csGRAY, alpha=False)
        ink = 255 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(width, width)
    regions, count = label(ink >= otsu_cutoff(ink), FG8)
    rgb = np.full((width, width, 3), 255, np.uint8)
    rgb[regions > 0] = PALETTE[(regions[regions > 0] - 1) % len(PALETTE)]
    return rgb, count


def draw(poster, component_path, euler_dir, output):
    assert sha256(poster.read_bytes()).hexdigest() == DIGEST
    baseline, corrected = numbers(component_path, euler_dir)
    root = ET.parse(poster).getroot()
    link = next(a for a in root.findall(NS+'a') if a.get('href')=='knots/10_3.svg')
    fitz.TOOLS.set_aa_level(8)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'mathtext.fontset':'stix',
                         'pdf.fonttype':42,'axes.linewidth':.6})
    fig = plt.figure(figsize=(6.2, 1.86), facecolor='white')
    top, bottom = .82, .19
    left = fig.add_gridspec(1,3,left=.007,right=.445,bottom=bottom,top=top,wspace=.02)
    for i,(width,expected) in enumerate(((128,5),(512,7),(2048,10))):
        axis = fig.add_subplot(left[0,i]);rgb,count = picture(link,width)
        assert count == expected
        axis.imshow(rgb,interpolation='nearest' if width==128 else 'antialiased')
        axis.set(xlim=(width*.065,width*.935),ylim=(width*.935,width*.065))
        axis.axis('off')
        axis.text(.5,1.018,f'{width:,} px',transform=axis.transAxes,ha='center',va='bottom',fontsize=7.7)
        axis.text(.5,-.012,rf'$b={count}$',transform=axis.transAxes,ha='center',va='top',fontsize=9.4)
    axis = fig.add_axes((.536,bottom,.448,top-bottom))
    x = np.arange(len(SIZES))
    navy, brown = '#1E526D', '#8B482F'
    axis.plot(x, baseline, '-o',linewidth=1.45,markersize=3.7,color=navy,label='Components',zorder=3)
    axis.plot(x, corrected, '--s',linewidth=1.25,markersize=4.0,markerfacecolor='white',
              markeredgewidth=.95,color=brown,label=r'$e-b+f$',zorder=3)
    axis.set_xticks(x,[str(v) for v in SIZES]);axis.set_yticks((0,44,88))
    axis.set(xlim=(-.17,5.20),ylim=(-5,99))
    axis.set_xlabel('Raster width (pixels)',labelpad=2.7)
    axis.set_ylabel('Correct out of 88',labelpad=2.2)
    axis.grid(axis='y',color='#D3D5D5',linewidth=.5,zorder=0)
    axis.spines[['top','right']].set_visible(False)
    axis.tick_params(width=.55,length=2.5,pad=2)
    axis.legend(loc='lower center',bbox_to_anchor=(.49,1.003),ncol=2,frameon=False,
                handlelength=2.2,columnspacing=1.1,handletextpad=.45,borderpad=0,fontsize=7.6)
    for xx, val, dy in zip(x,baseline,(3.5,3.5,3.5,-9.5,3.5,3.5)):
        axis.annotate(str(val),(xx,val),xytext=(0,dy),textcoords='offset points',
                      ha='center',va='bottom' if dy>0 else 'center',fontsize=6.9,color=navy)
    for xx, val, dy in zip(x,corrected,(3.5,3.5,3.5,3.5,-8.5,-8.5)):
        axis.annotate(str(val),(xx,val),xytext=(0,dy),textcoords='offset points',
                      ha='center',va='bottom' if dy>0 else 'center',fontsize=6.9,color=brown)
    fig.savefig(output.with_suffix('.pdf'),dpi=350,metadata={'Title':'Crossing counts on the 88 Rideout diagrams'})
    fig.savefig(output.with_suffix('.png'),dpi=200)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poster',type=Path,default=WORK/'legacy/knottable_v1.svg')
    p.add_argument('--components',type=Path,default=WORK/'legacy/results/poster_counts.csv')
    p.add_argument('--euler-dir',type=Path,default=WORK/'paired')
    p.add_argument('--out',type=Path,default=HERE/'poster-transfer.pdf')
    a=p.parse_args(); draw(a.poster,a.components,a.euler_dir,a.out)
