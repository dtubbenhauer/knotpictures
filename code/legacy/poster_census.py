"""Rasterize and count the complete named knot table printed on Philip Rideout's CC0 poster."""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET

import fitz
import numpy as np

from legacy_audit import FG8, counts, crop_nonzero, otsu_cutoff, write_rows

URL = 'https://prideout.net/blog/svg_knots/knottable_v1.svg'
DIGEST = 'bd34a6d0334013847388113f82ef9729664ab585a5b7f7694260bd9323541831'
SIZES = (64, 128, 256, 512, 1024, 2048)
NS = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', NS[1:-1])
SOURCE_CLASS_COUNTS = (1,0,0,1,1,2,3,7,21,49,3)

def standalone(link, width):
    # The parent SVG's affine transform sets poster location/size. Recover original SVG
    # geometry and viewport from the inner drawing group and its white bounding rectangle.
    group = deepcopy(link.find(NS+'g'))
    del group.attrib['transform']
    rect = group.find(NS+'rect')
    box = ' '.join(rect.get(s) for s in ('x','y','width','height'))
    attrs = {'width':str(width), 'height':str(width), 'viewBox':box,
             'preserveAspectRatio':'xMidYMid meet'}
    svg = ET.Element(NS+'svg', attrs)
    svg.append(group)
    return ET.tostring(svg, encoding='utf-8')


def run(poster, out):
    if not poster.exists():
        poster.write_bytes(urlopen(URL, timeout=45).read())
    if sha256(poster.read_bytes()).hexdigest() != DIGEST:
        raise ValueError('Poster SHA256 did not match the reference input')
    links=ET.parse(poster).getroot().findall(NS+'a')
    fitz.TOOLS.set_aa_level(8)
    rows=[]
    for link in links:
        href=link.get('href')
        name=Path(href).stem
        crossings,index=map(int,name.split('_'))
        for size in SIZES:
            with fitz.open(stream=standalone(link,size),filetype='svg') as doc:
                pix=doc[0].get_pixmap(colorspace=fitz.csGRAY,alpha=False)
                assert pix.width==size and pix.height==size
                gray=np.frombuffer(pix.samples, dtype=np.uint8).reshape(size,size)
            ink=255-gray
            cutoff=otsu_cutoff(ink)
            trimmed=crop_nonzero(ink)
            o=counts(trimmed,cutoff)
            fixed=counts(trimmed,128)
            rows.append(dict(knot=name,label=crossings,index=index,width=size,
                             source_element=href,otsu_min_foreground=cutoff,
                             components_otsu=o[f'components_{cutoff}_1'],
                             prediction_otsu=o[f'prediction_{cutoff}_1'],
                             components_128=fixed['components_128_1'],
                             prediction_128=fixed['prediction_128_1']))
    expected={f'{c}_{i}' for c,n in enumerate(SOURCE_CLASS_COUNTS) for i in range(1,n+1)}
    assert {row['knot'] for row in rows}==expected and len(rows)==len(expected)*len(SIZES)
    write_rows(out/'poster_counts.csv',rows)
    summary=[]
    for size in SIZES:
        data=[row for row in rows if row['width']==size]
        for method in ('otsu','128'):
            ok=[r['knot'] for r in data if r[f'prediction_{method}']==r['label']]
            misses=[r['knot'] for r in data if r['knot'] not in ok]
            summary.append(dict(size=size,threshold=method,correct=len(ok),n=len(data),
                                correct_nontrivial=len(ok)-('0_1' in ok),n_nontrivial=87,
                                errors=' '.join(misses)))
    write_rows(out/'poster_summary.csv',summary)
    print('PyMuPDF',fitz.VersionBind,'MuPDF',fitz.VersionFitz)
    for x in summary:print(x['size'],x['threshold'],x['correct'],x['n'])
if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--poster',type=Path,default=Path('knottable_v1.svg'))
    p.add_argument('--out',type=Path,default=Path('results'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);run(a.poster,a.out)
