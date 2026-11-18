"""Explore the Euler/endpoints count on images with mixed crossing conventions."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, convolve

from generate_pairs import SIDE, SUPERSAMPLE, curve, intersections, otsu_count, polylines_to_draw


def thin_zhang_suen(mask):
    """Thin a binary mask with the parallel two-subiteration Zhang–Suen rule."""
    ink = np.pad(mask, 1).astype(np.uint8)
    for _ in range(max(ink.shape)):
        changes = 0
        for step in (0, 1):
            north = ink[:-2, 1:-1]
            north_east = ink[:-2, 2:]
            east = ink[1:-1, 2:]
            south_east = ink[2:, 2:]
            south = ink[2:, 1:-1]
            south_west = ink[2:, :-2]
            west = ink[1:-1, :-2]
            north_west = ink[:-2, :-2]
            ring = [north, north_east, east, south_east, south, south_west, west, north_west]
            neighbors = sum(ring)
            transitions = sum((v == 0).astype(np.uint8) * w for v, w in zip(ring, ring[1:] + ring[:1]))
            if step == 0:
                triple_a, triple_b = north * east * south, east * south * west
            else:
                triple_a, triple_b = north * east * west, north * south * west
            center = ink[1:-1, 1:-1]
            remove = (center == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & (triple_a == 0) & (triple_b == 0)
            changes += remove.sum()
            center[remove] = 0
        if changes == 0:
            return ink[1:-1, 1:-1]
    raise RuntimeError('thinning did not converge')


def measure(im):
    x = np.asarray(im)
    threshold, components = otsu_count(im)
    ink = x <= threshold
    _, white_components = label(~ink)
    bounded = white_components - 1
    skeleton = thin_zhang_suen(ink)
    neighbors = convolve(skeleton, np.ones((3, 3), dtype=np.uint8), mode='constant') - skeleton
    tips = np.count_nonzero((skeleton != 0) & (neighbors == 1))
    return tips, bounded, components, tips + bounded - components


def mixed_image(p, info, group, k, curl, force_joined=False):
    """Open either both underpass half-edges or neither at each intersection."""
    s, visits, first_over, sine, _ = info
    under = visits[np.arange(len(visits)),first_over.astype(int)]
    generator = np.random.default_rng(np.random.SeedSequence([948271,group,k]))
    # Mark about half of the crossings as open, independently of the geometry.
    opened = generator.random(len(visits)) < 0.5
    if force_joined:
        opened[:] = False
    gaps = list(zip(under[opened],3.2 / sine[opened]))
    im = Image.new('L',(SIDE*SUPERSAMPLE,)*2,255)
    drawing = ImageDraw.Draw(im)
    paths = polylines_to_draw(p,s,gaps) if gaps else [np.vstack([p,p[0]])]
    for line in paths:
        drawing.line([tuple(a) for a in line*SUPERSAMPLE],fill=0,width=round(2.1*SUPERSAMPLE),joint='curve')
    return im.resize((SIDE,SIDE),Image.Resampling.LANCZOS), int(opened.sum())


def main(root):
    with (root/'manifest.csv').open() as stream:
        rows = [rec for rec in csv.DictReader(stream) if rec['split']=='test']
    output = []
    paths = []
    (root/'images/mixed').mkdir(parents=True, exist_ok=True)
    for rec in rows:
        im = Image.open(root/rec['image'])
        tips, bounded, components, estimate = measure(im)
        output.append(dict(style=rec['style'],group=rec['group'],k=rec['k'],curl=rec['curl'],count=rec['count'],opened=int(rec['count']),tips=tips,white_bounded=bounded,ink_components=components,estimate=estimate))
        if rec['style']=='clean':
            g,k,c = (int(rec[key]) for key in ['group','k','curl'])
            p,z = curve(g,k,c);info=intersections(p,z)
            im,n_open = mixed_image(p,info,g,k,c)
            relative_path = Path('images/mixed') / Path(rec['image']).name
            im.save(root / relative_path)
            paths.append(dict(image=relative_path.as_posix(), group=g, k=k, curl=c, count=k+c, opened=n_open))
            tips, bounded, components, estimate = measure(im)
            output.append(dict(style='mixed',group=g,k=k,curl=c,count=k+c,opened=n_open,tips=tips,white_bounded=bounded,ink_components=components,estimate=estimate))
    with (root/'mixed-images.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paths[0]))
        writer.writeheader()
        writer.writerows(paths)
    with (root/'mixed-counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    for style in ['clean','break','thin','mixed']:
        selected=[r for r in output if r['style']==style]
        err=[r['estimate']-int(r['count']) for r in selected]
        print(style,'exact',err.count(0),'of',len(selected),'absolute error sum',sum(map(abs,err)),'correct tip count',sum(r['tips']==2*r['opened'] for r in selected))


def illustrate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    group,k,curl=38,5,1
    p,z=curve(group,k,curl);info=intersections(p,z)
    images=[Image.open(root/'images/clean/g038_k05_c1.png'),
            mixed_image(p,info,group,k,curl)[0],
            mixed_image(p,info,group,k,curl,force_joined=True)[0]]
    titles=['Underpass gap at every crossing','Two crossings with gaps','All crossings joined']
    ink_colors=['#242A34','#346D78','#664E81','#397157','#925A35','#3F618E']
    white_colors=['#C4DFF9','#EBCCC3','#D3CAE5','#D1E5C5','#FBE1B9','#F1DBE7','#B8DEDD']
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.55))
    yy,xx=np.where(np.asarray(images[2])<225);bounds=(xx.min()-5,xx.max()+5,yy.max()+6,yy.min()-5)
    for ax,title,im in zip(axes,titles,images):
        x=np.asarray(im);threshold,n_ink=otsu_count(im); ink=x<=threshold
        smooth=thin_zhang_suen(ink);neighbors=convolve(smooth,np.ones((3,3),dtype=np.uint8),mode='constant')-smooth
        ends_y,ends_x=np.where((smooth>0)&(neighbors==1)); tips=len(ends_y)
        white,n_white=label(~ink);outside=white[0,0]; bounded=[i for i in range(1,n_white+1) if i!=outside]
        rgb=np.ones((*x.shape,3));fg,_=label(ink,structure=np.ones((3,3)))
        for component in range(1,n_ink+1): rgb[fg==component]=to_rgb(ink_colors[(component-1)%len(ink_colors)])
        for area,color in zip(bounded,white_colors): rgb[white==area]=to_rgb(color)
        ax.imshow(rgb,interpolation='nearest');ax.scatter(ends_x,ends_y,s=14,c='#A01929',edgecolors='white',linewidths=.35)
        ax.set_title(title,fontsize=9);ax.set_xlabel(f'{tips} + {len(bounded)} − {n_ink} = {tips + len(bounded) - n_ink}',fontsize=11)
        ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=.012,right=.988,top=.89,bottom=.15,wspace=.05)
    fig.savefig(root/'examples/mixed-euler.pdf',dpi=230);plt.close(fig)


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    main(root)
    illustrate(root)
