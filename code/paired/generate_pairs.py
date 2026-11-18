"""Render, geometrically verify, and measure a seed-grouped closed-braid benchmark."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.ndimage import label

SIDE = 160
SUPERSAMPLE = 4
SEED = 73129
BODY_VERTICES = 241


def torus_path(t, k):
    radius = 1 + 0.24 * np.cos(k * t)
    return np.column_stack((radius * np.cos(2 * t), radius * np.sin(2 * t)))


def curve(group, k, curl):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, group]))
    delta = rng.uniform(0.085, 0.117)
    # Both versions use the same arc from just after the insertion to just before it.
    t = np.linspace(delta, 2 * np.pi - delta, BODY_VERTICES)
    body = torus_path(t, k)
    if curl:
        # Local coordinates run along the chord and outwards from the braid.
        u = [-1, -.67, -.26, .52, .67, .33, -.23, -.67, -.54, .21, .68, 1]
        v = [0, .10, .41, 1.19, 1.75, 2.11, 2.16, 1.79, 1.23, .52, .13, 0]
        control = np.column_stack((body[0, 0] + body[0, 1] * np.array(v),
                                   body[0, 1] * np.array(u)))
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(control, axis=0), axis=1))]
        patch = CubicSpline(s, control, axis=0)(np.linspace(0, s[-1], 68))[1:-1]
        zp = np.linspace(1., -1., len(patch))
    else:
        tp = np.linspace(-delta, delta, 17)[1:-1]
        patch, zp = torus_path(tp, k), np.sin(k * tp)
    p = np.vstack((body, patch))
    z = np.r_[np.sin(k*t), zp]
    # One shared smooth radial deformation, twist, affine map, and translation
    # per group. The maps are injective: the radius ripple depends only on the
    # original angle; the subsequent twist depends only on the resulting radius.
    angle = np.arctan2(p[:, 1], p[:, 0]); radius = np.linalg.norm(p, axis=1)
    angle_a, angle_b = rng.uniform(0, 2*np.pi, 2)
    radius *= 1 + rng.uniform(.035, .13)*np.cos(3*angle + angle_a) + rng.uniform(.02, .075)*np.sin(2*angle + angle_b)
    angle += rng.uniform(-.42, .42)*(radius-1.) + rng.uniform(0, 2*np.pi)
    p = radius[:,None]*np.column_stack((np.cos(angle),np.sin(angle)))
    p = p @ np.array([[rng.uniform(.86, 1.11),rng.uniform(-.12,.12)], [0, rng.uniform(.87,1.11)]])
    # Make pixels y-down while preserving the mathematical diagram's handedness.
    p[:, 1] *= -1
    p = p*rng.uniform(30.6, 36.0) + SIDE/2 + rng.uniform(-3,3,2)
    return p, z


def cross(a,b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def intersections(p, z):
    """Enumerate all proper non-neighbor segment intersections and both visits."""
    n = len(p); d = np.roll(p,-1,axis=0)-p
    dist = np.linalg.norm(d,axis=1); arclength = np.r_[0,np.cumsum(dist)]
    i,j = np.triu_indices(n,k=2); valid = ~((i==0)&(j==n-1)); i,j=i[valid],j[valid]
    delta = p[j]-p[i]; denominator = cross(d[i],d[j]); valid=abs(denominator)>1e-9
    i,j,delta,denominator=i[valid],j[valid],delta[valid],denominator[valid]
    a,b=cross(delta,d[j])/denominator,cross(delta,d[i])/denominator
    valid=(a>0)&(b>0)&(a<1)&(b<1); i,j,a,b,denominator=i[valid],j[valid],a[valid],b[valid],denominator[valid]
    zi=z[i]+a*(z[(i+1)%n]-z[i]); zj=z[j]+b*(z[(j+1)%n]-z[j])
    assert np.all(abs(zi-zj)>0.05), 'ambiguous strand heights'
    visits=np.column_stack((arclength[i]+a*dist[i],arclength[j]+b*dist[j]))
    first_over=zi>zj
    sins=abs(denominator)/(dist[i]*dist[j]); points=p[i]+a[:,None]*d[i]
    return arclength,visits,first_over,sins,points


def polylines_to_draw(p, s, gaps):
    """Cut an oriented circle at the given arclength intervals; return ink paths."""
    total=s[-1]; P=np.vstack((p,p[0])); gaps = sorted((pos%total,half) for pos,half in gaps)
    for idx,(pos,half) in enumerate(gaps):
        nxt,next_half=gaps[(idx+1)%len(gaps)]
        lo=pos+half; hi=nxt-next_half+(total if idx==len(gaps)-1 else 0.)
        if lo>=hi:
            raise ValueError('two underpass gaps overlap')
        internal=np.r_[s[(s>lo)&(s<hi)],total+s[(total+s>lo)&(total+s<hi)]]
        sample=np.r_[lo,np.unique(internal),hi]%total
        yield np.column_stack((np.interp(sample,s,P[:,0]),np.interp(sample,s,P[:,1])))


def render(p, info, style='clean', switch=False):
    s,visits,first_over,sins,points=info
    keep=first_over.copy()
    if switch:
        keep[0]=~keep[0]
    under=visits[np.arange(len(visits)),keep.astype(int)]
    width=2.1 if style!='thin' else 1.2
    halo=1.1 if style!='thin' else .5
    gaps=list(zip(under,(width+halo)/sins))
    if style=='break':
        # Insert one accidental gap midway along the longest surviving run,
        # away from every crossing on either strand.
        candidates=np.linspace(0,s[-1],200,endpoint=False)
        visits_flat=visits.flatten(); sep=abs(candidates[:,None]-visits_flat)
        sep=np.minimum(sep,s[-1]-sep)
        gaps.append((candidates[np.argmax(sep.min(axis=1))],2.5))
    im=Image.new('L',(SIDE*SUPERSAMPLE,)*2,255); draw=ImageDraw.Draw(im)
    for line in polylines_to_draw(p,s,gaps):
        xy=[tuple(v) for v in line*SUPERSAMPLE]; draw.line(xy,fill=0,width=round(width*SUPERSAMPLE),joint='curve')
    if style=='thin':
        im=im.resize((72,72),Image.Resampling.LANCZOS).resize((SIDE,SIDE),Image.Resampling.BILINEAR)
    else:
        im=im.resize((SIDE,SIDE),Image.Resampling.LANCZOS)
    return im


def otsu_count(im):
    x=np.asarray(im); h=np.bincount(x.ravel(),minlength=256).astype(float); w=h.cumsum(); a=(np.arange(256)*h).cumsum()
    valid=(w>0)&(w<w[-1]); score=np.full(256,-1.); score[valid]=(a[-1]*w[valid]-a[valid]*w[-1])**2/(w[valid]*(w[-1]-w[valid]))
    threshold=int(score.argmax()); _,n=label(x<=threshold,structure=np.ones((3,3)))
    return threshold,n


def main(output):
    output.mkdir(parents=True,exist_ok=True); rows=[]
    for group in range(54):
        split='train' if group<30 else ('val' if group<38 else 'test')
        for k in [1,3,5,7,9]:
            for curl in [0,1]:
                p,z=curve(group,k,curl); info=intersections(p,z)
                # A genuine local type-I insertion has one self-crossing in its
                # inserted arc and no new intersection with the retained braid.
                visits_in_patch = info[1] > info[0][BODY_VERTICES - 1]
                assert np.count_nonzero(~visits_in_patch.any(axis=1)) == k
                assert np.count_nonzero(visits_in_patch.all(axis=1)) == curl
                assert not np.any(visits_in_patch[:, 0] != visits_in_patch[:, 1])
                assert min(p.min(),SIDE-p.max())>2,(group,k,curl,'clipped')
                for style in (['clean','thin','break'] if split=='test' else ['clean']):
                    im=render(p,info,style); threshold,count=otsu_count(im)
                    filename=f'images/{style}/g{group:03d}_k{k:02d}_c{curl}.png'; path=output/filename; path.parent.mkdir(parents=True,exist_ok=True); im.save(path)
                    rows.append(dict(image=filename,count=k+curl,split=split,group=group,k=k,curl=curl,style=style,geometric_count=len(info[1]),components=count,otsu=threshold))
    with (output/'manifest.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    # Same immersed curve and rendering geometry; exactly one crossing is switched.
    figs=output/'examples'; figs.mkdir(exist_ok=True)
    for name,k,curl,switch in [('trefoil',3,0,False),('trefoil-curl',3,1,False),('switched-unknot',3,0,True)]:
        p,z=curve(38,k,curl); info=intersections(p,z); im=render(p,info,switch=switch); im.save(figs/(name+'.png'))
    print('wrote',len(rows),'images')
    for split in ['train','val','test']:
        for style in ['clean','thin','break']:
            sub=[r for r in rows if r['split']==split and r['style']==style]
            if sub: print(split,style,len(sub),sum(r['count']==r['components'] for r in sub),'mean absolute error',np.mean([abs(r['count']-r['components']) for r in sub]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    main(parser.parse_args().output)
