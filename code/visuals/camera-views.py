"""Draw three orthographic views of the same trefoil polygon. No external data.
Run with numpy and matplotlib: python camera-views.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VIEWS = [(90, 0), (19, 103.1), (13, 91.1)]
EXPECTED = (3, 4, 6)


def determinant(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def projection(size, elevation, azimuth):
    t = (np.arange(size) + .17) * 2*np.pi/size
    r = 2 + np.cos(3*t)
    xyz = np.column_stack([r*np.cos(2*t), r*np.sin(2*t), np.sin(3*t)])
    a, b = np.deg2rad([azimuth, elevation])
    right = [-np.sin(a), np.cos(a), 0]
    up = [-np.sin(b)*np.cos(a), -np.sin(b)*np.sin(a), np.cos(b)]
    toward = [np.cos(b)*np.cos(a), np.cos(b)*np.sin(a), np.sin(b)]
    xy = xyz @ np.transpose([right, up])
    depth = xyz @ toward
    d = np.roll(xy, -1, axis=0) - xy
    dz = np.roll(depth, -1) - depth
    lengths = np.linalg.norm(d, axis=1)
    s = np.r_[0, np.cumsum(lengths)]
    first, second = np.triu_indices(size, k=2)
    ok = second-first < size-1
    first, second = first[ok], second[ok]
    offset = xy[second]-xy[first]
    denom = determinant(d[first], d[second])
    generic = abs(denom) > 1e-10
    fac = np.where(generic, denom, 1.)
    t1 = determinant(offset, d[second]) / fac
    t2 = determinant(offset, d[first]) / fac
    cross = generic & (t1>0) & (t1<1) & (t2>0) & (t2<1)
    gaps = []
    for i,j,u,v,area in zip(first[cross],second[cross],t1[cross],t2[cross],denom[cross]):
        z1, z2 = depth[i]+u*dz[i], depth[j]+v*dz[j]
        assert abs(z1-z2) > .015
        h, param = (i,u) if z1 < z2 else (j,v)
        sine = abs(area)/(lengths[i]*lengths[j])
        gaps.append((s[h]+param*lengths[h], .145/sine))
    # Duplicate a periodic wrapping gap before and after the drawing period.
    mask_intervals = sorted((max(0, g-width+shift), min(s[-1],g+width+shift))
                            for g,width in gaps for shift in (-s[-1],0,s[-1])
                            if g+width+shift > 0 and g-width+shift < s[-1])
    curves = []
    start = 0
    full = np.vstack([xy,xy[0]])
    for end,resume in mask_intervals + [(s[-1],s[-1])]:
        if end > start:
            locs = np.r_[start,s[(s>start)&(s<end)],end]
            curves.append(np.column_stack([np.interp(locs,s,full[:,i]) for i in (0,1)]))
        start = max(start,resume)
    return curves, len(gaps)


def main():
    fig, axes = plt.subplots(1,3,figsize=(6.4,1.95))
    for ax, view, expected in zip(axes,VIEWS,EXPECTED):
        paths, count = projection(720,*view)
        assert count == expected == projection(481,*view)[1]
        for segment in paths:
            ax.plot(*segment.T, color='#274B5D', lw=1.65, solid_capstyle='round')
        ax.set_aspect('equal')
        ax.set(xlim=(-3.05,3.05),ylim=(-3,3))
        ax.set_axis_off()
        ax.text(.5,-.005, f'{count} crossings', transform=ax.transAxes,ha='center',va='top',
                fontsize=10.5, family='serif')
    fig.subplots_adjust(left=.01,right=.99,bottom=.15,top=.96,wspace=.03)
    path = Path(__file__).resolve().with_name('camera-views.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=.035, metadata={'Title':'Three views of the same trefoil'})
    plt.close(fig)

if __name__=='__main__':
    main()
