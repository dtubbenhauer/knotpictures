"""Render and independently check a trefoil with successively filled crossings."""
from collections import Counter
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from figures import INK, TAU, save, xy

HERE = Path(__file__).resolve().parent
CHECKS = ((6,3,0), (4,1,0), (2,1,2), (0,1,4))
T_VISITS = np.pi / 6 + np.arange(6) * np.pi / 3
GAP = .062


def polylines(filled):
    gaps = [t for i,t in enumerate(T_VISITS) if i % 3 >= filled and np.sin(3*t) < 0]
    if not gaps:
        return [xy(np.linspace(0,TAU,3500))]
    boundaries = gaps + [gaps[0] + TAU]
    return [xy(np.linspace(a + GAP,b - GAP,1600)) for a,b in zip(boundaries[:-1],boundaries[1:])]


def graph_counts(filled):
    # Traverse the parameter circle in increasing order, marking all six visits.
    # A filled crossing identifies both visits. At an unfilled crossing the
    # underpassing visit splits into two endpoints; the overpass stays continuous.
    incoming, outgoing = [], []
    for i,t in enumerate(T_VISITS):
        crossing = i % 3
        if crossing < filled:
            before = after = ('junction',crossing)
        elif np.sin(3*t) > 0:
            before = after = ('over',crossing)
        else:
            before, after = ('left',crossing), ('right',crossing)
        incoming.append(before)
        outgoing.append(after)
    edges = [(outgoing[i], incoming[(i + 1) % 6]) for i in range(6)]
    degree = Counter(vertex for edge in edges for vertex in edge)
    groups = [{vertex} for vertex in degree]
    for left,right in edges:
        group_l = next(g for g in groups if left in g)
        group_r = next(g for g in groups if right in g)
        if group_l is not group_r:
            group_l.update(group_r)
            groups.remove(group_r)
    endpoints = sum(val == 1 for val in degree.values())
    bounded_regions = len(edges) - len(degree) + len(groups)
    return endpoints, len(groups), bounded_regions


def raster_counts(lines):
    image = Image.new('L', (900,900))
    drawing = ImageDraw.Draw(image)
    for line in lines:
        pixels = np.rint(127*line + 450).astype(int)
        drawing.line(list(map(tuple,pixels)),fill=255,width=6,joint='curve')
    ink = np.asarray(image) > 0
    _,b = label(ink, structure=np.ones((3,3),dtype=int))
    _,white_regions = label(~ink)  # four-connected complement
    return b,white_regions-1


def main():
    results=[]
    fig, axes=plt.subplots(1,4,figsize=(6.6,1.75),gridspec_kw={'wspace':.09})
    for filled,axis in enumerate(axes):
        lines = polylines(filled)
        e,b,f = graph_counts(filled)
        rb,rf = raster_counts(lines)
        assert (e,b,f) == CHECKS[filled]
        assert (rb,rf) == (b,f)
        assert e-b+f == 3
        results.append((filled,e,b,f,rb,rf))
        for points in lines:
            axis.plot(points[:,0],points[:,1],color=INK,lw=1.95,solid_capstyle='round')
        axis.set(xlim=(-3.0,3.0),ylim=(-2.64,3.00),aspect='equal')
        axis.axis('off')
        axis.text(.5,1.03,f'{filled} filled',transform=axis.transAxes,ha='center',va='bottom',fontsize=9.0)
        axis.text(.5,-.05,rf'$(e,b,f)=({e},{b},{f})$',transform=axis.transAxes,ha='center',va='top',fontsize=11.2)
    save(fig, HERE, 'mixed-junctions')
    with (HERE/'mixed-junctions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('filled','endpoints','components','bounded_white_regions','raster_components','raster_bounded_regions'))
        writer.writerows(results)


if __name__=='__main__':
    main()
