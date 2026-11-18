"""Redraw Millett's Culprit from the crossing data of Henrich–Kauffman Fig. 15.

The ordered integer column pairs below are the horizontal-endpoint data of
the grid in Fig. 15 of arXiv:1006.4176, read from the top down (zero-based).
Vertical passes go over horizontal passes. Only this combinatorial information
is retained in the drawing; positions are recomputed from its faces.
"""
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROWS = [(16,18),(12,14),(13,16),(14,17),(0,11),(6,12),(11,15),
        (13,17),(7,15),(6,8),(4,7),(2,3),(3,5),(1,4),(0,2),
        (9,18),(8,10),(1,9),(5,10)]
INC = [(1,0),(0,-1),(-1,0),(0,1)]  # counterclockwise in image coordinates
grid = defaultdict(set)

def connect_line(points):
    for a,b in zip(points,points[1:]):
        grid[a].add(b)
        grid[b].add(a)

for row,(left,right) in enumerate(ROWS):
    connect_line([(col,row) for col in range(left,right+1)])
for col in range(len(ROWS)):
    lo,hi = [row for row,ends in enumerate(ROWS) if col in ends]
    connect_line([(col,row) for row in range(lo,hi+1)])

crosses = sorted(v for v in grid if len(grid[v]) == 4)
assert len(crosses) == 10
twin = {}
for i,(x,y) in enumerate(crosses):
    for d,(dx,dy) in enumerate(INC):
        prev,v = (x,y),(x+dx,y+dy)
        while v not in crosses:
            (w,) = grid[v] - {prev}
            prev,v = v,w
        back = INC.index((prev[0]-v[0],prev[1]-v[1]))
        twin[i,d] = (crosses.index(v),back)
assert all(twin[twin[p]] == p for p in twin)
faces,seen = [],set()
for start in twin:
    face=[]
    p=start
    while p not in seen:
        seen.add(p)
        face.append(p)
        v,d = twin[p]
        p = (v,(d+1)%4)
    if face:
        assert p == start
        faces.append(face)
        assert len({v for v,d in face}) == len(face)
assert len(faces) == 12

# Every edge of a bigon or triangle alternates over/under between its ends.
# In this grid vertical at the crossing means over.
def mixed(edge):
    return edge[1]%2 != twin[edge][1]%2

assert min(map(len,faces)) == 2
assert all(all(map(mixed,face)) for face in faces if len(face) <= 3)
print('Boundary lengths', dict(sorted(Counter(map(len,faces)).items())))

# Barycentrically subdivide the spherical shadow. Harmonic extension from
# the boundary of an external face determines every position in the figure.
def V(p): return ('v',p[0])
def E(p): return ('e',min(p,twin[p]))
edges = defaultdict(set)
def link(p,q):
    edges[p].add(q)
    edges[q].add(p)

def grid_area(face):
    points=[]
    for i,d in face:
        start=crosses[i]
        dx,dy=INC[d]
        prev,curr=start,(start[0]+dx,start[1]+dy)
        points.append(start)
        while curr not in crosses:
            points.append(curr)
            (nxt,)=grid[curr]-{prev}
            prev,curr=curr,nxt
    return sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1]))

# Keep the unbounded face of the reference arc presentation. It is the unique
# boundary walk with negative area in the top-to-bottom grid coordinates.
(outer,) = [f for f,F in enumerate(faces) if grid_area(F)<0]
for f,face in enumerate(faces):
    for p in face:
        link(V(p),E(p))
        link(E(p),V(twin[p]))
        if f != outer:
            link(('f',f),V(p))
            link(('f',f),E(p))
rim = [z for p in faces[outer] for z in (V(p),E(p))]
angle = np.linspace(np.pi*.1,np.pi*2.1,len(rim),endpoint=False)
coords = {z: np.array([np.cos(t),np.sin(t)]) for z,t in zip(rim,angle)}
interior=[z for z in edges if z not in coords]
matrix=np.zeros((len(interior),len(interior)))
rhs=np.zeros((len(interior),2))
for i,z in enumerate(interior):
    matrix[i,i]=len(edges[z])
    for w in edges[z]:
        if w in coords: rhs[i]+=coords[w]
        else: matrix[i,interior.index(w)]-=1
coords.update(zip(interior,np.linalg.solve(matrix,rhs)))

# Share drawing space more evenly amongst the little triangles while retaining
# their orientation; this leaves room between neighbouring overpass gaps.
def determinant(u,v):
    return u[...,0]*v[...,1]-u[...,1]*v[...,0]

nodes=interior+rim
all_xy=np.array([coords[z] for z in nodes])
triangles=[]
for i,F in enumerate(faces):
    if i == outer: continue
    for p in F:
        for a,b in ((V(p),E(p)),(E(p),V(twin[p]))):
            tri=[nodes.index(x) for x in (('f',i),a,b)]
            x,y,z=all_xy[tri]
            if determinant(y-x,z-x)<0: tri[1],tri[2]=tri[2],tri[1]
            triangles.append(tri)
triangles=np.array(triangles)
def spacing(xy):
    xy=np.vstack([xy.reshape(-1,2),all_xy[len(interior):]])
    a,b,c=xy[triangles].transpose((1,0,2))
    area=determinant(b-a,c-a)
    if np.any(area<=0):
        return 1e20,np.zeros(len(interior)*2)
    cost=-np.log(area).sum()
    g=np.zeros_like(xy)
    for k,d in enumerate((b-c,c-a,a-b)):
        # d area / d a = (b_y-c_y,c_x-b_x)
        dg=np.stack([d[:,1],-d[:,0]],axis=1)
        np.add.at(g,triangles[:,k],-dg/area[:,None])
    return cost,g[:len(interior)].ravel()
opt=minimize(spacing,all_xy[:len(interior)].ravel(),jac=True,
             method='BFGS',options={'maxiter':500,'gtol':1e-7})
coords.update(zip(interior,.53*opt.x.reshape(-1,2)+.47*all_xy[:len(interior)]))

# Each macroscopic edge uses its own midpoint. Rounding the embedded
# polygon locally follows the same cyclic order at every vertex.
def tangent(p):
    return coords[E(p)]-coords[V(p)]

def ray(p):
    opposite=(p[0],(p[1]+2)%4)
    u=tangent(p);v=tangent(opposite)
    u=u/np.linalg.norm(u)-v/np.linalg.norm(v)
    return u/np.linalg.norm(u)

def curve(p,q,trim=True):
    a,b,m=coords[V(p)],coords[V(q)],coords[E(p)]
    ta,tb=ray(p),ray(q)
    ca=a+ta*np.linalg.norm(a-m)*.47
    cb=b+tb*np.linalg.norm(b-m)*.47
    delta=(cb-ca)*.16
    pieces=[]
    for x,y,z,w in ((a,ca,m-delta,m),(m,m+delta,cb,b)):
        t=np.linspace(0,1,200)[:,None]
        pieces.append((1-t)**3*x+3*(1-t)**2*t*y+3*(1-t)*t**2*z+t**3*w)
    points=np.vstack((pieces[0],pieces[1][1:]))
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if not trim:
        return points
    # Leave a small physical gap only at an endpoint traversed horizontally.
    gap=.038
    keep=(dist>(0 if p[1]%2 else gap)) & (dist<dist[-1]-(0 if q[1]%2 else gap))
    return points[keep]

def draw():
    fig,ax=plt.subplots(figsize=(3.15,2.24))
    for p,q in twin.items():
        if p<q:
            points=curve(p,q)
            ax.plot(*points.T,lw=1.35,color='#233c49',solid_capstyle='round')
    # Labels identify one bounded bigon and one bounded triangle.
    for n,letter in ((2,'B'),(3,'T')):
        f=max((i for i,F in enumerate(faces) if i!=outer and len(F)==n),
              key=lambda i:abs(determinant(coords[E(faces[i][0])]-coords[V(faces[i][0])],
                                        coords[E(faces[i][1])]-coords[V(faces[i][0])])))
        xy=coords['f',f]
        ax.text(*xy,letter,ha='center',va='center',fontsize=7,weight='bold',
                color='#a34e2e',fontfamily='DejaVu Sans')
    ax.set_aspect('equal');ax.autoscale_view();ax.axis('off')
    fig.tight_layout(pad=.01)
    out=Path(__file__).resolve().parent/'culprit.pdf'
    fig.savefig(out,bbox_inches='tight',pad_inches=.01)
    plt.close(fig)

if __name__ == '__main__': draw()
