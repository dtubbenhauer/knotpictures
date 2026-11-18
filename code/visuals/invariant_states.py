"""Three views of the closed braid (sigma_1 sigma_2^{-1})^2."""
from pathlib import Path
from itertools import product
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch

COLORS = ['#146b7e','#a34e2e','#765788']
INK = '#243a46'
BRAID = (0, 1, 0, 1)

class UnionFind:
    def __init__(self): self.parents={}
    def root(self,x):
        self.parents.setdefault(x,x)
        while self.parents[x]!=x: x=self.parents[x]
        return x
    def connect(self,a,b): self.parents[self.root(a)]=self.root(b)


def braid(state):
    pieces=[];graph=UnionFind()
    def append(a,b,points,over=False):
        graph.connect(a,b);pieces.append((a,points,over))
    for j,switch in enumerate(BRAID):
        top,bottom=4-j,3-j
        for i in range(3):
            if i not in (switch,switch+1):
                append((j,i),(j+1,i),[(i,top),(i,top-.3),(i,bottom+.3),(i,bottom)])
        kind=state[j] if state else None
        vertical=(kind=='A') if switch==0 else (kind=='B')
        i=switch
        if kind is None:
            left=((j,i),(j+1,i+1),[(i,top),(i,top-.32),(i+1,bottom+.32),(i+1,bottom)])
            right=((j,i+1),(j+1,i),[(i+1,top),(i+1,top-.32),(i,bottom+.32),(i,bottom)])
            under,over=(right,left) if i==0 else (left,right)
            append(*under);append(*over,over=True)
        elif vertical:
            for x in (i,i+1): append((j,x),(j+1,x),[(x,top),(x,top-.3),(x,bottom+.3),(x,bottom)])
        else:
            append((j,i),(j,i+1),[(i,top),(i,top-.6),(i+1,top-.6),(i+1,top)])
            append((j+1,i),(j+1,i+1),[(i,bottom),(i,bottom+.6),(i+1,bottom+.6),(i+1,bottom)])
    for x in range(3):
        r=2.48+(2-x)*.36; h=.23+(2-x)*.20
        pts=[(x,0),(x,-h),(r,-h),(r,0),(r,1),(r,3),(r,4),(r,4+h),(x,4+h),(x,4)]
        append((4,x),(0,x),pts)
    components={graph.root(a) for a,_,_ in pieces}
    colors={component:COLORS[i] for i,component in enumerate(sorted(components))}
    return [(points,colors[graph.root(a)] if state else INK,over) for a,points,over in pieces],len(components)


def render(ax,state):
    for points,color,over in braid(state)[0]:
        path=PlotPath(points,[PlotPath.MOVETO]+[PlotPath.CURVE4]*(len(points)-1))
        if over:
            ax.add_patch(PathPatch(path,fill=False,lw=6.5,color='white',capstyle='butt'))
        ax.add_patch(PathPatch(path,fill=False,lw=1.85,color=color,capstyle='round'))
    ax.set(xlim=(-.33,3.35),ylim=(-.60,4.60));ax.set_aspect('equal');ax.axis('off')


if __name__=='__main__':
    histogram=Counter((s.count('B'),braid(s)[1]) for s in product('AB',repeat=4))
    assert histogram=={(0,3):1,(1,2):4,(2,1):5,(2,3):1,(3,2):4,(4,3):1}
    fig,axes=plt.subplots(1,3,figsize=(5.53,2.36)); titles=['The figure-eight knot',r'$ABAB$: three circles',r'$BABA$: one circle']
    for ax,state,title in zip(axes,[None,'ABAB','BABA'],titles):
        render(ax,state);ax.set_title(title,fontsize=9,pad=2)
    fig.subplots_adjust(left=.015,right=.99,top=.91,bottom=.008,wspace=.12)
    fig.savefig(Path(__file__).with_name('invariant_states.pdf'));plt.close(fig)
