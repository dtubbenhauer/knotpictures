"""Draw the trefoil of the printed PD code as an editable vector autograph."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# These are the coordinates used in the checked six-edge drawing in Appendix A.1.
# Smaller parameter index is the starting point for its corresponding edge.
angles = np.pi / 6 + np.arange(6) * np.pi / 3
under = angles[[1,3,5]]
fig,ax = plt.subplots(figsize=(1,1))
for start, stop in zip(under, np.r_[under[1:], under[0]+2*np.pi]):
    t = np.linspace(start+.074,stop-.074,800)
    r=2+.86*np.cos(3*t)
    arg=2*t+np.pi/2
    ax.plot(-r*np.cos(arg),r*np.sin(arg),color='#243a46',lw=1.5,solid_capstyle='round')
ax.set_aspect('equal'); ax.axis('off'); fig.subplots_adjust(0,0,1,1)
fig.savefig(Path(__file__).resolve().with_name('trefoil-programme.pdf'),bbox_inches='tight',pad_inches=.015)
plt.close(fig)
