"""Check that claimed within-drawing rotations/reflections preserve radial ink profiles.

Original drawings are first deformed individually. Directory-order numbering of
successive rotated/reflected variants is tested against an image-only metric
(rotation/reflection-invariant radial mass about the pixel-grid center). Compare
each nonrepresentative base variant with the first variant of every candidate
source block from the same crossing class; exclude untestable single-source
classes and single-variant blocks. Run: python validate_source_blocks.py TAR OUT
"""
from collections import Counter
from pathlib import Path
import io
import sys
import tarfile
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from ink_features import dump_csv
from source_folds import ORIGINALS

def main(archive, output):
    y,x=np.ogrid[:512,:512]
    radius=np.floor(np.hypot(y-255.5,x-255.5)).astype(int).ravel()
    profiles={}
    with tarfile.open(archive) as tar:
        for m in tar:
            p=Path(m.name)
            if len(p.parts)!=3 or p.parts[0]!='train' or p.suffix!='.png' or not p.stem.isdigit():continue
            c=int(p.parts[1]);n=int(p.stem)
            if n >= ORIGINALS[c]*round(49/ORIGINALS[c]):continue
            with Image.open(io.BytesIO(tar.extractfile(m).read())) as img:
                mass=(255-np.asarray(img.convert('L'))).ravel()
            hist=np.bincount(radius,weights=mass)
            pr=gaussian_filter1d(hist,2)
            profiles[c,n]=pr/pr.sum()
    output_rows=[]
    for (c,n),profile in sorted(profiles.items()):
        k=round(49/ORIGINALS[c]); true_slot=n//k
        if ORIGINALS[c]==1 or n%k==0:continue
        distances=[np.linalg.norm(profile-profiles[c,candidate*k]) for candidate in range(ORIGINALS[c])]
        closest=int(np.argmin(distances))
        second_nearest=np.min([dist for i,dist in enumerate(distances) if i!=true_slot])
        output_rows.append(dict(label=c,variant=n,source_slot=true_slot,nearest_slot=closest,exact=int(closest==true_slot),
                           within_block_distance=distances[true_slot],nearest_other_block_distance=second_nearest))
    dump_csv(output/'source_block_checks.csv',output_rows)
    print('Number matched',sum(r['exact'] for r in output_rows),'/',len(output_rows))
    print('Class totals',Counter(r['label'] for r in output_rows))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
