"""Compare every archived scaling descendant to the reconstructed bilinear resize.
Run: python validate_lineage.py DATASET.tar.gz OUTPUT_DIRECTORY
Work in dark-ink coordinates on white and use floating-point PIL to avoid an
intermediate integer rounding stage. A difference of one level captures
implementation/rounding differences in torchvision vs Pillow.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import io
import sys
import tarfile
import numpy as np
from PIL import Image
from ink_features import dump_csv
from source_folds import ORIGINALS


def main(archive,output):
    bins=defaultdict(dict)
    with tarfile.open(archive) as tar:
        for m in tar:
            path=Path(m.name)
            if len(path.parts)==3 and path.parts[0]=='train' and path.suffix=='.png' and path.stem.isdigit():
                bins[int(path.parts[1])][int(path.stem)]=tar.extractfile(m).read()
    rows=[]
    for c,items in sorted(bins.items()):
        n=len(items)//21
        assert n==ORIGINALS[c]*round(49/ORIGINALS[c])
        for parent in range(n):
            with Image.open(io.BytesIO(items[parent])) as orig:
                image=orig.convert('F')
            for trim in range(1,21):
                child=n-1+20*parent+trim
                with Image.open(io.BytesIO(items[child])) as im:
                    actual=np.array(im,dtype=np.int16)
                inner=np.asarray(image.resize((512-2*trim,512-2*trim),Image.Resampling.BILINEAR))
                expected=np.pad(np.rint(inner).astype(np.int16),trim,constant_values=255)
                diff=abs(actual-expected)
                ratio=diff.sum()/max(1,(255-actual).sum())
                rows.append(dict(label=c,parent=parent,source_slot=parent//round(49/ORIGINALS[c]),child=child,trim=trim,relative_abs_pixel_error=ratio,max_8bit_error=int(diff.max()),px_error_gt_1=int((diff>1).sum())))
    dump_csv(output/'resize_lineage_checks.csv',rows)
    print('Validated',len(rows),'children, relative error min/median/max',np.quantile([r['relative_abs_pixel_error'] for r in rows],[0,0.5,1]))
    print('Max 8-bit pixel error',max(r['max_8bit_error'] for r in rows),'files with err>1:',sum(r['px_error_gt_1']>0 for r in rows))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
