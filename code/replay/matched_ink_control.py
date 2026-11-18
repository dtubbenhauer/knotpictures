"""Apply the four-feature classifier to precisely the compact CNN's train/test manifest.
Run after ink_features.py: python matched_ink_control.py GLOBAL_FEATURES_DIR CNN_MANIFEST_DIR
"""
from pathlib import Path
from collections import Counter
import csv
import sys
import numpy as np
from ink_features import FEATURE_NAMES,dump_csv,run

def main(directory,cnn):
    with (directory/'global_ink_features.csv').open() as f:
        allrows={r['path']:r for r in csv.DictReader(f)}
    for r in allrows.values():
        r['expected']=int(r['expected'])
        for k in FEATURE_NAMES:r[k]=float(r[k])
    with (cnn/'archive_split.csv').open() as f:
        train=[allrows[r['image']] for r in csv.DictReader(f) if r['split']=='train']
    assert len(train)==1192
    with (cnn/'matched_17.csv').open() as f:cases=list(csv.DictReader(f))
    out=[];stats=[];class_stats=[]
    for arm in ['seen_source','unseen_source']:
        group=[r for r in cases if r['arm']==arm]
        assert len(group)==1054
        rows=[allrows[r['image']] for r in group]
        preds=run(rows,train,arm)
        assert not any(r['same_file_as_training'] for r in preds)
        out+=preds
        counts=Counter(r['expected'] for r in preds);by_c=[]
        for c,n in sorted(counts.items()):
            s=[r for r in preds if r['expected']==c]
            acc=np.mean([r['exact'] for r in s]); mae=np.mean([r['abs_error'] for r in s])
            by_c.append(dict(arm=arm,label=c,n=n,correct=sum(r['exact'] for r in s),accuracy=acc,mae=mae))
        class_stats+=by_c
        stats.append(dict(arm=arm,n=len(group),correct=sum(r['exact'] for r in preds),
                          mae=np.mean([r['abs_error'] for r in preds]),macro_label_accuracy=np.mean([r['accuracy'] for r in by_c]),
                          macro_label_mae=np.mean([r['mae'] for r in by_c])))
    dump_csv(directory/'matched_ink_predictions.csv',out)
    dump_csv(directory/'matched_ink_summary.csv',stats)
    dump_csv(directory/'matched_ink_classes.csv',class_stats)
    print(stats)
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
