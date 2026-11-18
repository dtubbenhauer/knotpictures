"""Ten prespecified splits of the archived training folder, by image or distinct SHA-256.
Requires the feature table produced by ink_features.py. Run: python stability.py OUTPUT_DIR
"""
from collections import defaultdict
from pathlib import Path
import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from ink_features import dump_csv,FEATURE_NAMES,run

def report_eval(rows,split,seed,subset):
    return dict(split=split,seed=seed,subset=subset,correct=sum(r['exact'] for r in rows),
                total=len(rows),accuracy=np.mean([r['exact'] for r in rows]),
                mae=np.mean([r['abs_error'] for r in rows]),same_file_as_training=sum(r['same_file_as_training'] for r in rows))

def main(directory):
    with (directory/'global_ink_features.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['source']=='train']
    by_hash=defaultdict(list)
    for i,row in enumerate(rows):
        row['expected']=int(row['expected'])
        for k in FEATURE_NAMES:row[k]=float(row[k])
        by_hash[row['sha256']].append(i)
    keys=sorted(by_hash)
    for k in keys:assert len({rows[i]['expected'] for i in by_hash[k]})==1
    stats=[];out=[]
    for seed in range(10):
        for split in ('image','sha256'):
            if split=='image':
                tr,te=train_test_split(np.arange(len(rows)),test_size=.2,stratify=[r['expected'] for r in rows],random_state=seed)
            else:
                tr_keys,te_keys=train_test_split(keys,test_size=.2,stratify=[rows[by_hash[k][0]]['expected'] for k in keys],random_state=seed)
                tr=[i for k in tr_keys for i in by_hash[k]];te=[i for k in te_keys for i in by_hash[k]]
            pred=run([rows[i] for i in te],[rows[i] for i in tr],f'{split}_{seed}')
            out+=pred
            for sub,rr in [('all',pred),('classes_5_to_10',[r for r in pred if 5<=r['expected']<=10])]:
                stats.append(report_eval(rr,split,seed,sub))
    dump_csv(directory/'stability_summary.csv',stats);dump_csv(directory/'stability_predictions.csv',out)
    for split in ('image','sha256'):
        x=[r for r in stats if r['split']==split and r['subset']=='all']
        print(split,min(r['accuracy'] for r in x), max(r['accuracy'] for r in x), np.mean([r['accuracy'] for r in x]),sum(r['correct'] for r in x),sum(r['total'] for r in x))
if __name__=='__main__':main(Path(sys.argv[1]))
