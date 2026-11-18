"""Identify byte-identical PNG files and repeated ordered augmentation families."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from legacy_audit import write_rows

def audit(manifest, out):
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    train=[r for r in rows if r['file'].startswith('train/')]
    test=[r for r in rows if r['file'].startswith('test/')]
    counts=Counter(r['label'] for r in train)
    train_paths=defaultdict(list)
    families=defaultdict(dict)
    for row in train:
        train_paths[row['sha256']].append(row['file'])
        c=row['label']
        n=counts[c]//21
        assert counts[c]==n*21
        i=int(Path(row['file']).stem)
        j,slot=(i,0) if i<n else ((i-n)//20,(i-n)%20+1)
        assert slot not in families[c,j]
        families[c,j][slot]=row['sha256']
    sig=defaultdict(list)
    for family, f in families.items():
        assert set(f)==set(range(21))
        sig[tuple(f[i] for i in range(21))].append(family)
    assert len(sig)==len({s[0] for s in sig})
    # Every repeated first PNG coincides in all twenty associated transformed PNG files.
    expected=0
    for paths in train_paths.values():
        n=len(paths)
        # Condition on a uniform selection of a test image before drawing the other 1,810.
        expected+=n*(1-math.prod((1810-i)/(9050-i) for i in range(n-1)))
    expected=1811*expected/len(train)
    dup=[dict(file=r['file'],group=r['group'],label=r['label'],sha256=r['sha256'],
              train_matches=len(train_paths.get(r['sha256'],[])),
              train_files=' '.join(sorted(train_paths.get(r['sha256'],[])))) for r in test]
    write_rows(out/'test_overlap.csv',dup)
    d=dict(train_pngs=len(train),distinct_train_pngs=len(train_paths),
           training_files_in_multiple_occurrence_groups=sum(len(p) for p in train_paths.values() if len(p)>1),
           hash_multiplicity_counts=dict(sorted(Counter(len(p) for p in train_paths.values()).items())),
           numerical_families=len(families),distinct_21_member_families=len(sig),
           test_pngs=len(test),distinct_test_pngs=len(set(r['sha256'] for r in test)),
           test_overlap_with_train=sum(int(r['train_matches'])>0 for r in dup),
           expected_training_byte_matches_in_uniform_random_1811_file_holdout=expected)
    (out/'overlap_summary.json').write_text(json.dumps(d,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.manifest,a.out)
