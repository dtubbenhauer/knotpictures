"""Compare the frozen archive CNNs at identical label and resize indices.

The same-source arm uses unseen resized copies of variants whose source drawing
was used in training. The held-out arm uses unseen source drawings. There are
no fitting hashes in either arm, no repeated hashes within an arm, and exactly
the same joint frequency of (class label, shrink index) in both arms.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path
import tarfile

import numpy as np

from archive import ORIGINALS, PATTERN, group_partition
from cnn import CountCNN, batched_prediction, read_image, save_csv


def select(archive, fit_hashes):
    groups = group_partition()
    examples = defaultdict(dict)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            m = PATTERN.fullmatch(member.name)
            if not m or m[1] != 'train':
                continue
            label, number = int(m[2]), int(m[3])
            n = ORIGINALS[label]
            if n == 1:
                continue
            variants_per_source = round(49/n)
            base = n * variants_per_source
            if number < base:
                continue
            variant, index = divmod(number - base, 20)
            scale = 1 + index
            if scale in (5, 10, 15):
                continue
            group = variant // variants_per_source
            partition = groups[label, group]
            if partition == 'validation':
                continue
            arm = 'seen_source' if partition == 'train' else 'unseen_source'
            payload = tar.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in fit_hashes:
                continue
            # Keep a canonical pathname if more than one archive entry is identical.
            bucket = examples[label, scale, arm]
            if digest not in bucket or bucket[digest][0]['image'] > member.name:
                rows = {
                    'image': member.name, 'count': label, 'source_id': group, 'variant_id': variant,
                    'scale': scale, 'arm': arm, 'sha256': digest,
                }
                bucket[digest] = rows, payload
    output = []
    for label, scale in sorted({(l, s) for l, s, arm in examples}):
        keys = {arm: sorted(examples[label, scale, arm]) for arm in ['seen_source', 'unseen_source']}
        n_common = min(map(len, keys.values()))
        for arm, digests in keys.items():
            for digest in digests[:n_common]:
                row, payload = examples[label, scale, arm][digest]
                output.append((row, read_image(io.BytesIO(payload))))
    assert len(set(r['sha256'] for r, _ in output)) == len(output)
    return [r for r, x in output], np.array([x for r, x in output])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--models', type=Path, default=Path('archive_results'))
    args = parser.parse_args()
    with (args.models/'archive_split.csv').open() as f:
        fit_hashes = {r['sha256'] for r in csv.DictReader(f) if r['split'] in ('train', 'validation')}
    rows, images = select(args.archive, fit_hashes)
    summary = []
    for model_path in sorted(args.models.glob('model_*.npz')):
        seed = int(model_path.stem.split('_')[-1])
        model = CountCNN(seed)
        with np.load(model_path) as dump:
            model.weights = [dump[f'w{i}'] for i in range(len(model.weights))]
            raw = batched_prediction(model, images) * dump['target_sd'] + dump['target_mean']
        records = [{**r, 'raw': float(p), 'rounded': int(np.rint(p))} for r, p in zip(rows, raw)]
        save_csv(args.models/f'matched_{seed}.csv', records)
        for arm in ['seen_source', 'unseen_source']:
            sub = [r for r in records if r['arm'] == arm]
            by_label = {}
            for label in sorted({r['count'] for r in sub}):
                by_label[label] = [r['rounded'] - r['count'] for r in sub if r['count'] == label]
            errors = sum(by_label.values(), [])
            summary.append({
                'seed': seed, 'arm': arm, 'n': len(errors), 'correct': errors.count(0),
                'macro_label_accuracy': float(np.mean([x.count(0)/len(x) for x in by_label.values()])),
                'mae': float(np.abs(errors).mean()),
                'macro_label_mae': float(np.mean([np.abs(x).mean() for x in by_label.values()])),
            })
    save_csv(args.models/'matched_summary.csv', summary)
    print(summary)


if __name__ == '__main__':
    main()
