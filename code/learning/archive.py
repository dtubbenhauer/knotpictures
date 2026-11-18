"""Fit the four-file-per-variant CNN on original-drawing splits of the published tar.

Usage: python archive.py path/to/knot_crossings_dataset.tar.gz --out archive-seed17 --seeds 17
The archive layout derives from cells 5, 11, 13 of Latest.ipynb in commit 46fcbd90
of annedranowski/knot-cnn. No anonymous source index is assigned a knot name.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from cnn import batched_prediction, fit, read_image, save_csv

ORIGINALS = {0: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 3}
PATTERN = re.compile(r'^(train|test)/(\d+)/(\d+)\.png$')


def group_partition():
    """Split drawings before using transformed versions or their resized siblings."""
    rng = np.random.default_rng(20261118)
    groups = {}
    for count, n in ORIGINALS.items():
        ordered = rng.permutation(n)
        num_test = max(1, round(.2 * n)) if n >= 2 else 0
        num_val = max(1, round(.15 * n)) if n >= 3 else 0
        for order, source in enumerate(ordered):
            if order < num_test:
                fold = 'test_source'
            elif order < num_test + num_val:
                fold = 'validation'
            else:
                fold = 'train'
            groups[count, int(source)] = fold
    return groups


def sample_archive(archive, output):
    groups = group_partition()
    train_hashes = defaultdict(set)
    loaded = []
    with tarfile.open(archive, 'r|gz') as src:
        for member in src:
            match = PATTERN.fullmatch(member.name)
            if not match:
                continue
            kind, label, number = match.groups()
            label, number = int(label), int(number)
            payload = src.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if kind == 'train':
                n = ORIGINALS[label]
                variants_per_source = round(49 / n)
                base_count = n * variants_per_source
                variant = number if number < base_count else (number - base_count) // 20
                scale = 0 if number < base_count else 1 + (number - base_count) % 20
                group = variant // variants_per_source
                assert group < n
                fold = groups[label, group]
                train_hashes[digest].add((label, group))
                if fold == 'train' and scale not in {0, 5, 10, 15}:
                    continue
            else:
                group = variant = scale = ''
                fold = 'public_test'
            row = {
                'image': member.name, 'count': label, 'split': fold, 'source_id': group,
                'variant_id': variant, 'scale': scale, 'sha256': digest,
            }
            loaded.append((row, read_image(io.BytesIO(payload))))
    assert max(map(len, train_hashes.values())) == 1
    for row, image in loaded:
        if row['split'] == 'public_test':
            row['split'] = 'public_duplicate' if row['sha256'] in train_hashes else 'public_new'
    loaded.sort(key=lambda pair: pair[0]['image'])
    rows = [r for r, x in loaded]
    # Check the known archive counts and ensure distinct files were not split by source.
    assert len({(r['count'], r['source_id']) for r in rows if r['source_id'] != ''}) == 88
    assert len(train_hashes) == 6657
    assert sum(r['split'] == 'public_duplicate' for r in rows) == 18
    assert sum(r['split'] == 'public_new' for r in rows) == 31
    save_csv(output / 'archive_split.csv', rows)
    return rows, np.array([x for r, x in loaded])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, default=Path('archive_results'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, images = sample_archive(args.archive, args.out)
    target = np.array([r['count'] for r in rows], dtype='f')
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    validation = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    # One fixed size per validation variant suffices for selection. Report across all scales.
    selection = np.array([i for i in validation if rows[i]['scale'] == 0])
    for seed in args.seeds:
        model, mean, sd, epoch, history = fit(images, target, train, selection, seed, args.epochs)
        pred = batched_prediction(model, images) * sd + mean
        np.savez_compressed(
            args.out / f'model_{seed}.npz', **{f'w{i}': w for i, w in enumerate(model.weights)},
            target_mean=mean, target_sd=sd,
        )
        save_csv(args.out / f'history_{seed}.csv', history)
        exported = [{**r, 'raw': float(p), 'rounded': int(np.rint(p))} for r, p in zip(rows, pred)]
        save_csv(args.out / f'predictions_{seed}.csv', exported)
        summary = {}
        for fold in sorted({r['split'] for r in rows}):
            chosen = [r for r in exported if r['split'] == fold]
            errors = [r['rounded'] - r['count'] for r in chosen]
            summary[fold] = {
                'correct': errors.count(0), 'total': len(chosen),
                'mae': float(np.abs(errors).mean()),
            }
        report = {'best_epoch': epoch, 'selection_examples': len(selection), 'metrics': summary}
        (args.out / f'summary_{seed}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Archive fit {seed}: {report}', flush=True)


if __name__ == '__main__':
    main()
