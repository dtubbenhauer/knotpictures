"""Count foreground components in the released crossing corpus and the manuscript figures.

Usage: python legacy_audit.py --dataset knot_crossings_dataset.tar.gz --paper paper.zip --out results
All rasters are read in their native dimensions; figures are never read from PDF screenshots.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

FG8 = np.ones((3, 3), dtype=bool)
BG4 = ndi.generate_binary_structure(2, 1)
THRESHOLDS = (1, 32, 64, 128)
DATA_SHA = '67ce51264d06fb9bb0381d49a1763973e23dc831c5863ff847466bb6690c5099'

# Keep file identities in the CSV; derived versions are repeated observations of one drawing.
STAGES = ('knot_orig', 'knot_skeletonize', 'knot_persp', 'knot_tr', 'knot_final')
SINGLES = {
    'arXiv/figs/knot_orig.png': (9, 'training-illustration'),
    'arXiv/figs/knot_skeletonize.png': (9, 'training-illustration'),
    'arXiv/figs/knot_persp.png': (9, 'training-illustration'),
    'arXiv/figs/knot_tr.png': (9, 'training-illustration'),
    'arXiv/figs/knot_final.png': (9, 'training-illustration'),
    'arXiv/figs/9_46.png': (9, 'named-table-diagram'),
    'arXiv/figs/9_48.png': (9, 'named-table-diagram'),
    'arXiv/figs/trefoil-knot.png': (3, 'three-crossing-trefoil'),
    'arXiv/figs/f8.png': (4, 'four-crossing-figure-eight'),
    'arXiv/figs/culprit2.png': (10, 'ten-crossing-unknot'),
    'cjm/figs/bad-diagram.png': (10, 'ten-crossing-unknot-derived'),
    'cjm/figs/prediction2.png': (4, 'four-crossing-figure-eight-derived'),
    'cjm/figs/prediction3.png': (9, 'nine-crossing-redraw'),
}
# Exclude multi-panel images, open strands, ropes in photographs, PD labels/dots,
# differential masks, diagrams containing text, and model plots from this single-diagram experiment.


def read_ink(data):
    im = Image.open(BytesIO(data)).convert('RGBA')
    white = Image.new('RGBA', im.size, 'white')
    white.alpha_composite(im)
    gray = np.asarray(white.convert('L'))
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return 255-gray if np.median(border) >= 128 else gray


def crop_nonzero(ink):
    rows = np.flatnonzero(np.any(ink, axis=1))
    cols = np.flatnonzero(np.any(ink, axis=0))
    if len(rows) == 0:
        return ink[:1, :1]
    return ink[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]


def counts(ink, threshold):
    mask = ink >= threshold
    labels, n = ndi.label(mask, FG8)
    sizes = np.bincount(labels[mask], minlength=n+1)
    result = {}
    for min_pixels in (1, 4):
        keep = sizes >= min_pixels
        count = int(keep.sum())
        prediction = count
        if count == 0:
            prediction = -1  # blank is an invalid observation
        elif count == 1:
            retained = keep[labels]
            # A circular stroke has one hole; use complementary 4-neighbour connectivity.
            bg = np.pad(~retained, 1, constant_values=True)
            holes = ndi.label(bg, BG4)[1]-1
            if holes == 1:
                prediction = 0
        result[f'components_{threshold}_{min_pixels}'] = count
        result[f'prediction_{threshold}_{min_pixels}'] = prediction
    return result


def otsu_cutoff(ink):
    """Minimal foreground ink value; first between-class-variance maximum, strict > Otsu."""
    h = np.bincount(ink.ravel(), minlength=256).astype(float)
    left = np.cumsum(h)
    mass = np.cumsum(h*np.arange(256))
    denom = left*(left[-1]-left)
    score = np.divide((mass[-1]*left-mass*left[-1])**2, denom,
                      out=np.full(256, -1.), where=denom>0)
    return int(np.argmax(score)) + 1


def all_counts(ink):
    trimmed = crop_nonzero(ink)
    cutoff = otsu_cutoff(ink)
    fields = {'otsu_min_foreground':cutoff}
    for threshold in THRESHOLDS:
        fields.update(counts(trimmed, threshold))
    fields.update({k.replace(f'_{cutoff}_', '_otsu_'):v
                   for k,v in counts(trimmed, cutoff).items()})
    return fields


def test_style(label, j):
    if label == 11:
        return 'smooth-line'
    if label in (0, 3, 4):
        if j == 0:
            return 'outlined'
        return 'small-line' if label == 0 or j in (1, 2) else 'smooth-line'
    return ('outlined', 'small-line', 'smooth-line')[j // 2]


def measure(item):
    name, data, label, group = item
    ink = read_ink(data)
    row = dict(file=name, sha256=sha256(data).hexdigest(), label=label, group=group,
               width=ink.shape[1], height=ink.shape[0], ink_min=int(ink.min()), ink_max=int(ink.max()))
    row.update(all_counts(ink))
    return row


def write_rows(path, rows):
    with path.open('w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def released(dataset, out):
    digest = sha256(dataset.read_bytes()).hexdigest()
    if digest != DATA_SHA:
        raise ValueError(f'Unexpected dataset SHA256: {digest}')
    results = []
    with tarfile.open(dataset, mode='r|gz') as tar, ThreadPoolExecutor(max_workers=4) as pool:
        work = []
        for entry in tar:
            m = re.fullmatch(r'(train|test)/(\d+)/(\d+)\.png', entry.name)
            if not entry.isfile() or not m:
                continue
            split, label, idx = m[1], int(m[2]), int(m[3])
            group = 'train' if split == 'train' else test_style(label, idx)
            work.append((entry.name, tar.extractfile(entry).read(), label, group))
            if len(work) == 36:
                results.extend(pool.map(measure, work))
                work = []
        results.extend(pool.map(measure, work))
    results.sort(key=lambda r:r['file'])
    assert len(results) == 9100
    assert Counter(r['file'].split('/')[0] for r in results) == {'train':9051,'test':49}
    write_rows(out/'released_counts.csv', results)
    return results


def manuscript(paper, out):
    with ZipFile(paper) as z:
        mandatory = {f'arXiv/figs/{f}.png' for f in (*STAGES,'9_46','9_48')}
        names = set(z.namelist())
        missing = mandatory-names
        if missing:
            raise ValueError(f'Archive lacks required illustrations: {sorted(missing)}')
        results = [measure((name, z.read(name), truth, group))
                   for name,(truth,group) in SINGLES.items() if name in names]
        write_rows(out/'figure_counts_release.csv', [r for r in results if r['file'] in mandatory])
        if len(results) == len(SINGLES):
            write_rows(out/'figure_counts.csv', results)
        perturb = []
        for name in ('arXiv/figs/knot_orig.png', 'arXiv/figs/9_46.png', 'arXiv/figs/9_48.png'):
            ink = Image.fromarray(read_ink(z.read(name)))
            for width in (512, 256, 128, 64):
                for angle in (0, 17, 37, 90):
                    # Expand avoids clipping; use a single fixed bilinear interpolant at each step.
                    scaled = ink.resize((width, width), Image.Resampling.BILINEAR)
                    transformed = scaled.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=0)
                    row = dict(file=name, label=9, width=width, angle=angle)
                    row.update(all_counts(np.asarray(transformed)))
                    perturb.append(row)
        write_rows(out/'figure_perturbations.csv', perturb)
    return results


def summary(rows, out):
    tables = {'all train':[r for r in rows if r['group']=='train'],
              'all test':[r for r in rows if r['group']!='train']}
    for name in ('outlined', 'small-line', 'smooth-line'):
        tables['test '+name] = [r for r in rows if r['group']==name]
    results = []
    for group, subset in tables.items():
        for t in (*THRESHOLDS, 'otsu'):
            for a in (1, 4):
                key = f'prediction_{t}_{a}'
                pred = [r[key] for r in subset]
                correct = sum(r[key] == r['label'] for r in subset)
                results.append(dict(group=group, threshold=t, minimum_component_pixels=a, correct=correct,
                                    total=len(subset), invalid=pred.count(-1)))
    write_rows(out/'released_summary.csv', results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset:
        summary(released(args.dataset, args.out), args.out)
    if args.paper:
        manuscript(args.paper, args.out)

if __name__ == '__main__':
    main()
