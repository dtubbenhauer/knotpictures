"""Score the fixed archive-trained CNNs directly on held-out synthetic layouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cnn import CountCNN, batched_prediction, read_image, save_csv, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--models', type=Path, default=Path('archive_results'))
    args = parser.parse_args()
    with args.manifest.open() as file:
        rows = [r for r in csv.DictReader(file) if r['split'] == 'test']
    images = np.array([read_image(args.manifest.parent/r['image']) for r in rows])
    all_stats = {}
    for model_path in sorted(args.models.glob('model_*.npz')):
        seed = int(model_path.stem.split('_')[-1])
        model = CountCNN(seed)
        with np.load(model_path) as dump:
            model.weights = [dump[f'w{i}'] for i in range(len(model.weights))]
            preds = batched_prediction(model, images) * dump['target_sd'] + dump['target_mean']
        data = [{**r, 'raw': float(p), 'rounded': int(np.rint(p))} for r,p in zip(rows, preds)]
        save_csv(args.models/f'paired_transfer_{seed}.csv', data)
        seed_stats = {}
        for style in sorted({r['style'] for r in rows}):
            # Original training included {0,3,...,10}; 1 and 2 are a separate extrapolation.
            indices = [i for i,r in enumerate(rows) if r['style']==style and int(r['count'])>=3]
            seed_stats[style] = stats([rows[i] for i in indices], preds[indices])
        all_stats[seed] = seed_stats
    (args.models/'paired_transfer_summary.json').write_text(json.dumps(all_stats, indent=2)+'\n')
    print(all_stats)


if __name__ == '__main__':
    main()
