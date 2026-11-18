"""Evaluate independently trained CNNs on the 15 archived knot-table illustrations."""

import argparse
import csv
from pathlib import Path

import numpy as np

from cnn import CountCNN, batched_prediction, read_image, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--models', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--training-set', choices=['paired', 'archive'], default='paired')
    args = parser.parse_args()
    support = set(range(1, 11)) if args.training_set == 'paired' else {0, *range(3, 11)}
    with args.manifest.open(newline='') as file:
        rows = list(csv.DictReader(file))
    images = np.array([read_image(args.manifest.parent / r['image']) for r in rows])
    exported = []
    for model_path in sorted(args.models.glob('model_*.npz')):
        seed = int(model_path.stem.split('_')[-1])
        model = CountCNN(seed)
        with np.load(model_path) as dump:
            model.weights = [dump[f'w{i}'] for i in range(len(model.weights))]
            prediction = batched_prediction(model, images) * dump['target_sd'] + dump['target_mean']
        for r, p in zip(rows, prediction):
            exported.append({
                'seed': seed, 'name': r['name'], 'count': int(r['expected']),
                'raw': float(p), 'rounded': int(np.rint(p)),
                'in_training_support': int(r['expected']) in support,
            })
    save_csv(args.out/'external_predictions.csv', exported)
    for seed in sorted({r['seed'] for r in exported}):
        selected = [r for r in exported if r['seed'] == seed and r['in_training_support']]
        errors = [r['rounded'] - r['count'] for r in selected]
        print(seed, 'exact', errors.count(0), '/', len(errors), 'rounded MAE', np.abs(errors).mean())


if __name__ == '__main__':
    main()
