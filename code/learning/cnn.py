"""Train a small convolutional regressor on paired knot drawings using NumPy.

Run OPENBLAS_NUM_THREADS=1 python cnn.py ../paired/manifest.csv --out results
from the learning directory. All model selection uses the validation split.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


def windows(x, kernel):
    """Return flattened patches for a zero-padded convolution with stride two."""
    pad = kernel // 2
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    view = sliding_window_view(x, (kernel, kernel), axis=(1, 2))[:, ::2, ::2]
    n, height, width = view.shape[:3]
    ordered = np.ascontiguousarray(view.transpose(0, 1, 2, 4, 5, 3))
    return ordered.reshape(n, height, width, -1)


class CountCNN:
    def __init__(self, seed):
        self.shapes = [(5, 1, 8), (3, 8, 12), (3, 12, 16)]
        rng = np.random.default_rng(seed)
        self.weights = []
        for kernel, channels, filters in self.shapes:
            fan_in = kernel ** 2 * channels
            w = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, filters)).astype('f')
            self.weights.extend([w, np.zeros(filters, dtype='f')])
        self.weights.extend([
            rng.normal(0, .125, (64, 1)).astype('f'),
            np.zeros(1, dtype='f'),
        ])

    def forward(self, image, keep=False):
        x = image
        self.cache = []
        for layer, (kernel, channels, filters) in enumerate(self.shapes):
            patches = windows(x, kernel)
            w, bias = self.weights[2 * layer:2 * layer + 2]
            z = patches @ w + bias
            if keep:
                self.cache.append((patches, z, x.shape))
            x = np.maximum(z, 0)
        n, h, w, c = x.shape
        features = x.reshape(n, 2, h // 2, 2, w // 2, c).mean(axis=(2, 4)).reshape(n, -1)
        if keep:
            self.pool = (features, x.shape)
        return (features @ self.weights[-2] + self.weights[-1]).ravel()

    def gradients(self, predictions, target):
        error = (2.0 / len(target) * (predictions - target)).reshape(-1, 1)
        features, (n, h, w, c) = self.pool
        result = [None] * len(self.weights)
        result[-2:] = [features.T @ error, error.sum(0)]

        # Reverse independent spatial averaging in each of the four quadrants.
        dx = (error @ self.weights[-2].T).reshape(n, 2, 2, c)
        dx = dx.repeat(h // 2, axis=1).repeat(w // 2, axis=2) / ((h // 2) * (w // 2))
        for layer in range(2, -1, -1):
            patches, z, input_shape = self.cache[layer]
            dz = np.ascontiguousarray(dx * (z > 0)).reshape(-1, z.shape[-1])
            patch_matrix = patches.reshape(-1, patches.shape[-1])
            result[2 * layer:2 * layer + 2] = [patch_matrix.T @ dz, dz.sum(axis=0)]
            if layer == 0:
                continue
            kernel, channels, _ = self.shapes[layer]
            pad = kernel // 2
            n, height, width, _ = input_shape
            projected = (dz @ self.weights[2 * layer].T).reshape(
                *patches.shape[:3], kernel, kernel, channels
            )
            # Overlapping patches contribute additively to the input gradient.
            padded = np.zeros((n, height + 2 * pad, width + 2 * pad, channels), dtype='f')
            _, out_h, out_w = patches.shape[:3]
            for a in range(kernel):
                for b in range(kernel):
                    padded[:, a:a + 2 * out_h:2, b:b + 2 * out_w:2] += projected[:, :, :, a, b]
            dx = padded[:, pad:pad + height, pad:pad + width]
        return result


def read_image(path):
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        flat = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('L')
        x = np.asarray(flat.resize((80, 80), Image.Resampling.LANCZOS), dtype='f') / 255.
    border = np.concatenate([x[0], x[-1], x[1:-1, 0], x[1:-1, -1]])
    return (1 - x if np.median(border) > .5 else x)[:, :, None]


def batched_prediction(model, images, batch_size=24):
    batches = [model.forward(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
    return np.concatenate(batches)


def fit(images, target, train, validation, seed, epochs):
    rng = np.random.default_rng(seed + 10000)
    model = CountCNN(seed)
    first_moment = [np.zeros_like(w) for w in model.weights]
    second_moment = [np.zeros_like(w) for w in model.weights]
    mean, sd = target[train].mean(), target[train].std()
    norm = (target - mean) / sd
    losses = []
    iteration = 0
    best = np.inf
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train)
        batch_errors = []
        for start in range(0, len(order), 20):
            idx = order[start:start + 20]
            pred = model.forward(images[idx], keep=True)
            batch_errors.extend((pred - norm[idx]) ** 2)
            gradients = model.gradients(pred, norm[idx])
            iteration += 1
            for j, g in enumerate(gradients):
                first_moment[j] = .9 * first_moment[j] + .1 * g
                second_moment[j] = .999 * second_moment[j] + .001 * g * g
                m = first_moment[j] / (1 - .9 ** iteration)
                v = second_moment[j] / (1 - .999 ** iteration)
                model.weights[j] -= .001 * m / (np.sqrt(v) + 1.e-8)
        prediction = batched_prediction(model, images[validation]) * sd + mean
        val_mse = float(np.mean((prediction - target[validation]) ** 2))
        losses.append({
            'epoch': epoch,
            'train_minibatch_mse': float(np.mean(batch_errors) * sd ** 2),
            'validation_mse': val_mse,
        })
        if val_mse < best:
            best = val_mse
            best_epoch = epoch
            best_weights = [w.copy() for w in model.weights]
        if epoch == 1 or epoch % 10 == 0:
            print(f'Seed {seed}, epoch {epoch}, validation MSE {val_mse:.5f}', flush=True)
    model.weights = best_weights
    return model, mean, sd, best_epoch, losses


def stats(rows, raw):
    counts = np.array([int(r['count']) for r in rows])
    pred = np.rint(raw).astype(int)
    pairs = {}
    for r, x, p in zip(rows, raw, pred):
        key = r['group'], r['k']
        pairs.setdefault(key, {})[r['curl']] = (float(x), int(p))
    complete = [(p['0'], p['1']) for p in pairs.values() if '0' in p and '1' in p]
    return {
        'n': len(counts), 'correct': int((pred == counts).sum()),
        'mae': float(np.abs(pred - counts).mean()),
        'raw_mae': float(np.abs(raw - counts).mean()), 'pairs': len(complete),
        'pairs_exact_step': sum(int(b[1] - a[1] == 1) for a, b in complete),
        'mean_raw_step': float(np.mean([b[0] - a[0] for a, b in complete])) if complete else None,
    }


def save_csv(path, records):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline='') as file:
        rows = list(csv.DictReader(file))
    train_tags = {'train', 'training'}
    validation_tags = {'val', 'valid', 'validation'}
    test_tags = {'test', 'testing'}
    rows = [
        r for r in rows if r['split'] in (train_tags | validation_tags | test_tags)
        and r['curl'] in {'0', '1'}
    ]
    images = np.array([read_image(args.manifest.parent / r['image']) for r in rows])
    target = np.array([int(r['count']) for r in rows], dtype='f')
    train = np.array([i for i, r in enumerate(rows) if r['split'] in train_tags])
    validation = np.array([i for i, r in enumerate(rows) if r['split'] in validation_tags])
    if not len(train) or not len(validation):
        raise ValueError('Need training and validation folds')
    train_groups = {rows[i]['group'] for i in train}
    validation_groups = {rows[i]['group'] for i in validation}
    test_groups = {r['group'] for r in rows if r['split'] in test_tags}
    if (train_groups & validation_groups) or (train_groups & test_groups) or (validation_groups & test_groups):
        raise ValueError('Source groups overlap across folds')
    summary = {}
    for seed in args.seeds:
        model, mean, sd, epoch, history = fit(images, target, train, validation, seed, args.epochs)
        predictions = batched_prediction(model, images) * sd + mean
        np.savez_compressed(
            args.out / f'model_{seed}.npz',
            **{f'w{i}': w for i, w in enumerate(model.weights)},
            target_mean=mean, target_sd=sd,
        )
        save_csv(args.out / f'history_{seed}.csv', history)
        fields = ['image', 'split', 'group', 'k', 'curl', 'style', 'count']
        exported = [
            {**{k: r[k] for k in fields}, 'raw': float(p), 'rounded': int(np.rint(p))}
            for r, p in zip(rows, predictions)
        ]
        save_csv(args.out / f'predictions_{seed}.csv', exported)
        fold_stats = {}
        for split, style in sorted({(r['split'], r['style']) for r in rows}):
            idx = [i for i, r in enumerate(rows) if r['split'] == split and r['style'] == style]
            fold_stats[f'{split}_{style}'] = stats([rows[i] for i in idx], predictions[idx])
        summary[seed] = {'best_epoch': epoch, 'metrics': fold_stats}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Seed {seed} finished, best validation epoch {epoch}', flush=True)


if __name__ == '__main__':
    main()
