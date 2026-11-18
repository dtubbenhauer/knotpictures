"""Score all labeled diagrams on Jkasd's public-domain knot table through 7 crossings."""

import csv
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = (
    "0_1", "3_1", "4_1", "5_1", "5_2",
    "6_1", "6_2", "6_3", "7_1", "7_2",
    "7_3", "7_4", "7_5", "7_6", "7_7",
)
ROW_TOPS = (12, 309, 614)
SIDE = 230
SIZES = (230, 160, 128, 96, 64, 48, 32)
OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
EIGHT = np.ones((3, 3), dtype=bool)
FOUR = ndimage.generate_binary_structure(2, 1)


def measure(image):
    """Return Otsu cutoff, raw components, holes, predicted crossings, and component map."""
    ink = 255 - np.asarray(image, dtype=np.uint8)
    histogram = np.bincount(ink.ravel(), minlength=256)
    counts = np.cumsum(histogram)
    intensity_sum = np.cumsum(np.arange(256) * histogram)
    denominator = counts * (counts[-1] - counts)
    differences = (intensity_sum[-1] / counts[-1] * counts - intensity_sum) ** 2
    between_class_variance = np.divide(
        differences, denominator, out=np.zeros(256), where=denominator > 0
    )
    cutoff = int(np.argmax(between_class_variance))  # Choose the first cutoff on ties.
    foreground = ink > cutoff
    labels, components = ndimage.label(foreground, structure=EIGHT)
    # Padding joins all exterior background even when it meets multiple edges.
    _, background_regions = ndimage.label(~np.pad(foreground, 1), structure=FOUR)
    holes = background_regions - 1
    predicted = 0 if (components, holes) == (1, 1) else components
    return cutoff, components, holes, predicted, labels


def write_csv(filename, records):
    with (ROOT / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def illustrate(previews):
    sizes = (230, 96, 64, 32)
    names = ("3_1", "6_3", "7_4")
    fig, axes = plt.subplots(3, 4, figsize=(10.2, 7.5), layout="constrained")
    colors = plt.colormaps["tab10"].colors
    for row, name in enumerate(names):
        for column, size in enumerate(sizes):
            labels, components = previews[name, size]
            rgb = np.ones((*labels.shape, 3))
            for label in range(1, components + 1):
                rgb[labels == label] = colors[(label - 1) % len(colors)]
            axis = axes[row, column]
            axis.imshow(rgb, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
            if row == 0:
                axis.set_title(f"{size} × {size} px", fontsize=12)
            if column == 0:
                axis.set_ylabel(f"${name[0]}_{{{name[2:]}}}$", fontsize=19, rotation=0, labelpad=22)
            axis.set_xlabel(f"{components} ink components", fontsize=10)
    fig.savefig(ROOT / "resolution-example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    with Image.open(ROOT / "knots.png") as image:
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image.convert("RGBA"))
        sheet = white.convert("L")
    assert sheet.size == (1280, 954)
    (ROOT / "crops").mkdir(exist_ok=True)
    manifest, phase_rows, observations, previews = [], [], [], {}
    for index, name in enumerate(NAMES):
        row, column = divmod(index, 5)
        x = 25 + 250 * column
        y = ROW_TOPS[row]
        image_path = f"crops/{name}.png"
        expected = int(name.split("_")[0])
        manifest.append(dict(image=image_path, source="knots.png", name=name, expected=expected,
                             left=x, top=y, right=x + SIDE, bottom=y + SIDE))
        for dx, dy in OFFSETS:
            left, top = x + dx, y + dy
            crop = sheet.crop((left, top, left + SIDE, top + SIDE))
            values = np.asarray(crop)
            border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
            assert np.all(border == 255), f"Crop touches a mark: {name}, {dx}, {dy}"
            if (dx, dy) == (0, 0):
                crop.save(ROOT / image_path)
            for size in SIZES:
                image = crop if size == SIDE else crop.resize((size, size), Image.Resampling.LANCZOS)
                cutoff, components, holes, prediction, labels = measure(image)
                exact = int(prediction == expected)
                phase_rows.append(dict(name=name, expected=expected, side_px=size, dx=dx, dy=dy,
                                       prediction=prediction, exact=exact))
                if (dx, dy) == (0, 0):
                    observations.append(dict(name=name, expected=expected, side_px=size,
                                             otsu_ink_cutoff=cutoff, components=components, holes=holes,
                                             prediction=prediction, exact=exact))
                    previews[name, size] = labels, components

    write_csv("manifest.csv", manifest)
    write_csv("observations.csv", observations)
    write_csv("phase-observations.csv", phase_rows)
    summary, phase_summary = [], []
    for size in SIZES:
        rows = [r for r in observations if r["side_px"] == size]
        summary.append(dict(side_px=size, correct_all=sum(r["exact"] for r in rows), total=len(rows),
                            correct_primes=sum(r["exact"] for r in rows if r["expected"] > 0), total_primes=14,
                            mean_absolute_error=np.mean([abs(r["prediction"] - r["expected"]) for r in rows])))
        for dx, dy in OFFSETS:
            group = [r for r in phase_rows if (r["side_px"], r["dx"], r["dy"]) == (size, dx, dy)]
            phase_summary.append(dict(side_px=size, dx=dx, dy=dy,
                                      correct_primes=sum(r["exact"] for r in group if r["expected"] > 0),
                                      of_primes=14, correct_all=sum(r["exact"] for r in group), of_all=15))
    write_csv("summary.csv", summary)
    write_csv("phase-summary.csv", phase_summary)
    illustrate(previews)


if __name__ == "__main__":
    main()
