# Reproducing the raster audit

`poster_census.py` renders every named source diagram embedded in Philip Rideout's CC0 prime-knots poster. This single public SVG contains the 88 source names enumerated in the historical training notebook. For each entry, the script removes its poster-placement transform and uses the drawing's own bounding rectangle as its viewport. The knot fills, white overpasses, and stroke paths are retained. MuPDF renders the vector directly at each square resolution with preserved aspect ratio, 8-bit gray samples, and antialias level 8. The line plot in `poster_resolution.pdf` counts the successful Otsu predictions. Its upper panels use the actual binary masks of knot 10_3 at widths 128, 512, and 2048. Colors distinguish components independently within each panel. The rasterizer and its version matter. [Original poster and license](https://prideout.net/blog/svg_knots/)

```sh
python poster_census.py --poster knottable_v1.svg --out results
python poster_figure.py --poster knottable_v1.svg --results results --out results
```

`legacy_audit.py` reads released PNGs, composites transparent pixels onto white, converts to Pillow luminance, and infers polarity from the median of border pixels. It works with ink intensity from 0 (background) to 255 (maximum ink). The primary method maximizes Otsu's between-class variance on all pixels, choosing the first maximizing threshold. Foreground consists of samples strictly greater than this threshold. Foreground uses 8-neighbor connectivity, background uses 4-neighbor connectivity. One foreground component enclosing one hole predicts zero. An empty mask yields -1 (invalid). The fixed-threshold sensitivity runs retain ink intensities at least 1, 32, 64 or 128. The optional area filter keeps components of at least four foreground pixels. Correctness always compares to the original folder label on released images.

`released_counts.csv` contains all 9,100 numeric PNGs in the upstream compressed dataset, one row per file, with SHA-256 digests. AppleDouble metadata is excluded. `released_summary.csv` distinguishes the 9,051 training files from the 49 nominal test files. The test groups describe visual rendering style, assigned by inspection of all 49 images; they do not identify independent geometries. In the archive the 18 small-line test PNGs occur byte-for-byte in training. The 31 remaining PNGs comprise 15 outlined and 16 smooth-line views. The 1,811-image evaluation cohort in the original paper was a separate internal image split.

```sh
python legacy_audit.py --dataset knot_crossings_dataset.tar.gz --out results
python overlap_audit.py --manifest results/released_counts.csv --out results
```

`overlap_audit.py` groups training files by exact PNG bytes. It reproduces the ordinal 21-image families from cell 13 of the historical notebook, checks repeated full family signatures, and reports corresponding train paths for every nominal test example. For a tied byte identity with multiplicity k, conditional on picking one member for the test set, the chance that every other member also fell in the uniform 1,811-file test set is the product of (1810-i)/(9050-i) for i from 0 to k-2. The reported overlap expectation uses this expression. It does not identify the particular historical split.

`figure_counts_release.csv` audits the seven cited manuscript examples. The optional expanded audit covers additional manuscript PNGs when the full original author archive is available. The table covers the five consecutive preprocessing illustrations, both named nine-crossing examples, individual gapped knot drawings and their available standalone rerenders. Related images reappear as different observations of the same drawing. The audit excludes composites, photos, outlines of open strands, annotated PD codes, masks, and illustrations with axis labels. `figure_perturbations.csv` uses the three native 512-pixel source illustrations, independently resizes to four widths, then rotates by 0, 17, 37 or 90 degrees with bilinear interpolation and expanded bounds. These 48 runs form a sensitivity test on three drawings. The archived stage sequence itself is read from the five exact PNGs, without attempting to reconstruct the authors' random augmentation draws.

```sh
python legacy_audit.py --paper paper_illustrations.zip --out results
python legacy_figure.py --paper paper_illustrations.zip --out results
```

Running `python code/fetch_inputs.py paper` at the repository root builds `paper_illustrations.zip` from the pinned arXiv v1 source. It contains the five preprocessing rasters and both named nine-crossing figures at their original paths. The same commands work on the complete supplied manuscript archive. Either input reproduces `figure_counts_release.csv` and `figure_perturbations.csv`. The complete author archive additionally regenerates the optional 13-image `figure_counts.csv`.

The figure recolors the nine connected components in the authors' original raster and magnifies a 50-by-30-pixel region of their saved final raster, with brightness inverted for legibility. Its lower detail applies the same full-image Otsu threshold used for the count. The raw diagrams derive from Rideout's 2020 poster. Dranowski, Kabkov and Tubbenhauer provided the saved preprocessing rasters.

## Inputs and environment

The pinned inputs and individual illustration digests are listed in [inputs.json](../inputs.json). From the repository root, run `python code/fetch_inputs.py` to retrieve the raw PNG archive, CC0 poster, and cited plate images. The code uses the public arXiv v1 tarball to recover the original pixel data. For the historical generator see [the original notebook](https://github.com/annedranowski/knot-cnn/blob/46fcbd90a34b73775950090d91afe85d71e1f3db/notebooks/Latest.ipynb).

The calculations used Python 3.12.14, Pillow 13.0.0, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and PyMuPDF 1.26.6 (MuPDF 1.26.11). The full rerun uses the [experiment driver](../reproduce.py).
