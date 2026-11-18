# Paired knot drawings

The 1,020 newly generated PNGs are distributed in `drawings.zip`. The top-level result checker reads that ZIP directly. For direct per-image work, run `python -m zipfile -e drawings.zip .` from this directory, or regenerate the images with the commands below.


The experiment uses closed two-strand braids in a family of deformed annular layouts. The independently checked target is the number of transverse crossings in the drawing. The data are generated from scratch and contain no external images.

## Reproduce

```sh
python generate_pairs.py
python summarize.py
python measure_faces.py
```

These scripts write the images, per-image manifest, held-out face counts, summary table and illustrative figures beside the scripts. They were run with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Pillow 13.0.0 and Matplotlib 3.10.8. Set `--output DIR` on the first script and `--root DIR` on the other two to use another output directory.

## Generative design

For odd `k` in `{1,3,5,7,9}`, the closed curve has planar polar radius `1 + 0.24 cos(kt)`, polar angle `2t`, height `sin(kt)` and `0 <= t < 2*pi`. Two visits can share a point only when their parameters differ by `pi`. They then have opposite heights and meet where `cos(kt)=0`. Each crossing accounts for two visits, giving `k` crossings in total. This is the usual two-strand torus braid with `k` crossings, including the one-crossing unknot.

For each layout the generator compares a short ordinary arc with a replacement spline carrying a Reidemeister-I curl. The body outside that arc is identical in the pair. The seed controls the length of the original arc, a smooth radial distortion, a twist, an affine map, size, and translation. It never changes the braid word. Since the random transformations are shared across all five `k` and both curl states, the group, preferably also respected in any uncertainty estimates, consists of ten drawings. There are 54 groups. Groups 0–29 train, 30–37 validate, and 38–53 test. Every count from 1 through 10 occurs once in each group. All knot types and counts recur across splits. The held-out units are layouts.

Every underlying centerline is a closed polygon. The independent geometry check enumerates proper intersections of non-neighboring segments before any gaps are drawn. It requires exactly `k` crossings wholly on the retained braid, zero mixed braid–replacement crossings, and exactly `curl` self-crossings on the replacement. At each intersection the two height values determine the overpass. All 540 geometries also pass a frame-containment check. A separate pair switches one crossing of the ordinary trefoil drawing. Its centerline is unchanged; its two-strand braid reduces to one crossing by canceling two opposite signs.

## Raster conditions

Each output image is 160 by 160 grayscale pixels with a white margin. The bare polygon is drawn at four times the target resolution. Gaps are removed along the underpassing strand; the half-length is `(width + clearance) / |sin(angle between strands)|` in output-pixel units. The specified width is rounded on the fourfold grid when stroking. The `clean` condition uses width 2.1 and clearance 1.1, followed by Lanczos resizing to 160. The `break` condition also removes one five-pixel interval at a location far from geometric crossing visits on the same curve. The location maximizes the arclength distance to the nearest visit over 200 equally spaced samples. The `thin` condition uses width 1.2, clearance 0.5, Lanczos resizing to 72, then bilinear enlarging to 160. These coupled rendering changes are evaluated as one condition.

The 300 training and 80 validation files use `clean`. All 160 held-out geometries are rendered in all three conditions, yielding 860 PNGs in total. The artificial braid family and degradation mechanisms delimit the claims one can draw from the experiment.

## Pixel measurements

The threshold maximizes Otsu's between-class variance, choosing the first maximizing gray level. The component counter counts 8-connected sets of pixels at or below this threshold. The retrospective Euler diagnostic on the test set counts 4-connected sets of white pixels (strictly above threshold), discards the unique exterior region touching the frame, and subtracts one from the remaining count. This is the face-count formula for a connected four-valent shadow. A raster with open gaps need not satisfy it, and the raw estimator can return `-1`.

`manifest.csv` records original relative path, diagram count, split, group, `k`, curl flag, style, independently enumerated crossings, ink-component count and threshold. `summary.csv` reports the exact count and sum of absolute errors, and the number of within-layout Reidemeister pairs where the prediction increases by one. `faces.csv` contains the retrospective background-region estimates for the 480 test images. In `examples/paired-comparison.pdf` and `examples/shadow-faces.pdf`, the examples are always group 38, `k=5`. Colors in the first figure are assigned anew after connected-component labeling in each panel.

## Mixed-convention diagnostic

`python measure_mixed.py` recomputes an additional exploratory test on the same 160 held-out geometries. A fixed seed selects each crossing with probability one half to have its two under-half-branches detached. All remaining crossings are drawn through. To avoid changing the locations already selected in the braid body, the same random stream is used for both members of an RI pair; full per-row selections are determined by the seed and code. `mixed-counts.csv` records the additional 160 results alongside the original three test conditions. The `opened` column counts intentionally gapped crossings before rasterization, including for thin and broken styles. For the broken condition, the extra gap is an unrelated cut on an arc. No supervised model saw the mixed rendering during training or model selection.

When an ideal closed diagram is drawn through at every crossing, its graph has Euler characteristic minus the crossing count. Detaching an underpassing half-edge from a crossing adds one degree-one endpoint, leaving the edge count fixed. If there are `E` free ends, `B` bounded complementary regions and `C` ink components, the resulting graph thus has `C-B = E-n`; its crossing count is `E+B-C`. For the raster estimator we count the 8-connected ink and 4-connected white components at Otsu's threshold, thin the ink with Zhang and Suen's standard two-pass algorithm (Communications of the ACM 27, 236–239, 1984, DOI 10.1145/357994.358023), and count skeleton pixels with exactly one ink neighbor in their 8-neighborhood. The corresponding ideal/digital distinction applies to endpoints as well as components. `examples/mixed-euler.pdf` shows the same six-crossing curve with all gaps, a mixed selection, and all crossings joined. Free tips appear as red dots.

## Published artworks

`python measure_jkasd.py` applies the same endpoint rule to the fifteen public-domain knot-table crops in `../corpus`, using the exact masks recorded by that experiment at 32, 48, 64, 96, 128, 160 and 230 pixels. It compares component and hole counts against `../corpus/observations.csv` before estimating endpoints. `jkasd-euler.csv` contains all 105 per-image readings.

`python measure_poster.py` rasterizes Philip Rideout's CC0 poster with the renderer in `../legacy/poster_census.py`. It uses the 528 Otsu masks defined in `../legacy/results/poster_counts.csv` (88 underlying diagrams at 64, 128, 256, 512, 1024 and 2048 pixels), checking the components exactly against the existing census at every size. Pixel mask padding declares the image exterior, and endpoint thinning is performed on the ink bounding box with a one-pixel white margin. `poster-euler.csv` records all raw components, bounded regions, endpoints and the three readouts. Both external scripts accept alternate input directories on the command line.

The build in this archive also saves the 160 randomly mixed rasters in `images/mixed/` and their relative paths in `mixed-images.csv` when `measure_mixed.py` runs. They supplement the 860 main experiment files. The emitted pixel arrays and per-image scores are unchanged by this export.
