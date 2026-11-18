# Global ink control

The four-feature classifier uses no neural weights. It fits a standard scaler on
the training rows and uses Euclidean one-nearest-neighbor in standardized feature
space. Every evaluation uses the same four features. The source-fold and repeated
split analyses are descriptive checks. Repeated test views of a source are not
independent observations.

Images are converted to 8-bit grayscale. Pixel weight is (255 - gray)/255.
The four summaries are fractional ink (total weight divided by pixel count),
ink-weighted root mean square radius about the centroid, squared relative
anisotropy of the coordinate covariance (squared eigenvalue difference divided
by squared trace), and the fourth radial moment divided by squared second
radial moment. Coordinates are normalized by image width and height. All inputs
in these experiments are square. There is no zero-suppression rule in the
classifier; zero is an ordinary training label.

## Inputs

* The published `knot_crossings_dataset.tar.gz` from the Hugging Face dataset
  `tr33hugg3r/knot-crossings`, SHA-256
  `67ce51264d06fb9bb0381d49a1763973e23dc831c5863ff847466bb6690c5099`.
* Printed table image `knots.png` and its 15-row crop manifest from
  `table_baseline.py`.
* For the matched comparison, the compact CNN's `archive_split.csv` and
  `matched_17.csv`. All model seeds use exactly these same cases.

## Commands

Run from the directory containing the scripts, supplying the actual paths.

```sh
python ink_features.py DATASET.tar.gz TABLE_MANIFEST.csv KNOTS.png OUTPUT
python source_folds.py OUTPUT
python stability.py OUTPUT
python validate_lineage.py DATASET.tar.gz OUTPUT
python validate_source_blocks.py DATASET.tar.gz OUTPUT
python matched_ink_control.py OUTPUT CNN_MANIFEST_DIRECTORY
```

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and
Pillow 13.0.0 were used. All scripts run on CPU. The historical notebook
used to reconstruct original drawing groups is at
https://github.com/annedranowski/knot-cnn/blob/46fcbd90a34b73775950090d91afe85d71e1f3db/notebooks/Latest.ipynb .

## Grouping checks

The generator reads 88 drawings, appends each drawing's rotated and flipped
variants consecutively to a class-specific directory, and finally appends 20
rescales per numbered variant. On the deposited images this gives 431 numbered
variants with 21 views each. The 8,620 nonparent views all agree with
floating-point bilinear rescaling of the assigned parent to within one level
on the 8-bit grayscale scale at every pixel. For the 199 nonrepresentative
variants where the class has multiple drawings and a drawing has multiple
variants, the closest radial ink profile among the class's candidate source
representatives matched the assigned original in all 199 cases. All 6,657
distinct raw-file SHA-256 hashes occur within one reconstructed source
group each. These checks validate the group mapping; anonymous order within
a crossing class does not identify which table knot goes with a slot.

The 21-view block grouping included as `resizing_family_split` in
`global_ink_summary.csv` allows byte-identical bases to occur across partitions.
Only the `source_fold_*` outputs enforce original-drawing separation, and
the `stability_*` rows labeled `sha256` test byte-level deduplication alone.
