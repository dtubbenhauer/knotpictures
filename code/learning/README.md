# A small convolutional crossing-count regressor

This directory contains two independently trained instances of the same new model, each fitted with three initialization seeds. One experiment uses the published image archive. The other uses only the generated paired drawings. The original preprint's models and weights are separate. All fitted parameters, per-image predictions, and training histories for the two experiments are included.

## Reproduce

Use Python 3 with NumPy and Pillow. The runs recorded here used NumPy 2.3.5, Pillow 13.0.0, CPU and one BLAS thread. From this directory run the commands below. Adjust the archive path if necessary. The paired and table manifests use image paths relative to their respective parent directories. Regenerate the paired images and table crops with their companion scripts if they are absent.

```sh
export OPENBLAS_NUM_THREADS=1
python archive.py ../legacy/knot_crossings_dataset.tar.gz
python matched.py ../legacy/knot_crossings_dataset.tar.gz
python cross_paired.py ../paired/manifest.csv
python transfer.py ../corpus/manifest.csv --models archive_results --out archive_results --training-set archive
python cnn.py ../paired/manifest.csv --out results
python transfer.py ../corpus/manifest.csv --models results --out results
```

The SHA-256 checksum of the 9,100-image public archive used here is
`67ce51264d06fb9bb0381d49a1763973e23dc831c5863ff847466bb6690c5099`.
The archive-source assignment follows cells 5, 11, and 13 of the historical
`notebooks/Latest.ipynb` in commit `46fcbd90` of `annedranowski/knot-cnn`.
Each source index is anonymous within its label. File enumeration in the
historical generator determines the order.

## Fixed network and optimization

Each image is composited on white, converted to grayscale, and resized to 80 × 80 with Pillow's Lanczos resampling. The median of the border picks the polarity. The model sees foreground intensity in [0,1]. There is no per-image contrast stretching.

Three zero-padded, stride-two convolutional layers use 8 filters of size 5 × 5, 12 filters of size 3 × 3, and 16 filters of size 3 × 3. Each has a bias and a ReLU. The last activation map is 10 × 10. Averaging in four spatial quadrants leaves 64 features. An affine scalar readout yields a total of 2,893 trainable parameters. Every convolutional coefficient and the readout are fitted by backpropagation.

Labels are standardized using the training mean and population standard deviation. The objective is squared error. Training uses Adam with batch size 20, step size 0.001, moment coefficients 0.9 and 0.999 and epsilon 1e-8. There is no weight decay. Initialization seeds are 17, 29 and 43. A separate shuffling stream uses the seed plus 10,000. Each run trains for 45 epochs. Selection takes the earliest epoch attaining minimum validation mean squared error. The maximum duration, optimizer, architecture, and three seeds were chosen before observing predictions on the test sets. For exact accuracy and integer MAE, we round to nearest integer without clipping the output.

## Archive-trained runs

Within each class, the source IDs are shuffled by NumPy's default generator with seed 20261118, processing classes in ascending order. For a class with at least two sources, max(1, round(0.2 times the source count)) go to the held-out-source fold. A class with at least three sources contributes an additional max(1, round(0.15 times the source count)) to validation. Remaining sources go to fitting. This gives 57 fitting, 13 validation and 18 held-out sources. The three classes represented by a single drawing can only contribute to fitting. Their inclusion makes no claim about generalization to new source drawings for those classes.

The data generator provides 21 sizes of each transformed variant. The unscaled variant and shrink indices 5, 10 and 15 go into fitting for the fitting sources (1,192 image files). The 52 unscaled variants of the validation sources select checkpoints. Predictions on all 1,092 validation renders and all 1,701 held-out-source renders are archived. Duplicate hashes occur within source groups. The companion generator checks confirm that no identical bytes receive different source IDs.

`matched.py` creates the two directly comparable test arms. They share the joint distribution of class (5–10) and shrink index. Neither arm includes shrink indices 0, 5, 10, or 15. For each class and remaining index (1–20), files generated from fitting sources form one pool and those from held-out sources form the other. We exclude all validation and fitting hashes, collapse repeats within each pool, and sort by hash. Keep the first m examples from each pool where m is the smaller pool size. No hash repeats either within or across the final arms. Each arm contains 1,054 files. The familiar-source arm includes resized siblings of fitted variants. The held-out-source arm consists entirely of fresh original drawings. Neither the 1,054 files nor the training seeds are independent draws of a knot diagram. Interpret this as a descriptive comparison of these drawings. The stochasticity across seeds concerns only the optimizer and initial parameters.

The folder labels serve as targets for the public 49-file test split. Its 18 exact duplicates of public train files and 31 byte-novel images are reported separately. The archive-trained neural regressor can extrapolate far outside its fitted label range on unfamiliar line weights. Reported scores retain these outputs. Agreement with folder labels on this set is distinct from scoring independently certified visible counts on nonminimal drawings.

The three checkpoints in `archive_results/model_*.npz` were selected at epochs 15, 14 and 19 for seeds 17, 29 and 43. `matched_summary.csv` gives exact counts, micro-MAE and averages of the within-class accuracy and MAE. `paired_transfer_summary.json` scores the same frozen models on the independent generator's test drawings for targets 3–10 (inside archive label support). `external_predictions.csv` scores all 15 separately cropped table drawings. Transfer runs use exactly the same resize and polarity handling as training.

## Separately trained paired-drawing runs

All renders descended from the same layout seed stay in one fold. Fit on 30 layouts × 10 counts = 300 clean images and choose checkpoints on 8 layouts × 10 counts = 80 additional clean images. Each test style has 16 fresh layouts × 10 counts = 160 images. The underlying topology is controlled by the paired generator. Each pair has the same layout and braid, with an added Reidemeister-I curl in the second member. A single split is used across all three training seeds. Each model chose epoch 45.

`results/summary.json` provides per-style exact counts, raw and rounded absolute error, the number of pairs recovering a +1 step after rounding, and mean raw step. `predictions_*.csv` retain all per-image outputs. `external_predictions.csv` also records transfer to the independent table crops, including the zero-crossing drawing outside the paired training labels. These transfer drawings did not participate in checkpoint selection.
