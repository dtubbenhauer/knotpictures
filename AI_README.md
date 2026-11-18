# Reading the knot picture experiments

Read the [main article](ai/main-article.md) or the short [paper guide](ai/paper.md), then run `python code/verify.py`. The checker uses only the standard library and prints the expected numerators, denominators and error measures from the released rows.

| Claim to check | Evidence |
| --- | --- |
| 9,051 archived files come from 88 drawings | Historical source counts and grouping in [the generator](https://github.com/annedranowski/knot-cnn/blob/46fcbd90a34b73775950090d91afe85d71e1f3db/notebooks/Latest.ipynb), new [resize checks](code/replay/resize_lineage_checks.csv) |
| The public 49-file set includes 18 byte-identical training images | [Exact file matches](code/legacy/results/test_overlap.csv) |
| Global ink measurements perform well on the random image split | [Per-image predictions](code/replay/global_ink_predictions.csv), [split summary](code/replay/global_ink_summary.csv) |
| Familiar and unseen evaluation arms share label and scale distributions | [Image assignments and source IDs](code/learning/archive_results/archive_split.csv), [paired arm verification](code/verify.py) |
| Six separately trained networks are available | [Archive-trained checkpoints and predictions](code/learning/archive_results/), [braid-trained checkpoints and predictions](code/learning/results/) |
| The independent geometric ground truth comes from segment intersections | [Generator](code/paired/generate_pairs.py), [per-drawing labels](code/paired/manifest.csv) |
| The CvT error total reported in the first preprint can be read from its confusion matrix | Original paper, Figure 14. Archived weights are not used to generate the new CNN scores. [arXiv v1](https://arxiv.org/abs/2510.06284v1) |

Use a diagram as the unit when comparing the historical images, and a layout as the unit for held-out braid experiments. Renderings of the same geometry and the predictions from several initializations share their underlying drawing. Numerical classification uses diagram labels in the published corpus. The controlled curl changes the number of visible crossings by one and preserves the knot type.
