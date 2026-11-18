# Replaying the experiments

From the repository root, run `python code/verify.py` to recompute the reported results using only Python's standard library. This checks the included observations, image assignments, generated images and trained-model files in place. [Human overview](../README.md) · [Claim-to-file index](../AI_README.md)

For a new calculation, run `python code/fetch_inputs.py` followed by `python code/verify.py --raw`. The downloader verifies the exact compressed archive from Hugging Face, the Rideout poster and ten author illustrations in the arXiv v1 source, then keeps the inputs needed by the scripts. Each upstream input also has a separate mode, for example `python code/fetch_inputs.py paper`. Use `python code/fetch_inputs.py --check` for a network-free check of the local inputs.

The experiment driver is `python code/reproduce.py ../knotpictures-rerun`. Install the versions in [requirements.txt](requirements.txt) first. Each run uses a new output directory. By default it evaluates the supplied CNN checkpoints. Append `--retrain` to refit all six networks. The 2025 CNN weights are distinct from these models. Code and instructions for the earlier training project remain at their original URL. [Original project](https://github.com/annedranowski/knot-cnn)

| Directory | Contents |
| --- | --- |
| [corpus](corpus/README.md) | Jkasd table crop coordinates, resolution and phase observations |
| [legacy](legacy/README.md) | Independent measurements of published images; duplicate audit; poster census |
| [replay](replay/README.md) | Global ink control, several split designs and source-level cross-validation |
| [learning](learning/README.md) | Compact CNN implementation, six checkpoints, predictions and training curves |
| [paired](paired/README.md) | Two-strand diagrams with known geometry, curls, extra breaks and filled underpasses |
| [visuals](visuals/README.md) | Reproducible figure scripts and numerical checks |

The hosted Knot Crossing Dataset is declared Unlicense, the source poster is CC0, and Jkasd released the knot table into the public domain. Source credits and identifiers are linked from the [main README](../README.md).
