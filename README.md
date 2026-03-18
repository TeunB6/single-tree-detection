# tree-cover-segmentation

Neural network for individual tree crown detection using aerial RGB imagery
and LiDAR canopy height data from the
[NeonTreeEvaluation](https://github.com/weecology/NeonTreeEvaluation) dataset.

## Background

The [National Ecological Observatory Network (NEON)](https://www.neonscience.org/)
collects high-resolution aerial imagery across ecological sites in the United
States. The NeonTreeEvaluation benchmark provides expert-annotated bounding
boxes around individual tree crowns in these images, along with co-registered
LiDAR-derived canopy height models (CHM). This project trains an object
detection model to reproduce those annotations automatically.

## Data

The dataset is downloaded automatically on first run. It includes:

- **RGB imagery** -- three-channel aerial photographs
- **Canopy height model (CHM)** -- LiDAR-derived height above ground (metres),
  normalised to [0, 1] using a dataset-wide maximum of ~60.6 m
- **Annotations** -- Pascal VOC XML files with bounding boxes around individual
  tree crowns

There are approximately 20 large source images per split. During preprocessing
these are tiled into 400 x 400 pixel crops, each saved as a `.pt` file, which
is what the model actually trains on.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run main.py
```

`main.py` will download and verify the dataset (SHA-256 checked), unpack the
nested archives, fix the directory structure, and preprocess all source images
into cached crop files under `data/pt_data/`.

## Project structure

```
src/
  const.py                   -- project-wide paths and constants
  dataset.py                 -- PyTorch Dataset wrapping the cached crops
  data/
    setup.py                 -- preprocessing: load TIFFs, merge CHM, tile crops
  utils/
    download.py              -- streaming download, checksum, recursive unzip
    logger.py                -- file + terminal logging
    normalization_params.py  -- compute channel means/stdevs from cached data
    singleton.py             -- SingletonMeta base class
    transforms.py            -- ToTensor (uint8 -> float32)
tex/                         -- onboarding PDFs for non-tech team members
```

## Documentation

Two reference documents in `tex/` are aimed at team members new to ML:

- **ml_primer.pdf** -- explains the libraries (NumPy, PyTorch, Torchvision)
  from first principles, grounded in tree crown detection examples.
- **codebase_guide.pdf** -- walks through every source file, explaining what
  problem each one solves and why it is written the way it is.

## Normalisation constants

`CHANNEL_MEANS` and `CHANNEL_STDEVS` in `const.py` are per-channel statistics
used to normalise images before they enter the model. The current values were
computed from the training and test splits using `normalization_params.py`.
If the dataset or preprocessing changes, rerun that script and update the
constants.
