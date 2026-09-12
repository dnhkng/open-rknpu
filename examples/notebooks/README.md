<!-- SPDX-License-Identifier: MIT -->

# Notebooks

## `open_rknpu_walkthrough.ipynb`

A host-only walkthrough of the compiler: compile a single Conv, inspect the container,
compile a chain with an interior pool and a diamond, run the profile's Python integer
reference and compare bytes, and calibrate a small graph with `minmax`, `percentile` and
`kl`. It closes with the board steps from [`docs/board.md`](../../docs/board.md) as text.

Every code cell runs without the board, the vendor toolchain, PyTorch or the network, and
the notebook commits no outputs. The final markdown cell is the only hardware material.

### Open it in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnhkng/open-rknpu/blob/main/examples/notebooks/open_rknpu_walkthrough.ipynb)

Colab checks the repository out for you; the first markdown cell shows the install step and
the notebook needs nothing else.

### Run it locally

```sh
git clone https://github.com/dnhkng/open-rknpu && cd open-rknpu
python -m pip install -e .
jupyter notebook examples/notebooks/open_rknpu_walkthrough.ipynb
```

`jupyter` is not a project dependency; install it in the environment you use for the
walkthrough. The cells only write to temporary directories.

### Just read the cells

The notebook is plain `nbformat` 4 JSON, so GitHub and most editors render it. The code
cells are the same `open_rknpu` API calls that [`examples/primitives/`](../primitives/README.md)
makes; read them in order if you do not want to run them.

## The board steps need hardware

The last markdown cell of the walkthrough and `docs/board.md` cross-compile the libc-only
runtime and drive a suite over `adb`. They need the Luckfox Pico Mini B (RV1103) and must
not be run on the host; the notebook never executes them.
