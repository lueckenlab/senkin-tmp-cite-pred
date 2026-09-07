# senkin-tmp-cite-pred

[![Release](https://img.shields.io/github/v/release/lueckenlab/senkin-tmp-cite-pred)](https://img.shields.io/github/v/release/lueckenlab/senkin-tmp-cite-pred)
[![Build status](https://img.shields.io/github/actions/workflow/status/lueckenlab/senkin-tmp-cite-pred/main.yml?branch=main)](https://github.com/lueckenlab/senkin-tmp-cite-pred/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/lueckenlab/senkin-tmp-cite-pred/branch/main/graph/badge.svg)](https://codecov.io/gh/lueckenlab/senkin-tmp-cite-pred)
[![Commit activity](https://img.shields.io/github/commit-activity/m/lueckenlab/senkin-tmp-cite-pred)](https://img.shields.io/github/commit-activity/m/lueckenlab/senkin-tmp-cite-pred)
[![License](https://img.shields.io/github/license/lueckenlab/senkin-tmp-cite-pred)](https://img.shields.io/github/license/lueckenlab/senkin-tmp-cite-pred)

Reimplementation of the model for surface protein prediction from transcriptomics data, which won OpenProblems 2022 competition in CITE-seq task.

# Installation

Create a conda environment and install the package:

```bash
conda create -n cite_pred python=3.13 -y
conda activate cite_pred
pip install git+https://github.com/lueckenlab/senkin-tmp-cite-pred.git@main
```

# Faithfulness to the original solution

Version 0.2.0 fixes several deviations from the original pipeline (see [CHANGELOG.md](./CHANGELOG.md)). The
preprocessing was validated against the feature matrices of the original solution computed from the competition
raw counts: the CLR-TSVD components, the custom normalization with its TSVD/PCA components, and the selected
correlated genes are reproduced. Remember that the neural networks expect every feature block to be z-scored per
cell, which `senkin_tmp_cite_pred.nn_models.prepare_nn_inputs` does for you.

# Running the preprocessing and the model

See [example.ipynb](./example.ipynb) for a detailed description and an example on how to run the model.

See [open_problems_example.ipynb](./open_problems_example.ipynb) to see how to run the model with OpenProblems data format.

- **Github repository**: <https://github.com/lueckenlab/senkin-tmp-cite-pred/>

