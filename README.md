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

# Running the preprocessing and the model

See [example.ipynb](./example.ipynb) for a detailed description and an example on how to run the model.

See [open_problems_example.ipynb](./open_problems_example.ipynb) to see how to run the model with OpenProblems data format.

- **Github repository**: <https://github.com/lueckenlab/senkin-tmp-cite-pred/>

