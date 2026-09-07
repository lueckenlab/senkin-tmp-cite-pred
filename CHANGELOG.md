# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.2.0

This release makes the reimplementation match the original senkin13 pipeline. The previous versions deviated from
it in several places, which is why their results did not match the original model's output.

### Fixed

- `senkin_normalize` divided the z-scores by the variance instead of the standard deviation, and subtracted the
  per-batch median of the *raw counts* (mostly 0) instead of the median of the z-scored values. Both are now as in
  the original notebook. The function returns a dense `np.ndarray` (float32 by default) instead of an `np.matrix`
  and no longer needs the constant genes to be removed beforehand.
- `get_top_correlated_features` computed correlations on the raw counts. The original used the competition's
  log-normalized inputs; the matrix is now selected with `rna_key` (default `"X_log_normalized"`) and the protein
  matrix with `prot_key` (default `"dsb"`). Genes with an undefined correlation in any group (e.g. not expressed in
  a donor) were ranked first by `argsort` (NaN sorts last) and hence *selected*; they are now excluded as in the
  original. Correlations are computed in gene chunks, so whole-transcriptome inputs no longer need hundreds of GB.
- The neural network inputs were not z-scored. In the original every feature block (CLR-TSVD, selected raw genes,
  normalized TSVD/PCA, LightGBM predictions) is z-scored per cell before concatenation; use `prepare_nn_inputs`.
- `zscore` returns 0 instead of NaN for constant rows.
- Adam's `epsilon` for the cosine model is `1e-7` (the Keras 2 default the original relied on) instead of `1e-9`.
- `log_normalize` defaults to counts per million (`target_sum=1e6`), which is how the 2022 competition inputs
  were normalized. `preprocess_data` uses it.
- `preprocess_data` computes both the 100 components TSVD (`X_sqrt_norm_tsvd`) and the 64 components PCA
  (`X_sqrt_norm_pca`) of the normalized data, as in the original, instead of a single 100 components PCA
  (`X_pca_sqrt_norm`). `train_lightgbm_models` uses both for the second model.
- `preprocess_data` accepts `known_features=None`.

### Changed

- `get_lgbm_predictions` bins the features once per fold and reuses the LightGBM datasets for all targets
  (identical models, several times faster on large inputs). Sparse inputs are supported.
- `get_lgbm_predictions` limits the number of TSVD components to what the predictions can fit.
- `nn_kfold` and `train_nn_models` accept `models_dir` and create it.
- Added `get_matrix`, `to_dense` helpers and unit tests that compare the functions against literal
  re-implementations of the original notebooks.

## 0.0.2

### Added

- Example notebook for the open problem data format

### Changed

- Fix a bug when number of SVD/PCA components is bigger than the number of features or observations
- Fix a bug when batch size for neural networks is bigger than the data size
