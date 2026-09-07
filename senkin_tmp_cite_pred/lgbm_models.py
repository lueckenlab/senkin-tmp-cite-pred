import logging

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from senkin_tmp_cite_pred.metrics import correlation_score
from senkin_tmp_cite_pred.preprocess import get_matrix, to_dense

logger = logging.getLogger(__name__)

common_lgbm_params = {
    "objective": "rmse",
    "metric": "mse",
    "num_leaves": 33,
    "min_data_in_leaf": 30,
    "learning_rate": 0.01,
    "max_depth": 7,
    "boosting": "gbdt",
    "bagging_freq": 1,
    "verbosity": -1,
    "bagging_seed": 42,
}

lgbm_params_1 = {
    **common_lgbm_params,
    "feature_fraction": 0.05,
    "bagging_fraction": 0.9,
}

lgbm_params_2 = {
    **common_lgbm_params,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "lambda_l1": 0.1,
    "lambda_l2": 1,
}

lgbm_params_3 = {
    **common_lgbm_params,
    "feature_fraction": 0.08,
    "bagging_fraction": 0.9,
}

lgbm_params_4 = {
    **common_lgbm_params,
    "feature_fraction": 0.1,
    "bagging_fraction": 0.9,
    "lambda_l1": 1,
    "lambda_l2": 10,
}


def _build_fold_datasets(train_cite_X, folds, params):
    """Create LightGBM datasets for every fold once, so that the expensive feature binning is not
    repeated for every target. Labels are placeholders and are set per target with `set_label`.
    Binning does not depend on the labels, so the models are identical to the ones trained on
    freshly created datasets."""
    fold_datasets = []
    for train_idx, valid_idx in folds.split(train_cite_X):
        # The raw data slices are freed after the (lazy) construction, only the binned datasets are kept
        dtrain = lgb.Dataset(train_cite_X[train_idx], label=np.zeros(len(train_idx)), params=params)
        dval = lgb.Dataset(train_cite_X[valid_idx], label=np.zeros(len(valid_idx)), reference=dtrain, params=params)
        fold_datasets.append((train_idx, valid_idx, dtrain, dval))
    return fold_datasets


def train_lightgbm_kfold(
    train_cite_X, train_cite_y, test_cite_X, folds, params, num_boost_round=10000, early_stopping_rounds=100
):
    """Train LightGBM model using k-fold cross validation.

    The model is trained once per fold. Out-of-fold predictions are returned for the training data,
    and predictions for the test data are averaged across all folds.

    Parameters
    ----------
    train_cite_X : np.ndarray or scipy.sparse matrix
        Training features
    train_cite_y : np.ndarray
        Training target values (a single target)
    test_cite_X : np.ndarray or scipy.sparse matrix
        Test features
    folds : sklearn.model_selection._split.KFold
        K-fold cross validation splitter
    params : dict
        LightGBM model parameters
    num_boost_round : int
        Maximum number of LightGBM boosting rounds
    early_stopping_rounds : int
        Number of boosting rounds with no improvement on the validation fold after which training stops

    Returns
    -------
    tuple
        (train_predictions, test_predictions) - Model predictions on train and test sets
    """
    fold_datasets = _build_fold_datasets(train_cite_X, folds, params)
    return _train_single_target(train_cite_X, train_cite_y, test_cite_X, fold_datasets, params, num_boost_round, early_stopping_rounds)


def _train_single_target(
    train_cite_X, train_cite_y, test_cite_X, fold_datasets, params, num_boost_round, early_stopping_rounds
):
    train_preds = np.zeros(train_cite_X.shape[0])
    test_preds = np.zeros(test_cite_X.shape[0])
    n_splits = len(fold_datasets)

    train_cite_y = np.asarray(train_cite_y, dtype=np.float64)

    for n_fold, (train_idx, valid_idx, dtrain, dval) in enumerate(fold_datasets):
        logger.debug(f"Fold: {n_fold}")
        dtrain.set_label(train_cite_y[train_idx])
        dval.set_label(train_cite_y[valid_idx])

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )

        train_preds[valid_idx] = bst.predict(train_cite_X[valid_idx], num_iteration=bst.best_iteration)
        test_preds += bst.predict(test_cite_X, num_iteration=bst.best_iteration) / n_splits

    return train_preds, test_preds


def get_lgbm_predictions(
    train_cite_X,
    train_cite_y,
    test_cite_X,
    folds,
    params,
    n_tsvd_components=100,
    num_boost_round=10000,
    early_stopping_rounds=100,
):
    """Train LightGBM models for each target, return TSVD-reduced predictions

    Parameters
    ----------
    train_cite_X : np.ndarray or scipy.sparse matrix
        Training features
    train_cite_y : np.ndarray
        Training target values, e.g. protein expression values, shape (n_train_cells, n_targets)
    test_cite_X : np.ndarray or scipy.sparse matrix
        Test features
    folds : sklearn.model_selection._split.KFold
        K-fold cross validation splitter
    params : dict
        LightGBM model parameters
    n_tsvd_components : int
        Number of TSVD components. It is reduced if there are not enough targets.
    num_boost_round : int
        Number of LightGBM boosting rounds
    early_stopping_rounds : int
        Number of LightGBM boosting rounds with no improvement after which training will be stopped

    Returns
    -------
    np.ndarray
        TSVD-reduced predictions of shape (n_train_cells + n_test_cells, n_tsvd_components). Training
        cells (out-of-fold predictions) come first, test cells (averaged over folds) second.
    """
    train_cite_y = to_dense(train_cite_y, dtype=np.float64)
    if train_cite_y.ndim == 1:
        train_cite_y = train_cite_y[:, None]
    n_targets = train_cite_y.shape[1]

    train_preds = np.zeros(shape=(train_cite_X.shape[0], n_targets))
    test_preds = np.zeros(shape=(test_cite_X.shape[0], n_targets))

    logger.info(f"Training LightGBM models for {n_targets} targets")

    if sparse.issparse(train_cite_X):
        train_cite_X = sparse.csr_matrix(train_cite_X)
    if sparse.issparse(test_cite_X):
        test_cite_X = sparse.csr_matrix(test_cite_X)

    fold_datasets = _build_fold_datasets(train_cite_X, folds, params)

    for i in range(n_targets):
        logger.debug(f"Training LightGBM model for target {i}")

        train_preds[:, i], test_preds[:, i] = _train_single_target(
            train_cite_X,
            train_cite_y[:, i],
            test_cite_X,
            fold_datasets,
            params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    if n_targets > 1:
        cv = correlation_score(train_cite_y, train_preds)
        logger.info(f"CV score: {cv}")

    lgbm_predictions = np.concatenate([train_preds, test_preds], axis=0)

    n_tsvd_components = min(n_tsvd_components, min(lgbm_predictions.shape) - 1)
    logger.info(f"TSVD-reducing predictions to {n_tsvd_components} components")
    tsvd = TruncatedSVD(n_components=n_tsvd_components, algorithm="arpack")
    lgbm_predictions_svd = tsvd.fit_transform(lgbm_predictions)

    return lgbm_predictions_svd


def train_lightgbm_models(
    adata_rna,
    adata_prot,
    train_cell_ids,
    test_cell_ids,
    folds,
    num_boost_round=10000,
    early_stopping_rounds=100,
    n_tsvd_components=100,
    prot_key="dsb",
):
    """Train the 4 LightGBM models of the original solution and store TSVD-reduced predictions in `adata_rna.obsm`

    The models predict:
    1. normalized proteins from log-normalized RNA (`.obsm["X_log_normalized"]`)
    2. normalized proteins from CLR-TSVD, selected raw genes, and TSVD + PCA of the customly normalized RNA
    3. normalized proteins from raw RNA counts (`.X`)
    4. raw protein counts (`.X`) from raw RNA counts

    Parameters
    ----------
    adata_rna : AnnData
        RNA data preprocessed with `preprocess_data` (raw counts in .X, feature arrays in .obsm)
    adata_prot : AnnData
        Protein data with raw counts in .X and normalized values in `.layers[prot_key]`
    train_cell_ids, test_cell_ids : array-like
        Observation names of the training and test cells
    folds : sklearn.model_selection.KFold
        K-fold cross validation splitter
    num_boost_round, early_stopping_rounds, n_tsvd_components : int
        See `get_lgbm_predictions`
    prot_key : str = "dsb"
        Layer of `adata_prot` with the normalized protein expression, used as the target of the models 1-3

    Returns
    -------
    AnnData
        `adata_rna` with `X_lgbm_1`, ..., `X_lgbm_4` arrays added to `.obsm`
    """
    assert adata_rna.shape[0] == adata_prot.shape[0], "RNA and protein data must have the same number of cells"
    assert adata_rna.shape[0] == len(train_cell_ids) + len(
        test_cell_ids
    ), "RNA and protein data must have the same number of cells as train and test cell ids"
    assert adata_rna.obs_names.isin(train_cell_ids).sum() == len(train_cell_ids), "All train cell ids must be in the data"
    assert adata_rna.obs_names.isin(test_cell_ids).sum() == len(test_cell_ids), "All test cell ids must be in the data"

    # Create numerical indices for train and test cells to save info to the arrays in obsm correctly
    train_indices = adata_rna.obs_names.get_indexer(train_cell_ids)
    test_indices = adata_rna.obs_names.get_indexer(test_cell_ids)

    n_tsvd_components = min(n_tsvd_components, adata_prot.shape[1] - 1)  # Make sure there are less components than features to prevent an error

    prot_train = to_dense(get_matrix(adata_prot, prot_key)[train_indices], dtype=np.float64)
    prot_raw_train = to_dense(adata_prot.X[train_indices], dtype=np.float64)

    def _split(matrix):
        return matrix[train_indices], matrix[test_indices]

    logger.info("Initializing arrays in obsm with zeros")
    for i in range(1, 5):
        adata_rna.obsm[f"X_lgbm_{i}"] = np.zeros((adata_rna.shape[0], n_tsvd_components))

    def _store(key, predictions):
        adata_rna.obsm[key][train_indices] = predictions[: len(train_indices)]
        adata_rna.obsm[key][test_indices] = predictions[len(train_indices) :]

    lgbm_kwargs = {
        "folds": folds,
        "n_tsvd_components": n_tsvd_components,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
    }

    logger.info("Training LightGBM models")

    # Source: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/cite_lgb_transformed_sparse_matrix.ipynb
    logger.info("Training LightGBM model 1 for predicting normalized protein expression from log-normalized RNA expression")
    train_X, test_X = _split(adata_rna.obsm["X_log_normalized"])
    _store("X_lgbm_1", get_lgbm_predictions(train_X, prot_train, test_X, params=lgbm_params_1, **lgbm_kwargs))

    # Source: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/cite_lgb_raw_clr_pca.ipynb
    logger.info("Preparing datasets for LightGBM model 2")
    combined = np.concatenate(
        [
            to_dense(adata_rna.obsm["X_clr_tsvd"]),
            to_dense(adata_rna.obsm["X_raw_selected"]),
            to_dense(adata_rna.obsm["X_sqrt_norm_tsvd"]),
            to_dense(adata_rna.obsm["X_sqrt_norm_pca"]),
        ],
        axis=1,
    )
    train_X, test_X = _split(combined)
    del combined

    logger.info(
        "Training LightGBM model 2 for predicting normalized protein expression from customly normalized RNA expression data and selected features"
    )
    _store("X_lgbm_2", get_lgbm_predictions(train_X, prot_train, test_X, params=lgbm_params_2, **lgbm_kwargs))

    # Source: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/cite_lgb_raw_sparse_matrix.ipynb
    logger.info("Training LightGBM model 3 for predicting normalized protein expression from raw RNA expression")
    train_X, test_X = _split(adata_rna.X)
    _store("X_lgbm_3", get_lgbm_predictions(train_X, prot_train, test_X, params=lgbm_params_3, **lgbm_kwargs))

    # Source: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/cite_lgb_raw_target.ipynb
    logger.info("Training LightGBM model 4 for predicting raw protein expression from raw RNA expression")
    _store("X_lgbm_4", get_lgbm_predictions(train_X, prot_raw_train, test_X, params=lgbm_params_4, **lgbm_kwargs))

    return adata_rna
