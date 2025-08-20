import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import logging

common_lgbm_params = {
    "objective" : "rmse", 
    "metric" : "mse",
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
    "lambda_l1":0.1,
    "lambda_l2":1,
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
    "lambda_l1":1,
    "lambda_l2":10,
}

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the OpenProblems2022 competition rules. 
    
    It is assumed that the predictions are not constant.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values

    Returns
    -------
    float
        Average Pearson correlation coefficient across cells
    """
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    
    corrsum = 0

    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]

    return corrsum / len(y_true)

def train_lightgbm_kfold(train_cite_X, train_cite_y, test_cite_X, folds, params, num_boost_round=10000, early_stopping_rounds=100):
    """Train LightGBM model using k-fold cross validation.

    The model is trained 5 times, each time with a different fold.
    Predictions are averaged across all folds.

    Parameters
    ----------
    train_cite_X : np.ndarray
        Training features
    train_cite_y : np.ndarray
        Training target values
    test_cite_X : np.ndarray
        Test features
    folds : sklearn.model_selection._split.KFold
        K-fold cross validation splitter
    params : dict
        LightGBM model parameters

    Returns
    -------
    tuple
        (train_predictions, test_predictions) - Model predictions on train and test sets
    """
    train_preds = np.zeros(train_cite_X.shape[0])
    test_preds = np.zeros(test_cite_X.shape[0])

    logger = logging.getLogger(__name__)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_cite_X)): 
        logger.debug(f"Fold: {n_fold}")
        train_x = train_cite_X[train_idx]
        valid_x = train_cite_X[valid_idx]
        train_y = train_cite_y[train_idx]
        valid_y = train_cite_y[valid_idx]

        dtrain = lgb.Dataset(train_x, label=train_y,)
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain,)
        bst = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            ]
        )

        train_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        test_preds += bst.predict(test_cite_X, num_iteration=bst.best_iteration) / folds.n_splits         
        
    return train_preds, test_preds


def get_lgbm_predictions(train_cite_X, train_cite_y, test_cite_X, folds, params, n_tsvd_components=100, num_boost_round=10000, early_stopping_rounds=100):
    """Train LightGBM models for each target, return TSVD-reduced predictions

    Parameters
    ----------
    train_cite_X : np.ndarray
        Training features
    train_cite_y : np.ndarray
        Training target values, e.g. protein expression values
    test_cite_X : np.ndarray
        Test features
    folds : sklearn.model_selection._split.KFold
        K-fold cross validation splitter
    params : dict
        LightGBM model parameters
    n_tsvd_components : int
        Number of TSVD components
    num_boost_round : int
        Number of LightGBM boosting rounds
    early_stopping_rounds : int
        Number of LightGBM boosting rounds with no improvement after which training will be stopped

    Returns
    -------
    np.ndarray
        TSVD-reduced predictions
    """
    logger = logging.getLogger(__name__)
    
    train_preds = np.zeros(shape=(train_cite_X.shape[0], train_cite_y.shape[1]))
    test_preds = np.zeros(shape=(test_cite_X.shape[0], train_cite_y.shape[1]))

    logger.info(f"Training LightGBM models for {train_cite_y.shape[1]} targets")

    for i in range(train_cite_y.shape[1]):
        logger.debug(f"Training LightGBM model for target {i}")

        train_cite_y_single = train_cite_y[:, i]

        train_preds[:, i], test_preds[:, i] = train_lightgbm_kfold(
            train_cite_X,
            train_cite_y_single,
            test_cite_X,
            folds,
            params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds
        )

    cv = correlation_score(train_cite_y, train_preds)
    logger.info(f"CV score: {cv}")

    lgbm_predictions = np.concatenate([train_preds, test_preds], axis=0)

    n_tsvd_components = min(n_tsvd_components, train_cite_y.shape[1] - 1)  # Make sure there are less components than features
    logger.info(f"TSVD-reducing predictions to {n_tsvd_components} components")
    tsvd = TruncatedSVD(n_components=n_tsvd_components, algorithm="arpack")
    lgbm_predictions_svd = tsvd.fit_transform(lgbm_predictions)
    
    return lgbm_predictions_svd


def train_lightgbm_models(adata_rna, adata_prot, train_cell_ids, test_cell_ids, folds, num_boost_round=10000, early_stopping_rounds=100, n_tsvd_components=100):

    logger = logging.getLogger(__name__)

    assert adata_rna.shape[0] == adata_prot.shape[0], "RNA and protein data must have the same number of cells"
    assert adata_rna.shape[0] == len(train_cell_ids) + len(test_cell_ids), "RNA and protein data must have the same number of cells as train and test cell ids"
    assert train_cell_ids.isin(adata_rna.obs_names).all(), "All train cell ids must be in the data"
    assert test_cell_ids.isin(adata_rna.obs_names).all(), "All test cell ids must be in the data"

    logger.info("Training LightGBM models")

    # cite_lgb_transformed_sparse_matrix.ipynb
    logger.info("Training LightGBM model 1 for predicting DSB-normalized protein expression from log-normalized RNA expression")
    lgbm_1_predictions = get_lgbm_predictions(
        adata_rna[train_cell_ids].obsm["X_log_normalized"],
        adata_prot[train_cell_ids].layers["dsb"],
        adata_rna[test_cell_ids].obsm["X_log_normalized"],
        folds,
        lgbm_params_1,
        n_tsvd_components=n_tsvd_components,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    adata_rna[train_cell_ids].obsm["X_lgbm_1"] = lgbm_1_predictions[:len(train_cell_ids), :]
    adata_rna[test_cell_ids].obsm["X_lgbm_1"] = lgbm_1_predictions[len(train_cell_ids):, :]

    # cite_lgb_raw_clr_pca.ipynb
    logger.info("Preparing datasets for LightGBM model 2")
    train_cite_X = np.concatenate([
        adata_rna[train_cell_ids].obsm["X_clr_tsvd"],
        adata_rna[train_cell_ids].obsm["X_raw_selected"].toarray(),
        #train_cite_inputs_bio_norm_2_svd,  # We excluded TSVD preprocessing, because it is very similar to PCA. We use 100 PCs instead
        adata_rna[train_cell_ids].obsm["X_pca_sqrt_norm"],
        ], axis=1)

    test_cite_X = np.concatenate([
        adata_rna[test_cell_ids].obsm["X_clr_tsvd"],
        adata_rna[test_cell_ids].obsm["X_raw_selected"].toarray(),
        #test_cite_inputs_bio_norm_2_svd,
        adata_rna[test_cell_ids].obsm["X_pca_sqrt_norm"]
    ], axis=1)

    logger.info("Training LightGBM model 2 for predicting DSB-normalized protein expression from customly normalized RNA expression data and selected features")
    lgbm_2_predictions = get_lgbm_predictions(
        train_cite_X,
        adata_prot[train_cell_ids].layers["dsb"],
        test_cite_X,
        folds,
        lgbm_params_2,
        n_tsvd_components=n_tsvd_components,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
        )
    
    adata_rna[train_cell_ids].obsm["X_lgbm_2"] = lgbm_2_predictions[:len(train_cell_ids), :]
    adata_rna[test_cell_ids].obsm["X_lgbm_2"] = lgbm_2_predictions[len(train_cell_ids):, :]

    # cite_lgb_raw_sparse_matrix.ipynb
    logger.info("Training LightGBM model 3 for predicting DSB-normalized protein expression from raw RNA expression")
    lgbm_3_predictions = get_lgbm_predictions(
        adata_rna[train_cell_ids].X.toarray(),
        adata_prot[train_cell_ids].layers["dsb"],
        adata_rna[test_cell_ids].X.toarray(),
        folds,
        lgbm_params_3,
        n_tsvd_components=n_tsvd_components,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    adata_rna[train_cell_ids].obsm["X_lgbm_3"] = lgbm_3_predictions[:len(train_cell_ids), :]
    adata_rna[test_cell_ids].obsm["X_lgbm_3"] = lgbm_3_predictions[len(train_cell_ids):, :]

    # cite_lgb_raw_target.ipynb
    logger.info("Training LightGBM model 4 for predicting raw protein expression from raw RNA expression")
    lgbm_4_predictions = get_lgbm_predictions(
        adata_rna[train_cell_ids].X.toarray(),
        adata_prot[train_cell_ids].X.toarray(),
        adata_rna[test_cell_ids].X.toarray(),
        folds,
        lgbm_params_4,
        n_tsvd_components=n_tsvd_components,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )

    adata_rna[train_cell_ids].obsm["X_lgbm_4"] = lgbm_4_predictions[:len(train_cell_ids), :]
    adata_rna[test_cell_ids].obsm["X_lgbm_4"] = lgbm_4_predictions[len(train_cell_ids):, :]

    return adata_rna
