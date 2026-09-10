import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold

from senkin_tmp_cite_pred.lgbm_models import _build_fold_datasets, _train_single_target, get_lgbm_predictions, lgbm_params_2

PARAMS = {**lgbm_params_2, "num_threads": 1, "deterministic": True, "seed": 1, "learning_rate": 0.1}


def _fresh_datasets_reference(train_X, train_y, test_X, folds, params, num_boost_round, early_stopping_rounds):
    """Training with a new lgb.Dataset per fold and target, as in the original notebooks"""
    train_preds = np.zeros(train_X.shape[0])
    test_preds = np.zeros(test_X.shape[0])
    for train_idx, valid_idx in folds.split(train_X):
        dtrain = lgb.Dataset(train_X[train_idx], label=train_y[train_idx])
        dval = lgb.Dataset(train_X[valid_idx], label=train_y[valid_idx], reference=dtrain)
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        train_preds[valid_idx] = bst.predict(train_X[valid_idx], num_iteration=bst.best_iteration)
        test_preds += bst.predict(test_X, num_iteration=bst.best_iteration) / folds.n_splits
    return train_preds, test_preds


def _data(rng, as_sparse=False):
    train_X = rng.poisson(1.0, size=(300, 25)).astype(np.float32)
    test_X = rng.poisson(1.0, size=(80, 25)).astype(np.float32)
    weights = rng.normal(size=(25, 3))
    train_y = train_X @ weights + rng.normal(scale=0.1, size=(300, 3))
    if as_sparse:
        train_X, test_X = sparse.csr_matrix(train_X), sparse.csr_matrix(test_X)
    return train_X, train_y, test_X


def test_reused_fold_datasets_give_identical_models(rng):
    train_X, train_y, test_X = _data(rng)
    folds = KFold(n_splits=3, shuffle=True, random_state=0)
    fold_datasets = _build_fold_datasets(train_X, folds, PARAMS)

    for target in range(train_y.shape[1]):
        expected_train, expected_test = _fresh_datasets_reference(train_X, train_y[:, target], test_X, folds, PARAMS, 30, 5)
        result_train, result_test = _train_single_target(train_X, train_y[:, target], test_X, fold_datasets, PARAMS, 30, 5)
        np.testing.assert_allclose(result_train, expected_train, rtol=1e-6)
        np.testing.assert_allclose(result_test, expected_test, rtol=1e-6)


def test_get_lgbm_predictions_shapes_sparse_and_dense_agree(rng):
    train_X, train_y, test_X = _data(rng)
    folds = KFold(n_splits=2, shuffle=True, random_state=0)
    dense = get_lgbm_predictions(train_X, train_y, test_X, folds, PARAMS, n_tsvd_components=10, num_boost_round=10, early_stopping_rounds=3)
    sparse_result = get_lgbm_predictions(
        sparse.csr_matrix(train_X), train_y, sparse.csr_matrix(test_X), folds, PARAMS, n_tsvd_components=10, num_boost_round=10, early_stopping_rounds=3
    )
    # only 3 targets -> 2 TSVD components at most
    assert dense.shape == (train_X.shape[0] + test_X.shape[0], 2)
    np.testing.assert_allclose(np.abs(dense), np.abs(sparse_result), atol=1e-5)
