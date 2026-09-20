import lightgbm as lgb
import numpy as np
import pytest
from scipy import sparse
from sklearn.model_selection import KFold

from senkin_tmp_cite_pred.lgbm_models import get_lgbm_predictions, lgbm_params_2, train_lightgbm_kfold

PARAMS = {**lgbm_params_2, "num_threads": 1, "deterministic": True, "seed": 1, "learning_rate": 0.1}


def _fresh_datasets_reference(train_X, train_y, test_X, folds, params, num_boost_round, early_stopping_rounds):
    """Training with a new lgb.Dataset per fold and target, as in the original notebooks"""
    train_preds = np.zeros(train_X.shape[0])
    test_preds = np.zeros(test_X.shape[0])
    for train_idx, valid_idx in folds.split(train_X):
        dtrain = lgb.Dataset(train_X[train_idx], label=train_y[train_idx])
        dval = lgb.Dataset(train_X[valid_idx], label=train_y[valid_idx], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        train_preds[valid_idx] = booster.predict(train_X[valid_idx], num_iteration=booster.best_iteration)
        test_preds += booster.predict(test_X, num_iteration=booster.best_iteration) / folds.n_splits
    return train_preds, test_preds


def _data(rng, as_sparse=False, n_targets=3):
    train_X = rng.poisson(1.0, size=(300, 25)).astype(np.float32)
    test_X = rng.poisson(1.0, size=(80, 25)).astype(np.float32)
    weights = rng.normal(size=(25, n_targets))
    train_y = train_X @ weights + rng.normal(scale=0.1, size=(300, n_targets))
    if as_sparse:
        train_X, test_X = sparse.csr_matrix(train_X), sparse.csr_matrix(test_X)
    return train_X, train_y, test_X


def test_reused_fold_datasets_give_identical_models(rng):
    train_X, train_y, test_X = _data(rng)
    folds = KFold(n_splits=3, shuffle=True, random_state=0)

    result_train, result_test, best_iterations = train_lightgbm_kfold(train_X, train_y, test_X, folds, PARAMS, 30, 5)
    assert best_iterations.shape == (3, train_y.shape[1])

    for target in range(train_y.shape[1]):
        expected_train, expected_test = _fresh_datasets_reference(
            train_X, train_y[:, target], test_X, folds, PARAMS, 30, 5
        )
        np.testing.assert_allclose(result_train[:, target], expected_train, rtol=1e-6)
        np.testing.assert_allclose(result_test[:, target], expected_test, rtol=1e-6)


@pytest.mark.parametrize("n_jobs", [2, 3, -1])
def test_parallel_targets_match_serial(rng, n_jobs):
    """Workers load the folds from binary datasets saved by the parent; models must be identical to the serial path."""
    train_X, train_y, test_X = _data(rng, as_sparse=True, n_targets=5)
    folds = KFold(n_splits=3, shuffle=True, random_state=0)
    params = {**PARAMS, "num_threads": 3}

    serial = train_lightgbm_kfold(train_X, train_y, test_X, folds, params, 30, 5, n_jobs=1)
    parallel = train_lightgbm_kfold(train_X, train_y, test_X, folds, params, 30, 5, n_jobs=n_jobs)

    for serial_part, parallel_part in zip(serial, parallel):
        np.testing.assert_allclose(parallel_part, serial_part, rtol=1e-6)


def test_memory_budget_caps_workers(rng, caplog):
    train_X, train_y, test_X = _data(rng, as_sparse=True, n_targets=4)
    folds = KFold(n_splits=2, shuffle=True, random_state=0)
    params = {**PARAMS, "num_threads": 4}

    serial = train_lightgbm_kfold(train_X, train_y, test_X, folds, params, 20, 5, n_jobs=1)
    with caplog.at_level("INFO"):
        # a budget that fits the parent process plus about one worker
        budget = 0.75 + 0.5 + resource_peak_gb()
        capped = train_lightgbm_kfold(train_X, train_y, test_X, folds, params, 20, 5, n_jobs=4, memory_budget_gb=budget)
    assert "Reducing the number of worker processes" in caplog.text
    for serial_part, capped_part in zip(serial, capped):
        np.testing.assert_allclose(capped_part, serial_part, rtol=1e-6)


def resource_peak_gb():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def test_get_lgbm_predictions_shapes_sparse_and_dense_agree(rng):
    train_X, train_y, test_X = _data(rng)
    folds = KFold(n_splits=2, shuffle=True, random_state=0)
    dense = get_lgbm_predictions(
        train_X, train_y, test_X, folds, PARAMS, n_tsvd_components=10, num_boost_round=10, early_stopping_rounds=3
    )
    sparse_result = get_lgbm_predictions(
        sparse.csr_matrix(train_X),
        train_y,
        sparse.csr_matrix(test_X),
        folds,
        PARAMS,
        n_tsvd_components=10,
        num_boost_round=10,
        early_stopping_rounds=3,
    )
    # only 3 targets -> 2 TSVD components at most
    assert dense.shape == (train_X.shape[0] + test_X.shape[0], 2)
    np.testing.assert_allclose(np.abs(dense), np.abs(sparse_result), atol=1e-5)
