import logging
import os
import resource
import tempfile
import time

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
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
    # Build histograms column-wise. With feature_fraction 0.05-0.1 only a few percent of the genes are used per
    # tree, but LightGBM's automatic choice is often row-wise, which scans every nonzero of every cell each round.
    # Column-wise is 2-3x faster on the whole-transcriptome inputs and gives identical results.
    "force_col_wise": True,
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


def _available_cpus():
    """Number of CPUs this process may use (respects the affinity mask set by schedulers and containers)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on every platform
        return os.cpu_count() or 1


def _bin_fold(train_cite_X, train_idx, valid_idx, params):
    """Bin the features of one fold into constructed LightGBM datasets.

    Binning is the expensive, label-independent part of LightGBM training, so it is done once per fold and the
    datasets are reused for every target (labels are swapped with `set_label`). The models are identical to the
    ones trained on freshly created datasets. Each dataset is constructed right away so that the raw slice of the
    feature matrix is freed immediately and only the (much smaller) binned data is kept."""
    dtrain = lgb.Dataset(train_cite_X[train_idx], label=np.zeros(len(train_idx)), params=params).construct()
    dval = lgb.Dataset(
        train_cite_X[valid_idx], label=np.zeros(len(valid_idx)), reference=dtrain, params=params
    ).construct()
    return dtrain, dval


def _load_binned_fold(train_path, valid_path, params):
    """Load the binned datasets of one fold saved with `Dataset.save_binary`."""
    dtrain = lgb.Dataset(train_path, params=params).construct()
    dval = lgb.Dataset(valid_path, reference=dtrain, params=params).construct()
    return dtrain, dval


def _train_targets_on_fold(
    dtrain, dval, valid_x, test_cite_X, train_y_fold, valid_y_fold, params, num_boost_round, early_stopping_rounds
):
    """Train one LightGBM model per target on binned datasets of one fold.

    Returns the predictions for the validation cells of the fold, the predictions for the test cells, and the best
    iteration of every target, each with one column per target.
    """
    n_targets = train_y_fold.shape[1]
    valid_preds = np.zeros((valid_x.shape[0], n_targets))
    test_preds = np.zeros((test_cite_X.shape[0], n_targets))
    best_iterations = np.zeros(n_targets, dtype=int)

    for i in range(n_targets):
        dtrain.set_label(train_y_fold[:, i])
        dval.set_label(valid_y_fold[:, i])

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )

        valid_preds[:, i] = booster.predict(valid_x, num_iteration=booster.best_iteration)
        test_preds[:, i] = booster.predict(test_cite_X, num_iteration=booster.best_iteration)
        best_iterations[i] = booster.best_iteration

    return valid_preds, test_preds, best_iterations


def _train_fold(
    train_cite_X,
    train_cite_y,
    test_cite_X,
    train_idx,
    valid_idx,
    params,
    num_boost_round,
    early_stopping_rounds,
    binned_paths=None,
):
    """Train one LightGBM model per target on a single fold, binning the fold here or loading it from
    `binned_paths` (a (train, valid) pair of binary dataset files)."""
    train_cite_y = np.asarray(train_cite_y, dtype=np.float64)
    if train_cite_y.ndim == 1:
        train_cite_y = train_cite_y[:, None]

    if binned_paths is None:
        dtrain, dval = _bin_fold(train_cite_X, train_idx, valid_idx, params)
    else:
        dtrain, dval = _load_binned_fold(*binned_paths, params)
    valid_x = train_cite_X[valid_idx]

    return _train_targets_on_fold(
        dtrain, dval, valid_x, test_cite_X, train_cite_y[train_idx], train_cite_y[valid_idx],
        params, num_boost_round, early_stopping_rounds,
    )


def _save_binned_folds(train_cite_X, splits, params, directory):
    """Bin every fold once (with all threads) and save the datasets as LightGBM binary files, which worker
    processes load without holding the raw feature slices or LightGBM's binning buffers."""
    paths = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        start_time = time.time()
        dtrain, dval = _bin_fold(train_cite_X, train_idx, valid_idx, params)
        train_path = os.path.join(directory, f"fold_{fold}_train.bin")
        valid_path = os.path.join(directory, f"fold_{fold}_valid.bin")
        dtrain.save_binary(train_path)
        dval.save_binary(valid_path)
        del dtrain, dval
        paths.append((train_path, valid_path))
        logger.info(
            f"Binned fold {fold} in {time.time() - start_time:.0f} s "
            f"({(os.path.getsize(train_path) + os.path.getsize(valid_path)) / 1e9:.2f} GB)"
        )
    return paths


def _matrix_bytes(matrix):
    if sparse.issparse(matrix):
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return np.asarray(matrix).nbytes


def _process_rss_gb():
    """Resident memory of this process right now (Linux); falls back to the peak elsewhere."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _worker_layout(n_jobs, total_threads, memory_budget_gb, train_cite_X, binned_paths, splits):
    """Number of worker processes and threads per worker.

    Every worker holds one binned fold (the size of its binary file, which LightGBM keeps roughly as is in memory)
    plus the validation slice of the feature matrix it predicts on, on top of the parent process and the shared
    copy of the feature matrix. When `memory_budget_gb` allows fewer workers than threads, the threads are spread
    over the affordable workers instead of staying idle.
    """
    threads_per_worker = max(1, total_threads // n_jobs)
    if memory_budget_gb is None:
        return n_jobs, threads_per_worker

    largest_fold_gb = max(
        (os.path.getsize(train_path) + os.path.getsize(valid_path)) / 1e9 for train_path, valid_path in binned_paths
    )
    largest_valid_gb = max(_matrix_bytes(train_cite_X[valid_idx]) / 1e9 for _, valid_idx in splits)
    # measured on the NeurIPS 2022 CITE inputs: 4.2 GB private per worker for a 0.8 GB binary fold and a 1.2 GB slice
    worker_gb = 1.0 + 2.5 * largest_fold_gb + largest_valid_gb
    reserved_gb = _process_rss_gb() + _matrix_bytes(train_cite_X) / 1e9
    affordable = max(1, int((memory_budget_gb - reserved_gb) // worker_gb))
    logger.info(
        f"Memory budget {memory_budget_gb:.0f} GB: parent + shared inputs {reserved_gb:.1f} GB, "
        f"{worker_gb:.1f} GB per worker -> at most {affordable} worker(s)"
    )
    if affordable < n_jobs:
        threads_per_worker = -(-total_threads // affordable)  # ceil
        n_jobs = min(affordable, max(1, total_threads // threads_per_worker))
        logger.info(f"Using {n_jobs} worker(s) x {threads_per_worker} thread(s) to stay within the memory budget")
    return n_jobs, threads_per_worker


def train_lightgbm_kfold(
    train_cite_X,
    train_cite_y,
    test_cite_X,
    folds,
    params,
    num_boost_round=10000,
    early_stopping_rounds=100,
    n_jobs=1,
    memory_budget_gb=None,
):
    """Train LightGBM models using k-fold cross validation, one model per fold and target.

    Out-of-fold predictions are returned for the training data, and predictions for the test data are averaged
    across folds.

    Parameters
    ----------
    train_cite_X : np.ndarray or scipy.sparse matrix
        Training features
    train_cite_y : np.ndarray
        Training target values, shape (n_train_cells,) or (n_train_cells, n_targets)
    test_cite_X : np.ndarray or scipy.sparse matrix
        Test features
    folds : sklearn.model_selection._split.KFold
        K-fold cross validation splitter
    params : dict
        LightGBM model parameters. `num_threads` is the total number of threads to use; with `n_jobs > 1` it is
        divided among the worker processes. If it is missing, the CPUs available to the process are used.
    num_boost_round : int
        Maximum number of LightGBM boosting rounds
    early_stopping_rounds : int
        Number of boosting rounds with no improvement on the validation fold after which training stops
    n_jobs : int = 1
        Number of worker processes. The folds are binned once in this process and saved as LightGBM binary
        datasets; every worker loads one fold and trains a chunk of the targets on it, so its memory footprint is
        one binned fold. Training several targets in parallel with a few threads each is much more efficient than
        one target with many threads, because a single LightGBM model on tens of thousands of cells parallelizes
        poorly beyond a handful of threads. `-1` uses all available CPUs.
    memory_budget_gb : float, optional
        Total memory the training may use. With `n_jobs > 1` the number of workers is reduced if the parent
        process, the shared feature matrix and the workers' binned folds would not fit.

    Returns
    -------
    tuple
        (train_predictions, test_predictions, best_iterations): out-of-fold predictions of shape
        (n_train_cells, n_targets), fold-averaged test predictions of shape (n_test_cells, n_targets), and the best
        iteration of every (fold, target) model of shape (n_folds, n_targets).
    """
    train_cite_y = to_dense(train_cite_y, dtype=np.float64)
    if train_cite_y.ndim == 1:
        train_cite_y = train_cite_y[:, None]
    n_targets = train_cite_y.shape[1]

    total_threads = params.get("num_threads") or _available_cpus()
    if n_jobs == -1:
        n_jobs = total_threads
    n_jobs = max(1, min(n_jobs, n_targets, total_threads))

    if sparse.issparse(train_cite_X):
        train_cite_X = sparse.csr_matrix(train_cite_X)
    if sparse.issparse(test_cite_X):
        test_cite_X = sparse.csr_matrix(test_cite_X)

    splits = list(folds.split(train_cite_X))
    train_preds = np.zeros((train_cite_X.shape[0], n_targets))
    test_preds = np.zeros((test_cite_X.shape[0], n_targets))
    best_iterations = np.zeros((len(splits), n_targets), dtype=int)
    start_time = time.time()

    def _collect(results, n_tasks):
        for n_done, (fold, chunk, (valid_preds, fold_test_preds, fold_best_iterations)) in enumerate(results, start=1):
            _, valid_idx = splits[fold]
            train_preds[np.ix_(valid_idx, chunk)] = valid_preds
            test_preds[:, chunk] += fold_test_preds / len(splits)
            best_iterations[fold, chunk] = fold_best_iterations
            elapsed = time.time() - start_time
            logger.info(
                f"{n_done}/{n_tasks} fold-chunk tasks done in {elapsed / 60:.1f} min (fold {fold}, {len(chunk)} targets, "
                f"best iterations {int(fold_best_iterations.min())}-{int(fold_best_iterations.max())})"
            )

    if n_jobs == 1:
        all_targets = np.arange(n_targets)
        logger.info(f"Training LightGBM models for {n_targets} targets x {len(splits)} folds with {total_threads} thread(s)")
        results = (
            (fold, all_targets, _train_fold(
                train_cite_X, train_cite_y, test_cite_X, train_idx, valid_idx,
                {**params, "num_threads": total_threads}, num_boost_round, early_stopping_rounds,
            ))
            for fold, (train_idx, valid_idx) in enumerate(splits)
        )
        _collect(results, len(splits))
        return train_preds, test_preds, best_iterations

    with tempfile.TemporaryDirectory(prefix="lgbm_folds_") as directory:
        binned_paths = _save_binned_folds(train_cite_X, splits, {**params, "num_threads": total_threads}, directory)
        n_jobs, threads_per_worker = _worker_layout(n_jobs, total_threads, memory_budget_gb, train_cite_X, binned_paths, splits)
        worker_params = {**params, "num_threads": threads_per_worker}

        target_chunks = [chunk for chunk in np.array_split(np.arange(n_targets), n_jobs) if len(chunk)]
        tasks = [(fold, chunk) for fold in range(len(splits)) for chunk in target_chunks]

        def _run(fold, chunk):
            train_idx, valid_idx = splits[fold]
            return fold, chunk, _train_fold(
                train_cite_X, train_cite_y[:, chunk], test_cite_X, train_idx, valid_idx,
                worker_params, num_boost_round, early_stopping_rounds, binned_paths=binned_paths[fold],
            )

        logger.info(
            f"Training LightGBM models for {n_targets} targets x {len(splits)} folds "
            f"with {n_jobs} process(es) x {worker_params['num_threads']} thread(s)"
        )
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_run)(fold, chunk) for fold, chunk in tasks
        )
        _collect(results, len(tasks))

    return train_preds, test_preds, best_iterations


def get_lgbm_predictions(
    train_cite_X,
    train_cite_y,
    test_cite_X,
    folds,
    params,
    n_tsvd_components=100,
    num_boost_round=10000,
    early_stopping_rounds=100,
    n_jobs=1,
    memory_budget_gb=None,
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
        LightGBM model parameters (see `train_lightgbm_kfold` for `num_threads`)
    n_tsvd_components : int
        Number of TSVD components. It is reduced if there are not enough targets.
    num_boost_round : int
        Number of LightGBM boosting rounds
    early_stopping_rounds : int
        Number of LightGBM boosting rounds with no improvement after which training will be stopped
    n_jobs : int = 1
        Number of worker processes training different targets in parallel (see `train_lightgbm_kfold`)
    memory_budget_gb : float, optional
        Total memory the training may use (see `train_lightgbm_kfold`)

    Returns
    -------
    np.ndarray
        TSVD-reduced predictions of shape (n_train_cells + n_test_cells, n_tsvd_components). Training
        cells (out-of-fold predictions) come first, test cells (averaged over folds) second.
    """
    train_preds, test_preds, best_iterations = train_lightgbm_kfold(
        train_cite_X,
        train_cite_y,
        test_cite_X,
        folds,
        params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        memory_budget_gb=memory_budget_gb,
    )

    logger.info(
        f"Best iterations: median {int(np.median(best_iterations))}, max {int(best_iterations.max())} "
        f"(limit {num_boost_round})"
    )
    if train_preds.shape[1] > 1:
        cv = correlation_score(to_dense(train_cite_y, dtype=np.float64), train_preds)
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
    n_jobs=1,
    memory_budget_gb=None,
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
    n_jobs : int = 1
        Number of worker processes training different targets in parallel (see `train_lightgbm_kfold`)
    memory_budget_gb : float, optional
        Total memory the training may use (see `train_lightgbm_kfold`)

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
        "n_jobs": n_jobs,
        "memory_budget_gb": memory_budget_gb,
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
