import logging

import numpy as np
import scanpy as sc
from fast_array_utils.stats import mean_var as _get_mean_var
from muon import prot as pt
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

logger = logging.getLogger(__name__)


def get_matrix(adata, key=None):
    """Fetch a cells x features matrix stored in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Data object to take the matrix from.
    key : str, optional
        `None` or `"X"` returns `adata.X`. Otherwise the key is looked up in `adata.layers`
        first and in `adata.obsm` second.

    Returns
    -------
    matrix : array-like
        The requested matrix (sparse or dense, as stored).
    """
    if key is None or key == "X":
        return adata.X
    if key in adata.layers:
        return adata.layers[key]
    if key in adata.obsm:
        return adata.obsm[key]
    raise KeyError(
        f"'{key}' is neither in .layers {list(adata.layers.keys())} nor in .obsm {list(adata.obsm.keys())}"
    )


def to_dense(matrix, dtype=None):
    """Convert a (possibly sparse) matrix to a dense numpy array, optionally casting it."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if dtype is not None:
        dense = dense.astype(dtype, copy=False)
    return dense


def pairwise_corr(X, Y):
    """Compute pairwise Pearson correlation between columns of X and columns of Y.

    Parameters
    ----------
    X : array-like
        A matrix of shape (n_samples, n_features)
    Y : array-like
        A matrix of shape (n_samples, n_targets)

    Returns
    -------
    corr_matrix : array-like
        A matrix of shape (n_features, n_targets) with the correlation between each feature
        and each target. Constant columns yield NaN, the same as `numpy.corrcoef`.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # Scale X and Y so that the correlation is just the dot product
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    return X.T @ Y / X.shape[0]


def log_normalize(adata, target_sum: float | None = 1e6):
    """
    Normalize total counts per cell and apply log1p transformation.

    Parameters
    ----------
    adata : AnnotatedData object
        Data to transform. .X layer will be used, it must contain raw counts.
    target_sum : float, optional
        Total count each cell is normalized to before log1p. Default is 1e6 (counts per million),
        which is how the RNA inputs of the OpenProblems 2022 competition were normalized and hence
        what the original solution was trained on. Pass `None` to normalize to the median total
        count across cells instead.

    Returns
    -------
    X_log_normalized : array-like
        A matrix of shape (n_samples, n_features) containing the log-normalized data. Sparsity of
        the input is preserved.
    """
    normalized = sc.pp.normalize_total(adata, target_sum=target_sum, inplace=False)["X"]
    return sc.pp.log1p(normalized)


def clr_tsvd(adata, n_components=200, random_state=None):
    """
    Compute TSVD-transform of Centered-log-ratio (CLR)-normalized data.

    Parameters
    ----------
    adata : AnnotatedData object
        Data to transform. .X layer will be used, it must contain raw counts.
    n_components : int = 200
        The number of components to keep. Default is 200.
    random_state : int, optional
        Seed for the TSVD solver, for reproducible components across runs. Default is
        None, matching TruncatedSVD's default (non-deterministic with algorithm="arpack").

    Returns
    -------
    X_clr_tsvd : array-like
        A matrix of shape (n_samples, n_components) containing the TSVD-transformed CLR-normalized data.
    """
    n_components = min(n_components, min(adata.shape) - 1)  # Make sure there are not too many components than the data can fit
    tsvd = TruncatedSVD(n_components=n_components, algorithm="arpack", random_state=random_state)
    clr = pt.pp.clr(adata, inplace=False).X
    return tsvd.fit_transform(clr)


def remove_constant_vars(adata):
    """
    Remove constant variables from the data.

    Parameters
    ----------
    adata : AnnotatedData object
        Data to process. .X layer will be used

    Returns
    -------
    adata : AnnotatedData object
        Data with constant variables removed.
    """
    _, variances = _get_mean_var(adata.X, axis=0)
    non_constant_vars = variances != 0
    adata = adata[:, non_constant_vars]

    return adata.copy()


def senkin_normalize(adata, batch_key: str | None = "day", dtype=np.float32):
    """
    Apply Senkin normalization approach to the data.

    This reproduces the "fine tuned process" of the original preprocessing notebook
    (https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/preprocess_cite.ipynb):

    1. Division of each row (cell) by its mean
    2. Square root of the data
    3. Z-score transformation per column (gene), using the population standard deviation
    4. Batch effect correction: subtraction of the per-batch median of the z-scored values for each gene

    Parameters
    ----------
    adata : AnnotatedData object
        Data to process. .X layer will be used, it must contain raw counts. Constant genes should be
        removed beforehand (see `remove_constant_vars`), otherwise their z-scores are set to 0.
    batch_key : str, optional
        The key in adata.obs to use for batch effect correction. In the original notebook, it was "day".
        Pass `None` to treat all cells as a single batch.
    dtype : numpy dtype = np.float32
        Data type of the returned dense matrix. The result is dense, so for large datasets
        float32 halves the memory footprint compared to float64.

    Returns
    -------
    normalized_data : np.ndarray
        A dense matrix of shape (n_samples, n_features) containing the normalized data.
    """
    normalized_data = to_dense(adata.X, dtype=dtype)
    if normalized_data is adata.X:  # to_dense may return the very same array, do not modify adata in place
        normalized_data = normalized_data.copy()

    # Normalize each row by division per mean
    row_means = normalized_data.mean(axis=1, dtype=np.float64).astype(dtype)
    normalized_data /= row_means[:, None]
    np.sqrt(normalized_data, out=normalized_data)

    # Convert to Z-scores per column. The original used np.std, i.e. the population standard deviation
    means, variances = _get_mean_var(normalized_data, axis=0, correction=0)
    stds = np.sqrt(variances)
    n_constant = int((stds == 0).sum())
    if n_constant:
        logger.warning(f"{n_constant} constant genes found, their z-scores are set to 0. Consider removing them first.")
        stds[stds == 0] = 1
    normalized_data -= means.astype(dtype)
    normalized_data /= stds.astype(dtype)

    # Subtract per batch median of the z-scored values for each gene
    if batch_key is None:
        batches = np.zeros(adata.n_obs, dtype=int)
    else:
        batches = adata.obs[batch_key].astype(str).values

    for batch in np.unique(batches):
        batch_mask = batches == batch
        normalized_data[batch_mask] -= np.median(normalized_data[batch_mask], axis=0).astype(dtype)

    return normalized_data


def get_top_correlated_features(
    adata_rna,
    adata_prot,
    group_key: str | None = "donor",
    quantile_threshold: float = 0.1,
    top_n: int = 10,
    rna_key: str | None = "X_log_normalized",
    prot_key: str | None = "dsb",
    chunk_size: int = 2000,
):
    """
    Get list of top correlated genes for target proteins

    For every group of cells (e.g. donor), Pearson correlation between each gene and each
    (row-scaled) protein is computed. Genes are then ranked per protein by the `quantile_threshold`
    quantile of the correlation across groups (i.e. genes that are robustly correlated in all
    groups are preferred), and the `top_n` genes for each protein are pooled.

    Parameters
    ----------
    adata_rna : AnnotatedData object
        RNA data to process.
    adata_prot : AnnotatedData object
        Protein data to process. Must have the same cells in the same order as `adata_rna`.
    group_key : str, optional
        The key in adata_rna.obs to use for grouping. In the original notebook, it was combination of donor and day.
        Pass `None` to use a single group.
    quantile_threshold : float = 0.1
        The quantile threshold for the top correlated genes. Quantiles are computed for each gene-protein pair
        per group and are used to rank genes. For example, `q=0.1` means that genes are ranked by the
        10th percentile of the correlation values among groups.
    top_n : int = 10
        The number of top correlated genes to return. Note that the resulted number will likely be less
        than number of proteins * `top_n` because some of correlated genes overlap between proteins.
    rna_key : str, optional
        Where to take the RNA expression from (see `get_matrix`): `None` for `.X`, otherwise a key of
        `.layers` or `.obsm`. The original solution computed correlations on log-normalized data
        (see `log_normalize`), which is stored in `.obsm["X_log_normalized"]` by `preprocess_data`.
    prot_key : str, optional
        Where to take the protein expression from (see `get_matrix`). The original solution used
        DSB-normalized proteins.
    chunk_size : int = 2000
        Number of genes processed at once. Only affects memory consumption.

    Returns
    -------
    top_corr_genes : list
        Sorted list of top correlated genes. Names are taken from `adata_rna.var_names`.
        Genes with an undefined correlation (constant within any group) are never selected.
    """
    assert adata_rna.n_obs == adata_prot.n_obs, "RNA and protein data must have the same cells"

    prot_row_scaled = to_dense(get_matrix(adata_prot, prot_key), dtype=np.float64)
    prot_row_std = prot_row_scaled.std(axis=1, keepdims=True)
    # A cell with a constant protein vector (e.g. no protein counts at all) has no correlation with anything and
    # would turn every correlation of its group into NaN. Such cells are ignored.
    informative_cells = prot_row_std[:, 0] > 0
    if not informative_cells.all():
        logger.warning(
            f"{int((~informative_cells).sum())} cells have a constant protein vector and are ignored for the correlations"
        )
        prot_row_std[~informative_cells] = 1
    prot_row_scaled = (prot_row_scaled - prot_row_scaled.mean(axis=1, keepdims=True)) / prot_row_std

    rna = get_matrix(adata_rna, rna_key)
    if sparse.issparse(rna):
        rna = sparse.csc_matrix(rna)  # For efficient column slicing

    if group_key is None:
        groups = np.zeros(adata_rna.n_obs, dtype=int)
    else:
        groups = adata_rna.obs[group_key].astype(str).values
    group_masks = [(groups == group) & informative_cells for group in np.unique(groups)]
    group_masks = [mask for mask in group_masks if mask.sum() > 1]

    n_genes, n_proteins = adata_rna.n_vars, adata_prot.n_vars
    per_group_corr_quantile = np.full((n_genes, n_proteins), np.nan)

    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        rna_chunk = to_dense(rna[:, start:stop], dtype=np.float64)

        corr_matrices = np.empty((len(group_masks), stop - start, n_proteins))
        for i, group_mask in enumerate(group_masks):
            corr_matrices[i] = pairwise_corr(rna_chunk[group_mask], prot_row_scaled[group_mask])

        # NaN (a gene constant within a group) propagates, so such genes are excluded from the ranking.
        # This matches the original notebook, which dropped genes with a missing correlation in any group.
        per_group_corr_quantile[start:stop] = np.quantile(corr_matrices, q=quantile_threshold, axis=0)

    top_corr_genes = set()

    for i in range(n_proteins):
        quantiles = per_group_corr_quantile[:, i]
        valid_genes = np.flatnonzero(~np.isnan(quantiles))
        top_genes = valid_genes[np.argsort(-quantiles[valid_genes], kind="stable")[:top_n]]
        top_corr_genes.update(adata_rna.var_names[top_genes])

    if not top_corr_genes:
        logger.warning("No correlated genes selected: every gene has an undefined correlation in some group")

    return sorted(top_corr_genes)


def preprocess_data(
    mdata,
    empty_counts_range: tuple[float, float] = (1.5, 2.8),
    batch_key: str = "day",
    group_key: str = "donor",
    known_features: list | None = None,
    store_sqrt_norm: bool = False,
):
    """
    Preprocess data using senkin13 approach for RNA and basic preprocessing for protein data.

    Original notebook: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/preprocess_cite.ipynb

    It includes:
    - DSB normalization of the protein data
    - log1p(CPM) normalization of the RNA data (how the competition inputs were normalized)
    - Removal of constant features
    - 200 components TSVD of CLR-transformed data
    - 100 components TSVD and 64 components PCA of customly normalized data (see documentation of `senkin_normalize` for details)
    - Selection of genes correlated with target proteins (see `get_top_correlated_features`)

    In this function, we use the same parameters as in the original approach. If you want more flexibility,
    you can use the preprocessing functions separately.

    Parameters
    ----------
    mdata : MuData
        Data to process. Must contain modalities "rna" and "prot" with raw counts in .X.
    empty_counts_range : tuple[float, float], optional
        Range of empty counts to use for DSB transformation. In the OpenProblems 2022 competition, it was (1.5, 2.8),
        but this is data-dependent, so make sure to double check what makes sense for your data!
    batch_key : str, optional
        Key to correct for batch effects in the custom normalization. In the original notebook, it was "day".
    group_key : str, optional
        Key for grouping observations. In the original notebook, it was the combination of donor and day.
    known_features: list, optional
        List of known features to include in the selection. For example, genes encoding target proteins.
    store_sqrt_norm : bool = False
        Whether to keep the dense customly normalized matrix in `.obsm["X_sqrt_norm"]`. It has the same
        size as the RNA count matrix, so it is not stored by default.

    Returns
    -------
    MuData
        Processed data. RNA modality has additional arrays in obsm:
        - X_log_normalized: log1p(CPM)-normalized data
        - X_clr_tsvd: TSVD of CLR-transformed data
        - X_sqrt_norm_tsvd: TSVD of customly normalized data with 100 components
        - X_sqrt_norm_pca: PCA of customly normalized data with 64 components
        - X_raw_selected: Raw data with correlated and selected features
        Protein modality has a "dsb" layer with DSB-normalized data.
    """
    logger.info("DSB-normalizing protein data. The number of cells will be reduced.")
    logger.debug(f"Number of cells before DSB: {mdata.shape[0]}")
    mdata = pt.pp.dsb(
        mdata, add_layer=True, empty_counts_range=empty_counts_range, cell_counts_range=(empty_counts_range[1], np.inf)
    )
    adata_rna = mdata.mod["rna"]
    adata_prot = mdata.mod["prot"]

    logger.debug(f"Number of cells after DSB: {mdata.shape[0]}")

    logger.info("Starting RNA preprocessing")

    # This is the transformation used in the competition.
    logger.info("Normalization and log1p-transformation")
    adata_rna.obsm["X_log_normalized"] = log_normalize(adata_rna, target_sum=1e6)

    logger.info("Removing constant variables")
    adata_rna = remove_constant_vars(adata_rna)

    logger.info("Computing CLR-TSVD transformation with 200 components")
    adata_rna.obsm["X_clr_tsvd"] = clr_tsvd(adata_rna, n_components=200)

    logger.info("Applying Senkin normalization")
    sqrt_norm = senkin_normalize(adata_rna, batch_key=batch_key)
    if store_sqrt_norm:
        adata_rna.obsm["X_sqrt_norm"] = sqrt_norm

    # In the original notebook, both 100 components TSVD and 64 components PCA of the normalized data were used
    logger.info("Computing TSVD with 100 components and PCA with 64 components of the normalized data")
    n_tsvd = min(100, min(sqrt_norm.shape) - 1)
    adata_rna.obsm["X_sqrt_norm_tsvd"] = TruncatedSVD(n_components=n_tsvd, algorithm="arpack").fit_transform(sqrt_norm)
    n_pca = min(64, min(sqrt_norm.shape) - 1)
    adata_rna.obsm["X_sqrt_norm_pca"] = PCA(n_components=n_pca).fit_transform(sqrt_norm)
    del sqrt_norm

    logger.info("Verifying observation names match between RNA and protein data")
    assert (adata_rna.obs_names == adata_prot.obs_names).all()

    logger.info("Selecting features based on known proteins and correlations")

    top_corr_genes = get_top_correlated_features(
        adata_rna,
        adata_prot,
        group_key=group_key,
        quantile_threshold=0.1,
        top_n=10,
        rna_key="X_log_normalized",
        prot_key="dsb",
    )
    logger.info(f"Found {len(top_corr_genes)} top correlated genes")

    known_features = [] if known_features is None else list(known_features)
    missing_known_features = set(known_features) - set(adata_rna.var_names)
    if missing_known_features:
        logger.warning(f"{len(missing_known_features)} known features are not in the RNA data and are skipped")
    selected_features = sorted((set(top_corr_genes) | set(known_features)) - missing_known_features)
    logger.info(f"Total {len(selected_features)} features selected")
    adata_rna.uns["selected_features"] = selected_features

    adata_rna.obsm["X_raw_selected"] = adata_rna[:, selected_features].X
    logger.info("RNA preprocessing completed")

    mdata.mod["rna"] = adata_rna
    mdata.mod["prot"] = adata_prot

    return mdata
