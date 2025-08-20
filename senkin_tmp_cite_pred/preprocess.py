import logging

import numpy as np
import scanpy as sc
from sklearn.decomposition import TruncatedSVD, PCA
from muon import prot as pt
from scanpy.pp._utils import _get_mean_var


def pairwise_corr(X, Y):
    """Compute pairwise correlation between X and Y
    
    Parameters
    ----------
    X : array-like
        A matrix of shape (n_samples, n_features)
    Y : array-like
        A matrix of shape (n_samples, n_targets)
        
    Returns
    -------
    corr_matrix : array-like
        A matrix of shape (n_features, n_targets)
        The correlation between each feature and each target protein
    """
    # Scale X and Y so that the correlation is just the dot product
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    return X.T @ Y / X.shape[0]


def clr_tsvd(adata, n_components=200):
    """
    Compute TSVD-transform of Centered-log-ratio (CLR)-normalized data.
    
    Parameters
    ----------
    adata : AnnotatedData object
        Data to transform. .X layer will be used, it must contain raw counts.
    n_components : int = 200
        The number of components to keep. Default is 200.
        
    Returns
    -------
    X_clr_tsvd : array-like
        A matrix of shape (n_samples, n_components) containing the TSVD-transformed CLR-normalized data.
    """
    tsvd = TruncatedSVD(n_components=n_components, algorithm="arpack")    
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
    _, vars = _get_mean_var(adata.X)
    non_constant_vars = (vars != 0)
    adata = adata[:, non_constant_vars]
    
    return adata.copy()


def senkin_normalize(adata, batch_key: str = "day"):
    """
    Apply Senkin normalization approach to the data.

    This is a more efficient version of the original Senkin normalization approach. It includes:
    1. Division of each row by its mean
    2. Taking square root of the data
    3. Z-score transformation per column
    4. Subtraction of per batch raw count median for each gene
    
    Parameters
    ----------
    adata : AnnotatedData object
        Data to process. .X layer will be used, it must contain raw counts.
    batch_key : str = "day"
        The key in adata.obs to use for batch effect correction.
        
    Returns
    -------
    normalized_data : array-like
        A matrix of shape (n_samples, n_features) containing the normalized data.
    """
    # Normalize each row by division per mean
    normalized_data = (adata.X / adata.X.mean(axis=1).reshape(-1, 1)).tocsr()
    normalized_data = normalized_data.sqrt()

    means, vars = _get_mean_var(normalized_data)  # Efficient for different types
    # Convert to Z-scores per column
    normalized_data = (normalized_data - means) / vars

    # Subtract per batch raw count median for each gene. Note that it will leave >90% of the data unchanged because most of medians are 0.
    medians = sc.get.aggregate(adata, by=batch_key, func="median")

    for i, batch in enumerate(adata.obs[batch_key].unique()):
        batch_mask = (adata.obs[batch_key] == batch).values
        normalized_data[batch_mask, :] = normalized_data[batch_mask, :] - medians.layers["median"][i, :]

    return normalized_data


def get_top_correlated_features(adata_rna, adata_prot, group_key: str = "donor", quantile_threshold: float = 0.1, top_n: int = 10):
    """
    Get list of top correlated genes for target proteins

    Parameters
    ----------
    adata_rna : AnnotatedData object
        RNA data to process. .X layer will be used, it must contain raw counts.
    adata_prot : AnnotatedData object
        Protein data to process. .X layer will be used.
    group_key : str = "donor"
        The key in adata_rna.obs to use for grouping. In the original notebook, it was combination of donor and day.
    quantile_threshold : float = 0.1
        The quantile threshold for the top correlated genes. Quantiles are computed for each gene-protein pair
        per group and are used to rank genes. For example, `q=0.1` means that genes are ranked by the 
        10th percentile of the correlation values among groups.
    top_n : int = 10
        The number of top correlated genes to return. Note that the resulted number will likely be less
        than number of proteins * `top_n` because some of correlated genes overlap between proteins.

    Returns
    -------
    top_corr_genes : list
        List of top correlated genes. Names are taken from `adata_rna.var_names`.
    """

    prot_row_scaled = adata_prot.layers["dsb"]
    if not isinstance(prot_row_scaled, np.ndarray):
        prot_row_scaled = prot_row_scaled.toarray()
    prot_row_scaled = (prot_row_scaled - prot_row_scaled.mean(axis=1).reshape(-1, 1)) / prot_row_scaled.std(axis=1).reshape(-1, 1)

    n_groups, n_genes, n_proteins = adata_rna.obs[group_key].nunique(), adata_rna.shape[1], adata_prot.shape[1]

    corr_matrices = np.zeros((n_groups, n_genes, n_proteins))

    for i, group in enumerate(adata_rna.obs[group_key].unique()):
        group_mask = (adata_rna.obs[group_key] == group).values
        X = adata_rna.X[group_mask].toarray()
        Y = prot_row_scaled[group_mask]
        corr_matrices[i] = pairwise_corr(X, Y)

    per_group_corr_quantile = np.nanquantile(corr_matrices, q=quantile_threshold, axis=0)

    top_corr_genes = set()

    for i in range(n_proteins):
        top_genes = per_group_corr_quantile[:, i].argsort()[-top_n:]
        top_corr_genes.update(adata_rna.var_names[top_genes])

    return list(top_corr_genes)

def preprocess_data(mdata, empty_counts_range: tuple[float, float] = (1.5, 2.8), batch_key: str = "day", group_key: str = "donor", known_features: list = None):
    """
    Preprocess data using senkin13 approach for RNA and basic preprocessing for protein data.

    Original notebook: https://github.com/senkin13/kaggle/blob/master/Open-Problems-Multimodal-Single-Cell-Integration-2nd-Place-Solution/senkin13/preprocess_cite.ipynb

    It includes:
    - Removal of constant features
    - 200 components TSVD of CLR-transformed data
    - 100 components PCA of customly normalized data (see documentation of `senkin_normalize` for details)
    - Selection of genes correlated with target proteins

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

    Returns
    -------
    AnnData
        Processed RNA data with additional layers added to obsm:
        - X_clr_tsvd: TSVD of CLR-transformed data
        - X_sqrt_norm: Customly normalized data
        - X_pca_sqrt_norm: PCA of customly normalized data with 100 components
        - X_raw_selected: Raw data with correlated and selected features
    """
    logger = logging.getLogger(__name__)

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
    adata_rna.obsm["X_log_normalized"] = sc.pp.normalize_total(adata_rna, target_sum=None, inplace=False)["X"]
    sc.pp.log1p(adata_rna, obsm="X_log_normalized")

    logger.info("Removing constant variables")
    adata_rna = remove_constant_vars(adata_rna)

    logger.info("Computing CLR-TSVD transformation with 200 components")
    adata_rna.obsm["X_clr_tsvd"] = clr_tsvd(adata_rna, n_components=200)

    logger.info("Applying Senkin normalization")
    adata_rna.obsm["X_sqrt_norm"] = senkin_normalize(adata_rna, batch_key=batch_key)

    # In the original notebook, 100 components TSVD and 64 of PCA were used.
    # Up to centering and number of components, the results are the same.
    # We validated that they correlate perfectly or very well indeed.
    # Thus, here, we only use PCA with 100 components due to the fact that it is faster.
    logger.info("Computing PCA with 100 components")
    pca = PCA(n_components = 100, copy = False)
    adata_rna.obsm["X_pca_sqrt_norm"] = pca.fit_transform(adata_rna.obsm["X_sqrt_norm"])

    logger.info("Verifying observation names match between RNA and protein data")
    assert (adata_rna.obs_names == adata_prot.obs_names).all()

    logger.info("Selecting features based on known proteins and correlations")
    
    top_corr_genes = get_top_correlated_features(adata_rna, adata_prot, group_key=group_key, quantile_threshold=0.1, top_n=10)
    logger.info(f"Found {len(top_corr_genes)} top correlated genes")
    
    selected_features = list(set(top_corr_genes) | set(known_features))
    logger.info(f"Total {len(selected_features)} features selected")
    adata_rna.uns["selected_features"] = selected_features
    
    adata_rna.obsm["X_raw_selected"] = adata_rna[:, selected_features].X
    logger.info("RNA preprocessing completed")

    mdata.mod["rna"] = adata_rna
    mdata.mod["prot"] = adata_prot

    return mdata
