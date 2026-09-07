import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from senkin_tmp_cite_pred.preprocess import (
    get_matrix,
    get_top_correlated_features,
    log_normalize,
    pairwise_corr,
    remove_constant_vars,
    senkin_normalize,
)


def _counts_adata(rng, n_cells=300, n_genes=60, n_batches=3):
    counts = rng.poisson(lam=rng.gamma(0.5, 2.0, size=n_genes), size=(n_cells, n_genes)).astype(np.float32)
    counts[:, 0] = 1  # constant gene
    obs = pd.DataFrame(
        {
            "day": rng.choice([f"d{i}" for i in range(n_batches)], size=n_cells),
            "donor": rng.choice(["a", "b"], size=n_cells),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)


def _reference_senkin_normalize(counts, days):
    """Literal re-implementation of the 'fine tuned process' of the original preprocess_cite.ipynb"""
    normalized = counts / counts.mean(axis=1).reshape(-1, 1)
    normalized = np.sqrt(normalized)
    zscored = np.zeros_like(normalized)
    for j in range(normalized.shape[1]):
        column = normalized[:, j]
        zscored[:, j] = (column - np.mean(column)) / np.std(column)
    frame = pd.concat([pd.DataFrame({"day": days}), pd.DataFrame(zscored)], axis=1)
    day_median = frame.groupby(["day"]).transform("median").values
    return zscored - day_median


def test_senkin_normalize_matches_original_notebook(rng):
    adata = _counts_adata(rng)
    adata = remove_constant_vars(adata)
    counts = adata.X.toarray().astype(np.float64)

    expected = _reference_senkin_normalize(counts, adata.obs["day"].values)
    result = senkin_normalize(adata, batch_key="day", dtype=np.float64)

    assert isinstance(result, np.ndarray)
    assert result.shape == counts.shape
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_senkin_normalize_dense_and_sparse_agree_and_do_not_modify_input(rng):
    adata_sparse = remove_constant_vars(_counts_adata(rng))
    adata_dense = adata_sparse.copy()
    adata_dense.X = adata_dense.X.toarray()
    dense_before = adata_dense.X.copy()

    result_sparse = senkin_normalize(adata_sparse, batch_key="day", dtype=np.float32)
    result_dense = senkin_normalize(adata_dense, batch_key="day", dtype=np.float32)

    np.testing.assert_allclose(result_sparse, result_dense, atol=1e-5)
    np.testing.assert_array_equal(adata_dense.X, dense_before)
    assert result_sparse.dtype == np.float32


def test_senkin_normalize_zero_gene_gives_zero_and_single_batch(rng):
    adata = _counts_adata(rng)
    counts = adata.X.toarray()
    counts[:, 1] = 0  # an all-zero gene stays constant after the row normalization
    adata.X = sparse.csr_matrix(counts)
    result = senkin_normalize(adata, batch_key=None)
    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result[:, 1], 0)
    # with a single batch, the global median of every gene is subtracted
    np.testing.assert_allclose(np.median(result, axis=0), 0, atol=1e-6)


def test_remove_constant_vars(rng):
    adata = _counts_adata(rng)
    filtered = remove_constant_vars(adata)
    assert "gene0" not in filtered.var_names
    variances = adata.X.toarray().var(axis=0)
    assert filtered.n_vars == int((variances != 0).sum())
    assert (filtered.var_names == adata.var_names[variances != 0]).all()


def test_log_normalize_is_log1p_cpm_by_default(rng):
    adata = _counts_adata(rng)
    normalized = log_normalize(adata)
    assert sparse.issparse(normalized)
    np.testing.assert_allclose(np.expm1(normalized.toarray()).sum(axis=1), 1e6, rtol=1e-5)


def test_pairwise_corr_matches_corrcoef(rng):
    X = rng.normal(size=(50, 4))
    Y = rng.normal(size=(50, 3))
    expected = np.corrcoef(X.T, Y.T)[:4, 4:]
    np.testing.assert_allclose(pairwise_corr(X, Y), expected, atol=1e-12)
    X[:, 1] = 3  # constant column -> NaN like numpy
    assert np.isnan(pairwise_corr(X, Y)[1]).all()


def test_get_matrix(rng):
    adata = _counts_adata(rng)
    adata.layers["copy"] = adata.X.copy()
    adata.obsm["emb"] = rng.normal(size=(adata.n_obs, 2))
    assert get_matrix(adata) is adata.X
    assert get_matrix(adata, "X") is adata.X
    assert get_matrix(adata, "copy") is adata.layers["copy"]
    assert get_matrix(adata, "emb") is adata.obsm["emb"]
    with pytest.raises(KeyError):
        get_matrix(adata, "missing")


def _reference_top_correlated_features(rna, targets, groups, gene_names, protein_names, q=0.1, top_n=10):
    """Literal re-implementation of the correlated feature selection of the original preprocess_cite.ipynb"""
    targets = targets - targets.mean(axis=1).reshape(-1, 1)
    targets = targets / targets.std(axis=1).reshape(-1, 1)
    n_proteins = targets.shape[1]
    cor_list = {}
    for group in sorted(set(groups)):
        mask = groups == group
        corr = np.corrcoef(rna[mask].T, targets[mask].T)
        cor_list[group] = corr[:-n_proteins, -n_proteins:]
    cor_min = {col: [] for col in protein_names}
    for group in cor_list:
        frame = pd.DataFrame(cor_list[group], index=gene_names, columns=protein_names)
        for col in frame.columns:
            cor_min[col].append(frame[[col]])
    for col in protein_names:
        cor_min[col] = pd.concat(cor_min[col], axis=1).dropna()
        cor_min[col]["min"] = cor_min[col].quantile(axis=1, q=q)
    features = []
    for col in cor_min:
        features.extend(cor_min[col].sort_values("min", ascending=False).index.tolist()[:top_n])
    return sorted(set(features))


def test_get_top_correlated_features_matches_original_notebook(rng):
    n_cells, n_genes, n_proteins = 400, 70, 6
    protein = rng.normal(size=(n_cells, n_proteins))
    rna = rng.normal(size=(n_cells, n_genes))
    rna[:, :n_proteins] += 2 * protein  # correlated genes
    groups = rng.choice(["g1", "g2", "g3"], size=n_cells)
    rna[groups == "g2", 10] = 0  # gene constant within one group must never be selected
    gene_names = [f"gene{i}" for i in range(n_genes)]
    protein_names = [f"prot{i}" for i in range(n_proteins)]

    adata_rna = ad.AnnData(
        X=sparse.csr_matrix(np.abs(rna)),
        obs=pd.DataFrame({"group": groups}, index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )
    adata_rna.obsm["X_log_normalized"] = sparse.csr_matrix(rna)
    adata_prot = ad.AnnData(
        X=np.abs(protein),
        obs=adata_rna.obs.copy(),
        var=pd.DataFrame(index=protein_names),
        layers={"dsb": protein},
    )

    expected = _reference_top_correlated_features(rna, protein, groups, gene_names, protein_names, top_n=5)
    result = get_top_correlated_features(
        adata_rna, adata_prot, group_key="group", top_n=5, rna_key="X_log_normalized", prot_key="dsb", chunk_size=16
    )

    assert result == expected
    assert "gene10" not in result
    assert all(f"gene{i}" in result for i in range(n_proteins))


def test_get_top_correlated_features_single_group_and_x(rng):
    n_cells, n_genes, n_proteins = 200, 30, 3
    protein = rng.normal(size=(n_cells, n_proteins))
    rna = rng.normal(size=(n_cells, n_genes))
    rna[:, 0] += 5 * protein[:, 0]
    adata_rna = ad.AnnData(X=rna, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]))
    adata_prot = ad.AnnData(X=protein, obs=adata_rna.obs.copy())

    result = get_top_correlated_features(adata_rna, adata_prot, group_key=None, top_n=1, rna_key=None, prot_key=None)
    assert "0" in result
    assert len(result) <= n_proteins


def test_get_top_correlated_features_ignores_cells_with_constant_proteins(rng):
    n_cells, n_genes, n_proteins = 300, 40, 4
    protein = rng.normal(size=(n_cells, n_proteins))
    rna = rng.normal(size=(n_cells, n_genes))
    rna[:, :n_proteins] += 2 * protein
    groups = rng.choice(["g1", "g2"], size=n_cells)
    obs = pd.DataFrame({"group": groups}, index=[f"c{i}" for i in range(n_cells)])
    adata_rna = ad.AnnData(X=rna, obs=obs)
    adata_prot = ad.AnnData(X=protein.copy(), obs=obs.copy())
    expected = get_top_correlated_features(adata_rna, adata_prot, group_key="group", top_n=2, rna_key=None, prot_key=None)

    # cells without any protein signal (e.g. zero counts) must not turn every correlation into NaN
    protein_with_empty_cells = protein.copy()
    protein_with_empty_cells[:5] = 0
    adata_prot_empty = ad.AnnData(X=protein_with_empty_cells, obs=obs.copy())
    result = get_top_correlated_features(adata_rna, adata_prot_empty, group_key="group", top_n=2, rna_key=None, prot_key=None)

    assert len(result) > 0
    assert set(result) & set(expected)
