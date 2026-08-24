"""Tests for get_top_correlated_features.

The correlations are computed in gene chunks to bound memory. gene_chunk_size is a
pure memory/speed knob: the result must be identical for any chunk size, including a
single chunk that covers all genes (the original whole-array behaviour).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from senkin_tmp_cite_pred.preprocess import get_top_correlated_features, pairwise_corr

N_GENES = 57


def _make_data(n_cells=240, n_genes=N_GENES, n_proteins=9, n_groups=4, seed=0):
    """Small CITE-like pair. Continuous RNA avoids zero-variance gene columns within a
    group, so pairwise_corr's per-column standardization is well defined."""
    rng = np.random.RandomState(seed)
    X = rng.normal(loc=3.0, scale=1.0, size=(n_cells, n_genes))
    adata_rna = ad.AnnData(sp.csr_matrix(X))
    adata_rna.var_names = [f"gene{i}" for i in range(n_genes)]
    groups = rng.randint(0, n_groups, size=n_cells)
    adata_rna.obs["donor"] = pd.Categorical(groups)

    prot = rng.normal(size=(n_cells, n_proteins))
    adata_prot = ad.AnnData(np.zeros((n_cells, n_proteins), dtype=float))
    adata_prot.layers["dsb"] = prot
    return adata_rna, adata_prot


def _reference(adata_rna, adata_prot, group_key, quantile_threshold, top_n):
    """Independent whole-array implementation (the pre-chunking behaviour)."""
    prot = adata_prot.layers["dsb"]
    prot = prot if isinstance(prot, np.ndarray) else prot.toarray()
    prot = (prot - prot.mean(axis=1).reshape(-1, 1)) / prot.std(axis=1).reshape(-1, 1)

    groups = adata_rna.obs[group_key].unique()
    corr = np.zeros((len(groups), adata_rna.shape[1], adata_prot.shape[1]))
    for i, group in enumerate(groups):
        mask = (adata_rna.obs[group_key] == group).values
        corr[i] = pairwise_corr(adata_rna.X[mask].toarray(), prot[mask])
    per_group_corr_quantile = np.nanquantile(corr, q=quantile_threshold, axis=0)

    top = set()
    for i in range(adata_prot.shape[1]):
        top.update(adata_rna.var_names[per_group_corr_quantile[:, i].argsort()[-top_n:]])
    return set(top)


@pytest.mark.parametrize("gene_chunk_size", [1, 2, 7, 10, N_GENES, N_GENES + 5, 10_000])
def test_result_is_chunk_size_invariant(gene_chunk_size):
    """Any chunk size (down to 1 gene, up to a single chunk over all genes) yields the
    same selected genes as the whole-array reference."""
    adata_rna, adata_prot = _make_data()
    expected = _reference(adata_rna, adata_prot, "donor", 0.1, 10)

    result = set(
        get_top_correlated_features(
            adata_rna,
            adata_prot,
            group_key="donor",
            quantile_threshold=0.1,
            top_n=10,
            gene_chunk_size=gene_chunk_size,
        )
    )
    assert result == expected


def test_single_chunk_matches_many_chunks():
    """One chunk covering all genes and one-gene-per-chunk agree exactly."""
    adata_rna, adata_prot = _make_data(seed=1)
    kwargs = dict(group_key="donor", quantile_threshold=0.25, top_n=5)
    full = set(get_top_correlated_features(adata_rna, adata_prot, gene_chunk_size=10_000, **kwargs))
    per_gene = set(get_top_correlated_features(adata_rna, adata_prot, gene_chunk_size=1, **kwargs))
    assert full == per_gene


def test_default_matches_reference():
    """The default chunk size reproduces the whole-array reference."""
    adata_rna, adata_prot = _make_data(seed=2)
    expected = _reference(adata_rna, adata_prot, "donor", 0.1, 10)
    result = set(get_top_correlated_features(adata_rna, adata_prot, group_key="donor"))
    assert result == expected


def test_accepts_dense_rna():
    """A dense adata_rna.X gives the same result as the sparse case."""
    adata_rna, adata_prot = _make_data(seed=3)
    sparse_res = set(get_top_correlated_features(adata_rna, adata_prot, group_key="donor", gene_chunk_size=8))
    adata_dense = adata_rna.copy()
    adata_dense.X = adata_rna.X.toarray()
    dense_res = set(get_top_correlated_features(adata_dense, adata_prot, group_key="donor", gene_chunk_size=8))
    assert sparse_res == dense_res
