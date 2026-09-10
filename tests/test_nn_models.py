import numpy as np
from scipy import sparse

from senkin_tmp_cite_pred.nn_models import prepare_nn_inputs, zscore


def test_zscore_rows(rng):
    x = rng.normal(size=(5, 10)) * 3 + 2
    z = zscore(x)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)


def test_zscore_constant_row_is_zero_not_nan():
    x = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    z = zscore(x)
    np.testing.assert_array_equal(z[0], 0)
    assert np.isfinite(z).all()


def test_prepare_nn_inputs_zscores_every_block_separately(rng):
    block_a = rng.normal(size=(7, 4)) * 100
    block_b = sparse.csr_matrix(rng.poisson(2, size=(7, 6)).astype(float))
    result = prepare_nn_inputs(block_a, block_b)
    assert result.shape == (7, 10)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, :4], zscore(block_a), atol=1e-5)
    np.testing.assert_allclose(result[:, 4:], zscore(block_b.toarray()), atol=1e-5)
