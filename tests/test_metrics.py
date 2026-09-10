import numpy as np
import tensorflow as tf

from senkin_tmp_cite_pred.metrics import correlation_score, cosine_similarity_loss


def _original_cosine_similarity_loss(y_true, y_pred):
    """Loss exactly as written in the original notebook (double normalization through Keras' CosineSimilarity)"""
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    t1_norm = tf.math.l2_normalize(y_true - mx, axis=1)
    t2_norm = tf.math.l2_normalize(y_pred - my, axis=1)
    return tf.keras.losses.CosineSimilarity(axis=1)(t1_norm, t2_norm)


def test_cosine_similarity_loss_matches_original_formula(rng):
    y_true = rng.normal(size=(64, 140)).astype(np.float32)
    y_pred = rng.normal(size=(64, 140)).astype(np.float32) * 3 + 1
    expected = float(_original_cosine_similarity_loss(y_true, y_pred))
    result = float(tf.reduce_mean(cosine_similarity_loss(y_true, y_pred)))
    assert abs(result - expected) < 1e-5
    # per-cell values are minus the Pearson correlation of every cell
    per_cell = cosine_similarity_loss(y_true, y_pred).numpy()
    assert abs(-per_cell.mean() - correlation_score(y_true, y_pred)) < 1e-5


def test_cosine_similarity_loss_finite_gradient_for_constant_prediction(rng):
    y_true = tf.constant(rng.normal(size=(8, 140)).astype(np.float32))
    y_pred = tf.Variable(np.zeros((8, 140), dtype=np.float32))
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(cosine_similarity_loss(y_true, y_pred))
    grad = tape.gradient(loss, y_pred)
    assert np.isfinite(float(loss))
    assert np.isfinite(grad.numpy()).all()
    assert np.abs(grad.numpy()).max() < 1e4


def test_correlation_score_matches_corrcoef_and_skips_constant_cells(rng):
    y_true = rng.normal(size=(20, 30))
    y_pred = y_true + rng.normal(scale=0.5, size=(20, 30))
    expected = np.mean([np.corrcoef(y_true[i], y_pred[i])[0, 1] for i in range(20)])
    assert abs(correlation_score(y_true, y_pred) - expected) < 1e-12

    y_true_with_empty_cell = y_true.copy()
    y_true_with_empty_cell[0] = 0  # a cell without any protein signal
    expected_without = np.mean([np.corrcoef(y_true[i], y_pred[i])[0, 1] for i in range(1, 20)])
    result = correlation_score(y_true_with_empty_cell, y_pred)
    assert np.isfinite(result)
    assert abs(result - expected_without) < 1e-12
