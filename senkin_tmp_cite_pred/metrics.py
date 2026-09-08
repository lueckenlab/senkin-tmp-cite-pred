import numpy as np
import pandas as pd
import tensorflow as tf


def cosine_similarity_loss(y_true, y_pred):
    """Negative cosine similarity between the per-cell centered true and predicted protein vectors.

    This equals the original `CosineSimilarity(axis=1)(l2_normalize(xm), l2_normalize(ym))` of senkin13 for every
    non-degenerate prediction. The original normalized twice (once explicitly, once inside Keras' loss); with
    Keras 3 the second normalization has no epsilon guard, so an all-constant prediction (which is the exact
    starting point of `cite_cos_sim_model` because of its identity initialization) yields NaN/overflowing
    gradients and the training diverges. Normalizing once with a small epsilon avoids that.
    """
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.math.l2_normalize(xm, axis=1, epsilon=1e-6)
    t2_norm = tf.math.l2_normalize(ym, axis=1, epsilon=1e-6)
    return -tf.reduce_sum(t1_norm * t2_norm, axis=1)


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the OpenProblems2022 competition rules.

    Cells whose true or predicted vector is constant (e.g. cells without any protein counts) have an undefined
    correlation and are left out of the average instead of turning it into NaN.

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
    y_true = np.asarray(y_true.values if isinstance(y_true, pd.DataFrame) else y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred.values if isinstance(y_pred, pd.DataFrame) else y_pred, dtype=np.float64)

    true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = (true_centered * pred_centered).sum(axis=1) / np.sqrt(
            (true_centered**2).sum(axis=1) * (pred_centered**2).sum(axis=1)
        )

    return float(np.nanmean(correlations))
