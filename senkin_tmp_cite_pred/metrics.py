import numpy as np
import pandas as pd
import tensorflow as tf


def cosine_similarity_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.math.l2_normalize(xm, axis = 1)
    t2_norm = tf.math.l2_normalize(ym, axis = 1)
    cosine = tf.keras.losses.CosineSimilarity(axis = 1)(t1_norm, t2_norm)
    return cosine


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the OpenProblems2022 competition rules. 
    
    It is assumed that the predictions are not constant.

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
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    
    corrsum = 0

    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]

    return corrsum / len(y_true)
