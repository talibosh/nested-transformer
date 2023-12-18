import tensorflow as tf
import numpy as np

allowed_labels_long = tf.constant([9, 18, 20, 25, 26, 61, 62, 63, 80, 81])
allowed_labels_flat = tf.constant([3, 4, 91, 92, 94, 100, 102, 108])
allowed_labels = tf.constant([3, 4, 91, 92, 94, 100, 102, 108,9, 18, 20, 25, 26, 61, 62, 63, 80, 81])


def predicate(x, allowed_labels = allowed_labels):
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))


def update_label(label, allowed_long=allowed_labels_long, allowed_flat=allowed_labels_flat):
    # label = x['label']
    is_long = tf.equal(allowed_long, tf.cast(label, allowed_long.dtype))
    is_flat = tf.equal(allowed_flat, tf.cast(label, allowed_flat.dtype))
    reduced_long = tf.reduce_sum(tf.cast(is_long, tf.float32))
    reduced_flat = tf.reduce_sum(tf.cast(is_flat, tf.float32))
    if tf.greater(reduced_long, tf.constant(0.)):
        return np.full((1, 1), 1, dtype=int)
    else:
        return np.full((1, 1), 2, dtype=int)