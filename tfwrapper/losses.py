import tensorflow as tf
import numpy as np

def dice_loss(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):

    fg_dice = foreground_dice(logits, labels, epsilon, from_label=from_label, to_label=to_label)
    return 1.0 - fg_dice

def foreground_dice(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    """
    Pseudo-dice calculated from all voxels (from all subjects) and all non-background labels
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: scalar Dice
    """

    struct_dice = per_structure_dice(logits, labels, epsilon)
    foreground_dice = tf.slice(struct_dice, (0, from_label),(-1, to_label))

    return tf.reduce_mean(foreground_dice)


def per_structure_dice(logits, labels, epsilon=1e-10):
    """
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1])
    """

    prediction = tf.nn.softmax(logits)
    hard_pred = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])

    intersection = tf.multiply(hard_pred, labels)
    intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

    l = tf.reduce_sum(hard_pred, axis=[1, 2])
    r = tf.reduce_sum(labels, axis=[1, 2])

    dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

    return dices_per_subj


def pixel_wise_cross_entropy_loss(logits, labels):

    # Operates on logits and takes labels in normal format (i.e. not one-hot)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):

    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss

# I forgot what this one was doing:
# def pixel_wise_cross_entropy_loss_sparse(logits, labels, class_weights=None):
#
#     # Operates on logits and takes labels in normal format (i.e. not one-hot)
#
#     labels_squeezed = tf.cast(tf.squeeze(labels), tf.int32)
#
#     if class_weights is not None:
#         class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
#
#         weight_map = tf.multiply(tf.cast(labels_squeezed, tf.float32), class_weights)
#         weight_map = tf.reduce_sum(weight_map, axis=1)
#
#         loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_squeezed)
#         weighted_loss = tf.multiply(loss_map, weight_map)
#
#         loss = tf.reduce_mean(weighted_loss)
#
#     else:
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_squeezed))
#     return loss