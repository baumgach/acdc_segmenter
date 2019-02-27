# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np

def per_structure_dice(logits, labels, epsilon=1e-10, sum_over_batches=False, use_hard_pred=True):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :param sum_over_batches: Calculate intersection and union over whole batch rather than single images
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1]) (except when sum_over_batches is on)
    '''

    ndims = logits.get_shape().ndims

    prediction = tf.nn.softmax(logits)
    if use_hard_pred:
        # This casts the predictions to binary 0 or 1
        prediction = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])

    intersection = tf.multiply(prediction, labels)

    if ndims == 5:
        reduction_axes = [1,2,3]
    else:
        reduction_axes = [1,2]

    if sum_over_batches:
        reduction_axes = [0] + reduction_axes

    # Reduce the maps over all dimensions except the batch and the label index
    i = tf.reduce_sum(intersection, axis=reduction_axes)
    l = tf.reduce_sum(prediction, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dice_per_img_per_lab = 2 * i / (l + r + epsilon)

    return dice_per_img_per_lab


def dice_loss(logits, labels, epsilon=1e-10, only_foreground=False, sum_over_batches=False):
    '''
    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
     denotes background and the remaining labels are foreground. Note that the dice loss is computed
     on the softmax output directly (i.e. (0,1)) rather than the hard labels (i.e. {0,1}). This provides
     better gradients and facilitates training. 
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :param only_foreground: Exclude label 0 from evaluation
    :param sum_over_batches: calculate the intersection and union of the whole batch instead of individual images
    :return: Dice loss
    '''

    dice_per_img_per_lab = per_structure_dice(logits=logits,
                                              labels=labels,
                                              epsilon=epsilon,
                                              sum_over_batches=sum_over_batches,
                                              use_hard_pred=False)

    if only_foreground:
        if sum_over_batches:
            loss = 1 - tf.reduce_mean(dice_per_img_per_lab[1:])
        else:
            loss = 1 - tf.reduce_mean(dice_per_img_per_lab[:, 1:])
    else:
        loss = 1 - tf.reduce_mean(dice_per_img_per_lab)

    return loss

def pixel_wise_cross_entropy_loss(logits, labels):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):
    '''
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    '''

    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    #loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss

