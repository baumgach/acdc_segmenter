

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfwrapper import layers, losses

def inference(images, inference_handle, training=True):

    # return dialated_convs_nopool(images)
    # return inference_stack_decos(images)
    # return lisa_net(images)
    # return model_zoo.lisa_net_deeper(images, training)
    # return lisa_net_deeper_bn(images, training)
    # return inference_test(images)
    # return lisa_net_one_more_pool(images)
    # return stack_all_convs(images)
    # return dilation_after_max_pool(images)

    return inference_handle(images, training)


def loss(logits, labels, weight_decay=0.00005, loss_type='weighted_crossentropy'):

    with tf.variable_scope('weights_norm') as scope:

        weights_norm = tf.reduce_sum(
            input_tensor = weight_decay*tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )


    #tf.add_to_collection('losses', weights_norm)

    if loss_type == 'weighted_crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, labels,
                                                                          class_weights=[0.1, 0.3, 0.3, 0.3])
    elif loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, labels,
                                                                          class_weights=[0.25, 0.25, 0.25, 0.25])
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    #tf.add_to_collection('losses', cross_entropy_loss)

    #total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    total_loss = tf.add(segmentation_loss, weights_norm)

    return total_loss, segmentation_loss, weights_norm

def predict(images, inference_handle):

    logits = inference_handle(images, training=tf.constant(False, dtype=tf.bool))
    softmax = tf.nn.softmax(logits)
    mask = tf.arg_max(logits, dimension=3)

    return mask, softmax


def training(loss, optimizer_handle, learning_rate, **kwargs):

    if 'momentum' in kwargs:
        momentum = kwargs.get('momentum', 0.9)
        optimizer = optimizer_handle(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = optimizer_handle(learning_rate=learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):

    mask = tf.arg_max(tf.nn.softmax(logits, dim=-1), dimension=3)
    mask_gt = tf.arg_max(labels, dimension=3)


    tf.summary.image('example_segm', get_segmentation_summary(mask))
    tf.summary.image('example_gt', get_segmentation_summary(mask_gt))

    total_loss, nowd_loss, weights_norm = loss(logits, labels)

    cdice_structures = losses.per_structure_dice(logits, labels)
    cdice_foreground = tf.slice(cdice_structures, (0,1), (-1,-1))

    cdice = tf.reduce_mean(cdice_foreground)

    return nowd_loss, cdice

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

def get_segmentation_summary(img, idx=0):
    """
    Make an image summary for a segmentation mask
    """

    V = tf.cast(tf.slice(img, (idx, 0, 0), (1, -1, -1)), tf.uint8)
    # V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    #
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    # V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    # V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
