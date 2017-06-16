import tensorflow as tf
import math

import numpy as np

from tfwrapper import utils

def no_activation(x):
    return x

def activation_layer(bottom, name, activation=tf.nn.relu):

    with tf.name_scope(name):

        op = activation(bottom)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op

def max_pool_layer(x, kernel_size=(2,2), strides=(2,2), padding="SAME"):

    # TODO: Could be removed as it simply shadows the tf function

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


### CONVOLUTIONAL LAYERS ##############################################################################33

def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 **kwargs):

    bottom_num_filters = bottom.get_shape().as_list()[3]
    # bottom_num_filters = tf.shape(bottom)[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            # N = tf.cast(_get_rhs_dim(bottom), tf.float32)
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(3,3),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal',
                   **kwargs):


    # bottom_shape = bottom.get_shape().as_list()[1:]
    # batch_size = tf.shape(bottom)[0]
    # bottom_shape = [batch_size, bottom_shape[0], bottom_shape[1], bottom_shape[2]]

    bottom_shape = bottom.get_shape().as_list()
    batch_size = tf.shape(bottom)[0]
    fm_x = tf.shape(bottom)[1]*strides[0]
    fm_y = tf.shape(bottom)[2]*strides[1]
    if output_shape is None:

        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])
        # output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])
        # output_shape = tf.stack([batch_size, fm_x, fm_y, num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            # N = tf.cast(num_filters*fm_x*fm_y, tf.float32)
            # stddev = tf.sqrt(2.0 / N)
            # initial = tf.truncated_normal(weight_shape, stddev=stddev)
            # weights = tf.get_variable(name, initializer=initial)

            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        elif weight_init == 'bilinear':
            weights = _weight_variable_bilinear(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op


def conv2D_dilated_layer(bottom,
                         name,
                         kernel_size=(3,3),
                         num_filters=32,
                         rate=1,
                         activation=tf.nn.relu,
                         padding="SAME",
                         weight_init='he_normal',
                         **kwargs):

    bottom_num_filters = bottom.get_shape().as_list()[3]
    # bottom_num_filters = tf.shape(bottom)[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        if weight_init == 'he_normal':
            # N = tf.cast(_get_rhs_dim(bottom), tf.float32)
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op

### BATCH_NORM SHORTCUTS #####################################################################################


def batch_normalisation_layer(bottom, name, training, **kwargs):
    """
    Batch normalization on feedforward maps. (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
    Args:
        bottom:      input from last layer
        name:        name of layer
        training:    boolean tf.Variable, true indicates training phase (Note: cannot be a simple Python bool)
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name):

        n_out = bottom.get_shape().as_list()[-1]
        tensor_dim = len(bottom.get_shape().as_list())

        is_conv_layer = True if tensor_dim == 4 else False

        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float32)
        beta = tf.get_variable(name=name+'_beta', dtype=tf.float32, initializer=init_beta, regularizer=None, trainable=True)
        gamma = tf.get_variable(name=name+'_gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None, trainable=True)

        moments_over_axes = [0,1,2] if is_conv_layer else [0]

        batch_mean, batch_var = tf.nn.moments(bottom, moments_over_axes, name=name+'_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, 1e-3)

    return normed


def conv2D_layer_bn(bottom,
                    name,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool),
                    **kwargs):

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=no_activation,
                        padding=padding,
                        weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    relu = activation(conv_bn)

    return relu

def deconv2D_layer_bn(bottom,
                      name,
                      kernel_size=(3,3),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal',
                      training=tf.constant(True, dtype=tf.bool),
                      **kwargs):

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=no_activation,
                          padding=padding,
                          weight_init=weight_init)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    relu = activation(deco_bn)

    return relu

def conv2D_dilated_layer_bn(bottom,
                           name,
                           kernel_size=(3,3),
                           num_filters=32,
                           rate=1,
                           activation=tf.nn.relu,
                           padding="SAME",
                           weight_init='he_normal',
                           training=tf.constant(True, dtype=tf.bool),
                            **kwargs):

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=no_activation,
                                padding=padding,
                                weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    relu = activation(conv_bn)

    return relu

### GOOD OLD DENSE LAYER #####################################################################################

def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=tf.nn.relu,
                weight_init='he_normal',
                **kwargs):

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = bottom_rhs_dim
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.matmul(bottom_flat, weights)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op

### VARIABLE INITIALISERS ####################################################################################

def _weight_variable_simple(shape, stddev=0.02, name=None):

    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        weight = tf.Variable(initial)
    else:
        weight = tf.get_variable(name, initializer=initial)

    tf.add_to_collection('weight_variables', weight)

    return weight

def _weight_variable_he_normal(shape, N, name=None):

    stddev = math.sqrt(2.0/float(N))

    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        weight = tf.Variable(initial)
    else:
        weight = tf.get_variable(name, initializer=initial)

    tf.add_to_collection('weight_variables', weight)

    return weight


def _bias_variable(shape, name=None, init_value=0.0):
    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def _weight_variable_bilinear(shape, name=None):

    weights = bilinear_upsample_weights(shape)
    initial = tf.constant(weights, shape=shape, dtype=tf.float32)

    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(shape):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights