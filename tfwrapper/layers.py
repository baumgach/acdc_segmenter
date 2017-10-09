# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import math
import numpy as np

from tfwrapper import utils

from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer


def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def max_pool_layer3d(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], start_crop[3], 0),
                                     (-1, output_size[1], output_size[2], output_size[3], -1))
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], 0),
                                     (-1, output_size[1], output_size[2], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)


def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)


def dropout_layer(bottom, name, training, keep_prob=0.5):
    '''
    Performs dropout on the activations of an input
    '''

    keep_prob_pl = tf.cond(training,
                           lambda: tf.constant(keep_prob, dtype=bottom.dtype),
                           lambda: tf.constant(1.0, dtype=bottom.dtype))

    # The tf.nn.dropout function takes care of all the scaling
    # (https://www.tensorflow.org/get_started/mnist/pros)
    return tf.nn.dropout(bottom, keep_prob=keep_prob_pl, name=name)


# def batch_normalisation_layer(bottom, name, training):
#     '''
#     Alternative wrapper for tensorflows own batch normalisation function. I had some issues with 3D convolutions
#     when using this. The version below works with 3D convs as well.
#     :param bottom: Input layer (should be before activation)
#     :param name: A name for the computational graph
#     :param training: A tf.bool specifying if the layer is executed at training or testing time
#     :return: Batch normalised activation
#     '''
#
#     h_bn = tf.contrib.layers.batch_norm(inputs=bottom, decay=0.99, epsilon=1e-3, is_training=training,
#                                         scope=name, center=True, scale=True)
#
#     return h_bn

#
def batch_normalisation_layer(bottom, name, training, moving_average_decay=0.99, epsilon=1e-3):
    '''
    Batch normalisation layer (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    with tf.variable_scope(name):

        n_out = bottom.get_shape().as_list()[-1]
        tensor_dim = len(bottom.get_shape().as_list())

        if tensor_dim == 2:
            # must be a dense layer
            moments_over_axes = [0]
        elif tensor_dim == 4:
            # must be a 2D conv layer
            moments_over_axes = [0, 1, 2]
        elif tensor_dim == 5:
            # must be a 3D conv layer
            moments_over_axes = [0, 1, 2, 3]
        else:
            # is not likely to be something reasonable
            raise ValueError('Tensor dim %d is not supported by this batch_norm layer' % tensor_dim)

        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        init_gamma = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
        beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
                               trainable=True)
        gamma = tf.get_variable(name='gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
                                trainable=True)

        batch_mean, batch_var = tf.nn.moments(bottom, moments_over_axes, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normalised = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, epsilon)

        return normalised

### FEED_FORWARD LAYERS ##############################################################################33

def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 2D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(4,4),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()
    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]
    bias_shape = [num_filters]
    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv2D_dilated_layer(bottom,
                         name,
                         kernel_size=(3,3),
                         num_filters=32,
                         rate=1,
                         activation=tf.nn.relu,
                         padding="SAME",
                         weight_init='he_normal',
                         add_bias=True):

    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=tf.nn.relu,
                weight_init='he_normal',
                add_bias=True):

    '''
    Dense a.k.a. fully connected layer
    '''

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        biases = get_bias_variable(bias_shape, name='b')

        op = tf.matmul(bottom_flat, weights)
        if add_bias:
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    act = activation(conv_bn)

    return act


def conv3D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    act = activation(conv_bn)

    return act


def deconv2D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def deconv3D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):

    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def conv2D_dilated_layer_bn(bottom,
                           name,
                           training,
                           kernel_size=(3,3),
                           num_filters=32,
                           rate=1,
                           activation=tf.nn.relu,
                           padding="SAME",
                           weight_init='he_normal'):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=tf.identity,
                                padding=padding,
                                weight_init=weight_init,
                                add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    act = activation(conv_bn)

    return act



def dense_layer_bn(bottom,
                   name,
                   training,
                   hidden_units=512,
                   activation=tf.nn.relu,
                   weight_init='he_normal'):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=tf.identity,
                         weight_init=weight_init,
                         add_bias=False)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training)
    act = activation(batchnorm)

    return act

### VARIABLE INITIALISERS ####################################################################################

def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):

    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple_normal':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        weight = tf.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.add_to_collection('weight_variables', weight)

    return weight



def get_bias_variable(shape, name=None, init_value=0.0):

    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)



def _upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights

def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name, weights)
    tf.summary.histogram(biases.name, biases)
    tf.summary.histogram(op.op.name + '/activations', op)
    tf.summary.histogram(op.op.name + '/gradients', op)