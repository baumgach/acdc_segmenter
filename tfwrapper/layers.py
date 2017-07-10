# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import math
import numpy as np

from tfwrapper import utils

def linear_activation(x):
    '''
    A linear activation function (i.e. no non-linearity)
    '''
    return x

def activation_layer(bottom, name, activation=tf.nn.relu):
    '''
    Custom activation layer, with the tf.nn.relu as default
    '''

    with tf.name_scope(name):

        op = activation(bottom)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op

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

    if len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')

    elif len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        print('Padded shape:')
        print(padded.get_shape().as_list())


def batch_normalisation_layer(bottom, name, training):
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
        beta = tf.get_variable(name=name + '_beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
                               trainable=True)
        gamma = tf.get_variable(name=name + '_gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
                                trainable=True)

        batch_mean, batch_var = tf.nn.moments(bottom, moments_over_axes, name=name + '_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, 1e-3)

        return normed

### FEED_FORWARD LAYERS ##############################################################################33

def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal'):

    '''
    Standard 2D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
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


def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal'):

    '''
    Standard 3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(4,4),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):

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

    with tf.name_scope(name):

        if weight_init == 'he_normal':
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


def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):

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

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        elif weight_init == 'bilinear':
            weights = _weight_variable_bilinear(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv3d_transpose(bottom,
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
                         weight_init='he_normal'):

    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        if weight_init == 'he_normal':
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


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=tf.nn.relu,
                weight_init='he_normal'):

    '''
    Dense a.k.a. fully connected layer
    '''

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


### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    name,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool)):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    relu = activation(conv_bn)

    return relu


def conv3D_layer_bn(bottom,
                    name,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    relu = activation(conv_bn)

    return relu

def deconv2D_layer_bn(bottom,
                      name,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal',
                      training=tf.constant(True, dtype=tf.bool)):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=linear_activation,
                          padding=padding,
                          weight_init=weight_init)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    relu = activation(deco_bn)

    return relu


def deconv3D_layer_bn(bottom,
                      name,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal',
                      training=tf.constant(True, dtype=tf.bool),
                      **kwargs):

    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=linear_activation,
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
                           training=tf.constant(True, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=linear_activation,
                                padding=padding,
                                weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    relu = activation(conv_bn)

    return relu



def dense_layer_bn(bottom,
                   name,
                   hidden_units=512,
                   activation=tf.nn.relu,
                   weight_init='he_normal',
                   training=tf.constant(True, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=linear_activation,
                         weight_init=weight_init)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training)
    relu = activation(batchnorm)

    return relu

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
    '''
    Initialise weights with a billinear interpolation filter for upsampling
    '''

    weights = _bilinear_upsample_weights(shape)
    initial = tf.constant(weights, shape=shape, dtype=tf.float32)

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