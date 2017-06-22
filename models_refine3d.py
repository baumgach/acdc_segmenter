from tfwrapper import layers
import tensorflow as tf

NUM_CLASSES = 4

def unet_bn(images, training):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1), activation=layers.no_activation, training=training)

    return pred, conv9_2


def simple_refinement(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=16, training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=16, training=training)
    conv1_3 = layers.conv3D_layer_bn(conv1_2, 'conv1_3', num_filters=16, training=training)

    conv2_1 = layers.conv3D_layer_bn(conv1_3, 'conv2_1', num_filters=NUM_CLASSES, kernel_size=(1,1,1), training=training, activation=layers.no_activation)

    return conv2_1

def trivial(images, training):

    sm = tf.nn.softmax(images, dim=-1)
    hard = tf.one_hot(tf.argmax(sm, axis=-1), depth=4)

    conv = layers.conv3D_layer_bn(hard, 'conv2_1', num_filters=NUM_CLASSES, kernel_size=(1,1,1), training=training, activation=layers.no_activation)

    return conv

def do_nothing(images, training):

    # sm = tf.nn.softmax(images, dim=-1)

    return images


def residual_refinement(images, training):

        res1_1 = residual_unit(images, 'res1_1', num_filters=16, training=training)
        res1_2 = residual_unit(res1_1, 'res1_2', num_filters=16, training=training)
        res1_3 = residual_unit(res1_2, 'res1_3', num_filters=16, training=training)

        conv2_1 = layers.conv3D_layer_bn(res1_3, 'conv2_1', num_filters=NUM_CLASSES, kernel_size=(1,1,1), training=training, activation=layers.no_activation)

        return conv2_1


def residual_refinement_start_with_softmax(images, training):

    sm = tf.nn.softmax(images, dim=-1)

    res1_1 = residual_unit(sm, 'res1_1', num_filters=16, training=training)
    res1_2 = residual_unit(res1_1, 'res1_2', num_filters=16, training=training)
    res1_3 = residual_unit(res1_2, 'res1_3', num_filters=16, training=training)

    conv2_1 = layers.conv3D_layer_bn(res1_3, 'conv2_1', num_filters=NUM_CLASSES, kernel_size=(1, 1, 1),
                                     training=training, activation=layers.no_activation)

    return conv2_1


def residual_unit(input, name, training, num_filters=16, down_sample=False):

        c1 = layers.conv3D_layer_bn(input, name + '_c1', num_filters=num_filters, training=training)
        c2 = layers.conv3D_layer_bn(c1, name + '_c2', num_filters=num_filters, training=training, activation=layers.no_activation)

        projection = layers.conv3D_layer(input, name + '_cp', num_filters=num_filters, kernel_size=(1, 1, 1), stride=(1, 1, 1), activation=layers.no_activation)

        projection_bn = layers.batch_normalisation_layer(projection, name=name + 'bn2', training=training)

        sum_layer = tf.add(projection_bn, c2)
        act_layer = layers.activation_layer(sum_layer, activation=tf.nn.relu, name='act')

        return act_layer


# def residual_refinement(images, training):
#
#     name = 'res1'
#     num_filters = 16
#     input1 = images
#     c1_1 = layers.conv3D_layer_bn(input1, name + '_c1', num_filters=num_filters, training=training)
#     c1_2 = layers.conv3D_layer_bn(c1_1, name + '_c2', num_filters=num_filters, training=training,
#                                 activation=layers.no_activation)
#     projection = layers.conv3D_layer_bn(input1, name + '_cp', num_filters=num_filters, kernel_size=(1, 1, 1),
#                                         stride=(2, 2, 2), activation=layers.no_activation)
#     sum_layer_1 = tf.add(projection, c1_2)
#     act_layer_1 = layers.activation_layer(sum_layer_1, activation=tf.nn.relu, name='act')
#
#
#     name = 'res2'
#     num_filters = 16
#     input2 = act_layer_1
#     c2_1 = layers.conv3D_layer_bn(input2, name + '_c1', num_filters=num_filters, training=training)
#     c2_2 = layers.conv3D_layer_bn(c2_1, name + '_c2', num_filters=num_filters, training=training,
#                                 activation=layers.no_activation)
#     projection = layers.conv3D_layer_bn(input2, name + '_cp', num_filters=num_filters, kernel_size=(1, 1, 1),
#                                         stride=(2, 2, 2), activation=layers.no_activation)
#     sum_layer_2 = tf.add(projection, c2_2)
#     act_layer_2 = layers.activation_layer(sum_layer_2, activation=tf.nn.relu, name='act')
#
#
#     name = 'res3'
#     num_filters = 16
#     input3 = act_layer_2
#     c3_1 = layers.conv3D_layer_bn(input3, name + '_c1', num_filters=num_filters, training=training)
#     c3_2 = layers.conv3D_layer_bn(c3_1, name + '_c2', num_filters=num_filters, training=training,
#                                 activation=layers.no_activation)
#
#     projection = layers.conv3D_layer_bn(input3, name + '_cp', num_filters=num_filters, kernel_size=(1, 1, 1),
#                                         stride=(2, 2, 2), activation=layers.no_activation)
#
#     sum_layer_3 = tf.add(projection, c3_2)
#     act_layer_3 = layers.activation_layer(sum_layer_3, activation=tf.nn.relu, name='act')
#
#     conv2_1 = layers.conv3D_layer_bn(act_layer_3, 'conv2_1', num_filters=NUM_CLASSES, kernel_size=(1, 1, 1),
#                                      training=training, activation=layers.no_activation)
#
#     return conv2_1