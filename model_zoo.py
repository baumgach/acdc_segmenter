# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
from tfwrapper import layers


def VGG16_FCN_8(images, training, nlabels):

    conv1_1 = layers.conv2D_layer(images, 'conv1_1', num_filters=64)
    conv1_2 = layers.conv2D_layer(conv1_1, 'conv1_2', num_filters=64)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=128)
    conv2_2 = layers.conv2D_layer(conv2_1, 'conv2_2', num_filters=128)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=256)
    conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=256)
    conv3_3 = layers.conv2D_layer(conv3_2, 'conv3_3', num_filters=256)

    pool3 = layers.max_pool_layer2d(conv3_3)

    conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=512)
    conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=512)
    conv4_3 = layers.conv2D_layer(conv4_2, 'conv4_3', num_filters=512)

    pool4 = layers.max_pool_layer2d(conv4_3)

    conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=512)
    conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=512)
    conv5_3 = layers.conv2D_layer(conv5_2, 'conv5_3', num_filters=512)

    pool5 = layers.max_pool_layer2d(conv5_3)

    conv6 = layers.conv2D_layer(pool5, 'conv6', num_filters=4096, kernel_size=(3,3))
    conv7= layers.conv2D_layer(conv6, 'conv7', num_filters=4096, kernel_size=(1,1))

    score5 = layers.conv2D_layer(conv7, 'score5', num_filters=nlabels, kernel_size=(1,1))
    score4 = layers.conv2D_layer(pool4, 'score4', num_filters=nlabels, kernel_size=(1,1))
    score3 = layers.conv2D_layer(pool3, 'score3', num_filters=nlabels, kernel_size=(1,1))

    upscore1 = layers.deconv2D_layer(score5, name='upscore1', kernel_size=(4,4), strides=(2,2), num_filters=nlabels, weight_init='bilinear')

    sum1 = tf.add(upscore1, score4)

    upscore2 = layers.deconv2D_layer(sum1, name='upscore2', kernel_size=(4,4), strides=(2,2), num_filters=nlabels, weight_init='bilinear')

    sum2 = tf.add(upscore2, score3)

    upscore3 = layers.deconv2D_layer(sum2, name='upscore3', kernel_size=(16,16), strides=(8,8), num_filters=nlabels, weight_init='bilinear', activation=tf.identity)

    return upscore3


def VGG16_FCN_8_bn(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    conv3_3 = layers.conv2D_layer_bn(conv3_2, 'conv3_3', num_filters=256, training=training)

    pool3 = layers.max_pool_layer2d(conv3_3)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    conv4_3 = layers.conv2D_layer_bn(conv4_2, 'conv4_3', num_filters=512, training=training)

    pool4 = layers.max_pool_layer2d(conv4_3)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=512, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=512, training=training)
    conv5_3 = layers.conv2D_layer_bn(conv5_2, 'conv5_3', num_filters=512, training=training)

    pool5 = layers.max_pool_layer2d(conv5_3)

    conv6 = layers.conv2D_layer_bn(pool5, 'conv6', num_filters=4096, kernel_size=(7,7), training=training)
    conv7= layers.conv2D_layer_bn(conv6, 'conv7', num_filters=4096, kernel_size=(1,1), training=training)

    score5 = layers.conv2D_layer_bn(conv7, 'score5', num_filters=nlabels, kernel_size=(1,1), training=training)
    score4 = layers.conv2D_layer_bn(pool4, 'score4', num_filters=nlabels, kernel_size=(1,1), training=training)
    score3 = layers.conv2D_layer_bn(pool3, 'score3', num_filters=nlabels, kernel_size=(1,1), training=training)

    upscore1 = layers.deconv2D_layer_bn(score5, name='upscore1', kernel_size=(4,4), strides=(2,2), num_filters=nlabels, weight_init='bilinear', training=training)

    sum1 = tf.add(upscore1, score4)

    upscore2 = layers.deconv2D_layer_bn(sum1, name='upscore2', kernel_size=(4,4), strides=(2,2), num_filters=nlabels, weight_init='bilinear', training=training)

    sum2 = tf.add(upscore2, score3)

    upscore3 = layers.deconv2D_layer_bn(sum2, name='upscore3', kernel_size=(16,16), strides=(8,8), num_filters=nlabels, weight_init='bilinear', training=training, activation=tf.identity)

    return upscore3


def unet2D_bn_padding_same(images, training, nlabels):

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

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred


def unet2D_bn_padding_same_shallow(images, training, nlabels):

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

    upconv3 = layers.deconv2D_layer_bn(conv4_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred


def unet2D_padding_same(images, training, nlabels):

    conv1_1 = layers.conv2D_layer(images, 'conv1_1', num_filters=64)
    conv1_2 = layers.conv2D_layer(conv1_1, 'conv1_2', num_filters=64)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=128)
    conv2_2 = layers.conv2D_layer(conv2_1, 'conv2_2', num_filters=128)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=256)
    conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=256)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=512)
    conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=512)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=1024)
    conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=1024)

    upconv4 = layers.deconv2D_layer(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512)
    concat4 = tf.concat([conv4_2, upconv4], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer(concat4, 'conv6_1', num_filters=512)
    conv6_2 = layers.conv2D_layer(conv6_1, 'conv6_2', num_filters=512)

    upconv3 = layers.deconv2D_layer(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer(concat3, 'conv7_1', num_filters=256)
    conv7_2 = layers.conv2D_layer(conv7_1, 'conv7_2', num_filters=256)

    upconv2 = layers.deconv2D_layer(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer(concat2, 'conv8_1', num_filters=128)
    conv8_2 = layers.conv2D_layer(conv8_1, 'conv8_2', num_filters=128)

    upconv1 = layers.deconv2D_layer(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer(concat1, 'conv9_1', num_filters=64)
    conv9_2 = layers.conv2D_layer(conv9_1, 'conv9_2', num_filters=64)

    pred = layers.conv2D_layer(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity)

    return pred


def unet2D_padding_same_shallow(images, training, nlabels):

    conv1_1 = layers.conv2D_layer(images, 'conv1_1', num_filters=64)
    conv1_2 = layers.conv2D_layer(conv1_1, 'conv1_2', num_filters=64)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=128)
    conv2_2 = layers.conv2D_layer(conv2_1, 'conv2_2', num_filters=128)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=256)
    conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=256)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=512)
    conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=512)

    upconv3 = layers.deconv2D_layer(conv4_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer(concat3, 'conv7_1', num_filters=256)
    conv7_2 = layers.conv2D_layer(conv7_1, 'conv7_2', num_filters=256)

    upconv2 = layers.deconv2D_layer(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer(concat2, 'conv8_1', num_filters=128)
    conv8_2 = layers.conv2D_layer(conv8_1, 'conv8_2', num_filters=128)

    upconv1 = layers.deconv2D_layer(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer(concat1, 'conv9_1', num_filters=64)
    conv9_2 = layers.conv2D_layer(conv9_1, 'conv9_2', num_filters=64)

    pred = layers.conv2D_layer(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity)

    return pred


def unet2D_bn_padding_same_modified(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64//2, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64//2, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128//2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128//2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256//2, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256//2, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512//2, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512//2, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024//2, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024//2, training=training)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512//2, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512//2, training=training)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256//2, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256//2, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128//2, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128//2, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64//2, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64//2, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred






def unet2D_bn_modified(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, padding='VALID')

    return pred



def unet2D_bn(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, padding='VALID')

    return pred



def unet3D_bn(images, training, nlabels):

    images_padded = tf.pad(images, [[0, 0], [44, 44], [44, 44],[44, 44], [0, 0]], 'CONSTANT')

    conv1_1 = layers.conv3D_layer_bn(images_padded, 'conv1_1', num_filters=32, kernel_size=(3,3,3), training=training, padding='VALID')
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=512, training=training)
    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=4)

    conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=256, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=4)

    conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', kernel_size=(4, 4, 2), strides=(2, 2, 2), num_filters=128, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=4)

    conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')

    pred = layers.conv3D_layer_bn(conv8_2, 'pred', num_filters=nlabels, kernel_size=(1,1,1), activation=tf.identity, training=training, padding='VALID')

    return pred



def unet3D_bn_modified(images, training, nlabels):

    images_padded = tf.pad(images, [[0, 0], [44, 44], [44, 44], [16, 16], [0, 0]], 'CONSTANT')

    conv1_1 = layers.conv3D_layer_bn(images_padded, 'conv1_1', num_filters=32, kernel_size=(3,3,3), training=training, padding='VALID')
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=512, training=training)
    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=4)

    conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', kernel_size=(4, 4, 2), strides=(2, 2, 1), num_filters=256, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=4)

    conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=128, kernel_size=(3,3,3), training=training, padding='VALID')

    upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', kernel_size=(4, 4, 2), strides=(2, 2, 1), num_filters=128, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=4)

    conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=64, kernel_size=(3,3,3), training=training, padding='VALID')

    pred = layers.conv3D_layer_bn(conv8_2, 'pred', num_filters=nlabels, kernel_size=(1,1,1), activation=tf.identity, training=training, padding='VALID')

    return pred


def unet2D_bn_padding_same_modified_mp_stride1(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64//2, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64//2, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2, strides=(1,1))

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128//2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128//2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2, strides=(1,1))

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256//2, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256//2, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2, strides=(1,1))

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512//2, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512//2, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2, strides=(1,1))

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024//2, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024//2, training=training)

    concat4 = tf.concat([conv4_2, conv5_2], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512//2, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512//2, training=training)

    concat3 = tf.concat([conv3_2, conv6_2], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256//2, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256//2, training=training)

    concat2 = tf.concat([conv2_2, conv7_2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128//2, training=training)

    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128//2, training=training)
    concat1 = tf.concat([conv1_2, conv8_2], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64//2, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64//2, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred


def wormnet(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64//2, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64//2, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2, strides=(1,1))

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128//2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128//2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2, strides=(1,1))

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256//2, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256//2, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2, strides=(1,1))

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512//2, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512//2, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2, strides=(1,1))

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024//2, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024//2, training=training)

    conv6_1 = layers.conv2D_layer_bn(conv5_2, 'conv6_1', num_filters=512//2, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512//2, training=training)

    conv7_1 = layers.conv2D_layer_bn(conv6_2, 'conv7_1', num_filters=256//2, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256//2, training=training)

    conv8_1 = layers.conv2D_layer_bn(conv7_2, 'conv8_1', num_filters=128//2, training=training)

    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128//2, training=training)

    conv9_1 = layers.conv2D_layer_bn(conv8_2, 'conv9_1', num_filters=64//2, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64//2, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred