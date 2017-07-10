# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
from tfwrapper import losses


def inference(images, inference_handle, training=True):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module. 
    '''

    return inference_handle(images, training)


def loss(logits, labels, nlabels, loss_type, weight_decay=0.0):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'weighted_crossentropy'/'crossentropy'/'dice'/'crossentropy_and_dice'
    :param weight_decay: The weight for the L2 regularisation of the network paramters
    :return: The total loss including weight decay, the loss without weight decay, only the weight decay 
    '''

    labels = tf.one_hot(labels, depth=nlabels)

    with tf.variable_scope('weights_norm') as scope:

        weights_norm = tf.reduce_sum(
            input_tensor = weight_decay*tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

    if loss_type == 'weighted_crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, labels,
                                                                          class_weights=[0.1, 0.3, 0.3, 0.3])
    elif loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
    elif loss_type == 'crossentropy_and_dice':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels) + 0.2*losses.dice_loss(logits, labels)
    else:
        raise ValueError('Unknown loss: %s' % loss_type)


    total_loss = tf.add(segmentation_loss, weights_norm)

    return total_loss, segmentation_loss, weights_norm


def predict(images, inference_handle):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = inference_handle(images, training=tf.constant(False, dtype=tf.bool))
    softmax = tf.nn.softmax(logits)
    mask = tf.arg_max(softmax, dimension=-1)

    return mask, softmax


def training_step(loss, optimizer_handle, learning_rate, **kwargs):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :param momentum: Optionally, you can also pass a momentum term to the optimiser. 
    :return: The training operation
    '''


    if 'momentum' in kwargs:
        momentum = kwargs.get('momentum')
        optimizer = optimizer_handle(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = optimizer_handle(learning_rate=learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels, images, nlabels, loss_type):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    '''

    mask = tf.arg_max(tf.nn.softmax(logits, dim=-1), dimension=-1)  # was 3
    mask_gt = labels

    tf.summary.image('example_gt', prepare_tensor_for_summary(mask_gt, mode='mask', nlabels=nlabels))
    tf.summary.image('example_pred', prepare_tensor_for_summary(mask, mode='mask', nlabels=nlabels))
    tf.summary.image('example_zimg', prepare_tensor_for_summary(images, mode='image'))

    total_loss, nowd_loss, weights_norm = loss(logits, labels, nlabels=nlabels, loss_type=loss_type)

    cdice_structures = losses.per_structure_dice(logits, tf.one_hot(labels, depth=nlabels))
    cdice_foreground = tf.slice(cdice_structures, (0,1), (-1,-1))

    cdice = tf.reduce_mean(cdice_foreground)

    return nowd_loss, cdice


def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 10), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
