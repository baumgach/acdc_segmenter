import logging
import os.path
import sys
import time
import socket
import shutil

import tensorflow as tf
import cv2
import h5py
import numpy as np

import utils
import image_utils
import model as model
import tfwrapper.utils as tf_utils
from background_generator import BackgroundGenerator

from config.threedee.train import *
from config.threedee.system import *

from skimage import transform

# import matplotlib.pyplot as plt

### EXPERIMENT CONFIG FILE #############################################################
# from experiments import debug as exp_config
#from experiments.threedee import refine_residual_units as exp_config
from experiments.threedee import unet_2D_AND_3D as exp_config
# from experiments.threedee import unet_3d as exp_config
########################################################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

LOG_DIR = os.path.join(LOG_ROOT, exp_config.experiment_name)
DOWNSAMPLING_FACTOR = exp_config.down_sampling_factor
INPUT_CHANNELS = int(exp_config.input_channels)

# Find out if running locally or on grid engine. If GE then need to set cuda visible devices.
hostname = socket.gethostname()
print('Running on %s' % hostname)
if not hostname == local_hostname:
    logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
    logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])


def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, IMAGE_SIZE[0]/DOWNSAMPLING_FACTOR, IMAGE_SIZE[1]/DOWNSAMPLING_FACTOR, IMAGE_SIZE[2], INPUT_CHANNELS), name='images')
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, IMAGE_SIZE[0]/DOWNSAMPLING_FACTOR, IMAGE_SIZE[1]/DOWNSAMPLING_FACTOR, IMAGE_SIZE[2], 4), name='labels')
    return images_placeholder, labels_placeholder


def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in BackgroundGenerator(iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False)):  # No aug in evaluation

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      labels_placeholder: y,
                      training_time_placeholder: False}

        closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice



def resize_batch(images, labels, scale):

    batch_size = images.shape[0]

    new_images = []
    new_labels = []

    for ii in range(batch_size):

        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        scale_u = scale
        if img.ndim > len(scale):
            scale_u = scale + (img.ndim-len(scale))*[1]
        img = transform.rescale(img, scale_u, order=1, preserve_range=True, multichannel=False, mode='constant')

        if lbl.ndim > len(scale):
            scale_u = scale + (lbl.ndim-len(scale))*[1]
        lbl = transform.rescale(lbl, scale_u, order=1, preserve_range=True, multichannel=False, mode='constant')

        new_images.append(img)
        new_labels.append(lbl)

    img_arr = np.asarray(new_images, dtype=np.float32)
    lbl_arr = np.asarray(new_labels, dtype=np.uint8)

    return img_arr, lbl_arr


def iterate_minibatches(images, labels, batch_size=10, augment_batch=False):
    """

    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :return:
    """

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        y = tf_utils.labels_to_one_hot(y)

        if not DOWNSAMPLING_FACTOR == 1:
            X, y = resize_batch(X, y, scale=[1.0/DOWNSAMPLING_FACTOR,1.0/DOWNSAMPLING_FACTOR, 1])

        X = np.reshape(X, (batch_size, IMAGE_SIZE[0]//DOWNSAMPLING_FACTOR, IMAGE_SIZE[1]//DOWNSAMPLING_FACTOR, IMAGE_SIZE[2], INPUT_CHANNELS))

        # if augment_batch:
        #     X, y = augmentation_function(X, y,
        #                                  do_rotations=exp_config.do_rotations,
        #                                  do_scaleaug=exp_config.do_scaleaug,
        #                                  do_fliplr=exp_config.do_fliplr)

        #
        # if batch_size == 1:
        #     X = X[np.newaxis, ...]
        #     y = y[np.newaxis, ...]

        yield X, y


def run_training():

    data = h5py.File(os.path.join(PROJECT_ROOT, exp_config.data_file), 'r')

    # the following are HDF5 datasets, not numpy arrays
    # feature_maps_train = data['feature_maps_train']
    feature_maps_train = data['%s_train' % exp_config.input_dataset]
    labels_train = data['masks_train']

    # feature_maps_val = data['feature_maps_test']
    feature_maps_val = data['%s_test' % exp_config.input_dataset]
    labels_val = data['masks_test']

    if exp_config.use_data_fraction:
        num_images = feature_maps_train.shape[0]
        new_last_index = int(float(num_images)*exp_config.use_data_fraction)

        logging.warning('USING ONLY FRACTION OF DATA!')
        logging.warning(' - Number of imgs orig: %d, Number of imgs new: %d' % (num_images, new_last_index))
        feature_maps_train = feature_maps_train[0:new_last_index,...]
        labels_train = labels_train[0:new_last_index,...]

    logging.info('Data summary:')
    logging.info(' - Images:')
    logging.info(feature_maps_train.shape)
    logging.info(feature_maps_train.dtype)
    logging.info(' - Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)

    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        # images_placeholder, labels_placeholder = placeholder_inputs(None)  # or replace by none/batch_size
        images_placeholder, labels_placeholder = placeholder_inputs(exp_config.batch_size)  # or replace by none/batch_size

        learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
        training_time_placeholder = tf.placeholder(tf.bool, shape=[])

        tf.summary.scalar('learning_rate', learning_rate_placeholder)

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images_placeholder, exp_config.model_handle, training=training_time_placeholder)

        # Add to the Graph the Ops for loss calculation.
        [loss, _, weights_norm] = model.loss(logits,
                                             labels_placeholder,
                                             weight_decay=exp_config.weight_decay,
                                             loss_type=exp_config.loss_type)  # second output is unregularised loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        # Add to the Graph the Ops that calculate and apply gradients.
        if exp_config.momentum is not None:
            train_op = model.training(loss, exp_config.optimizer_handle, learning_rate_placeholder, momentum=exp_config.momentum)
        else:
            train_op = model.training(loss, exp_config.optimizer_handle, learning_rate_placeholder)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_loss = model.evaluation(logits,
                                     labels_placeholder,
                                     loss_type=exp_config.loss_type)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        saver_best_dice = tf.train.Saver()
        saver_best_xent = tf.train.Saver()

        # GPU settings
        gpu_memory_fraction = 1.0  # Fraction of GPU memory to use
        allow_growth = False
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,
                                                          allow_growth=allow_growth))


        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # with tf.name_scope('monitoring'):

        val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error')
        val_error_summary = tf.summary.scalar('validation_loss', val_error_)

        val_dice_ = tf.placeholder(tf.float32, shape=[], name='val_dice')
        val_dice_summary = tf.summary.scalar('validation_dice', val_dice_)

        val_summary = tf.summary.merge([val_error_summary, val_dice_summary])

        train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error')
        train_error_summary = tf.summary.scalar('training_loss', train_error_)

        train_dice_ = tf.placeholder(tf.float32, shape=[], name='train_dice')
        train_dice_summary = tf.summary.scalar('training_dice', train_dice_)

        train_summary = tf.summary.merge([train_error_summary, train_dice_summary])

        # Run the Op to initialize the variables.
        sess.run(init)

        step = 0
        curr_lr = exp_config.learning_rate

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_dice = 0

        for epoch in range(MAX_EPOCHS):

            logging.info('EPOCH %d' % epoch)

            for batch in BackgroundGenerator(iterate_minibatches(
                                                feature_maps_train,
                                                labels_train,
                                                batch_size=exp_config.batch_size,
                                                augment_batch=exp_config.augment_batch)):

                # logging.info('step: %d' % step)

                if exp_config.warmup_training:
                    if step < 50:
                        curr_lr = exp_config.learning_rate / 10.0
                    elif step == 50:
                        curr_lr = exp_config.learning_rate

                start_time = time.time()

                # batch = bgn_train.retrieve()
                x, y = batch

                # DEBUG:
                # print(x.shape)
                # print(y.shape)
                # xv = np.squeeze(x[0, :, :, 11, :])
                # yv = np.squeeze(y[0, :, :, 11, :])
                # print(yv.shape)
                #
                # plt.figure()
                # plt.imshow(xv)
                # plt.figure()
                # plt.imshow(yv)
                # plt.show()


                # TEMPORARY HACK (to avoid incomplete batches
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {
                    images_placeholder: x,
                    labels_placeholder: y,
                    learning_rate_placeholder: curr_lr,
                    training_time_placeholder: True
                }


                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                # debugging
                # logging.info('step: %d, loss value: %f' % (step, loss_value))

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 10 == 0:
                    # Print status to stdout.
                    logging.info('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % EVAL_FREQUENCY == 0:

                    checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_placeholder,
                                                       labels_placeholder,
                                                       training_time_placeholder,
                                                       feature_maps_train,
                                                       labels_train,
                                                       exp_config.batch_size)

                    train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                           train_dice_: train_dice}
                    )
                    summary_writer.add_summary(train_summary_msg, step)

                    # Evaluate against the validation set.
                    logging.info('Validation Data Eval:')
                    [val_loss, val_dice] = do_eval(sess,
                                                   eval_loss,
                                                   images_placeholder,
                                                   labels_placeholder,
                                                   training_time_placeholder,
                                                   feature_maps_val,
                                                   labels_val,
                                                   exp_config.batch_size)

                    val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                    )
                    summary_writer.add_summary(val_summary_msg, step)

                    loss_history.append(train_loss)
                    if len(loss_history) > 5:
                        loss_history.pop(0)
                        loss_gradient = (loss_history[-5]-loss_history[-1])/2

                    logging.info('loss gradient is currently %f' % loss_gradient)

                    if exp_config.schedule_lr and loss_gradient < SCHEDULE_GRADIENT_THRESHOLD:
                        logging.warning('Reducing learning rate!')
                        curr_lr /= 10.0
                        logging.info('Learning rate changed to: %f' % curr_lr)

                        # reset loss history to give the optimisation some time to start decreasing again
                        loss_gradient = np.inf
                        loss_history = []

                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(LOG_DIR, 'model_best_dice.ckpt')
                        saver_best_dice.save(sess, best_file, global_step=step)
                        logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                    if val_loss < best_val:
                        best_val = val_loss
                        best_file = os.path.join(LOG_DIR, 'model_best_xent.ckpt')
                        saver_best_xent.save(sess, best_file, global_step=step)
                        logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

                    if train_loss <= last_train: #best_train:
                        logging.info('Decrease in training error!')
                    else:
                        logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                    last_train = train_loss

                step += 1


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    # Copy experiment config file
    shutil.copy( exp_config.__file__, LOG_DIR)

    run_training()


if __name__ == '__main__':

    tf.app.run(main=main, argv=[sys.argv[0]])
