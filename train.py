
import os.path
import sys
import time
import socket
import logging

import tensorflow as tf
import numpy as np
import h5py
import cv2

import model as model
import image_utils
import model_zoo
import tfwrapper.utils as tf_utils
from background_generator import BackgroundGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### GLOBAL TRAINING SETTINGS: #####################################################

PROJECT_ROOT = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored'
LOG_DIR = os.path.join(PROJECT_ROOT, 'acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.001_aug')
# LOG_DIR = os.path.join(PROJECT_ROOT, 'acdc_logdir/lisa_net_deeper_sgd_sched_reg0.00005_lr0.1_aug_bn')

BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_FILE = 'data_288x288.hdf5'  #'data_288x288_plusregs.hdf5'
MODEL_HANDLE = model_zoo.lisa_net_deeper
OPTIMIZER_HANDLE = tf.train.AdamOptimizer
# OPTIMIZER_HANDLE = tf.train.GradientDescentOptimizer
SCHEDULE_LR = True
WARMUP_TRAINING = False
AUGMENT_BATCH = True
WEIGHT_DECAY = 0.00005

### GLOBAL CONSTANTS: #############################################################
IMAGE_SIZE = (288, 288)
NLABELS = 4
MAX_EPOCHS = 20000
###################################################################################

# Find out if running locally or on grid engine. If GE then need to set cuda visible devices.
hostname = socket.gethostname()
print('Running on %s' % hostname)
if not hostname == 'bmicdl03':
    logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
    logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])

def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], NLABELS), name='labels')
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

    for batch in iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False):  # No aug in evaluation

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


def shuffle_data(images, labels):

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)
    images = images[random_indices, ...]
    labels = labels[random_indices, ...]

    return images, labels


def augmentation_function(images, labels):

    # TODO: Create kwargs with augmentation options

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for ii in range(num_images):

        random_angle = np.random.uniform(-15, 15)
        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        img = image_utils.rotate_image(img, random_angle)
        lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # cv2.imshow('image', image_utils.convert_to_uint8(img))
        # cv2.imshow('labels', image_utils.convert_to_uint8(lbl))
        # cv2.waitKey(0)

        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch


def iterate_minibatches(images, labels, batch_size=10, augment_batch=False):

    images, labels = shuffle_data(images, labels)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        X = images[b_i:b_i+batch_size, ...]
        y = labels[b_i:b_i+batch_size, ...]

        if augment_batch:
            X, y = augmentation_function(X, y)

        y = tf_utils.labels_to_one_hot(y)

        if batch_size == 1:
            X = X[np.newaxis, ...]
            y = y[np.newaxis, ...]

        yield X, y


def run_training():

    data = h5py.File(os.path.join(PROJECT_ROOT, DATA_FILE), 'r')
    images_train = data['images_train'][:]
    labels_train = data['masks_train'][:]

    images_val = data['images_test'][:]
    labels_val = data['masks_test'][:]

    images_train = np.reshape(images_train, (images_train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    images_val = np.reshape(images_val, (images_val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    logging.info('Data summary:')
    logging.info(' - Images:')
    logging.info(images_train.shape)
    logging.info(images_train.dtype)
    logging.info(' - Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)

    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        # images_placeholder, labels_placeholder = placeholder_inputs(None)  # or replace by none/batch_size
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)  # or replace by none/batch_size

        learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
        training_time_placeholder = tf.placeholder(tf.bool, shape=[])

        tf.summary.scalar('learning_rate', learning_rate_placeholder)

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images_placeholder, MODEL_HANDLE, training=training_time_placeholder)

        # Add to the Graph the Ops for loss calculation.
        [loss, _, weights_norm] = model.loss(logits, labels_placeholder, weight_decay=WEIGHT_DECAY)  # second output is unregularised loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.training(loss, OPTIMIZER_HANDLE, learning_rate_placeholder)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_loss = model.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

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
        curr_lr = LEARNING_RATE

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf

        for epoch in range(MAX_EPOCHS):

            logging.info('EPOCH %d' % epoch)

            for batch in BackgroundGenerator(iterate_minibatches(images_train, labels_train, batch_size=BATCH_SIZE)):

                # logging.info('step: %d' % step)

                if SCHEDULE_LR and WARMUP_TRAINING:
                    if step < 50:
                        curr_lr = LEARNING_RATE / 10.0
                    elif step == 50:
                        curr_lr = LEARNING_RATE

                start_time = time.time()

                # batch = bgn_train.retrieve()
                x, y = batch

                # TEMPORARY HACK (to avoid incomplete batches
                if y.shape[0] < BATCH_SIZE:
                    step += 1
                    continue

                feed_dict = {
                    images_placeholder: x,
                    labels_placeholder: y,
                    learning_rate_placeholder: curr_lr,
                    training_time_placeholder: True
                }


                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

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
                if (step + 1) % 100 == 0:

                    checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_placeholder,
                                                       labels_placeholder,
                                                       training_time_placeholder,
                                                       images_train,
                                                       labels_train,
                                                       BATCH_SIZE)

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
                                                   images_val,
                                                   labels_val,
                                                   BATCH_SIZE)

                    val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                    )
                    summary_writer.add_summary(val_summary_msg, step)

                    loss_history.append(train_loss)
                    if len(loss_history) > 3:
                        loss_history.pop(0)
                        loss_gradient = np.abs((loss_history[-1]-loss_history[-2])+(loss_history[-2]-loss_history[-3]))/2

                    logging.info('loss gradient is currently %f' % loss_gradient)

                    if SCHEDULE_LR and loss_gradient < 0.000001:
                        logging.warning('Reducing learning rate!')
                        curr_lr /= 10.0
                        logging.info('Learning rate changed to: %f' % curr_lr)

                    if val_loss <= best_val:

                        best_val = val_loss

                        best_file = os.path.join(LOG_DIR, 'model_best.ckpt')
                        saver.save(sess, best_file, global_step=step)

                        logging.info('Found new best on validation set! ')

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
    run_training()


if __name__ == '__main__':

    tf.app.run(main=main, argv=[sys.argv[0]])
