# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np

import utils
import image_utils
import model as model

from background_generator import BackgroundGenerator
import config.system as sys_config
import acdc_data

### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:

# from experiments import FCN8_bn_wxent as exp_config
# from experiments import unet2D_bn_modified_dice as exp_config
# from experiments import unet2D_bn_modified_wxent as exp_config
# from experiments import unet2D_bn_modified_xent as exp_config
# from experiments import unet2D_bn_wxent as exp_config
from experiments import unet3D_bn_modified_wxent as exp_config

########################################################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')


def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

    init_step = 0

    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0

        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if hasattr(exp_config, 'train_on_all_data'):
        train_on_all_data = exp_config.train_on_all_data
    else:
        train_on_all_data = False

    # Load data
    data = acdc_data.load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False,
        split_test_train=(not train_on_all_data)
    )

    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train']
    labels_train = data['masks_train']

    if not train_on_all_data:
        images_val = data['images_test']
        labels_val = data['masks_test']

    if exp_config.use_data_fraction:
        num_images = images_train.shape[0]
        new_last_index = int(float(num_images)*exp_config.use_data_fraction)

        logging.warning('USING ONLY FRACTION OF DATA!')
        logging.warning(' - Number of imgs orig: %d, Number of imgs new: %d' % (num_images, new_last_index))
        images_train = images_train[0:new_last_index,...]
        labels_train = labels_train[0:new_last_index,...]

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

        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        mask_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)

        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')

        learning_rate_pl = tf.placeholder(tf.float32, shape=[])
        training_pl = tf.placeholder(tf.bool, shape=[])

        tf.summary.scalar('learning_rate', learning_rate_pl)

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images_pl, exp_config, training=training_pl)

        # Add to the Graph the Ops for loss calculation.
        [loss, _, weights_norm] = model.loss(logits,
                                             labels_pl,
                                             nlabels=exp_config.nlabels,
                                             loss_type=exp_config.loss_type,
                                             weight_decay=exp_config.weight_decay)  # second output is unregularised loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        # Add to the Graph the Ops that calculate and apply gradients.
        if exp_config.momentum is not None:
            train_op = model.training_step(loss, exp_config.optimizer_handle, learning_rate_pl, momentum=exp_config.momentum)
        else:
            train_op = model.training_step(loss, exp_config.optimizer_handle, learning_rate_pl)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_loss = model.evaluation(logits,
                                     labels_pl,
                                     images_pl,
                                     nlabels=exp_config.nlabels,
                                     loss_type=exp_config.loss_type)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.

        if train_on_all_data:
            max_to_keep = None
        else:
            max_to_keep = 5

        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver_best_dice = tf.train.Saver()
        saver_best_xent = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

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

        if continue_run:
            # Restore session
            saver.restore(sess, init_checkpoint_path)

        step = init_step
        curr_lr = exp_config.learning_rate

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_dice = 0

        for epoch in range(exp_config.max_epochs):

            logging.info('EPOCH %d' % epoch)


            for batch in BackgroundGenerator(iterate_minibatches(images_train,
                                             labels_train,
                                             batch_size=exp_config.batch_size,
                                             augment_batch=exp_config.augment_batch)):

            # You can run this loop with the BACKGROUND GENERATOR, which will lead to some improvements in the
            # training speed. However, be aware that currently an exception inside this loop may not be caught.
            # The batch generator may just continue running silently without warning eventhough the code has
            # crashed.
            # for batch in BackgroundGenerator(iterate_minibatches(images_train,
            #                                                      labels_train,
            #                                                      batch_size=exp_config.batch_size,
            #                                                      augment_batch=exp_config.augment_batch)):


                if exp_config.warmup_training:
                    if step < 50:
                        curr_lr = exp_config.learning_rate / 10.0
                    elif step == 50:
                        curr_lr = exp_config.learning_rate

                start_time = time.time()

                # batch = bgn_train.retrieve()
                x, y = batch

                # TEMPORARY HACK (to avoid incomplete batches
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {
                    images_pl: x,
                    labels_pl: y,
                    learning_rate_pl: curr_lr,
                    training_pl: True
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

                if (step + 1) % exp_config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_train,
                                                       labels_train,
                                                       exp_config.batch_size)

                    train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                           train_dice_: train_dice}
                                                 )
                    summary_writer.add_summary(train_summary_msg, step)

                    loss_history.append(train_loss)
                    if len(loss_history) > 5:
                        loss_history.pop(0)
                        loss_gradient = (loss_history[-5] - loss_history[-1]) / 2

                    logging.info('loss gradient is currently %f' % loss_gradient)

                    if exp_config.schedule_lr and loss_gradient < exp_config.schedule_gradient_threshold:
                        logging.warning('Reducing learning rate!')
                        curr_lr /= 10.0
                        logging.info('Learning rate changed to: %f' % curr_lr)

                        # reset loss history to give the optimisation some time to start decreasing again
                        loss_gradient = np.inf
                        loss_history = []

                    if train_loss <= last_train:  # best_train:
                        logging.info('Decrease in training error!')
                    else:
                        logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                    last_train = train_loss

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % exp_config.val_eval_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.

                    if not train_on_all_data:

                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')
                        [val_loss, val_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_val,
                                                       labels_val,
                                                       exp_config.batch_size)

                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                        )
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            best_file = os.path.join(log_dir, 'model_best_dice.ckpt')
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                        if val_loss < best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'model_best_xent.ckpt')
                            saver_best_xent.save(sess, best_file, global_step=step)
                            logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)


                step += 1

        sess.close()
    data.close()


def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in BackgroundGenerator(iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False)):  # No aug in evaluation
    # As before you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
    # but at the risk of not catching exceptions

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


def augmentation_function(images, labels, **kwargs):
    '''
    Function for augmentation of minibatches. It will transform a set of images and corresponding labels
    by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
    with random parameters. 
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size. 
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction. 
    :return: A mini batch of the same size but with transformed images and masks. 
    '''

    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    do_rotations = kwargs.get('do_rotations', False)
    do_scaleaug = kwargs.get('do_scaleaug', False)
    do_fliplr = kwargs.get('do_fliplr', False)


    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for ii in range(num_images):

        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        # ROTATE
        if do_rotations:
            angles = kwargs.get('angles', (-15,15))
            random_angle = np.random.uniform(angles[0], angles[1])
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # RANDOM CROP SCALE
        if do_scaleaug:
            offset = kwargs.get('offset', 30)
            n_x, n_y = img.shape
            r_y = np.random.random_integers(n_y-offset, n_y)
            p_x = np.random.random_integers(0, n_x-r_y)
            p_y = np.random.random_integers(0, n_y-r_y)

            img = image_utils.resize_image(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y))
            lbl = image_utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                lbl = np.fliplr(lbl)


        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch


def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :return: mini batches
    '''

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

        image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)

        if augment_batch:
            X, y = augmentation_function(X, y,
                                         do_rotations=exp_config.do_rotations,
                                         do_scaleaug=exp_config.do_scaleaug,
                                         do_fliplr=exp_config.do_fliplr)


        yield X, y


def main():

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description="Train a neural network.")
    # parser.add_argument("CONFIG_PATH", type=str, help="Path to config file (assuming you are in the working directory)")
    # args = parser.parse_args()

    main()
