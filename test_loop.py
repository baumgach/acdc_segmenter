
# Simple loop for displaying predictions for random slices from the test dataset
#
# Usage:
#
# python test_loop.py path/to/experiment_logs
#
#
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
import utils
import acdc_data
import image_utils
import model

def main(exp_config):

    # Load data
    data = acdc_data.load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False
    )

    batch_size = 1

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, exp_config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        while True:

            ind = np.random.randint(data['images_test'].shape[0])

            x = data['images_test'][ind,...]
            y = data['masks_test'][ind,...]

            x = image_utils.reshape_2Dimage_to_tensor(x)
            y = image_utils.reshape_2Dimage_to_tensor(y)

            feed_dict = {
                images_pl: x,
            }

            mask_out, softmax_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)

            fig = plt.figure()
            ax1 = fig.add_subplot(241)
            ax1.imshow(np.squeeze(x), cmap='gray')
            ax2 = fig.add_subplot(242)
            ax2.imshow(np.squeeze(y))
            ax3 = fig.add_subplot(243)
            ax3.imshow(np.squeeze(mask_out))

            ax5 = fig.add_subplot(245)
            ax5.imshow(np.squeeze(softmax_out[...,0]))
            ax6 = fig.add_subplot(246)
            ax6.imshow(np.squeeze(softmax_out[...,1]))
            ax7 = fig.add_subplot(247)
            ax7.imshow(np.squeeze(softmax_out[...,2]))
            ax8 = fig.add_subplot(248)
            ax8.imshow(np.squeeze(softmax_out[...,3]))

            plt.show()

    data.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    init_iteration = main(exp_config=exp_config)