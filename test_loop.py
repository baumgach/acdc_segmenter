import tensorflow as tf
import model as model
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import model_zoo

def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,288, 288, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,288, 288, 1), name='labels')
    return images_placeholder, labels_placeholder

data = h5py.File('data_288x288.hdf5', 'r')
images_val = data['images_test'][:]
labels_val = data['masks_test'][:]

images_val = np.reshape(images_val, (images_val.shape[0], 288, 288, 1))

images_placeholder, labels_placeholder = placeholder_inputs(1)

inference_handle = model_zoo.lisa_net_deeper
mask, softmax = model.predict(images_placeholder, inference_handle)

saver = tf.train.Saver()

init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)
    # saver.restore(sess, tf.train.latest_checkpoint('./acdc_logdir/lisa_net_deeper_mom0.9_sched_reg0.00000_lr0.1_aug_newbn'))
    saver.restore(sess, tf.train.latest_checkpoint('./acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.001_aug_refunweighted'))

    while True:

        ind = np.random.randint(images_val.shape[0])

        x = images_val[np.newaxis, ind,...]
        y = labels_val[ind,...]

        feed_dict = {
            images_placeholder: x,
        }

        mask_out, logits_out = sess.run([mask, softmax], feed_dict=feed_dict)

        logits_crf = np.squeeze(logits_out)
        logits_crf = np.transpose(logits_crf, (2, 0, 1))

        d = dcrf.DenseCRF2D(288, 288, 4)  # width, height, nlabels
        U = unary_from_softmax(logits_crf)
        U = U.reshape((4, -1))
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3)

        Q = d.inference(50)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        MAP = np.reshape(MAP, [288, 288])

        fig = plt.figure()
        ax1 = fig.add_subplot(241)
        ax1.imshow(np.squeeze(x), cmap='gray')
        ax2 = fig.add_subplot(242)
        ax2.imshow(np.squeeze(mask_out))
        ax3 = fig.add_subplot(243)
        ax3.imshow(np.squeeze(MAP))
        ax4 = fig.add_subplot(244)
        ax4.imshow(np.squeeze(y))

        logits_c0 = np.squeeze(logits_out[0,...,0])
        logits_c1 = np.squeeze(logits_out[0,...,1])
        logits_c2 = np.squeeze(logits_out[0,...,2])
        logits_c3 = np.squeeze(logits_out[0,...,3])

        ax5 = fig.add_subplot(245)
        ax5.imshow(logits_c0)

        ax6 = fig.add_subplot(246)
        ax6.imshow(logits_c1)

        ax7 = fig.add_subplot(247)
        ax7.imshow(logits_c2)

        ax8 = fig.add_subplot(248)
        ax8.imshow(logits_c3)

        plt.show()



