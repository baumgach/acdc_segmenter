import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform

from tfwrapper import losses, utils

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
        lbl = transform.rescale(lbl, scale_u, order=0, preserve_range=True, multichannel=False, mode='constant')

        new_images.append(img)
        new_labels.append(lbl)

    img_arr = np.asarray(new_images, dtype=np.float32)
    lbl_arr = np.asarray(new_labels, dtype=np.uint8)

    return img_arr, lbl_arr

f=h5py.File('feature_maps_and_pred.hdf5')

pred = f['predictions_test']
slice_pred = np.squeeze(pred[10,:,:,11,:])

mask = f['masks_test']
slice_mask = np.squeeze(mask[10,:,:,11])

# plt.figure()
# slice_argm = np.argmax(slice_pred, axis=2)
# plt.imshow(slice_argm)
#
# plt.figure()
# plt.imshow(slice_mask)
# plt.show()

FACTOR = 2

batch_size = 1
logits_pl = tf.placeholder(dtype=tf.float32, shape=(batch_size, 288 // FACTOR, 288 // FACTOR, 22, 4))
mask_pl = tf.placeholder(dtype=tf.float32, shape=(batch_size, 288//FACTOR, 288//FACTOR, 22, 4))

pred_sm_pl = tf.nn.softmax(logits_pl, dim=-1)
# pred_sm_pl = tf.argmax(tf.nn.softmax(input_pl, dim=-1)
pred_hard_pl = tf.one_hot(tf.argmax(pred_sm_pl, axis=-1), depth=tf.shape(pred_sm_pl)[-1])

# loss = losses.foreground_dice(logits_pl, mask_pl)
# loss = losses.pixel_wise_cross_entropy_loss(input_pl, mask_pl)
loss = losses.pixel_wise_cross_entropy_loss_weighted(logits_pl, mask_pl, class_weights=[0.1, 0.3, 0.3, 0.3])

# loss = losses.pixel_wise_cross_entropy_loss_weighted(logits_pl, pred_hard_pl, class_weights=[0.1, 0.3, 0.3, 0.3])
# loss = losses.pixel_wise_cross_entropy_loss(logits_pl, pred_hard_pl)
# loss = losses.foreground_dice(input_pl, pred_hard_pl)

pred_vol = pred[1,...]
mask_vol = utils.labels_to_one_hot(mask[2,...])

pred_vol = np.reshape(pred_vol, [batch_size, 288, 288, 22, 4])
mask_vol = np.reshape(mask_vol, [batch_size, 288, 288, 22, 4])

print(pred_vol.shape)
print(mask_vol.shape)

pred_vol, mask_vol = resize_batch(pred_vol, mask_vol, scale=[1.0/FACTOR, 1.0/FACTOR, 1])



with tf.Session() as sess:

    d = sess.run([loss], feed_dict={logits_pl: pred_vol, mask_pl: mask_vol})
    # d = sess.run([loss], feed_dict={logits_pl: pred_vol})

    print(d)
