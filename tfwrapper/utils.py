import tensorflow as tf
import numpy as np

def flatten(tensor):

    rhs_dim = get_rhs_dim(tensor)
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])

def to_onehot_image(labels, n_class=4):

    n_x, n_y = labels.shape
    onehot = np.zeros((n_x, n_y, n_class))

    for cc in range(n_class):
        mask_cc = (labels == cc).astype(int)
        onehot[:,:,cc] = mask_cc

    return onehot

def labels_to_one_hot(labels, n_class=4):

    N = labels.shape[0]
    out_labels = []
    for ii in range(N):
        out_labels.append(to_onehot_image(labels[ii, ...], n_class))
    return np.asarray(out_labels, dtype=np.float32)