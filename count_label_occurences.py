import tensorflow as tf
import model as model
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import model_zoo
import tfwrapper.utils as tf_utils
import image_utils
import cv2

import utils


data = h5py.File('betterinterp_212x212.hdf5', 'r')

labels_val = data['masks_train']

n_images = labels_val.shape[0]

label_occurences = {0: 0, 1: 0, 2: 0, 3: 0}

for ii in range(n_images):

    curr_lbl = np.squeeze(labels_val[ii,...])

    for lbl in [0,1,2,3]:

        label_occurences[lbl] += (curr_lbl == lbl).sum()


total_occurences = sum(label_occurences.values())

sm = 0
for lbl in [0,1,2,3]:

    percentage = float(label_occurences[lbl]) / total_occurences

    print('percentage label %d: %f' % (lbl, percentage))
    #print('lbl %d: %f' % (lbl, 1.0 / percentage))
    sm += 1.0 / percentage

print(sm)

for lbl in [0,1,2,3]:

    percentage = float(label_occurences[lbl]) / total_occurences
    print('lbl %d: %f' % (lbl, (1.0 / percentage) / sm))