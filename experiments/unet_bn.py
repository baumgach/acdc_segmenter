import model_zoo
import tensorflow as tf

experiment_name = 'unet_bn_unw_xent'

batch_size = 10
learning_rate = 0.01
data_file = 'data_288x288.hdf5'
model_handle = model_zoo.unet_bn
optimizer_handle = tf.train.AdamOptimizer

schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'crossentropy'  # crossentropy/weighted_crossentropy/dice

# Augmentation settings
augment_batch = False
do_rotations = True
do_scaleaug = False
