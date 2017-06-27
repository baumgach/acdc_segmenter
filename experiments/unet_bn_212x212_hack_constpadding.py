import model_zoo
import tensorflow as tf

experiment_name = 'unet_bn_212x212_hack_constpadding'

batch_size = 10
learning_rate = 0.01
data_file = 'betterinterp_212x212.hdf5'  # 'newdata_288x288.hdf5'
image_size = (212, 212)
model_handle = model_zoo.unet_bn_padded_hack_const_padding
optimizer_handle = tf.train.AdamOptimizer

schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'weighted_crossentropy'  # crossentropy/weighted_crossentropy/dice

# Augmentation settings
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely used settings
use_data_fraction = False  # Should normally be False


