import models3d
import tensorflow as tf

experiment_name = 'unet_2D_as_3D_2'

batch_size = 1
learning_rate = 0.01
data_file = 'feature_maps_and_pred.hdf5'  #'allvars_288x288x24.hdf5'
model_handle = models3d.unet_bn_translated_to_3D_half
optimizer_handle = tf.train.AdamOptimizer
input_dataset = 'images'
input_channels = 1
down_sampling_factor = 2  # 1 means no down samplign, 2 means half the size (must be int)

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
