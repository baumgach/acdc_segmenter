import models3d
import tensorflow as tf

experiment_name = 'unet_3D_224x224x24'

batch_size = 1
learning_rate = 0.01
data_file = 'data3D_224x224x24.hdf5'
image_size = (224,224,24)
model_handle = models3d.unet3D_bn
optimizer_handle = tf.train.AdamOptimizer
input_dataset = 'images'
input_channels = 1
down_sampling_factor = 1  # 1 means no down samplign, 2 means half the size (must be int)

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
