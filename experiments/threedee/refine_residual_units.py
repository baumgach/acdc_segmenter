import models_refine3d
import tensorflow as tf

experiment_name = 'refine_residual_on_FMs_2'

batch_size = 1
learning_rate = 0.01
data_file = 'allvars_288x288x24.hdf5'  #''feature_maps_and_pred.hdf5'  # 'newdata_288x288.hdf5'
input_dataset = 'feature_maps'
down_sampling_factor = 1  # 1.0 means no down samplign, 2.0 means half the size
input_channels = 64

model_handle = models_refine3d.residual_refinement
optimizer_handle = tf.train.AdamOptimizer
# optimizer_handle = tf.train.GradientDescentOptimizer

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
