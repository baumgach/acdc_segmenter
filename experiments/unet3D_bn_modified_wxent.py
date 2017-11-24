import model_zoo
import tensorflow as tf

experiment_name = 'unet3D_bn_modified_wxent_tfbn'

# Model settings
model_handle = model_zoo.unet3D_bn_modified

# Data settings
data_mode = '3D'  # 2D or 3D
image_size = (116, 116, 28)
target_resolution = (2.5, 2.5, 5.0)
nlabels = 4

# Training settings
batch_size = 1
learning_rate = 0.01
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'weighted_crossentropy'  # crossentropy/weighted_crossentropy/dice/dice_onlyfg

# Augmentation settings
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 20000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
