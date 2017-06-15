import model_zoo
import tensorflow as tf


### GLOBAL TRAINING SETTINGS: #####################################################
BATCH_SIZE = 16
LEARNING_RATE = 0.01
DATA_FILE = 'data_288x288.hdf5'  #'data_288x288_plusregs.hdf5'
# DATA_FILE = 'data_288x288_plusregs.hdf5'
MODEL_HANDLE = model_zoo.lisa_net_deeper_bn
MODEL_HANDLE = model_zoo.SonoNet32
# MODEL_HANDLE = model_zoo.lisa_net_one_more_pool
# MODEL_HANDLE = model_zoo.lisa_net_3pool_stack_convs
OPTIMIZER_HANDLE = tf.train.AdamOptimizer
# OPTIMIZER_HANDLE = tf.train.GradientDescentOptimizer
SCHEDULE_LR = False
WARMUP_TRAINING = False
AUGMENT_BATCH = True
WEIGHT_DECAY = 0.00005
MOMENTUM = None  #0.9

### GLOBAL CONSTANTS: #############################################################
IMAGE_SIZE = (288, 288)
NLABELS = 4
MAX_EPOCHS = 20000
SCHEDULE_GRADIENT_THRESHOLD = 0.00001
EVAL_FREQUENCY = 100
###################################################################################


