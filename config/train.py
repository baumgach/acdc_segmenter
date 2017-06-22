import tensorflow as tf

IMAGE_SIZE = (288, 288)
NLABELS = 4
MAX_EPOCHS = 20000
SCHEDULE_GRADIENT_THRESHOLD = 0.00001
EVAL_FREQUENCY = 100

# GPU settings
# gpu_memory_fraction = 1.0  # Fraction of GPU memory to use
# allow_growth = False
# gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,
#                                                       allow_growth=allow_growth))
gpu_config = tf.ConfigProto()