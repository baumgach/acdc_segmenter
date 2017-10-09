# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np

def flatten(tensor):
    '''
    Flatten the last N-1 dimensions of a tensor only keeping the first one, which is typically 
    equal to the number of batches. 
    Example: A tensor of shape [10, 200, 200, 32] becomes [10, 1280000] 
    '''
    rhs_dim = get_rhs_dim(tensor)
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    '''
    Get the multiplied dimensions of the last N-1 dimensions of a tensor. 
    I.e. an input tensor with shape [10, 200, 200, 32] leads to an output of 1280000 
    '''
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])

