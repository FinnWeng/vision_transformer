import math
import functools

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape

def weight_standardize(w, axis, eps):
  """Subtracts mean and divides by standard deviation."""
  w = w - tf.math.reduce_mean(w, axis=axis)
  w = w / (tf.math.reduce_std(w, axis=axis) + eps)
  return w

class Std_Conv(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Std_Conv, self).__init__()
        self.weight_standardized = False
        self.conv1 = tf.keras.layers.Conv2D(*args, **kwargs)
    def call(self, input):
        if self.weight_standardized == False:
            self.conv1(input)
            old_w = self.conv1.get_weights()
            new_kernel_w = weight_standardize(old_w[0],axis=[0,1,2],eps = 1e-5) 
            # print(old_w[0])
            # print(new_kernel_w)
            self.conv1.set_weights(old_w)
            self.weight_standardized = True
            output = self.conv1(input)
            # print(self.conv1.get_weights()[0])

        else:
            output = self.conv1(input)
        # print("final kernel w", self.conv1.get_weights()[0])
        return output


        
    

if __name__ == "__main__":
    conv1 = Std_Conv(1,2,1)
    a = tf.ones([5,256,256,3])
    b = conv1(a)
    b = conv1(a) # check if the kernel modify twice 