import math
import functools

import tensorflow as tf
import tensorflow_addons as tfa

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

class Residual_Unit(tf.keras.Model):
    """Bottleneck ResNet block."""
    def __init__(self, features, strides, name=None):
        super(Residual_Unit, self).__init__(name)

        self.features = features
        self.strides = strides
    
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    
    def call(self, x):
        needs_projection = (
            x.shape[-1] != self.features * 4 or self.strides != (1, 1))

        residual = x
        if needs_projection:
            residual = self.get("conv_proj",Std_Conv, self.features*4, (1,1),self.strides, 
                padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(residual)
            residual = self.get(tfa.layers.GroupNormalization, name='gn_proj', epsilon = 1e-6)(residual)

        y = self.get("conv1",Std_Conv, self.features, (1,1), 
                padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)
            
        y = self.get(tfa.layers.GroupNormalization, name='gn1', epsilon = 1e-6)(y)
        y = tf.keras.layers.ReLU()(y)

        y = self.get("conv2",Std_Conv, self.features, (3,3),self.strides, 
            padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)
        
        y = self.get(tfa.layers.GroupNormalization, name='gn2', epsilon = 1e-6)(y)
        y = tf.keras.layers.ReLU()(y)

        y = self.get("conv3",Std_Conv, self.features*4, (1,1), 
            padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)

        y = self.get(tfa.layers.GroupNormalization, name='gn3', epsilon = 1e-6, gamma_initializer= tf.keras.initializers.Zeros())(y)
        y = tf.keras.layers.ReLU()(residual + y)
        return y


class Res_Net_Stage(tf.keras.Model):
    """A ResNet stage."""

    def __init__(self, block_size, nout, first_stride, name=None):
        super(Res_Net_Stage, self).__init__(name)
        self.block_size = block_size
        self.nout = nout
        self.first_stride = first_stride
    
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    
    def __call__(self, x):
        x = self.get(Residual_Unit, self.nout, strides=self.first_stride, name='unit1')(x)
        for i in range(1, self.block_size):
            x = self.get(Residual_Unit, self.nout, strides=(1, 1), name=f'unit{i + 1}')(x)
        return x

        
    

if __name__ == "__main__":
    conv1 = Std_Conv(1,2,1)
    a = tf.ones([5,256,256,3])
    b = conv1(a)
    b = conv1(a) # check if the kernel modify twice 