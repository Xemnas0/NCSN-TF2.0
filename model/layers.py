import tensorflow as tf
import keras.layers as layers

class ResidualConvUnit(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResidualConvUnit, self).__init__()
        # todo: add conditional instance normalization ++
        # todo: dilated?
        self.conv1 = layers.Conv2D(filters, kernel_size)
        self.conv2 = layers.Conv2D(filters, kernel_size)

    def call(self, input, **kwargs):
        x = tf.nn.elu(input)
        x = self.conv1(x)
        x = tf.nn.elu(x)
        x = self.conv2(x)
        return tf.add(x, input)

class MultiResolutionFusion(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(MultiResolutionFusion, self).__init__()
