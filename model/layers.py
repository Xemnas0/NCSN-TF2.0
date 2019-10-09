import tensorflow as tf
import keras.layers as layers

class ConditionalFullPreActivationBlock(layers.Layer):
    def __init__(self, activation, downsample, filters, kernel_size, dilation, preactivation_blocks):
        super(ConditionalFullPreActivationBlock, self).__init__()
        # todo: check why 2 preactivation_blocks blocks, does it work with 1?

        self.C = None  # todo
        self.L = None  # todo
        self.norm1 = ConditionalInstanceNormalizationPlusPlus2D(self.C, self.L)
        self.conv1 = layers.Conv2D(filters, kernel_size, dilation_rate=dilation)
        self.norm2 = ConditionalInstanceNormalizationPlusPlus2D(self.C, self.L)
        self.conv2 = layers.Conv2D(filters, kernel_size, dilation_rate=dilation)
        self.activation = activation

    def call(self, inputs, **kwargs):
        skip_x, idx_sigmas = inputs
        x = self.norm1(skip_x, idx_sigmas)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        return skip_x + x


class ConditionalInstanceNormalizationPlusPlus2D(layers.Layer):
    def __init__(self, C, L):
        pass

# class ResidualConvUnit(layers.Layer):
#     def __init__(self, filters, kernel_size=3):
#         super(ResidualConvUnit, self).__init__()
#         # todo: add conditional instance normalization ++
#         # todo: dilated?
#         self.conv1 = layers.Conv2D(filters, kernel_size)
#         self.conv2 = layers.Conv2D(filters, kernel_size)
#
#     def call(self, input, **kwargs):
#         x = tf.nn.elu(input)
#         x = self.conv1(x)
#         x = tf.nn.elu(x)
#         x = self.conv2(x)
#         return tf.add(x, input)
#
# class MultiResolutionFusion(layers.Layer):
#     def __init__(self, filters, kernel_size=3):
#         super(MultiResolutionFusion, self).__init__()
