import tensorflow as tf
import keras.layers as layers


class ConditionalFullPreActivationBlock(layers.Layer):
    def __init__(self, activation, downsample, filters, kernel_size, dilation):
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
        x = self.norm1((skip_x, idx_sigmas))
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2((x, idx_sigmas))
        x = self.activation(x)
        x = self.conv2(x)

        return skip_x + x


class RCUBlock(ConditionalFullPreActivationBlock):
    def __init__(self, activation, downsample, filters, kernel_size, dilation):
        super(RCUBlock, self).__init__(activation, downsample, filters, kernel_size, dilation)


class ConditionalInstanceNormalizationPlusPlus2D(layers.Layer):
    def __init__(self, L):
        super(ConditionalInstanceNormalizationPlusPlus2D, self).__init__()
        self.L = L

    def build(self, input_shape):
        self.C = input_shape[1]  # FIXME: I might not be what you think I am. Zero?
        self.alpha = self.add_weight(shape=[self.L, self.C])
        self.beta = self.add_weight(shape=[self.L, self.C])
        self.gamma = self.add_weight(shape=[self.L, self.C])

    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        mu, s = tf.nn.moments(x, axes=[1])  # FIXME: I might not be what you think I am. One?
        m, v = tf.nn.moments(mu, axes=[0])

        # FIXME: probably will break
        first = self.gamma[idx_sigmas, :] * (x - mu) / s
        second = self.beta[idx_sigmas, :]
        third = self.alpha[idx_sigmas, :] * (mu - m) / v

        z = first + second + third

        return z


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
class MultiResolutionFusion(layers.Layer):
    def __init__(self, filters, kernel_size=3, has_low_input=True):
        super(MultiResolutionFusion, self).__init__()

        self.has_low_input = has_low_input

        if has_low_input:
            self.conv2d_low = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv2d_high = layers.Conv2D(filters, kernel_size, padding='same')

    def call(self, inputs, **kwargs):
        if len(inputs) == 1:
            high_input = inputs
            x = self.conv2d_high(high_input)
            return x
        elif len(inputs) == 2:
            high_input, low_input = inputs

            # FIXME: make me any beautiful
            upsample = layers.UpSampling2D(high_input.shape)

            low_input = self.conv2d_low(low_input)
            low_input = upsample(low_input)
            high_input = self.conv2d_high(high_input)

            return low_input + high_input
