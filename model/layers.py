import tensorflow as tf
import tensorflow.keras.layers as layers  # TODO: check if we should use keras or tf.keras
import configs


class DilatedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation=1, padding=1, strides=1):
        super(DilatedConv2D, self).__init__()

        self.padding = layers.ZeroPadding2D(padding)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation, padding='valid')
        self.filters = filters

    # def compute_output_shape(self, input_shape):
    #     # print(f"input shape: {input_shape}")
    #
    #     output_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters)
    #     return output_shape

    def call(self, inputs, **kwargs):
        x = self.padding(inputs)
        x = self.conv(x)
        return x


class ConditionalFullPreActivationBlock(layers.Layer):
    def __init__(self, activation, filters, kernel_size=3, dilation=1, padding=1, pool_size=0):
        super(ConditionalFullPreActivationBlock, self).__init__()
        # todo: check why 2 preactivation_blocks blocks, does it work with 1?

        self.norm1 = ConditionalInstanceNormalizationPlusPlus2D()
        self.conv1 = DilatedConv2D(filters, kernel_size, dilation, padding)
        self.norm2 = ConditionalInstanceNormalizationPlusPlus2D()
        self.conv2 = DilatedConv2D(filters, kernel_size, dilation, padding)
        self.pooling_size = pool_size
        self.activation = activation

        self.increase_channels_skip = layers.Conv2D(filters, kernel_size=1, padding='same')

        self.filters = filters

    # def build(self, input_shape):
    #     print(f"Output shape: {self.compute_output_shape(input_shape)}")

    # def compute_output_shape(self, input_shape):
    #     print("input shape residual block: "+str(input_shape))
    #     output_shape = (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.filters)
    #     return output_shape

    def call(self, inputs, **kwargs):
        skip_x, idx_sigmas = inputs
        x = self.norm1([skip_x, idx_sigmas])
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2([x, idx_sigmas])
        x = self.activation(x)
        x = self.conv2(x)

        if x.shape != skip_x.shape:
            skip_x = self.increase_channels_skip(skip_x)

        if self.pooling_size > 0:
            x = tf.nn.avg_pool2d(x, self.pooling_size, strides=1, padding='SAME')
            skip_x = tf.nn.avg_pool2d(skip_x, self.pooling_size, strides=1, padding='SAME')

        return skip_x + x


class RCUBlock(ConditionalFullPreActivationBlock):
    def __init__(self, activation, filters, kernel_size=3, dilation=1):
        super(RCUBlock, self).__init__(activation, filters, kernel_size, dilation)


class ConditionalInstanceNormalizationPlusPlus2D(layers.Layer):
    def __init__(self):
        super(ConditionalInstanceNormalizationPlusPlus2D, self).__init__()
        self.L = configs.config_values.num_L

        self.init_weights = 'random_normal'  # tf.initializers.RandomNormal(1, 0.02)
        self.init_bias = 'zeros'

    def build(self, input_shape):
        self.C = input_shape[0][-1]  # FIXME: I might not be what you think I am. Zero?
        self.alpha = self.add_weight(shape=(self.L, 1, 1, self.C),
                                     initializer=self.init_weights)  # TODO: maybe change init
        self.beta = self.add_weight(shape=(self.L, 1, 1, self.C), initializer=self.init_bias)
        self.gamma = self.add_weight(shape=(self.L, 1, 1, self.C), initializer=self.init_weights)

        super(ConditionalInstanceNormalizationPlusPlus2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        mu, s = tf.nn.moments(x, axes=[-1], keepdims=True)  # FIXME: I might not be what you think I am. One?
        m, v = tf.nn.moments(mu, axes=[0])

        # FIXED (maybe)
        first = tf.gather(self.gamma, idx_sigmas) * (x - mu) / s
        second = tf.gather(self.beta, idx_sigmas)
        third = tf.gather(self.alpha, idx_sigmas) * (mu - m) / v

        z = first + second + third

        return z


class ConditionalChainedResidualPooling2D(layers.Layer):
    def __init__(self, n_blocks, activation, filters, kernel_size=3, pooling_size=5):
        super(ConditionalChainedResidualPooling2D, self).__init__()
        self.activation1 = activation
        self.n_blocks = n_blocks
        self.pooling_size = pooling_size
        for n in range(n_blocks):
            setattr(self, f'norm{n}', ConditionalInstanceNormalizationPlusPlus2D())
            setattr(self, f'conv{n}', layers.Conv2D(filters, kernel_size, padding='same'))
        # self.pooling = layers.AveragePooling2D(pooling_size, padding='same')

    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        x_residual = self.activation1(x)
        x = x_residual
        for n in range(self.n_blocks):
            norm = getattr(self, f'norm{n}')
            conv = getattr(self, f'conv{n}')
            x = norm([x, idx_sigmas])
            # x = self.pooling(x)
            x = tf.nn.avg_pool2d(x, self.pooling_size, strides=1, padding='SAME')
            x = conv(x)
            x_residual += x
        return x_residual


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
    def __init__(self, filters, kernel_size=3):
        super(MultiResolutionFusion, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv2d_high = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.norm_high = ConditionalInstanceNormalizationPlusPlus2D()
        self.conv2d_low = None
        self.norm_low = None

    def build(self, input_shape):
        if len(input_shape[0]) == 2:
            self.norm_low = ConditionalInstanceNormalizationPlusPlus2D()
            self.conv2d_low = layers.Conv2D(self.filters, self.kernel_size, padding='same')

    def call(self, inputs, **kwargs):
        idx_sigmas = inputs[1]
        if len(inputs[0]) == 1:
            high_input = inputs[0][0]
            x = self.norm_high([high_input, idx_sigmas])
            x = self.conv2d_high(x)
            return x
        elif len(inputs[0]) == 2:
            high_input, low_input = inputs[0]

            # FIXME: make me any beautiful
            # upsample = layers.UpSampling2D(high_input.shape[1:-1])

            low_input = self.norm_low([low_input, idx_sigmas])
            low_input = self.conv2d_low(low_input)
            # low_input = upsample(low_input)
            low_input = tf.image.resize(low_input, high_input.shape[1:-1])
            high_input = self.norm_high([high_input, idx_sigmas])
            high_input = self.conv2d_high(high_input)

            return low_input + high_input


class RefineBlock(layers.Layer):
    def __init__(self, n_blocks_crp, activation, filters, kernel_size=3, pooling_size=5):
        super(RefineBlock, self).__init__()

        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size

        # NOTE: they use 2 block, we use 1 for now
        self.rcu_high = None
        self.rcu_low = None

        self.mrf = MultiResolutionFusion(filters, kernel_size)

        self.crp = ConditionalChainedResidualPooling2D(n_blocks_crp, activation, filters, kernel_size, pooling_size)

        self.rcu_end = RCUBlock(activation, filters, kernel_size)

    def build(self, input_shape):
        self.rcu_high = RCUBlock(self.activation, self.filters, self.kernel_size)
        if len(input_shape) == 2:
            self.rcu_low = RCUBlock(self.activation, self.filters, self.kernel_size)

    def call(self, inputs, **kwargs):
        idx_sigmas = inputs[1]
        if len(inputs[0]) == 1:
            high_input = inputs[0][0]

            high_input = self.rcu_high([high_input, idx_sigmas])
            x = self.mrf([[high_input], idx_sigmas])

        elif len(inputs[0]) == 2:
            high_input, low_input = inputs[0]

            high_input = self.rcu_high([high_input, idx_sigmas])
            low_input = self.rcu_low([low_input, idx_sigmas])
            x = self.mrf([[high_input, low_input], idx_sigmas])

        x = self.crp([x, idx_sigmas])
        x = self.rcu_end([x, idx_sigmas])
        return x

# class TestLayer(layers.Layer):
#     def __init__(self):
#         super(TestLayer, self).__init__()
#
#     def build(self, input_shape):
#         print(input_shape)
#
#     def call(self, inputs, **kwargs):
#         return inputs
#
#
# if __name__ == '__main__':
#     layer = TestLayer()
#     x = [tf.convert_to_tensor(list(range(10))), None]
#     output = layer(x)
