import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from model.layers import ConditionalInstanceNormalizationPlusPlus2D

class ResNet(keras.Model):
    def __init__(self, filters, activation):
        super(ResNet, self).__init__()

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.res_enc1 = ConditionalResidualBlock(activation, filters, is_encoder=True)
        self.res_enc2 = ConditionalResidualBlock(activation, filters * 2, dilation=2, is_encoder=True, resize=True)
        self.res_enc3 = ConditionalResidualBlock(activation, filters * 2, dilation=2, is_encoder=True)
        self.res_enc4 = ConditionalResidualBlock(activation, filters * 4, dilation=2, is_encoder=True, resize=True)
        self.res_enc5 = ConditionalResidualBlock(activation, filters * 4, dilation=2, is_encoder=True)

        self.res_dec1 = ConditionalResidualBlock(activation, filters * 4, dilation=2, is_encoder=False)
        self.res_dec2 = ConditionalResidualBlock(activation, filters * 4, dilation=2, is_encoder=False, resize=True)
        self.res_dec3 = ConditionalResidualBlock(activation, filters * 2, dilation=2, is_encoder=False)
        self.res_dec4 = ConditionalResidualBlock(activation, filters * 2, dilation=2, is_encoder=False, resize=True)
        self.res_dec5 = ConditionalResidualBlock(activation, filters, is_encoder=False)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[0][-1], kernel_size=3, strides=1, padding='same')

    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        x = self.increase_channels(x)

        x = self.res_enc1([x, idx_sigmas])
        x = self.res_enc2([x, idx_sigmas])
        x = self.res_enc3([x, idx_sigmas])
        x = self.res_enc4([x, idx_sigmas])
        x = self.res_enc5([x, idx_sigmas])

        x = self.res_dec1([x, idx_sigmas])
        x = self.res_dec2([x, idx_sigmas])
        x = self.res_dec3([x, idx_sigmas])
        x = self.res_dec4([x, idx_sigmas])
        x = self.res_dec5([x, idx_sigmas])

        output = self.norm([x, idx_sigmas])
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output


class ConditionalResidualBlock(layers.Layer):
    def __init__(self, activation, filters, is_encoder, kernel_size=3, dilation=1, resize=False):
        super(ConditionalResidualBlock, self).__init__()

        self.norm1 = ConditionalInstanceNormalizationPlusPlus2D()
        self.norm2 = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.resize = resize
        self.filters = filters
        self.is_encoder = is_encoder
        if is_encoder:
            self.conv1 = layers.Conv2D(filters, kernel_size, dilation_rate=(dilation, dilation), padding="same")
            self.conv2 = layers.Conv2D(filters, kernel_size, dilation_rate=(dilation, dilation), padding="same")
        else:
            self.conv1 = layers.Conv2DTranspose(filters, kernel_size, dilation_rate=(dilation, dilation),
                                                padding="same")
            self.conv2 = layers.Conv2DTranspose(filters, kernel_size, dilation_rate=(dilation, dilation),
                                                padding="same")
        self.adjust_skip = None

    def build(self, input_shape):
        # we might have different number of filters and/or dimensions after the block,
        # so we need to adjust the skip connection to match by a 1x1 convolution
        begin_filters = input_shape[0][-1]
        if (begin_filters != self.filters):
            if self.is_encoder:
                self.adjust_skip = layers.Conv2D(self.filters, kernel_size=1, padding='same')
            else:
                self.adjust_skip = layers.Conv2DTranspose(self.filters, kernel_size=1, padding='same')

    def call(self, inputs, **kwargs):
        skip_x, idx_sigmas = inputs
        x = self.norm1([skip_x, idx_sigmas])
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2([x, idx_sigmas])
        x = self.activation(x)
        x = self.conv2(x)

        if self.adjust_skip is not None:
            skip_x = self.adjust_skip(skip_x)

        if self.resize:
            if self.is_encoder:
                x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
                skip_x = tf.nn.avg_pool2d(skip_x, ksize=2, strides=2, padding='SAME')
            else:
                x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2))
                skip_x = tf.image.resize(skip_x, (x.shape[1], x.shape[2]))

        return skip_x + x


class ToyResNet(keras.Model):

    def __init__(self, activation):
        super(ToyResNet, self).__init__()

        self.increase_channels = layers.Conv2D(32, kernel_size=3, padding='same')

        self.res_enc1 = ResidualBlock(activation, 32, is_encoder=True)
        self.res_enc2 = ResidualBlock(activation, 64, is_encoder=True, resize=True)
        self.res_enc3 = ResidualBlock(activation, 64, is_encoder=True)
        self.res_enc4 = ResidualBlock(activation, 128, is_encoder=True, resize=True)
        self.res_enc5 = ResidualBlock(activation, 128, is_encoder=True)

        self.res_dec1 = ResidualBlock(activation, 128, is_encoder=False)
        self.res_dec2 = ResidualBlock(activation, 64, is_encoder=False, resize=True)
        self.res_dec3 = ResidualBlock(activation, 64, is_encoder=False)
        self.res_dec4 = ResidualBlock(activation, 32, is_encoder=False, resize=True)
        self.res_dec5 = ResidualBlock(activation, 32, is_encoder=False)

        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[-1], kernel_size=3, strides=1, padding='same')

    def call(self, inputs, mask=None):
        inputs = inputs
        out = self.increase_channels(inputs)

        out = self.res_enc1(out)
        out = self.res_enc2(out)
        out = self.res_enc3(out)
        out = self.res_enc4(out)
        out = self.res_enc5(out)

        out = self.res_dec1(out)
        out = self.res_dec2(out)
        out = self.res_dec3(out)
        out = self.res_dec4(out)
        out = self.res_dec5(out)

        out = self.decrease_channels(out)

        return out


class ResidualBlock(layers.Layer):
    def __init__(self, activation, filters, is_encoder, kernel_size=3, resize=False):
        super(ResidualBlock, self).__init__()

        # FIXME: THEY DON'T MENTION WHAT KIND OF NORMALIZATION IS USED, I ASSUMED BN, BUT THEY USE GROUP NORMALIZATION, NOT SURE IF THIS MAKES A DIFFERENCE?
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
        self.activation = activation

        self.filters = filters
        self.is_encoder = is_encoder
        self.resize = resize
        self.strides = strides = 2 if resize else 1
        padding = 'same'  # if strides == 1 else 'valid'

        if is_encoder:
            self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
            self.conv2 = layers.Conv2D(filters, kernel_size, strides=1, padding='same')
        else:
            self.conv1 = layers.Conv2DTranspose(filters, kernel_size, strides=1, padding='same')
            self.conv2 = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def build(self, input_shape):
        self.increase_skip_size = layers.Conv2D(self.filters, kernel_size=1, strides=2, padding='same')
        self.decrease_skip_size = layers.Conv2DTranspose(self.filters, kernel_size=1, strides=self.strides)

    def call(self, inputs, **kwargs):
        # x = self.norm1(inputs)
        # x = self.activation(x)
        # x = self.conv1(x)
        # x = self.norm2(x)
        # x = self.activation(x)
        # x = self.conv2(x)
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if x.shape != inputs.shape:
            if self.is_encoder:
                skip_x = self.increase_skip_size(inputs)
            else:
                skip_x = self.decrease_skip_size(inputs)
        else:
            skip_x = inputs

        return self.activation(skip_x + x)
