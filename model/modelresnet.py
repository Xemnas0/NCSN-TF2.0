import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class ModelResNet(keras.Model):

    def __init__(self, activation):
        super(ModelResNet, self).__init__()

        self.increase_channels = layers.Conv2D(32, kernel_size=3, padding='same')

        self.res_enc1 = ResidualBlock(activation, 32, is_encoder=True)
        self.res_enc2 = ResidualBlock(activation, 64, is_encoder=True, resize=True)
        self.res_enc3 = ResidualBlock(activation, 64, is_encoder=True)
        self.res_enc4 = ResidualBlock(activation, 128, is_encoder=True, resize=True)
        self.res_enc5 = ResidualBlock(activation, 128, is_encoder=True)

        self.res_dec1 = ResidualBlock(activation, 128, is_encoder=False)
        self.res_dec2 = ResidualBlock(activation, 128, is_encoder=False, resize=True)
        self.res_dec3 = ResidualBlock(activation, 64, is_encoder=False)
        self.res_dec4 = ResidualBlock(activation, 64, is_encoder=False, resize=True)
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

        # FIXME: THEY USE GROUP NORMALIZATION, NOT SURE IF THIS MAKES A DIFFERENCE?
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

        self.increase_channels_skip = layers.Conv2D(filters, kernel_size=1, strides=strides, padding=padding)

    def build(self, input_shape):
        self.increase_skip_size = layers.Conv2D(self.filters, kernel_size=1, strides=2)
        self.decrease_skip_size = layers.Conv2DTranspose(self.filters, kernel_size=1, strides=self.strides)

    def call(self, inputs, **kwargs):
        # TODO: CHECK IF INPUTS NEED TO BE SPLIT ?
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if x.shape != inputs.shape:
            if self.is_encoder:
                skip_x = self.increase_channels_skip(inputs)
            else:
                skip_x = self.decrease_skip_size(inputs)
        else:
            skip_x = inputs

        return skip_x + x
