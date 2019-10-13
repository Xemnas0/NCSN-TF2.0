import tensorflow as tf
import keras
import keras.layers as layers

from model.layers import RefineBlock, ConditionalFullPreActivationBlock, DilatedConv2D, ConditionalInstanceNormalizationPlusPlus2D

class RefineNet(keras.Model):

    def __init__(self, filters, activation):
        super(RefineNet, self).__init__()

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3, pool_size=7) #FIXME: check pool size
        self.preact_2 = ConditionalFullPreActivationBlock(activation, filters*2, kernel_size=3)
        self.preact_3 = ConditionalFullPreActivationBlock(activation, filters*2, kernel_size=3, dilation=2, padding=2)
        self.preact_4 = ConditionalFullPreActivationBlock(activation, filters*2, kernel_size=3, dilation=4, padding=4)
        # Increasing the dilation even more would not help

        self.refine_block_1 = RefineBlock(2, activation, filters)
        self.refine_block_2 = RefineBlock(2, activation, filters*2)
        self.refine_block_3 = RefineBlock(2, activation, filters*2)
        self.refine_block_4 = RefineBlock(2, activation, filters*2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.decrease_channels = layers.Conv2D(input_shape[1], kernel_size=3, stride=1, padding=1)

    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs

        x = self.increase_channels(inputs)

        output_1 = self.preact_1([x, idx_sigmas])
        output_2 = self.preact_2([output_1, idx_sigmas])
        output_3 = self.preact_3([output_2, idx_sigmas])
        output_4 = self.preact_4([output_3, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas])
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output



