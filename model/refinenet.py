import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tqdm import tqdm

from model.layers import RefineBlock, ConditionalFullPreActivationBlock, DilatedConv2D, \
    ConditionalInstanceNormalizationPlusPlus2D


class RefineNet(keras.Model):

    def __init__(self, filters, activation):
        super(RefineNet, self).__init__()
        self.in_shape = None

        # self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')  # TODO: (1) not mentioned in the paper. Can we assume it's understood we should have it?

        # TODO: They never say what the kernel_size is (see RefineNet for this? They are using 1 and 3.)

        # TODO: (I HAVE A QUESTION HERE): "use dilated conv to replace the subsampling layers EXCEPT the first one.
        #  ==> First one should be subsampling, but it's not? So, pooling = True in the preact_1 and dilations 2, 4, 6
        #  in others (note: dilation does not change the size of the image as we thought before, so this might be ok?)
        # TODO: If the first block is subsampling, then it should not have dilated according to their description ("REPLACE").
        self.preact_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        self.preact_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)
        self.preact_3 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2, padding=2)
        self.preact_4 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4, padding=4)
        # Increasing the dilation even more would not help - WHY THOUGH? Antonio, 24/10/2019

        # TODO: THEY DON'T SAY HOW MANY RCU BLOCKS TO USE? WHY DO WE HAVE 2 HERE?
        # TODO: NUMBER OF CRP BLOCKS TAKEN FROM RefineNet.
        self.refine_block_1 = RefineBlock(activation, filters, n_blocks_crp=2, n_blocks_rcu=2)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        # self.activation = activation  # TODO: This isn't mentioned in the paper.
        self.decrease_channels = None # TODO: Neither is this, see (1).

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[0][-1], kernel_size=3, strides=1, padding='same')

    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        # x = self.increase_channels(x)

        output_1 = self.preact_1([x, idx_sigmas])
        output_2 = self.preact_2([output_1, idx_sigmas])
        output_3 = self.preact_3([output_2, idx_sigmas])
        output_4 = self.preact_4([output_3, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas])
        # output = self.activation(output)
        output = self.decrease_channels(output)

        return output

    def summary(self):
        x = [layers.Input(name="images", shape=self.in_shape[0][1:]),
             layers.Input(name="idx_sigmas", shape=(), dtype=tf.int32)]
        return keras.Model(inputs=x, outputs=self.call(x)).summary()


if __name__ == '__main__':
    import utils
    import configs
    import tensorflow_datasets as tfds

    args = utils.get_command_line_args()
    configs.config_values = args

    data_generators = tfds.load(name="mnist", split="test", batch_size=-1)
    test = tf.cast(data_generators['image'], tf.float32)

    x = test[:3]
    idx_sigmas = tf.convert_to_tensor([3, 9, 3])
    model = RefineNet(16, tf.nn.elu)
    output = model([x, idx_sigmas])
    print(model.summary())
