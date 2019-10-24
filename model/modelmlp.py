import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class ModelMLP(keras.Model):

    def __init__(self, activation):
        super(ModelMLP, self).__init__()

        self.mlp1 = layers.Dense(128)  # dimension of the input, here 2D GMM samples
        self.mlp2 = layers.Dense(128)
        self.mlp3 = layers.Dense(2)
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.mlp1(inputs)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.activation(x)
        x = self.mlp3(x)

        return x
