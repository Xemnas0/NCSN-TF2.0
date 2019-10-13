import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils
import configs
from datasets.dataset_loader import get_data_generator
from model.refinenet import RefineNet
from losses.losses import loss_per_batch
from tqdm import tqdm

def train(model, inputs, learning_rate):
    pass

if __name__ == "__main__":
    args = utils.get_command_line_args()
    configs.config_values = args

    # loading dataset
    train_data = get_data_generator(args.dataset)
    num_batches = tf.data.experimental.cardinality(train_data)

    # initialize model
    model = RefineNet(filters=5, activation=tf.nn.elu)

    # declare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # 10 times larger than in their paper

    # TRAINING LOOP CAN GO HERE
    epochs = 3

    # array of sigma levles
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(args.sigma_low), tf.math.log(args.sigma_high), args.num_L))

    print("=========================================")
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_data))
        progress_bar.set_description(f'loss {0:.3f}')
        for i, data_batch in progress_bar:
            idx_sigmas = tf.random.uniform([data_batch.shape[0]], minval=0, maxval=args.num_L, dtype=tf.dtypes.int32)
            sigmas = tf.gather(sigma_levels, idx_sigmas)
            sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
            data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas

            with tf.GradientTape() as t:
                scores = model([data_batch_perturbed, idx_sigmas])
                batch_loss = loss_per_batch(scores, data_batch_perturbed, data_batch, sigmas)
                gradients = t.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            progress_bar.set_description(f'loss {batch_loss:.3f}')
    print("=========================================")

    # NOTE bad way to choose the best model - saving all checkpoints and then testing after