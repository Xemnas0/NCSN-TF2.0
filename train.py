import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils, os
from tqdm import tqdm

# our files
from datasets.dataset_loader import get_train_test_data
from model.refinenet import RefineNet
from losses.losses import loss_per_batch
import configs


def train():   
    # load dataset from tfds (or use downloaded version if exists)
    train_data, test_data = get_train_test_data(configs.config_values.dataset)
    num_examples = int(tf.data.experimental.cardinality(train_data))

    # split data into batches
    train_data = train_data.shuffle(1000).batch(configs.config_values.batch_size)
    test_data = test_data.batch(configs.config_values.batch_size)

    num_batches = int(tf.data.experimental.cardinality(train_data))
    num_filters = {'mnist': 16, 'cifar10': 128, 'celeb_a': 128} # NOTE change mnist back to 64

    # path for saving the model(s)
    save_dir = configs.config_values.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize model
    model = RefineNet(filters=num_filters[configs.config_values.dataset], activation=tf.nn.elu)

    # declare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # NOTE 10 times larger than in their paper

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    sigma_levels = tf.math.exp(tf.linspace( tf.math.log(configs.config_values.sigma_low), 
                                            tf.math.log(configs.config_values.sigma_high), 
                                            args.num_L ))

    # training loop
    print(f'dataset: {configs.config_values.dataset}, '
          f'number of examples: {num_examples}, '
          f'batch size: {configs.config_values.batch_size}\n'
          f'training...')
    
    epochs = configs.config_values.epochs
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_data), total=num_batches)
        progress_bar.set_description(f'epoch {epoch}/{epochs} | '
                                     f'current loss ? | average loss ?')

        total_loss = 0
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

            total_loss += batch_loss
            progress_bar.set_description(f'epoch {epoch}/{epochs} | '
                f'current loss {batch_loss:.3f} | average loss {total_loss/(i+1):.3f}')

            if epoch % configs.config_values.checkpoint_freq == 0:
                # TODO: maybe save also info about the sigmas
                model.save_weights(save_dir+f'refinenet_{configs.config_values.dataset}_epoch{epoch}.h5', 
                    overwrite=False, save_format='h5')
                print("Model saved successfully! Congratulations! Go celebrate.")

    # NOTE bad way to choose the best model - saving all checkpoints and then testing after

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    train()
    