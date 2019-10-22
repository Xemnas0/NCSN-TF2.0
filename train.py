import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils, os
from tqdm import tqdm
from datetime import datetime
from model.inception import compute_inception_score
import cProfile

# our files
from datasets.dataset_loader import get_train_test_data
from model.refinenet import RefineNet
from losses.losses import loss_per_batch, loss_per_batch_alternative
from generate import sample_many
import configs
from generate import plot_grayscale

def manage_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


manage_gpu_memory_usage()

device = utils.get_tensorflow_device()


def print_model_summary(model):
    batch = 2
    if configs.config_values.dataset in ["cifar10", "celeb_a"]:
        x = [tf.ones(shape=(batch, 32, 32, 3)), tf.ones(batch, dtype=tf.int32)]
    else:
        x = [tf.ones(shape=(batch, 28, 28, 1)), tf.ones(batch, dtype=tf.int32)]
    out = model(x)
    print(model.summary())


@tf.function
def train_one_step(model, optimizer, data_batch_perturbed, data_batch, idx_sigmas, sigmas):
    with tf.GradientTape() as t:
        scores = model([data_batch_perturbed, idx_sigmas])
        current_loss = loss_per_batch_alternative(scores, data_batch_perturbed, data_batch, sigmas)
        gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss

def train():   
    # load dataset from tfds (or use downloaded version if exists)
    train_data, test_data = get_train_test_data(configs.config_values.dataset)
    num_examples = int(tf.data.experimental.cardinality(train_data))

    # split data into batches
    train_data = train_data.shuffle(1000).batch(configs.config_values.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(configs.config_values.batch_size)

    num_batches = int(tf.data.experimental.cardinality(train_data))
    num_filters = {'mnist': 64, 'cifar10': 64, 'celeb_a': 128} # NOTE change mnist back to 64

    # path for saving the model(s)
    save_dir = configs.config_values.checkpoint_dir + configs.config_values.dataset + '/'
    # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)

    start_time = datetime.now().strftime("%y%m%d-%H%M")
    log_dir = configs.config_values.log_dir + configs.config_values.dataset + "/"
    summary_writer = tf.summary.create_file_writer(log_dir+start_time)

    # initialize models
    model = RefineNet(filters=num_filters[configs.config_values.dataset], activation=tf.nn.elu)
    print_model_summary(model)

    # declare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # NOTE 10 times larger than in their paper

    # if resuming training, overwrite model parameters from checkpoint
    if configs.config_values.resume:
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        print("loading model from checkpoint ", latest_checkpoint)
        step = tf.Variable(0)
        ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
        ckpt.restore(latest_checkpoint)
        step = int(step)
    else:
        step = 0

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    sigma_levels = tf.math.exp(tf.linspace( tf.math.log(configs.config_values.sigma_high),
                                            tf.math.log(configs.config_values.sigma_low),
                                            configs.config_values.num_L ))

    # training loop
    print(f'dataset: {configs.config_values.dataset}, '
          f'number of examples: {num_examples}, '
          f'batch size: {configs.config_values.batch_size}\n'
          f'training...')

    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step+1)
    progress_bar.set_description(f'iteration {step}/{total_steps} | current loss ?')

    with tf.device(device): # For some reason, this makes everything faster
        avg_loss = 0
        for data_batch in progress_bar:
            step += 1
            idx_sigmas = tf.random.uniform([data_batch.shape[0]], minval=0,
                                                maxval=configs.config_values.num_L,
                                                dtype=tf.dtypes.int32)
            sigmas = tf.gather(sigma_levels, idx_sigmas)
            sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
            data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas

            current_loss = train_one_step(model, optimizer, data_batch_perturbed, data_batch, idx_sigmas, sigmas)

            tf.summary.scalar('loss', float(current_loss), step=int(step))

            progress_bar.set_description(f'iteration {step}/{total_steps} | current loss {current_loss:.3f}')

            avg_loss += current_loss
            if step % configs.config_values.checkpoint_freq == 0:
                # TODO: maybe save also info about the sigmas
                ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
                ckpt.step.assign_add(step)
                ckpt.save(save_dir+f"{start_time}_step_{step}")
                print(f"\nSaved checkpoint. Average loss: {avg_loss/configs.config_values.checkpoint_freq:.3f}")
                avg_loss = 0
            if step == total_steps:
                return

            # Compute inception score mean and standard deviation
            images = sample_many(model, sigma_levels, n_images=1000)
            compute_inception_score(images, image_side_inception=199)
            # images = model.sample_and_save(sigma_levels, n_images=1000, T=100)
            # is_mean, is_stddev = inception.inception_score(images)

    # NOTE bad way to choose the best model - saving all checkpoints and then testing after

if __name__ == "__main__":

    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    train()
