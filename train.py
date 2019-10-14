import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils, os
from tqdm import tqdm
from datetime import datetime

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
    train_data = train_data.shuffle(1000).batch(configs.config_values.batch_size).repeat()
    test_data = test_data.batch(configs.config_values.batch_size)

    num_batches = int(tf.data.experimental.cardinality(train_data))
    num_filters = {'mnist': 16, 'cifar10': 128, 'celeb_a': 128} # NOTE change mnist back to 64

    # path for saving the model(s)
    save_dir = configs.config_values.checkpoint_dir + configs.config_values.dataset
    # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)

    start_time = datetime.now().strftime("%y%m%d-%H%M")
    log_dir = configs.config_values.log_dir + configs.config_values.dataset
    summary_writer = tf.summary.create_file_writer(log_dir+start_time)

    # initialize model
    model = RefineNet(filters=num_filters[configs.config_values.dataset], activation=tf.nn.elu)
    
    # declare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # NOTE 10 times larger than in their paper

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
    sigma_levels = tf.math.exp(tf.linspace( tf.math.log(configs.config_values.sigma_low), 
                                            tf.math.log(configs.config_values.sigma_high), 
                                            configs.config_values.num_L ))

    # training loop
    print(f'dataset: {configs.config_values.dataset}, '
          f'number of examples: {num_examples}, '
          f'batch size: {configs.config_values.batch_size}\n'
          f'training...')
    
    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step+1)
    progress_bar.set_description(f'iteration {step}/{total_steps} | current loss ?')

    for data_batch in progress_bar:
        step += 1
        idx_sigmas = tf.random.uniform([data_batch.shape[0]], minval=0, 
                                            maxval=configs.config_values.num_L, 
                                            dtype=tf.dtypes.int32)
        sigmas = tf.gather(sigma_levels, idx_sigmas)
        sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
        data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas

        with tf.GradientTape() as t:
            scores = model([data_batch_perturbed, idx_sigmas])
            current_loss = loss_per_batch(scores, data_batch_perturbed, data_batch, sigmas)
            gradients = t.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tf.summary.scalar('loss', float(current_loss), step=int(step))

        progress_bar.set_description(f'iteration {step}/{total_steps} | current loss {current_loss:.3f}')

        if step % configs.config_values.checkpoint_freq == 0:
            # TODO: maybe save also info about the sigmas
            ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
            ckpt.step.assign_add(step)
            ckpt.save(save_dir+f"{start_time}_step_{step}")
            print("saved checkpoint")

        if step == total_steps:
            return
    
    # NOTE bad way to choose the best model - saving all checkpoints and then testing after

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    train()
    