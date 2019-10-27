import csv
import os
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_train_test_data
from losses.losses import dsm_loss
# our files
from model.inception import Metrics

utils.manage_gpu_memory_usage()
device = utils.get_tensorflow_device()


@tf.function
def train_one_step(model, optimizer, data_batch_perturbed, data_batch, idx_sigmas, sigmas):
    with tf.GradientTape() as t:
        scores = model([data_batch_perturbed, idx_sigmas])
        current_loss = dsm_loss(scores, data_batch_perturbed, data_batch, sigmas)
        gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss


def train():
    # load dataset from tfds (or use downloaded version if exists)
    train_data = get_train_test_data(configs.config_values.dataset)[0]

    # split data into batches
    train_data = train_data.shuffle(buffer_size=10000)
    if configs.config_values.dataset != 'celeb_a':
        train_data = train_data.batch(configs.config_values.batch_size)
    train_data = train_data.repeat()
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    if configs.config_values.baseline:
        sigma_levels = tf.ones(1) * configs.config_values.sigma_low
        configs.config_values.num_L = 1
    else:
        sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                               tf.math.log(configs.config_values.sigma_low),
                                               configs.config_values.num_L))

    model, optimizer, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from, verbose=True)

    # # Compute inception score mean and standard deviation
    # sample_dir = configs.config_values.samples_dir + start_time + '_' + complete_model_name + '/'
    # sample_many_and_save(model, sigma_levels, n_images=2, save_directory=sample_dir)
    # sampled_images = sample_many(model, sigma_levels, n_images=2)
    # is_mean, is_stddev = metrics.compute_inception_score(sampled_images, image_side_inception=199)
    # print(f'Inception Score: {is_mean:.3} +- {is_stddev:.3}')
    # # Compute fid
    # test_data_inception = test_data.take(2176)
    # fid = metrics.compute_fid(images_1=sampled_images, data_2=test_data_inception, image_side_inception=199)
    # print(f'FID Score: {fid:.3}')

    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step + 1)
    progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, total_steps))

    loss_history = []
    with tf.device(device):  # For some reason, this makes everything faster
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
            loss_history.append([step, current_loss.numpy()])

            progress_bar.set_description('iteration {}/{} | current loss {:.3f}'.format(
                step, total_steps, current_loss
            ))

            avg_loss += current_loss
            if step % configs.config_values.checkpoint_freq == 0:
                # TODO: maybe save also info about the sigmas
                # Save checkpoint
                ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
                ckpt.step.assign_add(step)
                ckpt.save(save_dir + "{}_step_{}".format(start_time, step))
                # Append in csv file
                with open(save_dir + 'loss_history.csv', mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=';')
                    writer.writerows(loss_history)
                print("\nSaved checkpoint. Average loss: {:.3f}".format(avg_loss / configs.config_values.checkpoint_freq))
                loss_history = []
                avg_loss = 0
            if step == total_steps:
                return


if __name__ == "__main__":
    tf.random.set_seed(2019)

    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    train()
