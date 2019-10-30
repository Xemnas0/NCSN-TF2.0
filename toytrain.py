import csv, os
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

import configs
import utils
# our files
from datasets.dataset_loader import get_train_test_data
from losses.losses import ssm_loss
from model.resnet import ToyResNet


@tf.function
def train_one_step(model, data_batch, optimizer):
    with tf.GradientTape(persistent=True) as t:
        current_loss = ssm_loss(model, data_batch)
        gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss


def main():
    device = utils.get_tensorflow_device()
    tf.random.set_seed(2019)

    perturbed = True

    train_data, _ = get_train_test_data(configs.config_values.dataset)
    train_data = train_data.shuffle(60000).batch(configs.config_values.batch_size).repeat().prefetch(
                                                        buffer_size=tf.data.experimental.AUTOTUNE)

    save_dir = configs.config_values.checkpoint_dir + configs.config_values.dataset + 'toy1' + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = ToyResNet(activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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

    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step + 1)
    progress_bar.set_description(f'iteration {step}/{total_steps} | current loss ?')

    sigma = tf.convert_to_tensor(0.0001, dtype=tf.float32)
    sigmas = tf.fill((configs.config_values.batch_size, 1, 1, 1), sigma)

    loss_history = []
    with tf.device(device):
        for i, data_batch in enumerate(progress_bar):
            step += 1

            if perturbed:
                noise = tf.random.normal(shape=data_batch.shape) * sigmas
                data_batch = data_batch + noise

            current_loss = train_one_step(model, data_batch, optimizer)
            loss_history.append(current_loss)
            progress_bar.set_description(f'iteration {step}/{total_steps} | current loss {current_loss:.3f}')

            if step == total_steps:
                with open(save_dir + 'loss_history.csv', mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=';')
                    writer.writerows(loss_history)
                return
