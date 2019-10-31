import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import configs
import utils


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)


def save_as_grid(images, filename, spacing=2, rows=None):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param images:
    :return:
    """
    # Define grid dimensions
    n_images, height, width, channels = images.shape
    if rows is None:
        rows = np.floor(np.sqrt(n_images)).astype(int)
    cols = n_images // rows

    # Process image
    images = _preprocess_image_to_save(images)

    # Init image
    grid_cols = rows * height + (rows + 1) * spacing
    grid_rows = cols * width + (cols + 1) * spacing
    mode = 'LA' if channels == 1 else "RGB"
    im = Image.new(mode, (grid_rows, grid_cols))
    for i in range(n_images):
        col = i // rows
        row = i % rows
        x = col * height + (1 + col) * spacing
        y = row * width + (1 + row) * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(images[i]), (x, y))
        # im.show() # for debugging

    im.save(filename, format="PNG")


@tf.function
def sample_one_step(model, x, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1.0)  # TODO: check if stddev is correct
    score = model([x, idx_sigmas])
    noise = tf.sqrt(alpha_i) * z_t
    return x + alpha_i / 2 * score + noise


def sample_many(model, sigmas, batch_size=128, eps=2 * 1e-5, T=100, n_images=1):
    """
    Used for sampling big amount of images (e.g. 50000)
    :param model: model for sampling (RefineNet)
    :param sigmas: sigma levels of noise
    :param eps:
    :param T: iteration per sigma level
    :return: Tensor of dimensions (n_images, width, height, channels)
    """
    # Tuple for (n_images, width, height, channels)
    image_size = (n_images,) + utils.get_dataset_image_size(configs.config_values.dataset)
    batch_size = min(batch_size, n_images)

    with tf.device('CPU'):
        x = tf.random.uniform(shape=image_size)
    x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    x_processed = None

    n_processed_images = 0
    for i_batch, batch in enumerate(
            tqdm(x, total=tf.data.experimental.cardinality(x).numpy(), desc='Generating samples')):
        for i, sigma_i in enumerate(sigmas):
            alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
            idx_sigmas = tf.ones(batch.get_shape()[0], dtype=tf.int32) * i
            for t in range(T):
                batch = sample_one_step(model, batch, idx_sigmas, alpha_i)

        with tf.device('CPU'):
            if x_processed is not None:
                x_processed = tf.concat([x_processed, batch], axis=0)
            else:
                x_processed = batch

        n_processed_images += batch_size

    x_processed = _preprocess_image_to_save(x_processed)

    return x_processed


@tf.function
def _preprocess_image_to_save(x):
    x = tf.clip_by_value(x, 0, 1)
    x = x * 255
    x = x + 0.5
    x = tf.clip_by_value(x, 0, 255)
    # min = tf.reduce_min(x)
    # max = tf.reduce_max(x)
    # x = (x + min) / (max + min) * 255
    return x


def sample_many_and_save(model, sigmas, batch_size=128, eps=2 * 1e-5, T=100, n_images=1, save_directory=None):
    """
    Used for sampling big amount of images (e.g. 50000)
    :param model: model for sampling (RefineNet)
    :param sigmas: sigma levels of noise
    :param eps:
    :param T: iteration per sigma level
    :return: Tensor of dimensions (n_images, width, height, channels)
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Tuple for (n_images, width, height, channels)
    image_size = (n_images,) + utils.get_dataset_image_size(configs.config_values.dataset)
    batch_size = min(batch_size, n_images)

    with tf.device('CPU'):
        x = tf.random.uniform(shape=image_size)
    x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)

    idx_image = 0
    for i_batch, batch in enumerate(
            tqdm(x, total=tf.data.experimental.cardinality(x).numpy(), desc='Generating samples')):
        for i, sigma_i in enumerate(sigmas):
            alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
            idx_sigmas = tf.ones(batch.get_shape()[0], dtype=tf.int32) * i
            for t in range(T):
                batch = sample_one_step(model, batch, idx_sigmas, alpha_i)

        if save_directory is not None:
            batch = _preprocess_image_to_save(batch)
            for image in batch:
                im = Image.new('RGB', image_size[1:3])
                if image_size[-1] == 1:
                    image = tf.tile(image, [1, 1, 3])
                im.paste(tf.keras.preprocessing.image.array_to_img(image))
                im.save(save_directory + f'{idx_image}.png', format="PNG")
                idx_image += 1


def sample_and_save(model, sigmas, eps=2 * 1e-5, T=100, n_images=1, save_directory=None):
    """
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    image_size = (n_images,) + utils.get_dataset_image_size(configs.config_values.dataset)

    x = tf.random.uniform(shape=image_size)

    for i, sigma_i in enumerate(tqdm(sigmas, desc='Sampling for each sigma')):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        idx_sigmas = tf.ones(n_images, dtype=tf.int32) * i
        for t in range(T):
            x = sample_one_step(model, x, idx_sigmas, alpha_i)

            if (t + 1) % 10 == 0:
                save_as_grid(x, save_directory + f'sigma{i + 1}_t{t + 1}.png')
    return x


def main():
    save_dir, complete_model_name = utils.get_savemodel_dir()
    model, optimizer, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from, verbose=True)
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    if configs.config_values.sigma_sequence == 'linear':
        sigma_levels = tf.linspace(configs.config_values.sigma_high,
                                   configs.config_values.sigma_low,
                                   configs.config_values.num_L)
    else:
        sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                               tf.math.log(configs.config_values.sigma_low),
                                               configs.config_values.num_L))

    samples_directory = './samples/{}_{}_step{}/'.format(start_time, complete_model_name, step)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    sample_and_save(model, sigma_levels, n_images=12, T=100, save_directory=samples_directory)
