import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_data_k_nearest


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)


def save_as_grid(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param images:
    :return:
    """
    # Define grid dimensions
    n_images, height, width, channels = images.shape
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
        row = i // rows
        col = i % rows
        row_start = row * height + (1 + row) * spacing
        col_start = col * width + (1 + col) * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(images[i]), (row_start, col_start))
        # im.show()

    im.save(filename, format="PNG")


def save_as_grid_closest_k(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    """
    # images is of shape [ [ sample, [ closest, closest, ... ] ], [ sample, [ closest, closest, ... ] ]
    _, height, width, channels = images[0][0].shape
    rows = len(images)
    cols = len(images[0][1]) + 1

    # init image
    image_height = rows * height + (rows + 1) * spacing
    image_width = cols * width + (cols + 1) * spacing + spacing  # double spacing between samples and x/occluded_x
    mode = 'LA' if channels == 1 else "RGB"
    im = Image.new(mode, (image_width, image_height), color='white')

    for i in range(rows):  # i = row, j = column
        sample, closest = images[i]

        # plot the sample
        row_start = i * height + (1 + i) * spacing
        col_start = spacing

        im.paste(tf.keras.preprocessing.image.array_to_img(sample[0, :, :, :]), (col_start, row_start))

        # plot the closest images from training set
        for j in range(len(closest)):
            col_start = (j + 1) * width + (j + 2) * spacing + spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(closest[j][0, :, :, :]), (col_start, row_start))

    im.save(filename, format="PNG")


@tf.function
def sample_one_step(model, x, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1.0)  # TODO: check if stddev is correct
    score = model([x, idx_sigmas])
    noise = tf.sqrt(alpha_i * 2) * z_t
    return x + alpha_i * score + noise


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
    x = x * 255
    x = x + 0.5
    x = tf.clip_by_value(x, 0, 255)
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


def sample_and_save_intermediate(model, sigmas, eps=2 * 1e-5, T=100, n_images=1, save_directory=None):
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

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    samples_directory = './samples/{}_{}_step{}/'.format(start_time, complete_model_name, step)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    if configs.config_values.find_nearest:
        n_images = 10  # TODO make this not be hard-coded
        samples = tf.split(sample_many(model, sigma_levels, T=100, n_images=n_images), n_images)
        data_as_array = get_data_k_nearest(configs.config_values.dataset)
        data_as_array = data_as_array.batch(int(tf.data.experimental.cardinality(data_as_array)))
        data_as_array = tf.data.experimental.get_single_element(data_as_array)

        images = []
        for i, sample in enumerate(samples):
            # save_image(sample[0, :, :, 0], samples_directory + f'sample_{i}')
            k_closest_images, smallest_idx = utils.find_k_closest(sample, configs.config_values.k, data_as_array)
            # for j, img in enumerate(k_closest_images):
                # save_image(img[0, :, :, 0], samples_directory + f'sample_{i}_closest_{j}')

            print(smallest_idx)

            images.append([sample, k_closest_images])

        save_as_grid_closest_k(images, samples_directory+"k_closest_grid.png", spacing=4)
    else:
        n_images = 400
        sample_and_save(model, sigma_levels, n_images=n_images, T=100, save_directory=samples_directory)
