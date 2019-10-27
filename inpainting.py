import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_data_inpainting

utils.manage_gpu_memory_usage()


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def save_as_grid(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    """
    # images is of shape [ [occluded_x, [sample, sample, sample...], x],
    #                      [occluded_x, [sample, sample, sample...], x],
    #                      ...]
    _, height, width, channels = images[0][0].shape
    rows = len(images)
    cols = len(images[0][1]) + 2

    # init image
    image_height = rows * height + (rows + 1) * spacing
    image_width = cols * width + (cols + 1) * spacing + 2 * spacing  # double spacing between samples and x/occluded_x
    mode = 'L' if channels == 1 else "RGB"
    im = Image.new(mode, (image_width, image_height), color='white')

    for i in range(rows):  # i = row, j = column
        occluded_x, samples, x = images[i]

        # plot the occluded image
        row_start = i * height + (1 + i) * spacing
        col_start = spacing

        im.paste(tf.keras.preprocessing.image.array_to_img(occluded_x[0, :, :, :]), (col_start, row_start))

        # plot the samples
        for j in range(len(samples)):
            col_start = (j + 1) * width + (j + 2) * spacing + spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(samples[j][0, :, :, :]), (col_start, row_start))

        # plot the original image
        col_start = (len(samples) + 1) * width + (len(samples) + 2) * spacing + 2 * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(x[0, :, :, :]), (col_start, row_start))
        # im.save(filename+"_n", format="PNG")

    im.save(filename, format="PNG")


def save_image(image, filename):
    rgb = False if image.shape[-1] == 1 else True
    if len(image.shape) == 4:
        image = image[0, :, :, 0]
    if not rgb:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.savefig(filename)


@tf.function
def inpaint_one_step(model, x_t, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x_t.shape, mean=0, stddev=1.0)
    score = model([x_t, idx_sigmas])
    noise = tf.sqrt(alpha_i) * z_t
    return x_t + (alpha_i / 2) * score + noise


def inpaint_x(model, sigmas, m, x, eps=2 * 1e-5, T=100):
    x_t = tf.random.uniform(shape=x.shape)
    x_t = (x_t * (1 - m)) + (x * m)

    for i, sigma_i in enumerate(sigmas):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        z = tf.random.normal(shape=x.shape, mean=0, stddev=sigma_i ** 2)
        y = x + z
        idx_sigmas = tf.ones(1, dtype=tf.int32) * i
        for t in range(T):
            x_t = inpaint_one_step(model, x_t, idx_sigmas, alpha_i)
            x_t = (x_t * (1 - m)) + (y * m)
    return x_t


def main():
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # construct path and folder
    dataset = configs.config_values.dataset
    samples_directory = f'./inpainting_results/{dataset}_{start_time}'
    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    # load model from checkpoint
    save_dir, complete_model_name = utils.get_savemodel_dir()
    model, optimizer, step = utils.try_load_model(save_dir, verbose=True)

    # initialise sigmas
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    # TODO add these values to args
    N = 5  # number of images to occlude
    n = 5  # number of samples to generate for each occluded image
    # mask_style = 'vertical_split'  # what kind of occlusion to use
    mask_style = 'middle'  # what kind of occlusion to use

    # load data for inpainting (currently always N first data points from test data)
    data = get_data_inpainting(configs.config_values.dataset, N)

    images = []

    for i, x in enumerate(data):
        mask = np.zeros(x.shape)
        if mask_style == 'vertical_split':
            mask[:, :, :x.shape[2] // 2, :] += 1  # set left side to ones
        if mask_style == 'middle':
            fifth = x.shape[2] // 5
            mask[:, :, :2 * fifth, :] += 1  # set stripe in the middle to ones
            mask[:, :, -(2 * fifth):, :] += 1  # set stripe in the middle to ones
        elif mask_style == 'checkerboard':
            mask[:, ::2, ::2, :] += 1  # set every other value to ones
        else:
            pass  # TODO add options here

        occluded_x = x * mask

        save_dir = f'{samples_directory}/image_{i}'
        save_image(x, save_dir + '_original')
        save_image(occluded_x, save_dir + '_occluded')

        samples = []
        for j in tqdm(range(n)):
            sample = inpaint_x(model, sigma_levels, mask, x)
            samples.append(sample)
            save_image(sample, save_dir + f'_sample_{j}')

        images.append([occluded_x, samples, x])

    save_as_grid(images, samples_directory + '/grid.png')
