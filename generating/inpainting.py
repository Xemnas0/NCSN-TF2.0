import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_data_inpainting
from generating.generate import _preprocess_image_to_save

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
    height, width, channels = images[0][0].shape
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

        im.paste(tf.keras.preprocessing.image.array_to_img(occluded_x), (col_start, row_start))

        # plot the samples
        for j in range(len(samples)):
            col_start = (j + 1) * width + (j + 2) * spacing + spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(samples[j]), (col_start, row_start))

        # plot the original image
        col_start = (len(samples) + 1) * width + (len(samples) + 2) * spacing + 2 * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(x), (col_start, row_start))
        # im.save(filename+"_n", format="PNG")

    im.save(filename, format="PNG")


def save_image(image, filename):
    mode = 'L' if image.shape[-1] == 1 else 'RGB'
    im = Image.new(mode, utils.get_dataset_image_size(configs.config_values.dataset)[:2])
    im.paste(tf.keras.preprocessing.image.array_to_img(image))
    im.save(filename + '.png', format="PNG")


@tf.function
def inpaint_one_step(model, x_t, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x_t.shape, mean=0, stddev=1.0)
    score = model([x_t, idx_sigmas])
    noise = tf.sqrt(2 * alpha_i) * z_t
    return x_t + alpha_i * score + noise


def inpaint_x(model, sigmas, m, x, eps=2 * 1e-5, T=100):
    x_t = tf.random.uniform(shape=x.shape)
    x_t = (x_t * (1 - m)) + (x * m)

    for i, sigma_i in enumerate(sigmas):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        z = tf.random.normal(shape=x.shape, mean=0, stddev=sigma_i)
        y = (x + z) * m
        x_t = x_t * (1 - m) + y
        idx_sigmas = tf.ones(x.shape[0], dtype=tf.int32) * i
        for t in range(T):
            x_t = inpaint_one_step(model, x_t, idx_sigmas, alpha_i)
            x_t = x_t * (1 - m) + y
            # if (t+1) % 10 == 0:
            #     save_image(x_t[0], './samples/test/' + 'image_{}-{}_inpainted'.format(i, t))
    return x_t


def main():
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # load model from checkpoint
    save_dir, complete_model_name = utils.get_savemodel_dir()
    model, optimizer, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from, verbose=True)

    # construct path and folder
    dataset = configs.config_values.dataset
    # samples_directory = f'./inpainting_results/{dataset}_{start_time}'
    samples_directory = './samples/{}_{}_step{}_inpainting/'.format(start_time, complete_model_name, step)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    # initialise sigmas
    sigma_levels = utils.get_sigma_levels()

    # TODO add these values to args
    N_to_occlude = 10  # number of images to occlude
    n_reconstructions = 8  # number of samples to generate for each occluded image
    mask_style = 'horizontal_up'  # what kind of occlusion to use
    # mask_style = 'middle'  # what kind of occlusion to use

    # load data for inpainting (currently always N first data points from test data)
    data = get_data_inpainting(configs.config_values.dataset, N_to_occlude)

    images = []

    mask = np.zeros(data.shape[1:])
    if mask_style == 'vertical_split':
        mask[:, :data.shape[2] // 2, :] += 1  # set left side to ones
    if mask_style == 'middle':
        fifth = data.shape[2] // 5
        mask[:, :2 * fifth, :] += 1  # set stripe in the middle to ones
        mask[:, -(2 * fifth):, :] += 1  # set stripe in the middle to ones
    elif mask_style == 'checkerboard':
        mask[::2, ::2, :] += 1  # set every other value to ones
    elif mask_style == 'horizontal_down':
        mask[:data.shape[1] // 2, :, :] += 1
    elif mask_style == 'horizontal_up':
        mask[data.shape[1] // 2:, :, :] += 1
    elif mask_style =='centered':
        init_x, init_y = data.shape[1]//4, data.shape[2] // 4
        mask += 1
        mask[init_x:3*init_x, init_y:3*init_y, : ] -=1
    else:
        pass  # TODO add options here

    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    for i, x in enumerate(data):
        occluded_x = x * mask
        save_dir = f'{samples_directory}/image_{i}'
        save_image(x, save_dir + '_original')
        save_image(occluded_x, save_dir + '_occluded')

    reconstructions = [[] for i in range(N_to_occlude)]
    for j in tqdm(range(n_reconstructions)):
        samples_j = inpaint_x(model, sigma_levels, mask, data, T=100)
        samples_j = _preprocess_image_to_save(samples_j)
        for i, reconstruction in enumerate(samples_j):
            reconstructions[i].append(reconstruction)
            save_image(reconstruction, samples_directory + 'image_{}-{}_inpainted'.format(i, j))

    for i in range(N_to_occlude):
        images.append([data[i] * mask, reconstructions[i], data[i]])

    save_as_grid(images, samples_directory + '/grid.png', spacing=5)

    # for i, x in enumerate(data):
    #     occluded_x = x * mask
    #
    #     samples = []
    #     for j in tqdm(range(n_reconstructions)):
    #         sample = inpaint_x(model, sigma_levels, mask, tf.expand_dims(x, 0))[0]
    #         samples.append(sample)
    #         save_image(sample, save_dir + f'_sample_{j}')
    #
    #     images.append([occluded_x, samples, x])
    #
    # save_as_grid(images, samples_directory + '/grid.png')
