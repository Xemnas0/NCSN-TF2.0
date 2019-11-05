import os
from datetime import datetime

import tensorflow as tf
from PIL import Image

import configs
import utils
from datasets.dataset_loader import get_celeb_a32, get_data_k_nearest
from generating.generate import sample_many


def save_as_grid_closest_k(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    """
    # images is of shape [ [ sample, [ closest, closest, ... ] ], [ sample, [ closest, closest, ... ] ]
    height, width, channels = images[0][0].shape
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

        im.paste(tf.keras.preprocessing.image.array_to_img(sample), (col_start, row_start))

        # plot the closest images from training set
        for j in range(len(closest)):
            col_start = (j + 1) * width + (j + 2) * spacing + spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(closest[j]), (col_start, row_start))

    im.save(filename, format="PNG")


def main():
    save_dir, complete_model_name = utils.get_savemodel_dir()
    model, optimizer, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from, verbose=True)
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    sigma_levels = utils.get_sigma_levels()

    k = configs.config_values.k
    samples_directory = './samples/{}_{}_step{}_{}nearest/'.format(start_time, complete_model_name, step, k)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    n_images = 10  # TODO make this not be hard-coded
    samples = sample_many(model, sigma_levels, batch_size=configs.config_values.batch_size,
                                   T=100, n_images=n_images)

    if configs.config_values.dataset == 'celeb_a':
        data = get_celeb_a32()
    else:
        data = get_data_k_nearest(configs.config_values.dataset)
        data = data.batch(int(tf.data.experimental.cardinality(data)))
        # data = tf.data.experimental.get_single_element(data)

    images = []
    data_subsets = []
    for i, sample in enumerate(samples):
        for data_batch in data:
            k_closest_images, _ = utils.find_k_closest(sample, k, data_batch)
            data_subsets.append(k_closest_images)
        # data = tf.convert_to_tensor(data_subsets)
        # k_closest_images, smallest_idx = utils.find_k_closest(sample, k, data)
        # save_image(sample[0, :, :, 0], samples_directory + f'sample_{i}')
        # k_closest_images, smallest_idx = utils.find_k_closest(sample, configs.config_values.k, data_as_array)
        # for j, img in enumerate(k_closest_images):
        # save_image(img[0, :, :, 0], samples_directory + f'sample_{i}_closest_{j}')

        # print(smallest_idx)
        images.append([sample, k_closest_images])

    save_as_grid_closest_k(images, samples_directory + "k_closest_grid.png", spacing=5)
