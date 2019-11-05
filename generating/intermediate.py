import os
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

import configs
import utils
from generating.generate import sample_one_step, save_as_grid


def sample_and_save_intermediate(model, sigmas, x=None, eps=2 * 1e-5, T=100, n_images=1, save_directory=None):
    """
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if x is None:
        image_size = (n_images,) + utils.get_dataset_image_size(configs.config_values.dataset)
        x = tf.random.uniform(shape=image_size)
    else:
        image_size = x.shape
        n_images = image_size[0]

    x_all = None
    for i, sigma_i in enumerate(tqdm(sigmas, desc='Sampling for each sigma')):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        idx_sigmas = tf.ones(n_images, dtype=tf.int32) * i
        for t in range(T):
            x = sample_one_step(model, x, idx_sigmas, alpha_i)

        if x_all is None:
            x_all = x
        else:
            x_all = tf.concat([x_all, x], axis=0)

    save_as_grid(x_all, save_directory + 'intermediate.png', rows=n_images)
    return x


def main():
    save_dir, complete_model_name = utils.get_savemodel_dir()
    model, optimizer, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from, verbose=True)
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    samples_directory = './samples/{}_{}_step{}_intermediate/'.format(start_time, complete_model_name, step)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)
    x0 = utils.get_init_samples()
    sample_and_save_intermediate(model, sigma_levels, x=x0, eps=2 * 1e-5, T=100, n_images=5,
                                 save_directory=samples_directory)
