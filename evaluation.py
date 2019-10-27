"""
This file contains functions for evaluating which checkpoint (saved every 5000 steps) of a model is the best one.
This selection is based on a small FID score computed with 1000 images.
TODO: decide whether to use training set or test set.

"""
import tensorflow as tf
import configs

import utils
from generate import sample_many, sample_many_and_save
from model.inception import Metrics
import numpy as np

if __name__ == '__main__':
    args = utils.get_command_line_args()
    configs.config_values = args
    metric = Metrics()
    n_images_FID = 1000
    multiple = 5000
    i = 20

    dir_statistics = './statistics'
    save_dir, complete_model_name = utils.get_savemodel_dir()

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    while True:
        step_ckpt = i * multiple

        print("\n" + "=" * 30, "\nStep {}".format(step_ckpt))

        model, _, step = utils.try_load_model(save_dir, step_ckpt=step_ckpt, return_new_model=False, verbose=False)

        if model is None:
            break

        partial_filename = '{}/{}_step{}'.format(dir_statistics, complete_model_name, step_ckpt)
        sample_many_and_save(model, sigma_levels,
                             save_directory='{}/{}_step{}/samples/'.format(dir_statistics, complete_model_name,
                                                                           step_ckpt))
        samples = sample_many(model, sigma_levels, n_images=n_images_FID)


        # is_mean, is_stddev = metric.compute_inception_score(samples)
        # print("Inception score: {:.2}+-{:.2}".format(is_mean, is_stddev))
        #
        # mu, sigma = metric.compute_mu_sigma(samples)
        # np.savez(partial_filename)
        # i += 1
