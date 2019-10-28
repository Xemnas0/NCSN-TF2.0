"""
This file contains functions for evaluating which checkpoint (saved every 5000 steps) of a model is the best one.
This selection is based on a small FID score computed with 1000 images.
TODO: decide whether to use training set or test set.

"""
import csv
import os

import tensorflow as tf

import configs
import utils
from generate import sample_many_and_save
import fid
stat_files = {
    "cifar10": "./statistics/fid_stats_cifar10_train.npz"
}


def main():
    batch_FID = 1000
    multiple = 10000
    i = 1

    dir_statistics = './statistics'
    save_dir, complete_model_name = utils.get_savemodel_dir()

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    filename_stats_dataset = stat_files[configs.config_values.dataset]

    while True:
        step_ckpt = i * multiple

        print("\n" + "=" * 30, "\nStep {}".format(step_ckpt))

        save_directory = '{}/{}/step{}/samples/'.format(dir_statistics, complete_model_name, step_ckpt)

        if not os.path.exists(save_directory) and configs.config_values.eval_setting == 'sample':

            model, _, step = utils.try_load_model(save_dir, step_ckpt=step_ckpt, return_new_model=False, verbose=False)

            if model is None:
                break

            print("Generating samples...")
            sample_many_and_save(model, sigma_levels, save_directory=save_directory, n_images=batch_FID)
        elif configs.config_values.eval_setting == 'fid':
            print("Computing FID...")
            # fid_score = os.system('python fid.py {} {} --gpu GPU:0'.format(save_directory, filename_stats_dataset))
            fid_score = fid.main(save_directory, filename_stats_dataset)

            print("Steps {}, FID {}".format(step_ckpt, fid_score))

            with open('{}/{}/'.format(dir_statistics, complete_model_name) + 'all_FIDs.csv', mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([step_ckpt, fid_score])

        # is_mean, is_stddev = metric.compute_inception_score(samples)
        # print("Inception score: {:.2}+-{:.2}".format(is_mean, is_stddev))
        #
        # mu, sigma = metric.compute_mu_sigma(samples)
        # np.savez(partial_filename)

        # returned = os.system('python3 fid.py {} {} --gpu GPU:0'.format(save_directory, filename_stats_dataset))
        # print(returned)

        i += 1
