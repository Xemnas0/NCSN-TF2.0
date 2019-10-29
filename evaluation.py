"""
This file contains functions for evaluating which checkpoint (saved every 5000 steps) of a model is the best one.
This selection is based on a small FID score computed with 1000 images.
TODO: decide whether to use training set or test set.

"""
import csv
import os
import shutil

import tensorflow as tf

import configs
import fid
import utils
from generate import sample_many_and_save

stat_files = {
    "cifar10": "./statistics/fid_stats_cifar10_train.npz"
}


def main():
    batch_FID = 1000
    multiple = 10000
    i = step_ckpt = 0

    dir_statistics = './statistics'
    save_dir, complete_model_name = utils.get_savemodel_dir()

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    if configs.config_values.eval_setting == '50k':
        model, _, step = utils.try_load_model(save_dir, step_ckpt=configs.config_values.resume_from,
                                              return_new_model=False, verbose=False)
        save_directory = '{}/{}/step{}/is_50k/'.format(dir_statistics, complete_model_name, step_ckpt)
        sample_many_and_save(model, sigma_levels, save_directory=save_directory, n_images=50000)
        return

    filename_stats_dataset = stat_files[configs.config_values.dataset]

    csv_filename = '{}/{}/'.format(dir_statistics, complete_model_name) + 'all_FIDs.csv'
    # Remove csv if it already exists
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    while step_ckpt <= configs.config_values.steps:
        i += 1
        step_ckpt = i * multiple

        print("\n" + "=" * 30, "\nStep {}".format(step_ckpt))

        save_directory = '{}/{}/step{}/samples/'.format(dir_statistics, complete_model_name, step_ckpt)

        if configs.config_values.eval_setting == 'sample':
            # If the directory exists and has images, don't generate
            if os.path.exists(save_directory):
                if len(os.listdir(save_directory)) == batch_FID:
                    print(save_directory, " already exists and has samples")
                    continue
                else:
                    print("Removing existing samples that were not enough ({})".format(batch_FID))
                    shutil.rmtree(save_directory)

            model, _, step = utils.try_load_model(save_dir, step_ckpt=step_ckpt, return_new_model=False, verbose=False)

            if model is None:
                break

            print("Generating samples in ", save_directory)
            sample_many_and_save(model, sigma_levels, save_directory=save_directory, n_images=batch_FID)
        elif configs.config_values.eval_setting == 'fid':
            # Check if directory exists
            if not os.path.exists(save_directory):
                print("Sample directory ", save_directory, " not found")
                continue
            # Check if it's empty, and if it is, delete it
            if not len(os.listdir(save_directory)):
                print("Found empty ", save_directory, ". Deleting it...")
                os.rmdir(save_directory)
                continue

            print("Computing FID...")

            fid_score = fid.main(save_directory, filename_stats_dataset)

            print("Steps {}, FID {}".format(step_ckpt, fid_score))

            with open(csv_filename, mode='a',
                      newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([step_ckpt, fid_score])

        # is_mean, is_stddev = metric.compute_inception_score(samples)
        # print("Inception score: {:.2}+-{:.2}".format(is_mean, is_stddev))
        #
        # mu, sigma = metric.compute_mu_sigma(samples)
        # np.savez(partial_filename)

        # returned = os.system('python3 fid.py {} {} --gpu GPU:0'.format(save_directory, filename_stats_dataset))
        # print(returned)
