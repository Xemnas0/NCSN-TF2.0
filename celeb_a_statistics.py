import os

import tensorflow.compat.v1 as tf

import fid
from datasets.dataset_loader import get_celeb_a
from tqdm import tqdm
import numpy as np

from generating.inpainting import save_image


def main():
    data = get_celeb_a(random_flip=False)[0]
    print()
    i = 0
    dir_celeb_a = './statistics/celeb_a_images'
    if not os.path.exists(dir_celeb_a):
        os.makedirs(dir_celeb_a)
    for batch in data:
        for image in batch:
            i+=1
            save_image(image, dir_celeb_a +'/{}'.format(i))
        if i % 10000 == 0:
            print('{}/167000 something'.format(i))

    tf.disable_v2_behavior()
    inception_path = fid.check_or_download_inception(None)
    fid.create_inception_graph(inception_path)
    files = [dir_celeb_a + '/' + filename for filename in os.listdir(dir_celeb_a)]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics_from_files(files, sess, verbose=True)
        np.savez('./statistics/fid_stats_celeb_a_train.npz', mu=mu, sigma=sigma)