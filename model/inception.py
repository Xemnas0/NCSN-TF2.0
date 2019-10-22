"""
Inception Score based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
FID Score based on https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
"""
import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

class Metrics:
    def __init__(self):
        self.model = tf.keras.applications.inception_v3.InceptionV3()

    def compute_inception_score(self, images, n_splits=10, batch_size=128, image_side_inception=299):
        is_mean, is_stddev = _compute_inception_score(self.model, images, n_splits, batch_size, image_side_inception)
        return is_mean, is_stddev

    def compute_fid(self, images_1=None, data_1=None, images_2=None, data_2=None, batch_size=128,
                    image_side_inception=299):
        input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])
        # Compute the activations of Inception
        if data_1 is None:
            act_1 = np.zeros((images_1.shape[0], 1000))
            data_1 = _preprocess_dataset_inception(images_1, input_size, batch_size)
        else:
            act_1 = np.zeros((tf.data.experimental.cardinality(data_1), 1000))
            data_1 = data_1.batch(batch_size)
        if data_2 is None:
            act_2 = np.zeros((images_2.shape[0], 1000))
            data_2 = _preprocess_dataset_inception(images_2, input_size, batch_size)
        else:
            act_2 = np.zeros((tf.data.experimental.cardinality(data_2), 1000))
            data_2 = data_2.batch(batch_size)

        for i, batch in enumerate(tqdm(data_1, desc='First dataset fid')):
            i_batch_size = batch.get_shape()[0]
            act_1[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(self.model, batch)
        for i, batch in enumerate(tqdm(data_2, desc='Second dataset fid')):
            i_batch_size = batch.get_shape()[0]
            act_2[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(self.model, batch)
        # Compute moments of the activations
        mean_1, sigma_1 = tf.nn.moments(act_1, axis=0)
        mean_2, sigma_2 = tf.nn.moments(act_2, axis=0)
        # Calculate sum squared difference between means
        ssdif = tf.reduce_sum(tf.square(mean_1 - mean_2))
        # Compute covariance
        cov_mean = tf.linalg.sqrtm(tf.linalg.matmul(sigma_1, sigma_2))
        if cov_mean.dtype in [tf.complex, tf.complex64, tf.complex128]:
            print(f"Cov mean is of type ", cov_mean.dtype)
            cov_mean = tf.math.real(cov_mean)

        fid = ssdif + tf.linalg.trace(sigma_1 + sigma_2 - 2.0 * cov_mean)
        return fid


def _compute_inception_score(model, images, n_splits=10, batch_size=128, image_side_inception=299):
    assert images.dtype == tf.float32
    print("Computing Inception Score...\n")

    input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])

    N = images.get_shape()[0]

    data = _preprocess_dataset_inception(images, input_size, batch_size)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(tqdm(data, desc='Inception activations')):
        i_batch_size = batch.get_shape()[0]
        preds[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(model, batch)

    split_scores = []

    for k in tqdm(range(n_splits), desc='IS for every split'):
        part = preds[k * (N // n_splits): (k + 1) * (N // n_splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def _preprocess_dataset_inception(images, input_size, batch_size=128):
    """
    Transform the images in a tf.data.Dataset and resize them to fit InceptionV3
    :param images:
    :param batch_size:
    :param input_size: tuple (height, width). Input size of the InceptionV3 (299x299x3) usually
    :return:
    """
    data = tf.data.Dataset.from_tensor_slices(images).map(lambda x: tf.image.resize(x, input_size))

    if images.get_shape()[-1] == 1:  # If it's grayscale, duplicate to get 3 channels
        data = data.map(lambda x: tf.tile(x, [1, 1, 3]))
    data = data.batch(batch_size)
    return data


@tf.function
def _inception_one_step(model, batch):
    with tf.device('CPU'):  # TODO: change this to GPU for the cloud
        predictions = model(batch)
    return predictions
