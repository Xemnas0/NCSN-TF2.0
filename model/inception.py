"""
Inception Score based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
FID Score based on https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
"""
import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from scipy import linalg

SIZE_ACTIVATION = 2048


class Metrics:
    def __init__(self):
        self.model = tf.keras.applications.inception_v3.InceptionV3()
        self.model_act = tf.keras.models.Model(inputs=self.model.input,
                                               outputs=self.model.get_layer('avg_pool').output)

    def compute_inception_score(self, images, n_splits=10, batch_size=128, image_side_inception=299):
        is_mean, is_stddev = _compute_inception_score(self.model, images, n_splits, batch_size, image_side_inception)
        return is_mean, is_stddev

    def _compute_activations(self, images, batch_size=128, image_side_inception=299):
        input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])
        data = _preprocess_dataset_inception(input_size, images=images, batch_size=batch_size)

        activations = np.zeros((images.shape[0], SIZE_ACTIVATION))
        for i, batch in enumerate(
                tqdm(data, total=tf.data.experimental.cardinality(data).numpy(), desc='Computing activations')):
            i_batch_size = batch.get_shape()[0]
            activations[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(self.model_act, batch)
        return activations

    def compute_mu_sigma(self, images, batch_size=128, image_side_inception=299):
        activations = self._compute_activations(images, batch_size, image_side_inception)
        mean, sigma = np.mean(activations, axis=0), np.conv(activations, rowvar=False)
        return mean, sigma

    def compute_fid(self, images_1=None, data_1=None, images_2=None, data_2=None, batch_size=128,
                    image_side_inception=299, eps=1e-7):
        size_act = 2048
        input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])
        # Compute the activations of Inception
        if data_1 is None:
            act_1 = np.zeros((images_1.shape[0], size_act))
            data_1 = _preprocess_dataset_inception(input_size, images=images_1, batch_size=batch_size)
        else:
            act_1 = np.zeros((tf.data.experimental.cardinality(data_1), size_act))
            data_1 = _preprocess_dataset_inception(input_size, data=data_1, batch_size=batch_size)
        if data_2 is None:
            act_2 = np.zeros((images_2.shape[0], size_act))
            data_2 = _preprocess_dataset_inception(input_size, images=images_2, batch_size=batch_size)
        else:
            act_2 = np.zeros((tf.data.experimental.cardinality(data_2), size_act))
            data_2 = _preprocess_dataset_inception(input_size, data=data_2, batch_size=batch_size)

        for i, batch in enumerate(
                tqdm(data_1, total=tf.data.experimental.cardinality(data_1).numpy(), desc='First dataset fid')):
            i_batch_size = batch.get_shape()[0]
            act_1[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(self.model_act, batch)
        for i, batch in enumerate(
                tqdm(data_2, total=tf.data.experimental.cardinality(data_2).numpy(), desc='Second dataset fid')):
            i_batch_size = batch.get_shape()[0]
            act_2[i * batch_size:i * batch_size + i_batch_size] = _inception_one_step(self.model_act, batch)
        # Compute moments of the activations
        mean_1, sigma_1 = np.mean(act_1, axis=0), np.cov(act_1, rowvar=False)
        mean_2, sigma_2 = np.mean(act_2, axis=0), np.cov(act_2, rowvar=False)
        # Calculate sum squared difference between means
        ssdif = tf.reduce_sum(tf.square(mean_1 - mean_2))
        # Compute covariance
        cov_mean = linalg.sqrtm((sigma_1 + eps).dot(sigma_2 + eps))
        # cov_mean = tf.linalg.sqrtm(tf.linalg.matmul(sigma_1+eps, sigma_2+eps))
        # if cov_mean.dtype in [tf.complex, tf.complex64, tf.complex128]:
        #     print(f"Cov mean is of type ", cov_mean.dtype)
        #     cov_mean = tf.math.real(cov_mean)
        if np.iscomplexobj(cov_mean):
            if not np.allclose(np.diagonal(cov_mean).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_mean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            cov_mean = cov_mean.real

        fid = ssdif + tf.linalg.trace(sigma_1 + sigma_2 - 2.0 * cov_mean)
        return fid


def _compute_inception_score(model, images, n_splits=10, batch_size=128, image_side_inception=299):
    assert images.dtype == tf.float32
    print("\nComputing Inception Score...")

    input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])

    N = images.get_shape()[0]

    data = _preprocess_dataset_inception(input_size, images=images, batch_size=batch_size)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(
            tqdm(data, total=tf.data.experimental.cardinality(data).numpy(), desc='Inception activations')):
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


def _preprocess_dataset_inception(input_size, images=None, data=None, batch_size=128):
    """
    Transform the images in a tf.data.Dataset and resize them to fit InceptionV3
    :param images:
    :param batch_size:
    :param input_size: tuple (height, width). Input size of the InceptionV3 (299x299x3) usually
    :return:
    """
    if data is None:
        data = tf.data.Dataset.from_tensor_slices(images)
    data = data.map(lambda x: tf.image.resize(x, input_size))
    if data.output_shapes[-1] == 1:  # If it's grayscale, duplicate to get 3 channels
        data = data.map(lambda x: tf.tile(x, [1, 1, 3]))
    # data = data.map(lambda x: x / 255)
    data = data.batch(batch_size)
    return data


@tf.function
def _inception_one_step(model, batch):
    with tf.device('GPU:0'):
        predictions = model(batch)
    return predictions
