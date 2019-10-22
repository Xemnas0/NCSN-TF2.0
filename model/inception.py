"""
Based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
"""
import tensorflow as tf
import numpy as np
from scipy.stats import entropy


def compute_inception_score(images, n_splits=10, batch_size=128, image_side_inception=299):
    assert images.dtype == tf.float32

    model = tf.keras.applications.inception_v3.InceptionV3()
    input_size = tf.convert_to_tensor([image_side_inception, image_side_inception])

    N = images.get_shape()[0]

    data = tf.data.Dataset.from_tensor_slices(images).map(lambda x: tf.image.resize(x, input_size))

    if images.get_shape()[-1] == 1:  # If it's grayscale, duplicate to get 3 channels
        data = data.map(lambda x: tf.tile(x, [1, 1, 3]))
    data = data.batch(batch_size)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(data):
        i_batch_size = batch.get_shape()[0]
        inception_one_step(model, i, batch, batch_size, i_batch_size, preds)

    split_scores = []

    for k in range(n_splits):
        part = preds[k * (N // n_splits): (k + 1) * (N // n_splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


@tf.function
def inception_one_step(model, i, batch, batch_size, i_batch_size, preds):
    with tf.device('CPU'):
        predictions = model(batch)
    preds[i * batch_size:i * batch_size + i_batch_size] = predictions
    return preds
