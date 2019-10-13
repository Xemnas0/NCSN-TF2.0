import tensorflow as tf
from model.refinenet import RefineNet

def sample(model, n_images, sigmas, eps, T):
    """
    Only for MNIST, for now.
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    image_size = (n_images, 28, 28, 1)

    x = tf.random.uniform(shape=image_size)

    for i, sigma_i in enumerate(sigmas):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2

        for t in range(T):
            z_t = tf.random.normal(shape=image_size, mean=0, stddev=1.0)  # TODO: check if stddev is correct
            x += alpha_i / 2 * model([x, tf.ones(n_images)*i]) + tf.sqrt(alpha_i) * z_t

    return x


if __name__ == '__main__':
    model_name = ''
    model_directory = './saved_models/'

    model = RefineNet(filters=16, activation=tf.nn.elu)

    model = tf.keras.models.load_weights(model_directory + model_name)


