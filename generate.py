import tensorflow as tf
from model.refinenet import RefineNet
import utils
import configs
import matplotlib.pyplot as plt
from tqdm import tqdm

def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def sample(model, sigmas, eps=2 * 1e-5, T=100, n_images=1):
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
    plot_grayscale(x[0, :, :, 0])

    for i, sigma_i in enumerate(sigmas):
        print(f"sigma {i}/{len(sigmas)}")
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2

        for t in tqdm(range(T)):
            z_t = tf.random.normal(shape=image_size, mean=0, stddev=1.0)  # TODO: check if stddev is correct
            score = model([x, tf.ones(n_images, dtype=tf.int32) * i])
            x = x + alpha_i * score + tf.sqrt(alpha_i*2) * z_t

            plot_grayscale(clamped(x[0, :, :, 0]))
            
    return x


if __name__ == '__main__':
    args = utils.get_command_line_args()
    configs.config_values = args

    model_name = ''
    model_directory = './saved_models/'
    dataset = 'mnist/'

    step = tf.Variable(0)
    model = RefineNet(filters=16, activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    latest_checkpoint = tf.train.latest_checkpoint(model_directory + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(1.0),
                                           tf.math.log(0.01),
                                           10))

    samples = sample(model, sigma_levels, T=100)

