import tensorflow as tf
from model.refinenet import RefineNet
import utils
import configs
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from PIL import Image
import numpy as np


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)


def save_as_grid(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param images:
    :return:
    """
    # Define grid dimensions
    n_images, height, width, channels = images.shape
    rows = np.floor(np.sqrt(n_images)).astype(int)
    cols = n_images // rows

    # Init image
    grid_cols = rows * height + (rows + 1) * spacing
    grid_rows = cols * width + (cols + 1) * spacing
    mode = 'L' if channels == 1 else "RGB"
    im = Image.new(mode, (grid_rows, grid_cols))
    for i in range(n_images):
        row = i // rows
        col = i % rows
        row_start = row * height + (1 + row) * spacing
        col_start = col * width + (1 + col) * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(images[i]), (row_start, col_start))
        # im.show()

    im.save(filename, format="PNG")


@tf.function
def sample_one_step(model, x, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1.0)  # TODO: check if stddev is correct
    score = model([x, idx_sigmas])
    noise = tf.sqrt(alpha_i * 2) * z_t
    return x + alpha_i * score + noise


def sample_many(model, sigmas, batch_size=128, eps=2 * 1e-5, T=100, n_images=1):
    """
    Used for sampling big amount of images (e.g. 50000)
    :param model: model for sampling (RefineNet)
    :param sigmas: sigma levels of noise
    :param eps:
    :param T: iteration per sigma level
    :return: Tensor of dimensions (n_images, width, height, channels)
    """
    # Tuple for (n_images, width, height, channels)
    image_size = (n_images) + model.in_shape[0][1:]
    batch_size = min(batch_size, n_images)

    with tf.device('CPU'):
        x = tf.random.uniform(shape=image_size)
    x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    x_processed = None

    n_processed_images = 0
    for i_batch, batch in enumerate(tqdm(x, total=tf.data.experimental.cardinality(x).numpy(), desc='Generating samples')):
        for i, sigma_i in enumerate(sigmas):
            alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
            idx_sigmas = tf.ones(batch.get_shape()[0], dtype=tf.int32) * i
            for t in range(T):
                sample_one_step(model, batch, idx_sigmas, alpha_i)

        with tf.device('CPU'):
            if x_processed is not None:
                x_processed = tf.concat([x_processed, batch], axis=0)
            else:
                x_processed = batch

        n_processed_images += batch_size

    return x_processed


def sample_and_save(model, sigmas, image_size, eps=2 * 1e-5, T=100, n_images=1):
    """
    Only for MNIST, for now.
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    image_size = (n_images,) + image_size

    x = tf.random.uniform(shape=image_size)

    for i, sigma_i in enumerate(tqdm(sigmas, desc='Sampling for each sigma')):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        idx_sigmas = tf.ones(n_images, dtype=tf.int32) * i
        for t in tqdm(range(T)):
            x = sample_one_step(model, x, idx_sigmas, alpha_i)

            if (t + 1) % 10 == 0:
                save_as_grid(x, samples_directory + f'sigma{i + 1}_t{t + 1}.png')
    return x


if __name__ == '__main__':
    args = utils.get_command_line_args()
    configs.config_values = args

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    dataset_name = configs.config_values.dataset
    model_directory = './saved_models/'
    dataset = dataset_name + '/'
    filters = 16
    image_size = utils.get_dataset_image_size(configs.config_values.dataset)

    step = tf.Variable(0)
    model = RefineNet(filters=filters, activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    latest_checkpoint = tf.train.latest_checkpoint(model_directory + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(1.0),
                                           tf.math.log(0.01),
                                           10))

    samples_directory = './samples/' + f'{start_time}_{dataset_name}_{step.numpy()}steps_{filters}filters' + "/"  # TODO: add number of steps in name
    os.makedirs(samples_directory)

    samples = sample_and_save(model, sigma_levels, T=100, n_images=400, image_size=image_size)
