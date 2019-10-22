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
from datasets.dataset_loader import get_train_test_data
from generate import save_as_grid, sample_one_step
def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)

def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)


# def save_as_grid(images, filename, spacing=2):
#     """
#     Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
#     :param images:
#     :return:
#     """
#     # Define grid dimensions
#     n_images, height, width, channels = images.shape
#     rows = np.floor(np.sqrt(n_images)).astype(int)
#     cols = n_images // rows

#     # Init image
#     grid_cols = rows * height + (rows + 1) * spacing
#     grid_rows = cols * width + (cols + 1) * spacing
#     im = Image.new('L', (grid_rows, grid_cols))
#     for i in range(n_images):
#         row = i // rows
#         col = i % rows
#         row_start = row * height + (1 + row) * spacing
#         col_start = col * width + (1 + col) * spacing
#         im.paste(tf.keras.preprocessing.image.array_to_img(images[i]), (row_start, col_start))
#         # im.show()

#     im.save(filename, format="PNG")


@tf.function
def inpaint_one_step(model, x_t, idx_sigmas, alpha_i):
    z_t = tf.random.normal(shape=x_t.shape, mean=0, stddev=1.0)
    score = model([x_t, idx_sigmas])
    noise = tf.sqrt(alpha_i) * z_t
    return x_t + (alpha_i/2) * score + noise


def inpaint_x(model, sigmas, m, x, eps=2 * 1e-5, T=100):
    # almost the same loop as generation except with mask
    # TODO: merge this with generate? 
    x_t = tf.random.uniform(shape=x.shape)
    x_t = (x_t * (1 - m)) + (x * m)

    for i, sigma_i in enumerate(sigmas):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        z = tf.random.normal(shape=x.shape, mean=0, stddev=sigma_i**2)
        y = x + z
        idx_sigmas = tf.ones(1, dtype=tf.int32) * i
        for t in range(T):
            x_t = inpaint_one_step(model, x_t, idx_sigmas, alpha_i)
            x_t = (x_t * (1 - m)) + (y * m)
            if (t + 1) % 10 == 0:
                save_as_grid(x_t, samples_directory + f'sigma{i + 1}_t{t + 1}.png')
    return x_t


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = utils.get_command_line_args()
    configs.config_values = args

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    model_directory = './saved_models/'
    dataset = 'mnist/'
    samples_directory = './inpainting_results/' + start_time + "/"
    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    step = tf.Variable(0)
    model = RefineNet(filters=16, activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    latest_checkpoint = tf.train.latest_checkpoint(model_directory + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(1.0),
                                           tf.math.log(0.01),
                                           10))

    test_dataset = get_train_test_data('mnist')[1].batch(1) #TODO make this better!
    n = 1
    N = 1
    for i, x in enumerate(test_dataset):
        if i >= n:
            break
        rows = x.shape[1]
        cols = x.shape[2]

        plt.imshow(x[0,:,:,0], cmap=plt.get_cmap("gray"))
        plt.savefig(f"{samples_directory}image_{i}_original")

        m = np.concatenate((np.ones((1,rows,cols//2,1)), np.zeros((1,rows,cols//2,1))), axis=2)
        occluded_x = x*m

        plt.imshow(occluded_x[0,:,:,0], cmap=plt.get_cmap("gray"))
        plt.savefig(f"{samples_directory}image_{i}_masked")

        for j in tqdm(range(N)):
            sample = inpaint_x(model, sigma_levels, m, occluded_x)
            plt.imshow(sample[0,:,:,0], cmap=plt.get_cmap("gray"))
            plt.savefig(f"{samples_directory}image_{i}_sample_{j}")
