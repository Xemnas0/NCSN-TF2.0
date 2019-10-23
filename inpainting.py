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
from datasets.dataset_loader import get_data_inpainting


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


# def save_as_grid(images, filename, spacing=2):
#     """
#     Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
#     """
#     # images is of shape [ [occluded_x, [sample, sample, sample...], x],
#     #                      [occluded_x, [sample, sample, sample...], x],
#     #                      ...]
#     _, height, width, channels = images[0][0].shape
#     rows = len(images)
#     cols = len(images[0][1]) + 2
#
#     # init image
#     grid_cols = rows * height# + (rows + 1) * spacing
#     grid_rows = cols * width# + (cols + 1) * spacing
#     mode = 'L' if channels == 1 else "RGB"
#     im = Image.new(mode, (grid_rows, grid_cols))
#
#     for i in range(rows):  # i = row, j = column
#         occluded_x, samples, x = images[i]
#
#         # plot the occluded image
#         row_start = 0#i * height# + (1 + i) * spacing
#         col_start = 0#spacing
#         # im.paste(tf.keras.preprocessing.image.array_to_img(occluded_x[0,:,:,:]), (row_start, col_start))
#         # im.save(filename+"_0", format="PNG")
#
#         # plot the samples
#         for j in range(len(samples)):
#             col_start = (j+1) * width# + (1 + j+1) * spacing
#             im.paste(tf.keras.preprocessing.image.array_to_img(samples[j][0,:,:,:]), (row_start, col_start))
#             im.save(filename+f"_{j+1}", format="PNG")
#
#         # plot the original image
#         col_start = len(samples) * width# + (1 + len(samples)+1) * spacing
#         im.paste(tf.keras.preprocessing.image.array_to_img(x[0,:,:,:]), (row_start, col_start))
#         im.save(filename+"_n", format="PNG")
#
#     im.save(filename, format="PNG")


def save_image(image, filename):
    if len(image.shape) == 4:
        image = image[0,:,:,0]

    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(filename)


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
            # if (t + 1) % 10 == 0:
            #     save_as_grid(x_t, samples_directory + f'sigma{i + 1}_t{t + 1}.png')
    return x_t


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # construct path and folder TODO get this from some global function?
    checkpoint_dir = configs.config_values.checkpoint_dir
    dataset = configs.config_values.dataset
    samples_directory = f'./inpainting_results/{dataset}_{start_time}'
    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    # load model from checkpoint TODO also construct the path here using args?
    step = tf.Variable(0)
    model = RefineNet(filters=16, activation=tf.nn.elu) # TODO filters from args
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    # initialise sigmas
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    # TODO add these values to args
    N = 1  # number of images to occlude
    n = 2  # number of samples to generate for each occluded image
    mask_style = 'vertical_split'  # what kind of occlusion to use

    # load data for inpainting (currently always N first data points from test data)
    data = get_data_inpainting(configs.config_values.dataset, N)

    images = []

    for i, x in enumerate(data):
        mask = np.zeros(x.shape)
        if mask_style == 'vertical_split':
            mask[:, :, :x.shape[2]//2, :] += 1  # set left side to ones
        else:
            pass  # TODO add options here

        occluded_x = x * mask

        save_dir = f'{samples_directory}/image_{i}'
        save_image(x, save_dir + '_original')
        save_image(occluded_x, save_dir + '_occluded')

        samples = []
        for j in tqdm(range(n)):
            sample = inpaint_x(model, sigma_levels, mask, x)
            samples.append(sample)
            save_image(sample, save_dir + f'_sample_{j}')

        images.append([occluded_x, samples, x])

    # save_as_grid(images, samples_directory + '/grid.png')
