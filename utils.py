import argparse
import tensorflow as tf
import numpy as np
from model.refinenet import RefineNet
from model.resnet import ResNet
import configs

dict_datasets_image_size = {
    'mnist': (28, 28, 1),
    'cifar10': (32, 32, 3),
    'celeb_a': (32, 32, 3)
}


def find_k_closest(image, k, data_as_array):
    l2_distances = tf.norm(tf.norm(tf.norm(data_as_array - image, axis=3), axis=2), axis=1)
    _, smallest_idx = tf.math.top_k(-l2_distances, k)
    closest_k = tf.split(tf.gather(data_as_array, smallest_idx[:k]), k)
    return closest_k


def get_dataset_image_size(dataset_name):
    return dict_datasets_image_size[dataset_name]


def get_command_line_args():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='mnist',
                        help="tfds name of dataset (default: 'mnist')")
    parser.add_argument('--baseline', action='store_true',
                        help='whether to run baseline experiment with only one sigma (default: False)')
    parser.add_argument('--filters', default=128, type=int,
                        help='number of filters in the model. (default: 128)')
    parser.add_argument('--num_L', default=10, type=int,
                        help="number of levels of noise to use (default: 10)")
    parser.add_argument('--sigma_low', default=0.01, type=float,
                        help="lowest value for noise (default: 0.01)")
    parser.add_argument('--sigma_high', default=1.0, type=float,
                        help="highest value for noise (default: 1.0)")
    parser.add_argument('--steps', default=200000, type=int,
                        help="number of steps to train the model for (default: 200000)")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="learning rate for the optimizer")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="batch size (default: 128)")
    parser.add_argument('--samples_dir', default='./samples/',
                        help="folder for saving samples (default: ./samples/)")
    parser.add_argument('--checkpoint_dir', default='./saved_models/',
                        help="folder for saving model checkpoints (default: ./saved_models/)")
    parser.add_argument('--checkpoint_freq', default=5000, type=int,
                        help="how often to save a model checkpoint (default: 5000 iterations)")
    parser.add_argument('--resume', action='store_true',
                        help="whether to resume from latest checkpoint (default: False)")
    parser.add_argument('--find_nearest', action='store_true',
                        help="whether to find closest k neighbours in training set (default: False)")
    parser.add_argument('--k', default=10, type=int,
                        help='number of nearest neighbours to find from data (default: 10)')
    parser.add_argument('--resnet', action='store_true',
                        help='whether to run the experiment with ResNet architecture (default: False)')

    parser = parser.parse_args()
    print("=" * 20 + "\nParameters: \n")
    for key in parser.__dict__:
        print(key + ': ' + str(parser.__dict__[key]))
    print("=" * 20 + "\n")
    return parser


def get_tensorflow_device():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
    print("Using device {}".format(device))
    return device


def get_savemodel_dir():
    models_dir = configs.config_values.checkpoint_dir
    if configs.config_values.baseline:
        model_name = 'baseline'
    elif configs.config_values.resnet:
        model_name = 'resnet'
    else:
        model_name = 'refinenet'

    # Folder name: model_name+filters+dataset+L
    if not configs.config_values.baseline:
        complete_model_name = '{}{}_{}_L{}'.format(model_name, configs.config_values.filters,
                                                   configs.config_values.dataset, configs.config_values.num_L)
    else:
        complete_model_name = '{}{}_{}'.format(model_name, configs.config_values.filters, configs.config_values.dataset)
    folder_name = models_dir + complete_model_name + '/'
    return folder_name, complete_model_name


def print_model_summary(model):
    batch = 2
    input_shape = (batch,) + get_dataset_image_size(configs.config_values.dataset)
    x = [tf.ones(shape=input_shape), tf.ones(batch, dtype=tf.int32)]
    model(x)
    print(model.summary())


def try_load_model(save_dir, verbose=True):
    """
    Tries to load a model from the provided directory, otherwise returns a new initialized model.
    :param save_dir: directory with checkpoints
    :param verbose: true for printing the model summary
    :return:
    """
    # initialize return values
    if configs.config_values.resnet:
        model = ResNet(filters=configs.config_values.filters, activation=tf.nn.elu)
    else:
        model = RefineNet(filters=configs.config_values.filters, activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.config_values.learning_rate)
    step = 0

    # if resuming training, overwrite model parameters from checkpoint
    if configs.config_values.resume:
        print("Trying to load a model from " + save_dir)
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint is None:
            print("No model found. Using a new model")
        else:
            print("Loaded model: " + latest_checkpoint)
            step = tf.Variable(0)
            ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
            ckpt.restore(latest_checkpoint)
            step = int(step)

    if verbose:
        print_model_summary(model)

    return model, optimizer, step


def manage_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)