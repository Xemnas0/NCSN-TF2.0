import argparse
import tensorflow as tf

from model.refinenet import RefineNet

import configs

dict_datasets_image_size = {
    'mnist': (28, 28, 1),
    'cifar10': (32, 32, 3),
    'celeb_a': (32, 32, 3)
}


def get_dataset_image_size(dataset_name):
    return dict_datasets_image_size[dataset_name]


def get_command_line_args():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='mnist',
                        help="tfds name of dataset (default: 'mnist')")
    parser.add_argument('--baseline', action='store_true',
                        help='Whether different baseline experiment (default: False)')
    parser.add_argument('--filters', default=128, type=int, help='Number of filters in the model. (default: 128)')
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
    parser.add_argument('--log_dir', default='./logs/',
                        help="folder for saving logs (default: ./logs/)")
    parser.add_argument('--checkpoint_dir', default='./saved_models/',
                        help="folder for saving model checkpoints (default: ./saved_models/)")
    parser.add_argument('--checkpoint_freq', default=5000, type=int,
                        help="how often to save a model checkpoint (default: 5000 iterations)")
    parser.add_argument('--resume', action='store_true',
                        help="whether to resume from latest checkpoint (default: False)")

    return parser.parse_args()


def get_tensorflow_device():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
    print(f"Using device {device}")
    return device


def get_savemodel_dir():
    models_dir = configs.config_values.checkpoint_dir
    model_name = 'baseline' if configs.config_values.baseline else 'refinenet'

    # Folder name: model_name+filters+dataset+L
    if not configs.config_values.baseline:
        folder_name = f'{models_dir}{model_name}{configs.config_values.filters}_{configs.config_values.dataset}_L{configs.config_values.num_L}/'
    else:
        folder_name = f'{models_dir}{model_name}{configs.config_values.filters}_{configs.config_values.dataset}/'

    return folder_name


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
            print("Loaded model: "+latest_checkpoint)
            step = tf.Variable(0)
            ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
            ckpt.restore(latest_checkpoint)
            step = int(step)

    if verbose:
        print_model_summary(model)

    return model, optimizer, step
