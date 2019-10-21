import argparse
import tensorflow as tf
def get_command_line_args():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='mnist',
                        help="tfds name of dataset (default: 'mnist')")
    parser.add_argument('--num_L', default=10, type=int,
                        help="number of levels of noise to use (default: 10)")
    parser.add_argument('--sigma_low', default=0.01, type=float,
                        help="lowest value for noise (default: 0.01)")
    parser.add_argument('--sigma_high', default=1.0, type=float,
                        help="highest value for noise (default: 1.0)")
    parser.add_argument('--steps', default=200000, type=int,
                        help="number of steps to train the model for (default: 200000)")
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