import argparse

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
    parser.add_argument('--epochs', default=300, type=int,
                        help="number of epochs to train the model for (default: 300)")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="batch size (default: 128)")
    parser.add_argument('--checkpoint_dir', default='./saved_models/',
                        help="folder for saving model checkpoints")
    parser.add_argument('--checkpoint_freq', default=10,
                        help="how often to save a model checkpoint (default: 10 epochs)")

    return parser.parse_args()