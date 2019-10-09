import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='cifar10',
                        help="tfds name of dataset. Default: 'cifar10'")
    # TODO: add more command line arguments here!
    # possible: whether to use local data, where data is stored, checkpointing

    return parser.parse_args()