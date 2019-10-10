import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='mnist',
                        help="tfds name of dataset. Default: 'mnist'")
    # TODO: add more command line arguments here!
    # possible: whether to use local data, where data is stored, checkpointing

    return parser.parse_args()