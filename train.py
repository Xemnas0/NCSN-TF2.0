import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--dataset', default='cifar10',
                        help="Name of dataset to train the model: choose from \
                          'mnist', 'cifar10', 'celeb_a'. Default is 'cifar10'.")
    # TODO: add more command line arguments here!

    args = parser.parse_args()

    if args.dataset not in ['mnist', 'cifar10', 'celeb_a']:
        raise SystemExit("Error: dataset is not one of the supported options \
            ('mnist', 'cifar10' or 'celeb_a').")
    
    # if we get to this point then everything was ok

    # load the data
    data_generators = tfds.load(name=args.dataset)
    data = data_generators['train'].concatenate(data_generators['test'])