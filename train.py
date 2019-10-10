import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils
import configs
from datasets.dataset_loader import get_data_generator

if __name__ == "__main__":
    args = utils.get_command_line_args()
    configs.config_values = args

    # loading dataset
    train_data = get_data_generator(args.dataset)

    # TRAINING LOOP CAN GO HERE
    epochs = 3

    for epoch in range(epochs):
        print("epoch", epoch)

        for i, data_batch in enumerate(train_data):
            pass
