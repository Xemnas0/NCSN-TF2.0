import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils
import configs
from datasets.dataset_loader import get_data_generator
from model.refinenet import RefineNet

if __name__ == "__main__":
    # args = utils.get_command_line_args()
    # configs.config_values = args
    #
    # # loading dataset
    # train_data = get_data_generator(args.dataset)
    #
    # # TRAINING LOOP CAN GO HERE
    # epochs = 3
    #
    # for epoch in range(epochs):
    #     print("epoch", epoch)
    #
    #     for i, data_batch in enumerate(train_data):
    #         pass

    # TEST
    import tensorflow_datasets as tfds

    args = utils.get_command_line_args()
    configs.config_values = args

    data_generators = tfds.load(name="cifar10", split="test", batch_size=-1)
    test = tf.cast(data_generators['image'], tf.float32)

    x = test[:3]
    idx_sigmas = tf.convert_to_tensor([3, 9, 3])
    model = RefineNet(5, tf.nn.elu)
    output = model([x, idx_sigmas])
